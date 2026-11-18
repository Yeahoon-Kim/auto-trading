"""US 시가총액 verifier — shares.daily_shares() 결과가 상식 범위인지 자동 점검 (읽기 전용).

점검 항목 (각각 통과 기준이 있다):
  split_jump   분할마다, 분할 전 마지막 채택 공시와 분할 후 첫 채택 공시의 주식수(오늘 기준) 비율이 0.8~1.25배인가.
               (보정이 틀리면 분할 비율만큼 튄다.) 기준: 실패 비율 5% 이하.
  jump         종목별, 연속한 공시 사이 주식수가 1.5배 넘게/0.67배 밑으로 바뀐 비율. 기준: 전체 2% 이하.
  unit_outlier 종목별, 전 기간 중앙값 대비 10배 넘게/0.1배 밑인 날 비율. 기준: 0.
  rank         알려진 날짜의 시총 상위 10(cik로 묶음)에 반드시 있어야 할 종목이 다 있는가. 기준: 전부.
  absurd       시총 8조 달러 초과 또는 1천만 달러 미만인 날. 기준: 0.
  coverage     us_1d 봉 중 주식수가 있는 비율 (2010~). 기준: 큰 종목(BIG) 각각 90% 이상.
  multi_class  같은 cik 종목끼리 시총 차이(클래스별 가격 차이만큼만 달라야 함). 보고만.
naive_daily(): 전략부 L002가 쓴 방식(shares_outstanding 최신 공시값 그대로 × 종가) — 전후 비교용.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import shares as sh

# 날짜 → 그날 시총 상위 10에 반드시 있어야 할 종목 (널리 알려진 순위, 회사 단위)
RANK_SAMPLES = {
    "2012-06-29": ["AAPL", "XOM", "MSFT"],
    "2015-06-30": ["AAPL", "XOM", "MSFT", "GOOGL"],
    "2018-06-29": ["AAPL", "AMZN", "MSFT", "GOOGL"],
    "2020-12-31": ["AAPL", "MSFT", "AMZN", "GOOGL"],
    "2023-06-30": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "2025-06-30": ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META"],
}
BIG = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "XOM", "V", "JPM", "JNJ", "WMT", "PG",
       "AVGO", "DIS", "MA", "UNH", "HD", "LLY", "ORCL", "TSLA"]


def naive_daily(facts: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """L002 방식 재현: field=shares_outstanding, 공시일 다음 날부터 최신 공시의 tag_rank 최소 값, 분할 보정 없음."""
    f = facts[(facts["field"] == "shares_outstanding") & (facts["unit"] == "shares")]
    f = f.sort_values(["filed", "tag_rank", "end"], ascending=[True, True, False]).drop_duplicates(["symbol", "accn"])
    e = pd.DataFrame({"symbol": f["symbol"], "day": sh._day(f["filed"]) + pd.Timedelta(days=1),
                      "shares": f["val"].astype(float), "cik": f["cik"]}).sort_values("day")
    b = bars.assign(day=sh._day(bars["ts"])).sort_values("day")
    out = pd.merge_asof(b, e, on="day", by="symbol", direction="backward")
    out["market_cap"] = out["close"] * out["shares"]
    return out.drop(columns="day").sort_values(["symbol", "ts"]).reset_index(drop=True)


def _event_series(daily: pd.DataFrame) -> pd.DataFrame:
    """일별 → 주식수가 바뀐 지점만 (종목, ts, shares)."""
    d = daily.dropna(subset=["shares"])
    chg = d.groupby("symbol")["shares"].diff().fillna(1) != 0
    return d[chg][["symbol", "ts", "shares"]]


def check_split_jump(daily: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """분할마다 분할 전후 주식수 비율 (분할 전 마지막 값 vs 분할 뒤 180일 안에 처음 바뀐 값).
    기대값: 실제 분할 1배, 분사형 배수 1/ratio배. 0.8~1.25배 안이면 통과. splits에 true 열(shares.classify_splits)이
    없으면 비율 모양(is_standard)으로 정한다."""
    sp = splits if "true" in splits else sh.classify_splits(sh.load_splits(splits), pd.DataFrame())
    ev = _event_series(daily)
    ev = ev.assign(day=sh._day(ev["ts"]))
    rows = []
    for _, s in sp.iterrows():
        e = ev[ev["symbol"] == s["symbol"]]
        before = e[e["day"] < s["day"]]
        after = e[(e["day"] > s["day"]) & (e["day"] <= s["day"] + pd.Timedelta(days=180))]
        if before.empty or after.empty:
            continue
        r = after["shares"].iloc[0] / before["shares"].iloc[-1]
        exp = 1.0 if s["true"] else 1 / s["ratio"]  # 분사형: 오늘 기준 주식수가 분사일에 1/ratio배가 정상
        rows.append({"symbol": s["symbol"], "day": s["day"], "split": s["ratio"], "standard": s["true"], "ratio": r,
                     "ok": 0.8 <= r / exp <= 1.25})
    return pd.DataFrame(rows, columns=["symbol", "day", "split", "standard", "ratio", "ok"])


def check_jump(daily: pd.DataFrame) -> pd.DataFrame:
    ev = _event_series(daily)
    r = ev.groupby("symbol")["shares"].transform(lambda s: s / s.shift())
    ev = ev.assign(jump=(r > 1.5) | (r < 1 / 1.5))
    return ev.groupby("symbol").agg(changes=("shares", "size"), jumps=("jump", "sum")).reset_index()


def check_rank(daily: pd.DataFrame, samples: dict = RANK_SAMPLES) -> pd.DataFrame:
    rows = []
    d = daily.assign(day=sh._day(daily["ts"]))
    for date, must in samples.items():
        t = pd.Timestamp(date)
        x = d[(d["day"] <= t) & (d["day"] > t - pd.Timedelta(days=7))].dropna(subset=["market_cap"])
        x = x.sort_values("day").drop_duplicates("symbol", keep="last")
        key = x["cik"].fillna(-1).astype("int64").astype(str).where(x["cik"].notna(), x["symbol"]) \
            if "cik" in x else x["symbol"]
        x = x.assign(_k=key).sort_values("market_cap", ascending=False).drop_duplicates("_k")
        top = x.head(10)
        names = list(top["symbol"])
        # GOOG·GOOGL은 같은 회사 — 어느 쪽이 대표로 잡혀도 통과
        alias = {"GOOG": "GOOGL"}
        got = {alias.get(s, s) for s in names}
        miss = [m for m in must if m not in got]
        rows.append({"date": date, "ok": not miss, "missing": ",".join(miss),
                     "top10": " ".join(f"{s}:{v / 1e9:,.0f}" for s, v in zip(top["symbol"], top["market_cap"]))})
    return pd.DataFrame(rows)


def check_coverage(daily: pd.DataFrame, start: str = "2010-01-01") -> pd.DataFrame:
    d = daily[daily["ts"] >= pd.Timestamp(start, tz="UTC")]
    med = d.groupby("symbol")["shares"].transform("median")
    far = ((d["shares"] / med) > 10) | ((d["shares"] / med) < 0.1)
    mc = d["market_cap"]
    absurd = (mc > 8e12) | (mc < 1e7)
    return pd.DataFrame({"symbol": d["symbol"], "has": d["shares"].notna(), "far": far, "absurd": absurd}) \
        .groupby("symbol").agg(bars=("has", "size"), coverage=("has", "mean"), unit_outlier=("far", "mean"),
                               absurd=("absurd", "sum")).reset_index()


def check_multi_class(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.dropna(subset=["market_cap", "cik"])
    n = d.groupby("cik")["symbol"].nunique()
    multi = n[n > 1].index
    if not len(multi):
        return pd.DataFrame(columns=["cik", "symbols", "median_gap"])
    x = d[d["cik"].isin(multi)].pivot_table(index=["cik", "ts"], columns="symbol", values="market_cap")
    rows = []
    for c in multi:
        m = x.loc[c].dropna(axis=1, how="all").dropna()
        if m.shape[1] < 2 or m.empty:
            continue
        gap = (m.max(axis=1) / m.min(axis=1) - 1).median()
        rows.append({"cik": int(c), "symbols": ",".join(m.columns), "median_gap": gap})
    return pd.DataFrame(rows)


def run_all(daily: pd.DataFrame, splits: pd.DataFrame | None) -> dict:
    """전체 점검 → {항목: 표} + 'summary'(항목별 수치·통과 여부)."""
    splits = sh.splits_from_attrs(daily.attrs.get("splits")) if daily.attrs.get("splits") else splits
    sj = check_split_jump(daily, splits) if splits is not None and len(splits) else pd.DataFrame(columns=["ok"])
    jp, rk, cv, mc = check_jump(daily), check_rank(daily), check_coverage(daily), check_multi_class(daily)
    big = cv[cv["symbol"].isin(BIG)]
    summ = [
        ("split_jump 실패 비율", 1 - sj["ok"].mean() if len(sj) else np.nan, "≤5%",
         bool(len(sj) and 1 - sj["ok"].mean() <= 0.05)),
        ("jump 비율(주식수 변경 중 1.5배↑)", jp["jumps"].sum() / max(jp["changes"].sum(), 1), "≤2%",
         jp["jumps"].sum() / max(jp["changes"].sum(), 1) <= 0.02),
        ("unit_outlier 종목 수(10배↑ 날 있음)", int((cv["unit_outlier"] > 0).sum()), "0",
         int((cv["unit_outlier"] > 0).sum()) == 0),
        ("rank 표본 통과", f"{int(rk['ok'].sum())}/{len(rk)}", "전부", bool(rk["ok"].all())),
        ("absurd 날 수", int(cv["absurd"].sum()), "0", int(cv["absurd"].sum()) == 0),
        ("전체 coverage(봉 가중)", (cv["coverage"] * cv["bars"]).sum() / max(cv["bars"].sum(), 1), "보고", True),
        ("BIG 종목 중 coverage<90%", ",".join(big[big["coverage"] < 0.9]["symbol"]) or "-", "없음",
         bool((big["coverage"] >= 0.9).all())),
    ]
    summary = pd.DataFrame(summ, columns=["check", "value", "pass_rule", "ok"])
    return {"summary": summary, "split_jump": sj, "jump": jp, "rank": rk, "coverage": cv, "multi_class": mc}


def main(argv: list[str] | None = None) -> int:
    """uv run python -m autotrader.data.shares_check [--out 폴더] [--splits p] [--class-shares p] [--extra-facts p]
    저장소(AUTOTRADER_DATA_DIR 반영)의 us_1d·edgar_facts·edgar_class_shares·us_splits로 계산해 점검한다.
    --splits 등은 병합 전 분석용(저장소 대신/추가로 쓸 parquet). 요약은 표준출력, 종목별 표는 --out에 CSV.
    반환 코드: 통과 기준을 못 넘은 항목 수."""
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(prog="autotrader.data.shares_check")
    p.add_argument("--out", default=None)
    p.add_argument("--splits")
    p.add_argument("--class-shares")
    p.add_argument("--extra-facts")
    a = p.parse_args(argv)
    sp = pd.read_parquet(a.splits) if a.splits else None
    cls = pd.read_parquet(a.class_shares) if a.class_shares else None
    bars = sh._bars(None)
    facts = None
    if a.extra_facts:
        from . import store
        folder = (store.DATA_DIR / "edgar_facts" / "ALL").as_posix()
        facts = pd.concat([store.query(f"SELECT * FROM '{folder}/*.parquet' WHERE field LIKE 'shares%'"),
                           pd.read_parquet(a.extra_facts)], ignore_index=True)
    ev = sh.share_events(facts=facts, class_shares=cls, splits=sp, bars=bars)
    daily = sh.daily_shares(events=ev, bars=bars)
    r = run_all(daily, sp if sp is not None else sh.store.read("us_splits", "ALL"))
    pd.set_option("display.width", 200)
    print(r["summary"].to_string(index=False))
    print(r["rank"].to_string(index=False))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        sh.quality(ev, daily).merge(r["coverage"][["symbol", "absurd"]], on="symbol", how="left") \
            .to_csv(out / "us_shares_quality.csv", index=False)
        for k in ("split_jump", "jump", "rank", "multi_class", "summary"):
            r[k].to_csv(out / f"us_mcap_{k}.csv", index=False)
        print(f"종목별 표: {out}")
    return int((~r["summary"]["ok"].astype(bool)).sum())


if __name__ == "__main__":
    raise SystemExit(main())
