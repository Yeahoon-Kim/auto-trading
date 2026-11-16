"""리서치팀(펀더멘털): 기업 스냅샷 — 가격·밸류에이션·재무 추이·공시·내부자 거래를 한 번에 출력.

fundamental-analyst 에이전트가 숫자를 손으로 쿼리하다 틀리지 않도록, 로컬 데이터만으로
사실을 모아 주는 도구다. 해석·전망은 하지 않는다.

    uv run python -m autotrader.intel.company 005930
    uv run python -m autotrader.intel.company AAPL --market US
    uv run python -m autotrader.intel.company 005930 --peers 000660,042700   # 비교 기업 표
    uv run python -m autotrader.intel.company 005930 --asof 2025-06-30       # 그 시점에 알 수 있던 것만

재무 출처: KR은 dart_fin(DART 주요계정, 2015~, 상장사 전체, 매일 갱신), US는 legacy(2021~, 511사, 갱신 안 됨).
시점 규칙 (--asof): DART 재무·공시는 접수일 다음 날부터. legacy 재무는 분기 +45일·연간 +90일 추정.
뉴스·섹터 분류는 없다.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from ..data import store


def _d() -> str:
    """데이터 폴더(store.DATA_DIR, AUTOTRADER_DATA_DIR 반영). 실행 위치(cwd)와 무관하게."""
    return store.DATA_DIR.as_posix()


LAG = {"quarterly": 45, "annual": 90}


def _q(sql: str) -> pd.DataFrame:
    return store.query(sql)


def _market(symbol: str) -> str:
    return "KR" if symbol.isdigit() else "US"


def _info(sym: str, mkt: str) -> dict:
    df = _q(f"SELECT name, name_ko, exchange, shares_outstanding FROM '{_d()}/legacy/instruments.parquet' "
            f"WHERE symbol='{sym}' AND market='{mkt}'")
    return df.iloc[0].to_dict() if len(df) else {}


def _prices(sym: str, mkt: str, asof: pd.Timestamp) -> pd.DataFrame:
    """일봉 (ts, close, market_cap). KR은 krx_1d(시가총액 포함), US는 us_1d → 없으면 legacy."""
    if mkt == "KR":
        df = _q(f"SELECT ts, close, market_cap, name FROM '{_d()}/krx_1d/ALL/*.parquet' "
                f"WHERE symbol='{sym}' ORDER BY ts")
    else:
        df = store.read("us_1d", sym)
        if df.empty:
            df = _q(f"SELECT trade_date AS ts, close FROM '{_d()}/legacy/daily_prices_us.parquet' "
                    f"WHERE symbol='{sym}' ORDER BY trade_date")
        df = df[["ts", "close"]].copy()
        shares = _info(sym, mkt).get("shares_outstanding")
        df["market_cap"] = df["close"] * shares if shares and pd.notna(shares) else float("nan")
    if df.empty:
        return df
    # ts는 현지 자정을 UTC로 저장한 값(KRX: 전날 15:00 UTC) → 현지 날짜로 바꿔서 asof와 비교
    tz = "Asia/Seoul" if mkt == "KR" else "America/New_York"
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    return df[df["ts"] <= asof].reset_index(drop=True)


def _fundamentals(sym: str, mkt: str, asof: pd.Timestamp) -> pd.DataFrame:
    """KR은 DART 주요계정(dart_fin, 실제 접수일 기준), 없으면 legacy."""
    if mkt == "KR":
        f = _fund_dart(sym, asof)
        if not f.empty:
            return f
    return _fund_legacy(sym, mkt, asof)


FLOW = ("revenue", "operating_income", "net_income")
BS = ("total_assets", "total_liabilities", "total_equity")


def _fund_dart(sym: str, asof: pd.Timestamp) -> pd.DataFrame:
    if not any((store.DATA_DIR / "dart_fin" / "ALL").glob("*.parquet")):  # 아직 수집 전
        return pd.DataFrame()
    d = _q(f"SELECT * FROM '{_d()}/dart_fin/ALL/*.parquet' WHERE symbol='{sym}'")
    if d.empty:
        return d
    local = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)  # noqa: E731
    d["effective_date"], d["known_from"] = local(d["ts"]).dt.normalize(), local(d["filed"]) + pd.Timedelta(days=1)
    d = d[d["known_from"] <= asof].sort_values("effective_date")
    rows = []
    for y, g in d.groupby("bsns_year"):
        by = {r["period"]: r for _, r in g.iterrows()}
        for p, lab in (("Q1", "Q1"), ("H1", "Q2"), ("Q3", "Q3")):
            if p in by:
                r = by[p]
                rows.append({"period_label": f"{y}{lab}", "period_type": "quarterly", "flag": "",
                             **{k: r[k] for k in ("effective_date", "known_from", *FLOW, *BS) if k in r}})
        if "FY" in by:
            a = by["FY"]
            rows.append({"period_label": str(y), "period_type": "annual", "flag": "",
                         **{k: a[k] for k in ("effective_date", "known_from", *FLOW, *BS) if k in a}})
            if "Q3" in by:  # 4분기 보고서는 없다 → 연간 − 3분기 누적 (공개 시점은 사업보고서 접수일)
                q3 = by["Q3"]
                rows.append({"period_label": f"{y}Q4*", "period_type": "quarterly", "flag": "FY−Q3누적",
                             "effective_date": a["effective_date"], "known_from": a["known_from"],
                             **{k: a[k] - q3[f"{k}_cum"] for k in FLOW}, **{k: a[k] for k in BS if k in a}})
    f = pd.DataFrame(rows)
    f["eps"] = float("nan")
    f["source"] = "DART"
    return f.sort_values(["period_type", "effective_date"]).reset_index(drop=True)


def _fund_legacy(sym: str, mkt: str, asof: pd.Timestamp) -> pd.DataFrame:
    df = _q(f"SELECT period_label, period_type, effective_date, revenue, operating_income, net_income, eps "
            f"FROM '{_d()}/legacy/fundamentals.parquet' WHERE symbol='{sym}' AND market='{mkt}'")
    if df.empty:
        return df
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df["known_from"] = df["effective_date"] + pd.to_timedelta(df["period_type"].map(LAG).fillna(90), unit="D")
    df = df[df["known_from"] <= asof]
    df["flag"] = ""
    ann, q = df[df["period_type"] == "annual"], df[df["period_type"] == "quarterly"]
    # legacy US: 같은 기간이 라벨만 바꿔 중복 수록되고, 9개월 누적값이 분기 행으로 섞여 있다 → 기간당 최솟값 1행
    q = q.sort_values("revenue").drop_duplicates("effective_date", keep="first")
    # legacy KR: 4분기 행이 없다 → 연간 − 같은 회계연도 3개 분기로 계산
    extra = []
    for _, a_ in ann.iterrows():
        d = a_["effective_date"]
        same = q[(q["effective_date"] > d - pd.Timedelta(days=360)) & (q["effective_date"] < d - pd.Timedelta(days=20))]
        if len(same) == 3 and not ((q["effective_date"] - d).abs() < pd.Timedelta(days=20)).any():
            r = a_.copy()
            r["period_type"], r["period_label"], r["flag"] = "quarterly", f"{a_['period_label']}Q4*", "연간−3분기"
            for c in ("revenue", "operating_income", "net_income", "eps"):
                r[c] = a_[c] - same[c].sum()
            extra.append(r)
    q = pd.concat([q, pd.DataFrame(extra)], ignore_index=True) if extra else q
    # 급변 표시: 분기 매출이 다른 분기 중앙값의 1.8배 이상 (실제 급성장일 수도 있으니 제외하지 않고 원문 확인)
    q.loc[q["revenue"] > 1.8 * q["revenue"].median(), "flag"] = (q["flag"] + " 급변·원문확인").str.strip()
    out = pd.concat([ann, q]).sort_values(["period_type", "effective_date"]).reset_index(drop=True)
    out["source"] = "legacy(공개일 추정)"
    return out


def _ttm(f: pd.DataFrame) -> dict:
    """최근 4개 분기 합 (4개가 약 1년에 걸쳐 이어질 때만). 아니면 최근 연간."""
    q = f[f["period_type"] == "quarterly"].sort_values("effective_date").tail(4)
    span = (q["effective_date"].iloc[-1] - q["effective_date"].iloc[0]).days if len(q) else 0
    if len(q) == 4 and 240 <= span <= 300:
        return {"basis": f"TTM({q['period_label'].iloc[0]}~{q['period_label'].iloc[-1]})",
                **{c: q[c].sum() for c in ("revenue", "operating_income", "net_income")}}
    a = f[(f["period_type"] == "annual") & f["revenue"].notna()].tail(1)
    if len(a):
        return {"basis": f"연간 {a['period_label'].iloc[0]}",
                **{c: a[c].iloc[0] for c in ("revenue", "operating_income", "net_income")}}
    return {}


def _ret(p: pd.DataFrame, days: int) -> float:
    if p.empty:
        return float("nan")
    last = p["ts"].iloc[-1]
    past = p[p["ts"] <= last - pd.Timedelta(days=days)]
    return p["close"].iloc[-1] / past["close"].iloc[-1] - 1 if len(past) else float("nan")


def _disclosures(sym: str, mkt: str, asof: pd.Timestamp, days: int) -> pd.DataFrame:
    start = (asof - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end = (asof - pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # 접수일 다음 날부터 알 수 있음
    leg = _q(f"SELECT CAST(filed_at AS DATE) AS date, category, title FROM '{_d()}/legacy/disclosures.parquet' "
             f"WHERE symbol='{sym}' AND market='{mkt}' AND category <> 'insider' "
             f"AND filed_at BETWEEN '{start}' AND '{end}'")
    if mkt == "KR":
        new = _q(f"SELECT ts, '' AS category, report_nm AS title "
                 f"FROM '{_d()}/dart_disclosures/ALL/*.parquet' WHERE symbol='{sym}'")
        # ts = KST 접수일 자정의 UTC → KST 날짜로
        new["date"] = pd.to_datetime(new["ts"], utc=True).dt.tz_convert("Asia/Seoul").dt.date
        new = new[(new["date"] >= pd.Timestamp(start).date()) & (new["date"] <= pd.Timestamp(end).date())]
        leg = pd.concat([leg, new.drop(columns="ts")], ignore_index=True)
    if leg.empty:
        return leg
    leg["title"] = leg["title"].str.strip()
    return leg.sort_values("category").drop_duplicates(["date", "title"], keep="last").sort_values("date", ascending=False)


def _insiders(sym: str, mkt: str, asof: pd.Timestamp, days: int) -> pd.DataFrame:
    start = (asof - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    return _q(f"SELECT transaction, count(*) AS n, sum(shares) AS shares FROM '{_d()}/legacy/insider_events.parquet' "
              f"WHERE symbol='{sym}' AND market='{mkt}' AND event_date BETWEEN '{start}' AND '{asof:%Y-%m-%d}' "
              f"GROUP BY 1 ORDER BY n DESC")


def summary_row(sym: str, mkt: str, asof: pd.Timestamp) -> dict:
    p, f = _prices(sym, mkt, asof), _fundamentals(sym, mkt, asof)
    info, t = _info(sym, mkt), (_ttm(f) if len(f) else {})
    cap = p["market_cap"].iloc[-1] if len(p) else float("nan")
    rev, oi, ni = (t.get(k, float("nan")) for k in ("revenue", "operating_income", "net_income"))
    bs = f.dropna(subset=["total_equity"]).sort_values("effective_date").tail(1) if "total_equity" in f else f.head(0)
    eq = bs["total_equity"].iloc[0] if len(bs) else float("nan")
    liab = bs["total_liabilities"].iloc[0] if len(bs) else float("nan")
    name = info.get("name_ko") or info.get("name") or (p["name"].iloc[-1] if "name" in p and len(p) else "?")
    return {
        "symbol": sym, "name": name,
        "price_date": p["ts"].iloc[-1].strftime("%Y-%m-%d") if len(p) else "-",
        "close": p["close"].iloc[-1] if len(p) else float("nan"),
        "mcap": cap, "ret_3m": _ret(p, 91), "ret_1y": _ret(p, 365),
        "basis": t.get("basis", "재무 없음"), "revenue": rev,
        "op_margin": oi / rev if rev else float("nan"), "net_margin": ni / rev if rev else float("nan"),
        "PER": cap / ni if ni and ni > 0 else float("nan"), "PSR": cap / rev if rev else float("nan"),
        "PBR": cap / eq if eq and eq > 0 else float("nan"), "debt_ratio": liab / eq if eq and eq > 0 else float("nan"),
        "fin_source": f["source"].iloc[0] if len(f) else "-",
    }


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:,.3g}" if abs(x) < 10 else f"{x:,.0f}")


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # pandas×numpy timedelta 경고
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="기업 스냅샷 (로컬 데이터만, 해석 없음)")
    ap.add_argument("symbol")
    ap.add_argument("--market", choices=["KR", "US"])
    ap.add_argument("--peers", default="", help="비교 기업, 쉼표 구분")
    ap.add_argument("--asof", default=None, help="이 날짜에 알 수 있던 정보만 (기본: 오늘)")
    ap.add_argument("--days", type=int, default=180, help="공시·내부자 조회 기간")
    a = ap.parse_args()

    sym = a.symbol.upper()
    mkt = a.market or _market(sym)
    asof = pd.Timestamp(a.asof or pd.Timestamp.now()).normalize()  # 현지 날짜 (시간대 없음)

    print(f"# {sym} ({mkt}) 스냅샷 — 기준 {asof:%Y-%m-%d}\n")
    row = summary_row(sym, mkt, asof)
    print(_fmt(pd.DataFrame([row]).T.reset_index().rename(columns={"index": "항목", 0: "값"})), "\n")

    f = _fundamentals(sym, mkt, asof)
    src = f["source"].iloc[0] if len(f) else "-"
    print(f"## 재무 추이 (출처 {src}, 원/달러, 손익은 분기 3개월 값. effective_date=기간 종료일, "
          "known_from=알 수 있게 된 날(DART: 접수일 다음 날, legacy: 분기+45·연간+90일 추정), Q4*=연간−3분기)")
    if f.empty:
        print("재무제표 없음 — DART/SEC 원문이나 웹 자료로 보충할 것\n")
    else:
        cols = ["period_label", "period_type", "effective_date", "known_from", "revenue", "operating_income",
                "net_income", *[c for c in ("total_equity", "eps") if c in f and f[c].notna().any()], "flag"]
        out = f[cols].copy()
        for c in ("effective_date", "known_from"):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        print(_fmt(out.groupby("period_type").tail(12)), "\n")

    d = _disclosures(sym, mkt, asof, a.days)
    print(f"## 공시 최근 {a.days}일 ({len(d)}건, 최대 40건 표시)")
    print(_fmt(d.head(40)) if len(d) else "없음", "\n")

    ins = _insiders(sym, mkt, asof, a.days)
    print(f"## 내부자 거래 최근 {a.days}일 (거래 유형별)")
    print(_fmt(ins) if len(ins) else "없음 (legacy 수록 범위 밖일 수 있음)", "\n")

    if a.peers:
        peers = [sym] + [s.strip().upper() for s in a.peers.split(",") if s.strip()]
        print("## 비교 기업")
        print(_fmt(pd.DataFrame([summary_row(s, a.market or _market(s), asof) for s in peers])))


if __name__ == "__main__":
    main()
