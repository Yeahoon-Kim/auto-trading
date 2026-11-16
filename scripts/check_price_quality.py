"""주가 일봉 데이터 품질 점검 (결측치·이상치). 읽기 전용 — 데이터는 고치지 않는다.

대상: krx_1d(KRX 원본 일봉) + kr_adj_factor(KIS 수정계수) + kr_1d(KIS 원본 일봉, 교차검증), us_1d.
실행: uv run python scripts/check_price_quality.py [--out results/data_quality]
결과: 요약은 표준출력, 사례 목록은 --out 폴더에 CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from autotrader.data import store

ROOT = Path(__file__).resolve().parents[1]
D = store.DATA_DIR.as_posix()  # AUTOTRADER_DATA_DIR 반영
KST = "Asia/Seoul"
LIMIT_CHANGE = pd.Timestamp("2015-06-15")   # 가격제한폭 ±15% → ±30%
DELIST_TAIL = 15                            # 상장폐지 직전 정리매매로 보는 거래일 수


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET TimeZone='UTC'")
    return c


def sym_from_path(s: pd.Series) -> pd.Series:
    return s.str.replace("\\", "/", regex=False).str.split("/").str[-2]


def local_date(ts: pd.Series, tz: str) -> pd.Series:
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()


def section(title: str) -> None:
    print(f"\n## {title}")


def load_krx(c) -> pd.DataFrame:
    df = c.sql(f"SELECT ts, symbol, market, open, high, low, close, volume, value, market_cap, shares "
               f"FROM '{D}/krx_1d/ALL/*.parquet'").df()
    df["date"] = local_date(df["ts"], KST)
    df["symbol"] = df["symbol"].astype("category")
    return df.drop(columns="ts").sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def load_factor(c) -> pd.DataFrame:
    f = c.sql(f"SELECT ts, factor, filename FROM read_parquet('{D}/kr_adj_factor/*/*.parquet', filename=true)").df()
    f["symbol"] = sym_from_path(f["filename"])
    f["date"] = local_date(f["ts"], KST)
    return f[["symbol", "date", "factor"]]


def load_kis(c) -> pd.DataFrame:
    k = c.sql(f"SELECT ts, close, volume, filename FROM read_parquet('{D}/kr_1d/*/*.parquet', filename=true)").df()
    k["symbol"] = sym_from_path(k["filename"])
    k["date"] = local_date(k["ts"], KST)
    return k[["symbol", "date", "close", "volume"]]


def check_kr(out: Path) -> None:
    c = con()
    px = load_krx(c)
    g = px.groupby("symbol", observed=True)
    print(f"krx_1d: {len(px):,}행, {px['symbol'].nunique():,}종목, {px['date'].min():%Y-%m-%d} ~ {px['date'].max():%Y-%m-%d}, "
          f"{px['date'].nunique():,}거래일")

    # ── 1. 달력 ──
    section("1. 거래일 달력")
    days = pd.Series(sorted(px["date"].unique()))
    kis_days = set(local_date(c.sql(f"SELECT DISTINCT ts FROM '{D}/kr_1d/005930/*.parquet'").df()["ts"], KST))
    krx_days = set(days)
    miss_krx = sorted(d for d in kis_days if d not in krx_days and days.min() <= d <= days.max())
    miss_kis = sorted(d for d in krx_days if d not in kis_days)
    print(f"KIS(005930)에는 있고 KRX에는 없는 거래일: {len(miss_krx)} {[f'{d:%Y-%m-%d}' for d in miss_krx[:10]]}")
    print(f"KRX에는 있고 KIS(005930)에는 없는 거래일: {len(miss_kis)} {[f'{d:%Y-%m-%d}' for d in miss_kis[:10]]}")
    n_day = px.groupby("date").size()
    med = n_day.rolling(21, center=True, min_periods=5).median()
    thin = n_day[n_day < 0.9 * med]
    print(f"종목 수가 주변 중앙값의 90% 미만인 날: {len(thin)} {thin.head(10).to_dict()}")
    gaps = days.diff().dt.days
    print(f"거래일 간격 최대: {int(gaps.max())}일 ({days[gaps.idxmax() - 1]:%Y-%m-%d} → {days[gaps.idxmax()]:%Y-%m-%d})")

    # ── 2. 중복·결측 ──
    section("2. 중복·결측값")
    print(f"(date, symbol) 중복: {int(px.duplicated(['date', 'symbol']).sum())}")
    print("NULL 개수:", {k: int(v) for k, v in px.isna().sum().items() if v})
    # 상장 기간 중 빠진 날 (첫 거래일~마지막 거래일 사이에 행이 없는 거래일)
    day_idx = pd.Series(np.arange(len(days)), index=days)
    px["di"] = px["date"].map(day_idx).astype(int)
    span = g["di"].agg(["min", "max", "count"])
    span["holes"] = span["max"] - span["min"] + 1 - span["count"]
    holes = span[span["holes"] > 0].sort_values("holes", ascending=False)
    print(f"상장 기간 중 행이 빠진 종목: {len(holes)}종목, 총 {int(holes['holes'].sum())}일 "
          f"(상위: {holes['holes'].head(5).to_dict()})")
    holes.to_csv(out / "kr_missing_days_by_symbol.csv")

    # ── 3. 0·음수·거래 없음 ──
    section("3. 0 가격 / 거래 없음")
    zc = px["close"] <= 0
    zohl = (px[["open", "high", "low"]] <= 0).any(axis=1) & ~zc
    v0 = px["volume"] == 0
    print(f"종가 0 이하: {int(zc.sum()):,}행 ({px.loc[zc, 'symbol'].nunique()}종목)")
    print(f"시·고·저 중 0, 종가는 양수: {int(zohl.sum()):,}행 — 이 중 거래량 0: {int((zohl & v0).sum()):,}")
    print(f"거래량 0: {int(v0.sum()):,}행 ({v0.mean():.2%}) — 거래정지/무거래일. 이때 종가는 직전 종가 복사")
    v0_move = v0 & (px["close"] != g["close"].shift()) & ~zc
    print(f"거래량 0인데 종가가 직전과 다름: {int(v0_move.sum()):,}행 (기준가 조정일 가능성)")

    # ── 4. OHLC 관계 ──
    section("4. 시고저종 관계 (거래 있는 날)")
    t = px[(px["volume"] > 0) & (px[["open", "high", "low", "close"]] > 0).all(axis=1)]
    bad_hl = t["high"] < t["low"]
    bad_h = t["high"] < t[["open", "close"]].max(axis=1)
    bad_l = t["low"] > t[["open", "close"]].min(axis=1)
    vwap = t["value"] / t["volume"]
    bad_v = (vwap < t["low"] * 0.98) | (vwap > t["high"] * 1.02)
    print(f"고가<저가 {int(bad_hl.sum())}, 고가<max(시,종) {int(bad_h.sum())}, 저가>min(시,종) {int(bad_l.sum())}, "
          f"거래대금/거래량이 고저 범위 밖(±2%) {int(bad_v.sum())}")
    t[bad_hl | bad_h | bad_l | bad_v].assign(vwap=vwap).to_csv(out / "kr_ohlc_violations.csv", index=False)

    # ── 5. 시총 = 종가 × 상장주식수 ──
    section("5. 시가총액 정합성")
    ok = (px["close"] > 0) & (px["shares"] > 0)
    rel = (px.loc[ok, "market_cap"] / (px.loc[ok, "close"] * px.loc[ok, "shares"]) - 1).abs()
    print(f"|시총/(종가×주식수) − 1| > 1%: {int((rel > 0.01).sum()):,}행 / {int(ok.sum()):,}")
    print(f"상장주식수 0 또는 결측: {int((~(px['shares'] > 0)).sum()):,}행")

    # ── 6. 원본 가격의 가격제한폭 초과 ──
    section("6. 원본 종가 일간 변동이 가격제한폭 초과")
    prev = px["close"].where(px["close"] > 0).groupby(px["symbol"], observed=True).ffill()
    prev = prev.groupby(px["symbol"], observed=True).shift()
    r_raw = px["close"] / prev - 1
    lim = np.where(px["date"] < LIMIT_CHANGE, 0.15, 0.30) + 0.005
    first = g.cumcount() == 0
    last_date = g["date"].transform("max")
    from_end = g.cumcount(ascending=False)
    delisted = last_date < px["date"].max() - pd.Timedelta(days=7)
    tail = delisted & (from_end < DELIST_TAIL)
    over_raw = (r_raw.abs() > lim) & ~first & (px["close"] > 0)
    print(f"원본 기준 초과: {int(over_raw.sum()):,}건 ({px.loc[over_raw, 'symbol'].nunique()}종목) "
          f"— 정리매매 구간 {int((over_raw & tail).sum())}, 그 외 {int((over_raw & ~tail).sum())}")
    sh_prev = px["shares"].groupby(px["symbol"], observed=True).shift()
    sh_chg = (px["shares"] != sh_prev) & sh_prev.notna()
    print(f"  그 외 중 같은 날 상장주식수 변동 동반(분할·병합·감자 추정): {int((over_raw & ~tail & sh_chg).sum())}")

    # ── 7. 수정계수 ──
    section("7. 수정계수(kr_adj_factor)")
    f = load_factor(c)
    print(f"kr_adj_factor: {len(f):,}행, {f['symbol'].nunique()}종목")
    print(f"계수 NULL {int(f['factor'].isna().sum())}, 0 이하 {int((f['factor'] <= 0).sum())}, "
          f">1.5 {int((f['factor'] > 1.5).sum())}, <0.001 {int((f['factor'] < 0.001).sum())}")
    print(f"(date, symbol) 중복: {int(f.duplicated(['symbol', 'date']).sum())}")
    no_f_sym = set(px["symbol"].astype(str)) - set(f["symbol"])
    print(f"krx_1d에는 있지만 수정계수 파일이 아예 없는 종목: {len(no_f_sym)} (예: {sorted(no_f_sym)[:10]})")
    px["symbol"] = px["symbol"].astype(str)
    m = px.merge(f.drop_duplicates(["symbol", "date"], keep="last"), on=["symbol", "date"], how="left")
    m["r_raw"] = r_raw.values
    m["tail"], m["first"] = tail.values, first.values
    m["sh_chg"] = sh_chg.values
    nofac = m["factor"].isna()
    print(f"krx 행 중 그날 수정계수 없음: {int(nofac.sum()):,}행 ({nofac.mean():.2%}), "
          f"그중 거래량>0 {int((nofac & (m['volume'] > 0)).sum()):,}")
    by_year = m.groupby(m["date"].dt.year)["factor"].apply(lambda s: s.isna().mean())
    print("연도별 계수 결측률:", {int(k): f"{v:.1%}" for k, v in by_year.items()})
    fac = m["factor"].where(m["factor"] > 0)
    fs = fac.groupby(m["symbol"]).ffill().groupby(m["symbol"]).bfill().fillna(1.0)  # backtest/data.py와 같은 규칙
    m["adj_close"] = m["close"].where(m["close"] > 0) * fs

    # ── 8. 수정 후에도 남는 이상 변동 ──
    section("8. 수정주가 기준 가격제한폭 초과 (수정계수로 설명 안 되는 점프)")
    ap = m["adj_close"].groupby(m["symbol"]).ffill().groupby(m["symbol"]).shift()
    m["r_adj"] = m["adj_close"] / ap - 1
    m["lim"] = lim
    over = (m["r_adj"].abs() > m["lim"]) & ~m["first"] & m["adj_close"].notna()
    gap = m["di"] - m.groupby("symbol")["di"].shift() > 1
    halted_before = m.groupby("symbol")["volume"].shift() == 0
    print(f"수정 기준 초과: {int(over.sum()):,}건 ({m.loc[over, 'symbol'].nunique()}종목)")
    cats = {
        "정리매매 구간(상장폐지 직전 {}일)".format(DELIST_TAIL): over & m["tail"],
        "그날 수정계수 결측": over & ~m["tail"] & m["factor"].isna(),
        "직전 거래정지(거래량0)/행 누락 뒤 재개": over & ~m["tail"] & m["factor"].notna() & (halted_before | gap),
        "상장주식수 변동 동반": over & ~m["tail"] & m["factor"].notna() & ~(halted_before | gap) & m["sh_chg"],
        "그 외(원인 불명)": over & ~m["tail"] & m["factor"].notna() & ~(halted_before | gap) & ~m["sh_chg"],
    }
    for k, v in cats.items():
        print(f"  {k}: {int(v.sum()):,}건 ({m.loc[v, 'symbol'].nunique()}종목)")
    m["category"] = ""
    for k, v in cats.items():
        m.loc[v, "category"] = k
    ex = m.loc[over, ["symbol", "date", "market", "close", "volume", "shares", "factor", "r_raw", "r_adj", "category"]]
    ex.sort_values("r_adj", key=abs, ascending=False).to_csv(out / "kr_adj_jumps.csv", index=False)
    print("  절대값 상위 10:")
    print(ex.reindex(ex["r_adj"].abs().sort_values(ascending=False).index).head(10)
          .to_string(index=False, formatters={"r_raw": "{:+.1%}".format, "r_adj": "{:+.1%}".format}))
    print("  연도별 건수:", ex.groupby(ex["date"].dt.year).size().to_dict())

    # ── 9. KIS 원본 일봉과 교차검증 ──
    section("9. KRX vs KIS 원본 종가 교차검증")
    k = load_kis(c)
    print(f"kr_1d(KIS): {len(k):,}행, {k['symbol'].nunique()}종목")
    j = m[["symbol", "date", "close", "volume"]].merge(k, on=["symbol", "date"], how="outer",
                                                       suffixes=("_krx", "_kis"), indicator=True)
    common_syms = set(k["symbol"])
    j = j[j["symbol"].isin(common_syms)]
    print("행 일치 여부:", j["_merge"].value_counts().to_dict(), "(krx만=left_only, kis만=right_only)")
    b = j[j["_merge"] == "both"]
    diff = (b["close_krx"] != b["close_kis"])
    print(f"같은 날 종가 불일치: {int(diff.sum()):,} / {len(b):,} ({diff.mean():.3%}), "
          f"거래량 불일치: {int((b['volume_krx'] != b['volume_kis']).sum()):,}")
    b[diff].to_csv(out / "kr_krx_vs_kis_close_mismatch.csv", index=False)
    j[j["_merge"] != "both"].to_csv(out / "kr_krx_vs_kis_missing_rows.csv", index=False)


def check_us(out: Path) -> None:
    c = con()
    u = c.sql(f"SELECT ts, open, high, low, close, volume, adj_close, filename "
              f"FROM read_parquet('{D}/us_1d/*/*.parquet', filename=true, union_by_name=true)").df()
    u["symbol"] = sym_from_path(u["filename"])
    u = u.drop(columns="filename").sort_values(["symbol", "ts"]).reset_index(drop=True)
    u["date"] = local_date(u["ts"], "America/New_York")
    section("US. us_1d")
    print(f"{len(u):,}행, {u['symbol'].nunique()}종목, {u['date'].min():%Y-%m-%d} ~ {u['date'].max():%Y-%m-%d}")
    print(f"(date, symbol) 중복: {int(u.duplicated(['symbol', 'date']).sum())}")
    print("NULL 개수:", {k: int(v) for k, v in u.isna().sum().items() if v})
    print(f"가격 0 이하: {int((u[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum())}")
    bad = (u["high"] < u["low"]) | (u["high"] < u[["open", "close"]].max(axis=1) * 0.999) | \
          (u["low"] > u[["open", "close"]].min(axis=1) * 1.001)
    print(f"시고저종 관계 위반(0.1% 허용): {int(bad.sum())}")
    days = pd.Series(sorted(u["date"].unique()))
    di = pd.Series(np.arange(len(days)), index=days)
    u["di"] = u["date"].map(di)
    sp = u.groupby("symbol")["di"].agg(["min", "max", "count"])
    sp["holes"] = sp["max"] - sp["min"] + 1 - sp["count"]
    print(f"상장 기간 중 빠진 날이 있는 종목: {int((sp['holes'] > 0).sum())}, 총 {int(sp['holes'].sum())}일")
    r = u.groupby("symbol")["adj_close"].pct_change()
    big = r.abs() > 0.5
    print(f"수정종가 하루 ±50% 초과: {int(big.sum())}건 {sorted(set(u.loc[big, 'symbol']))[:15]}")
    rr = u.groupby("symbol")["close"].pct_change()
    print(f"원본종가 하루 ±50% 초과(분할 미반영 원본 특성): {int((rr.abs() > 0.5).sum())}건")
    u.loc[big | bad, ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]] \
        .assign(r_adj=r[big | bad]).to_csv(out / "us_anomalies.csv", index=False)
    stale = u.groupby("symbol")["date"].max()
    print(f"마지막 날짜가 전체 최신보다 7일 넘게 이전인 종목: {int((stale < u['date'].max() - pd.Timedelta(days=7)).sum())}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/data_quality")
    ap.add_argument("--only", choices=["kr", "us"])
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if a.only != "us":
        check_kr(out)
    if a.only != "kr":
        check_us(out)
