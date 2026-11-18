"""L003 라운드 3 국면 표 — 공식 `regime_1d` 정의를 lab 안에서 2010년부터 다시 계산한다.

정의 (복사하지 않고 import)
  `autotrader.intel.market`의 `_trend_block`(지수 종가 → 200일선 위/아래, 20일 수익, 20일 실현변동성, 756일 변동성 백분위)과
  `_labels`(추세 상승/하락/혼조, 변동성 낮음/보통/높음)를 그대로 부른다. 조립(concat → 관측 표시 → ffill(limit=5) → _labels)은
  `compute()`와 같은 순서다. `compute()` 자체는 부르지 않는다: (1) 2015년 이후만 남기고 (2) 원천을 끝까지 읽으며
  (3) BTC·펀딩비·김프·공포탐욕까지 읽기 때문이다.

공식 표와 다른 점 (의도)
  - 2015년 필터 없음 (2010년부터). 지수 원천이 2010-01-04부터라 200일선은 2010-10월께, 변동성 백분위(최소 250일)는 2011년부터 생긴다.
  - BTC·하이일드 스프레드(최근 3년만)·펀딩비·김프·공포탐욕은 뺀다. `_labels`가 요구하는 btc_*·hy_spread_chg_20d 열은 빈 값으로
    넣어 부르고 결과에서 지운다 (위험선호 점수 risk_off_* 도 지움 — 하이일드가 빠져 공식과 다르므로).
  - 추가 열: `dxy_chg_20d` = 달러지수(macro_1d DXY) 20일 변화. 공식 `usdkrw_chg_20d`와 같은 식(pct_change(20), 원천 날짜 기준).
  - 원천은 `end`(미포함)에서 잘라 읽는다. 탐색 구간 실행은 end = 경계 2018-03-12 → 확인 구간 값이 계산에 들어가지 않는다.
  - 인덱스: 공식 표는 BTC(2017-08~ 매일)가 섞여 달력 날짜 전부가 행이고, 이 표는 지수·거시 원천 날짜의 합집합이다.
    ffill(limit=5)은 '행 5개'라서, 원천이 관측되지 않은 날(휴장일)의 채운 값은 두 표가 다를 수 있다. 원천이 관측된 날의 값은 같아야 한다
    (`compare_official`이 둘을 나눠 센다).

시점 (lab에서 쓸 때, `lookup`)
  신호일 t에는 관측일 < t 인 값만 쓴다 (= 관측일 표를 shift(1) 한 뒤 t의 값). 원천마다 그 원천이 실제로 관측된 날짜만 쓰고
  (ffill로 채운 행은 쓰지 않음), 마지막 관측이 t보다 STALE_DAYS일 넘게 오래면 NaN.
  - NFCI는 금요일 날짜로 기록되지만 다음 주 수요일(관측 + 5일)에 공표된다 → 공표일을 관측일로 보고 shift(1) = 목요일부터 보이게
    더 늦춘다 (NFCI_LAG_DAYS = 5. lab 판단: 날짜 그대로 shift(1)만 하면 3~4거래일 미래참조).
  - 추세·변동성 라벨은 입력이 정의되지 않은 날(200일선·20일 수익·변동성 백분위가 아직 없음)은 공식 `_labels`가 '혼조'·'보통'으로
    채우지만, lab에서는 NaN(국면 모름)으로 둔다 → H3a에서 빠지는 이벤트로 센다.

    $env:AUTOTRADER_DATA_DIR = "C:\\Users\\apple\\Claude\\auto-trading\\data"
    uv run python labs/L003-chart-patterns/regime/__init__.py --check      # 공식 대조 + 잘라내기 검사
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.intel import market as M

LAB = Path(__file__).resolve().parents[1]
BOUNDARY = pd.Timestamp("2018-03-12")    # study/harness.py BOUNDARY와 같다
STALE_DAYS = 10
NFCI_LAG_DAYS = 5          # 금요일 관측 → 수요일 공표 (관측 + 5일). lookup은 공표일 < 신호일
PFX = {"kr_stock": "kospi", "us_stock": "spx"}
TREND_CODE = {"상승": 1.0, "혼조": 0.0, "하락": -1.0}
VOL_CODE = {"낮음": 0.0, "보통": 1.0, "높음": 2.0}
MACRO_COLS = ["vix", "vix_term", "t10y2y", "nfci", "usdkrw", "usdkrw_chg_20d", "dxy", "dxy_chg_20d"]
# lab에서 쓰는 열 (lookup 결과)
LOOKUP_COLS = ["trend", "vol", "rv_pctile", "vix", "vix_term", "t10y2y", "nfci", "usdkrw_chg_20d", "dxy_chg_20d"]
# 모델 특징 이름 (model/run.py --regime-features). 추세·변동성 라벨은 위 코드로 숫자화
FEATURES = ["rg_trend", "rg_vol", "rg_rv_pctile", "rg_vix", "rg_vix_term", "rg_t10y2y", "rg_nfci",
            "rg_usdkrw_chg20", "rg_dxy_chg20"]


# ───────────────────────── 원천 ─────────────────────────
def _read(dataset: str, symbol: str, col: str, end) -> pd.Series:
    """M._close와 같은 읽기(ts 날짜 → 값)에 end(미포함) 자르기만 더한다. 인덱스는 tz 없는 날짜."""
    from autotrader.data import store
    d = store.read(dataset, symbol, end=end)
    if d.empty:
        return pd.Series(dtype=float, name=symbol)
    s = d.set_index(d["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize())[col].rename(symbol)
    return s[~s.index.duplicated(keep="last")].astype(float)


def sources(end=BOUNDARY, reader=None) -> dict[str, pd.Series]:
    r = reader or _read
    return {"SPX": r("macro_1d", "SPX", "close", end), "KOSPI": r("macro_1d", "KOSPI", "close", end),
            "VIX": r("macro_1d", "VIX", "close", end), "VIX3M": r("macro_1d", "VIX3M", "close", end),
            "USDKRW": r("macro_1d", "USDKRW", "close", end), "DXY": r("macro_1d", "DXY", "close", end),
            "T10Y2Y": r("fred", "T10Y2Y", "value", end), "NFCI": r("fred", "NFCI", "value", end)}


# ───────────────────────── 표 ─────────────────────────
def table(end=BOUNDARY, src: dict | None = None) -> pd.DataFrame:
    """관측일별 국면 표 (공식 regime_1d와 같은 열·같은 시점 = shift 전). end 미포함까지의 원천만 쓴다.
    추가로 lab용 열: obs_<원천열>(그 날 원천이 실제 관측됐는지), {p}_trend_ok / {p}_vol_ok(라벨 입력이 정의됐는지)."""
    src = src if src is not None else sources(end)
    if end is not None:
        src = {k: v[v.index < pd.Timestamp(end)] for k, v in src.items()}
    spx, kospi = src["SPX"], src["KOSPI"]
    parts = [M._trend_block(spx, "spx"), M._trend_block(kospi, "kospi")]
    vix, vix3m, usdkrw, dxy = src["VIX"], src["VIX3M"], src["USDKRW"], src["DXY"]
    macro = pd.DataFrame({
        "vix": vix, "vix_term": vix / vix3m,
        "t10y2y": src["T10Y2Y"], "nfci": src["NFCI"],
        "usdkrw": usdkrw, "usdkrw_chg_20d": usdkrw.pct_change(20),
        "dxy": dxy, "dxy_chg_20d": dxy.pct_change(20),        # 추가 열 (공식 표에는 dxy 수준만)
    })
    # 라벨 입력이 정의된 날 (원천 날짜 기준; 공식 _labels는 NaN을 '혼조'·'보통'으로 채운다)
    ok = {}
    for p, px in (("spx", spx), ("kospi", kospi)):
        ok[f"{p}_trend_ok"] = px.rolling(200).mean().notna() & px.pct_change(20).notna()
    df = pd.concat(parts + [macro], axis=1).sort_index()
    for c in MACRO_COLS:
        df[f"obs_{c}"] = df[c].notna()
    for p in ("spx", "kospi"):   # compute()와 같다
        df[f"{p}_observed"] = df[f"{p}_close"].notna()
        df[f"{p}_trend_ok"] = ok[f"{p}_trend_ok"].reindex(df.index).astype("boolean").fillna(False).astype(bool)
        df[f"{p}_vol_ok"] = df[f"{p}_rv_pctile"].notna()
    df = df.ffill(limit=5)       # compute()와 같다 (행 5개)
    # _labels가 요구하는 열: BTC·하이일드는 빼므로 빈 값으로 넣고 부른 뒤 지운다
    df["btc_above_200d"], df["btc_ret_20d"], df["btc_rv_pctile"] = False, np.nan, np.nan
    df["hy_spread_chg_20d"] = np.nan
    df = M._labels(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("btc_")] + ["hy_spread_chg_20d", "risk_off_signals", "risk_regime"])
    return df


# ───────────────────────── 시점 맞춘 조회 ─────────────────────────
def _asof_before(obs_dates: np.ndarray, vals: np.ndarray, dates: np.ndarray, lag_days: int, stale_days: int) -> np.ndarray:
    """dates마다 '관측일 + lag_days < 날짜'인 마지막 관측값 (lag 0 = 관측일 < 날짜 = shift(1)). 너무 오래되면 NaN."""
    d = np.asarray(dates, dtype="datetime64[ns]")
    vis = np.asarray(obs_dates, dtype="datetime64[ns]") + np.timedelta64(lag_days, "D")
    pos = np.searchsorted(vis, d, side="left") - 1          # vis < d 인 마지막
    ok = pos >= 0
    pc = np.clip(pos, 0, None)
    if len(vis) == 0:
        return np.full(len(d), np.nan, dtype=object if vals.dtype == object else float)
    stale = (d - np.asarray(obs_dates, dtype="datetime64[ns]")[pc]) > np.timedelta64(stale_days + lag_days, "D")
    ok &= ~stale
    out = vals[pc].copy()
    if out.dtype == object:
        out[~ok] = None
    else:
        out = np.where(ok, out.astype(float), np.nan)
    return out


def lookup(tab: pd.DataFrame, dates, market: str) -> pd.DataFrame:
    """신호일 dates(그 시장 거래일; 주봉은 그 주 마지막 거래일)에 보이는 국면 값. 열 = LOOKUP_COLS.
    지수 열(trend·vol·rv_pctile)은 그 시장 지수(KR = KOSPI, US = S&P500)가 실제 관측된 행만, 거시 열은 그 원천이 관측된 행만 쓴다."""
    p = PFX[market]
    dates = np.asarray(dates, dtype="datetime64[ns]")
    out = {}
    rows = tab[tab[f"{p}_observed"].astype(bool)]
    od = rows.index.values
    tr = rows[f"{p}_trend"].astype(object).where(rows[f"{p}_trend_ok"].astype(bool), None).to_numpy(object)
    vo = rows[f"{p}_vol"].astype(object).where(rows[f"{p}_vol_ok"].astype(bool), None).to_numpy(object)
    out["trend"] = _asof_before(od, tr, dates, 0, STALE_DAYS)
    out["vol"] = _asof_before(od, vo, dates, 0, STALE_DAYS)
    out["rv_pctile"] = _asof_before(od, rows[f"{p}_rv_pctile"].to_numpy(float), dates, 0, STALE_DAYS)
    for c in ("vix", "vix_term", "t10y2y", "nfci", "usdkrw_chg_20d", "dxy_chg_20d"):
        r = tab[tab[f"obs_{c}"].astype(bool)]
        lag = NFCI_LAG_DAYS if c == "nfci" else 0
        out[c] = _asof_before(r.index.values, r[c].to_numpy(float), dates, lag, STALE_DAYS)
    return pd.DataFrame(out)


def features(tab: pd.DataFrame, dates, market: str) -> pd.DataFrame:
    """모델 특징 (FEATURES). 라벨은 TREND_CODE·VOL_CODE로 숫자화, 모르면 NaN."""
    lk = lookup(tab, dates, market)
    return pd.DataFrame({
        "rg_trend": pd.Series(lk["trend"]).map(TREND_CODE).astype(float).to_numpy(),
        "rg_vol": pd.Series(lk["vol"]).map(VOL_CODE).astype(float).to_numpy(),
        "rg_rv_pctile": lk["rv_pctile"].to_numpy(float), "rg_vix": lk["vix"].to_numpy(float),
        "rg_vix_term": lk["vix_term"].to_numpy(float), "rg_t10y2y": lk["t10y2y"].to_numpy(float),
        "rg_nfci": lk["nfci"].to_numpy(float), "rg_usdkrw_chg20": lk["usdkrw_chg_20d"].to_numpy(float),
        "rg_dxy_chg20": lk["dxy_chg_20d"].to_numpy(float)})


# ───────────────────────── 검사 ─────────────────────────
OFFICIAL_COLS = ["spx_close", "spx_above_200d", "spx_ret_20d", "spx_rv_20d", "spx_rv_pctile",
                 "kospi_close", "kospi_above_200d", "kospi_ret_20d", "kospi_rv_20d", "kospi_rv_pctile",
                 "vix", "vix_term", "t10y2y", "nfci", "usdkrw", "usdkrw_chg_20d", "dxy",
                 "spx_observed", "kospi_observed", "spx_trend", "spx_vol", "kospi_trend", "kospi_vol"]


def _src_of(col: str) -> str:
    """열이 의존하는 '관측 표시' 열."""
    if col.startswith("spx_"):
        return "spx_observed"
    if col.startswith("kospi_"):
        return "kospi_observed"
    return f"obs_{col}"


def _eq(a: pd.Series, b: pd.Series) -> np.ndarray:
    if a.dtype == object or b.dtype == object or a.dtype == bool or b.dtype == bool:
        an, bn = a.isna().to_numpy(), b.isna().to_numpy()
        return (an & bn) | (a.astype(str).to_numpy() == b.astype(str).to_numpy())
    x, y = a.to_numpy(float), b.to_numpy(float)
    return (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.maximum(1.0, np.abs(x)))


def compare_official(mine: pd.DataFrame, start="2015-01-01", end=BOUNDARY) -> dict:
    """공식 regime_1d(관측일 값, shift 전)와 [start, end) 공통 날짜에서 열마다 비교.
    '관측일 행' = 그 열의 원천이 그날 실제 관측된 행 (ffill과 무관 → 같아야 한다), '채운 행' = 휴장으로 ffill된 행."""
    from autotrader.data import store
    off = store.read("regime_1d", "ALL", start=start, end=end)
    off.index = off["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    off = off[~off.index.duplicated(keep="last")]
    m = mine[(mine.index >= pd.Timestamp(start)) & (mine.index < pd.Timestamp(end))]
    common = m.index.intersection(off.index)
    res = {"official_rows": len(off), "mine_rows": len(m), "common_rows": len(common),
           "official_only_rows": int(len(off.index.difference(m.index))), "mine_only_rows": int(len(m.index.difference(off.index))),
           "cols": {}, "mismatch_observed": 0, "mismatch_filled": 0, "cells_observed": 0, "cells_filled": 0, "examples": []}
    for c in OFFICIAL_COLS:
        a, b = m.loc[common, c], off.loc[common, c]
        eq = _eq(a, b)
        obs = m.loc[common, _src_of(c)].astype(bool).to_numpy()
        mo, mf = int((~eq & obs).sum()), int((~eq & ~obs).sum())
        res["cols"][c] = {"mismatch_observed": mo, "mismatch_filled": mf}
        res["mismatch_observed"] += mo
        res["mismatch_filled"] += mf
        res["cells_observed"] += int(obs.sum())
        res["cells_filled"] += int((~obs).sum())
        if (~eq).any() and len(res["examples"]) < 8:
            i = np.flatnonzero(~eq)[0]
            res["examples"].append({"col": c, "date": str(common[i].date()), "mine": str(a.iloc[i]), "official": str(b.iloc[i]),
                                    "source_observed": bool(obs[i])})
    return res


def check_truncation(src: dict, cuts, end=BOUNDARY) -> dict:
    """원천을 cut(포함)에서 잘라 다시 계산한 표가 cut 이하 행에서 전체 표와 같은지 (관측일 표),
    그리고 lab 조회값이 cut 다음 날까지(관측 < 날짜 ≤ cut+1일, 즉 cut 관측값까지만 필요한 날) 같은지."""
    full = table(end, src)
    lab_cols = [c for c in OFFICIAL_COLS + ["dxy_chg_20d"] if c in full.columns]
    res = {"cuts": [], "table_cells": 0, "table_mismatch": 0, "lookup_cells": 0, "lookup_mismatch": 0}
    for cut in cuts:
        cut = pd.Timestamp(cut)
        part = table(cut + pd.Timedelta(days=1), src)       # end 미포함 → cut 포함
        a = full[full.index <= cut]
        b = part.reindex(a.index)
        mm = sum(int((~_eq(a[c], b[c])).sum()) for c in lab_cols)
        cells = len(a) * len(lab_cols)
        # 조회: cut 이전 2년 영업일 + cut 다음 날 (KR·US)
        q = pd.bdate_range(cut - pd.Timedelta(days=730), cut + pd.Timedelta(days=1)).values
        lm = lc = 0
        for mk in PFX:
            x, y = lookup(full, q, mk), lookup(part, q, mk)
            for c in LOOKUP_COLS:
                lm += int((~_eq(pd.Series(x[c]), pd.Series(y[c]))).sum())
                lc += len(q)
        res["cuts"].append({"cut": str(cut.date()), "rows": len(a), "table_mismatch": mm, "lookup_mismatch": lm})
        res["table_cells"] += cells
        res["table_mismatch"] += mm
        res["lookup_cells"] += lc
        res["lookup_mismatch"] += lm
    res["ok"] = res["table_mismatch"] == 0 and res["lookup_mismatch"] == 0
    return res


def coverage(tab: pd.DataFrame, start="2010-01-01", end=BOUNDARY) -> dict:
    """영업일마다 lab 조회값이 있는 비율 (시장별)."""
    q = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1)).values
    out = {}
    for mk in PFX:
        lk = lookup(tab, q, mk)
        first = {c: (str(pd.Timestamp(q[np.flatnonzero(pd.notna(lk[c]).to_numpy())[0]]).date())
                     if pd.notna(lk[c]).any() else None) for c in LOOKUP_COLS}
        out[mk] = {"first_valid": first,
                   "trend_counts": pd.Series(lk["trend"]).value_counts(dropna=False).rename(str).to_dict(),
                   "vol_counts": pd.Series(lk["vol"]).value_counts(dropna=False).rename(str).to_dict()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="L003 R3 국면 표 검사 (공식 regime_1d 대조, 잘라내기)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cuts", type=int, default=6)
    ap.add_argument("--out", default=str(LAB / "results" / "R3" / "regime-check.json"))
    a = ap.parse_args(argv)
    import time
    t0 = time.time()
    src = sources(BOUNDARY)
    tab = table(BOUNDARY, src)
    res = {"rows": len(tab), "first": str(tab.index[0].date()), "last": str(tab.index[-1].date()),
           "max_source_date": {k: str(v.index.max().date()) for k, v in src.items()},
           "coverage": coverage(tab)}
    if a.check:
        res["official"] = compare_official(tab)
        cuts = pd.bdate_range("2011-06-01", BOUNDARY - pd.Timedelta(days=3))
        cuts = [cuts[int(i)] for i in np.linspace(0, len(cuts) - 1, a.cuts)]
        res["truncation"] = check_truncation(src, cuts)
    res["sec"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    o = res.get("official", {})
    print(json.dumps({k: v for k, v in o.items() if k != "cols"}, ensure_ascii=False, default=str))
    print({c: v for c, v in o.get("cols", {}).items() if v["mismatch_observed"] or v["mismatch_filled"]})
    print("truncation", {k: v for k, v in res.get("truncation", {}).items() if k != "cuts"})
    print("coverage", json.dumps(res["coverage"], ensure_ascii=False))
    print(f"{res['sec']}초 → {a.out}")


if __name__ == "__main__":
    sys.exit(main())
