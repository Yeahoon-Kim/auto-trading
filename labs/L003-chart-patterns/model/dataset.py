"""L003 라운드 2 데이터셋 — 탐지기 6종의 모든 신호 × (입력 특징, 목표값).

표본
  study 하네스(`study/harness.py`)와 같은 데이터·유니버스·봉·구간 규칙을 쓴다.
  KR = 그날 시총 상위 500 ∧ 20일 평균 거래대금 ≥ 10억 ∧ 거래 가능, US = us_1d 전체. 일봉 + W-FRI 주봉.
  탐지기 = `patterns.detect` (TOOLS 6종). 신호 봉 t의 유니버스 행에 붙인다 (`H.attach_events`).

목표값 y (방향 부호를 곱하지 않는다)
  ret20 = t 다음 일봉 시가 → 진입 봉을 1봉째로 20봉째 종가 (H.build_bars의 ret20, 상장폐지는 마지막 종가 청산).
  y = ret20 − (같은 날짜·같은 시장·같은 봉 유니버스 종목 ret20 평균).  평균에 쓰는 종목 수 xs_n ≥ 25인 날만 쓴다.
  탐색 구간 행 = 신호일과 20봉 청산일이 모두 경계(2018-03-12) 전 (H.build_bars의 period 0).

입력 특징 (모두 신호 봉 t의 종가 시점까지 아는 값) — FEATURES
  direction, is_kr, is_weekly           : 신호 방향(+1/−1), 시장, 봉 (탐지 결과·설정값)
  tool_variant (범주)                    : 도구·변형
  pattern_len                           : t − 패턴 시작 봉 (봉 수). 패턴 시작은 탐지기가 t까지 데이터로 정한 값
  height_atr                            : 패턴 높이 / ATR14(t)  (이동평균 도구는 높이 없음 → NaN)
  stop_atr                              : 방향 × (종가 t − 손절 기준가) / ATR14(t)
  ret_5, ret_20, ret_60, ret_250        : 직전 n봉 수익 (종가 t / 종가 t−n − 1)
  vol_20                                : 직전 20봉 로그 수익 표준편차
  liq_chg                               : log(직전 5봉 평균 거래대금 / 직전 60봉 평균). KR = krx value, US = 원본 종가 × 거래량
                                          (분할 보정 가격 × 원본 거래량은 분할 전후로 튀므로 원본 종가를 쓴다). 주봉은 그 주 합계
  ma_gap_short, ma_gap_long             : 종가 t / 이동평균 − 1. 일봉 50·200일, 주봉 10·40주 (탐지기 params와 같음)
  mcap_rank                             : 날짜 t 시가총액 횡단면 백분위 (KR = 그날 유니버스 안, US = 시총이 있는 전 종목, 1 = 가장 큼)
                                          US 시총은 load()의 market_cap (EDGAR 공시일 다음 거래일부터 반영 = 시점 맞음)
  idx_above200, idx_ret60               : 시장 지수(KR = KOSPI, US = S&P500)의 t 이하 마지막 종가 기준 200일선 위(1)/아래(0), 60일 수익
                                          `load()`는 지수를 주지 않으므로 macro_1d(Yahoo ^KS11, ^GSPC 일봉 종가)를 store.read(end=경계)로 읽는다.
                                          macro_1d ts = 현지 날짜 00:00 UTC (확인함) → 같은 날짜 지수 종가는 같은 날 주식 종가와 같은 시점

시점(미래참조) 근거
  - 종목 특징은 종목별 봉 배열에 뒤만 보는 rolling/ewm만 쓴다 → `stock_bar_features`의 k행은 0..k봉에만 의존.
  - 날짜 특징(mcap_rank, 지수)은 그날(또는 그 이전) 값만 쓴다.
  - `check_truncation`이 신호마다 데이터를 t에서 잘라 탐지·특징을 다시 만들어 같은지 확인한다.

라운드 3 H4 (horizon 60): 목표값·비용·청산일을 60봉으로 바꾼다 (build_dataset(horizon=60)). ret60은 H.build_bars가 이미 만든다
  (탐색 구간 행의 60봉 청산이 경계를 넘으면 NaN → 그 행은 뺀다). 열 이름이 exit60·ret60·cost60·delist60이 된다.

캐시: model/.cache/dataset_<period>_<null>_<dev_end>.parquet (git 제외, horizon 60은 _h60 붙음)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
for _p in (str(LAB / "study"), str(LAB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import harness as H  # noqa: E402
import patterns as P  # noqa: E402

CACHE = HERE / ".cache"
MARKETS = ("kr_stock", "us_stock")
TIMEFRAMES = ("1d", "1w")
INDEX = {"kr_stock": "KOSPI", "us_stock": "SPX"}
RET_WINS = (5, 20, 60, 250)
VOL_WIN = 20
LIQ_SHORT, LIQ_LONG = 5, 60
IDX_MA, IDX_RET = 200, 60
IDX_STALE_DAYS = 10          # 지수 마지막 값이 신호일보다 이만큼 넘게 오래되면 NaN
MIN_XS = H.MIN_XS            # 횡단면 평균에 쓸 최소 종목 수 (study와 같은 25)
H20 = H.MAIN_H
DS_VERSION = 2               # 데이터셋 정의가 바뀌면 올린다 (캐시 키)

STOCK_FEATS = [f"ret_{n}" for n in RET_WINS] + ["vol_20", "liq_chg", "ma_gap_short", "ma_gap_long"]
EVENT_FEATS = ["pattern_len", "height_atr", "stop_atr"]
DATE_FEATS = ["mcap_rank", "idx_above200", "idx_ret60"]
FLAG_FEATS = ["direction", "is_kr", "is_weekly"]
NUM_FEATURES = FLAG_FEATS + EVENT_FEATS + STOCK_FEATS + DATE_FEATS
CAT_FEATURE = "tool_variant"
FEATURES = NUM_FEATURES + [CAT_FEATURE]
PERIODS = {"explore": (0,), "confirm": (1,), "both": (0, 1)}


# ───────────────────────── 데이터 읽기 ─────────────────────────
def load_market(market: str, period: str, use_cache: bool = True) -> dict:
    """study 하네스와 같은 가격(load 기본 보정). 탐색 실행은 경계 전만 남긴다.
    KR = study 캐시(value·market_cap 포함). US = 거래량·원본 종가·시총이 더 필요해 model/.cache에 따로 캐시."""
    from autotrader.backtest import data as bdata
    if market == "kr_stock":
        wide = H.load_wide(market, use_cache=use_cache)
    else:
        end = bdata.dev_end()
        path = CACHE / f"{market}_model_{H.START}_{end.date()}.pkl"
        if use_cache and path.exists():
            wide = pd.read_pickle(path)
        else:
            p = bdata.load(market, start=H.START, fields=("high", "low", "volume", "raw_close", "market_cap"))
            wide = {k: v for k, v in p.items()}
            if use_cache:
                CACHE.mkdir(exist_ok=True)
                pd.to_pickle(wide, path)
    if period == "explore":
        wide = H.slice_wide(wide, H.BOUNDARY)     # 확인 구간 데이터는 쓰지 않는다
    return wide


def load_index(market: str, period: str) -> pd.Series:
    """시장 지수 일봉 종가 (macro_1d). 탐색 실행은 경계 전만 읽는다 (store.read end = 경계, 미포함)."""
    from autotrader.backtest import data as bdata
    from autotrader.data import store
    end = H.BOUNDARY if period == "explore" else bdata.dev_end()
    d = store.read("macro_1d", INDEX[market], start=H.START, end=end)
    s = pd.Series(d["close"].to_numpy(float), index=d["ts"].dt.tz_localize(None).dt.normalize())
    return s[~s.index.duplicated(keep="last")].sort_index()


def daily_value(market: str, wide: dict) -> pd.DataFrame | None:
    if "value" in wide:
        return wide["value"].astype(float)
    if "volume" in wide and "raw_close" in wide:
        return wide["raw_close"].astype(float) * wide["volume"].astype(float)
    return None


# ───────────────────────── 가짜 시장 ─────────────────────────
def remove_drift(s: H.Stock, boundary=H.BOUNDARY) -> H.Stock:
    """구간(탐색/확인)마다 종가 대 종가 총수익(c_t / c_t-1)의 산술 평균을 1로 맞춘 종목. 봉 안 비율(시가/전일종가, 고가/종가,
    저가/종가)은 그대로.
    이유: boot 가짜 시장은 날마다 독립이지만 종목별 평균 수익(drift)은 남는다. 종목마다 drift가 다르면 직전 수익이 앞으로의
    수익을 횡단면으로 예측하는 '가짜 예측력'이 생긴다 (drift를 구간 전체 표본에서 정하므로 일종의 미래 정보).
    로그 수익 평균을 0으로 맞추면 산술 기대수익이 σ²/2만큼 남아 '변동성이 큰 종목이 더 오른다'는 가짜 예측력이 생긴다
    (2026-11-18 첫 헛발견 실행에서 릿지 IC +0.016, p 6e-5) → 산술 평균을 1로 맞춘다. 독립 추출이면 E[20봉 총수익] = 1."""
    n = len(s.c)
    if n < 3:
        return s
    g = s.c[1:] / s.c[:-1]
    seg = s.dates[1:] < np.datetime64(boundary)
    for m in (seg, ~seg):
        if m.any():
            g[m] /= g[m].mean()
    c = np.empty(n)
    c[0] = s.c[0]
    c[1:] = s.c[0] * np.cumprod(g)
    o, h, l = s.o.copy(), s.h.copy(), s.l.copy()
    o[1:] = c[:-1] * (s.o[1:] / s.c[:-1])
    h = c * (s.h / s.c)
    l = c * (s.l / s.c)
    return H.Stock(s.sym, s.dates, o, h, l, c)


def null_stock(s: H.Stock, seed: int, drift: str = "remove") -> H.Stock:
    """헛발견 검사 가짜 시장 = study 하네스 shuffle_stock(boot) 재사용. drift='remove'면 먼저 종목 drift를 지운다."""
    if drift == "remove":
        s = remove_drift(s)
    elif drift != "keep":
        raise ValueError(drift)
    return H.shuffle_stock(s, seed, method="boot")


# ───────────────────────── 특징 ─────────────────────────
def stock_bar_features(s: H.Stock, val: np.ndarray | None, tf: str) -> pd.DataFrame:
    """종목 하나의 봉마다 특징 (k행 = k봉 종가까지의 값). 뒤만 보는 rolling/ewm만 쓴다."""
    bars, end = H.bars_of(s, tf)
    h, l, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    p = P.params(tf)
    cs = pd.Series(c)
    f = {"bar_date": bars.index.values, "close": c, "atr": P.atr(h, l, c)}
    for n in RET_WINS:
        f[f"ret_{n}"] = c / cs.shift(n).to_numpy() - 1
    f["vol_20"] = np.log(cs).diff().rolling(VOL_WIN, min_periods=VOL_WIN).std().to_numpy()
    f["ma_gap_short"] = c / cs.rolling(p["ma_short"], min_periods=p["ma_short"]).mean().to_numpy() - 1
    f["ma_gap_long"] = c / cs.rolling(p["ma_long"], min_periods=p["ma_long"]).mean().to_numpy() - 1
    if val is None:
        f["liq_chg"] = np.full(len(c), np.nan)
    else:
        start = np.r_[0, end[:-1] + 1]
        ok = np.isfinite(val) & (val > 0)
        bv = np.add.reduceat(np.where(ok, val, 0.0), start)
        bc = np.add.reduceat(ok.astype(float), start)
        bv = pd.Series(np.where(bc > 0, bv, np.nan))
        a = bv.rolling(LIQ_SHORT, min_periods=LIQ_SHORT).mean().to_numpy()
        b = bv.rolling(LIQ_LONG, min_periods=int(LIQ_LONG * 0.8)).mean().to_numpy()
        with np.errstate(all="ignore"):
            f["liq_chg"] = np.where((a > 0) & (b > 0), np.log(a / b), np.nan)
    return pd.DataFrame(f)


def event_features(fb: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """fb = stock_bar_features, ev = 그 종목의 신호 (k, pattern_start, height, stop_level, direction)."""
    k = ev["k"].to_numpy(int)
    row = fb.iloc[k]
    out = row[STOCK_FEATS].copy()
    out.index = ev.index
    bd = fb["bar_date"].to_numpy()
    ps = pd.to_datetime(ev["pattern_start"]).to_numpy(dtype="datetime64[ns]")
    ks = np.searchsorted(bd, ps)
    out["pattern_len"] = (k - ks).astype(float)
    a = row["atr"].to_numpy()
    cl = row["close"].to_numpy()
    with np.errstate(all="ignore"):
        out["height_atr"] = ev["height"].to_numpy(float) / a
        out["stop_atr"] = ev["direction"].to_numpy(float) * (cl - ev["stop_level"].to_numpy(float)) / a
    return out


def mcap_rank_table(wide: dict, univ: pd.DataFrame | None) -> pd.DataFrame | None:
    """날짜 × 종목 시총 백분위 (그날 값만)."""
    if "market_cap" not in wide:
        return None
    mc = wide["market_cap"].astype(float)
    mc = mc.where(mc > 0)
    if univ is not None:
        mc = mc.where(univ.reindex_like(mc).fillna(False).astype(bool))
    return mc.rank(axis=1, pct=True)


def index_table(idx: pd.Series) -> pd.DataFrame:
    ma = idx.rolling(IDX_MA, min_periods=IDX_MA).mean()
    above = (idx > ma).astype(float).where(ma.notna())
    return pd.DataFrame({"idx_above200": above, "idx_ret60": idx / idx.shift(IDX_RET) - 1})


def lookup_index(tab: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    """t 이하 마지막 지수 날짜의 값 (as-of)."""
    d = np.asarray(dates, dtype="datetime64[ns]")
    ix = tab.index.values
    pos = np.searchsorted(ix, d, side="right") - 1
    ok = pos >= 0
    pc = np.clip(pos, 0, None)
    stale = (d - ix[pc]) > np.timedelta64(IDX_STALE_DAYS, "D")
    ok &= ~stale
    out = {}
    for c in tab.columns:
        v = tab[c].to_numpy(float)[pc]
        out[c] = np.where(ok, v, np.nan)
    return pd.DataFrame(out)


def lookup_wide(tab: pd.DataFrame | None, dates: np.ndarray, syms: np.ndarray) -> np.ndarray:
    if tab is None:
        return np.full(len(dates), np.nan)
    r = np.searchsorted(tab.index.values, np.asarray(dates, dtype="datetime64[ns]"))
    rc = np.clip(r, 0, len(tab) - 1)
    hit = tab.index.values[rc] == np.asarray(dates, dtype="datetime64[ns]")
    cmap = {c: j for j, c in enumerate(tab.columns)}
    cj = np.array([cmap.get(s, -1) for s in syms])
    arr = tab.to_numpy(float)
    v = arr[rc, np.clip(cj, 0, None)]
    return np.where(hit & (cj >= 0), v, np.nan)


def stock_values(val_w: pd.DataFrame | None, s: H.Stock) -> np.ndarray | None:
    if val_w is None or s.sym not in val_w.columns:
        return None
    return val_w[s.sym].reindex(pd.DatetimeIndex(s.dates)).to_numpy(float)


# ───────────────────────── 조립 ─────────────────────────
def build_market(market: str, wide: dict, idx: pd.Series | None, periods=(0,), null_seed: int | None = None,
                 drift: str = "remove", workers: int = 6, tools=None, timeframes=TIMEFRAMES,
                 horizon: int = H20) -> tuple[pd.DataFrame, dict]:
    if horizon not in H.HORIZONS:
        raise ValueError(f"horizon은 {H.HORIZONS} 중 하나")
    hz = horizon
    t0 = time.time()
    tools = tuple(tools or P.TOOLS)
    costs = H.Costs(market)
    univ = H.universe_wide(market, wide)
    stocks = H.stocks_from_wide(wide)
    if null_seed is not None:
        stocks = [null_stock(s, null_seed, drift) for s in stocks]
    by_sym = {s.sym: s for s in stocks}
    val_w = daily_value(market, wide)
    mrank = mcap_rank_table(wide, univ)
    itab = index_table(idx) if idx is not None and len(idx) else None
    info = {"symbols": len(stocks), "data_start": str(wide["close"].index[0].date()),
            "data_end": str(wide["close"].index[-1].date()), "timing": {}}
    td = time.time()
    det = H.run_detection(stocks, tools, timeframes, str(LAB), "patterns", workers)
    info["timing"]["detect"] = round(time.time() - td, 1)
    panel_last = wide["close"].index[-1]
    parts = []
    for tf in timeframes:
        tb = time.time()
        bt = H.build_bars(market, tf, stocks, univ, panel_last, costs)
        ev, _, stat = H.attach_events(det[det["timeframe"] == tf], bt, periods)
        b = bt.df
        okb = b["period"].isin(periods).to_numpy() & np.isfinite(b[f"ret{hz}"].to_numpy())
        xs = b.loc[okb].groupby("date")[f"ret{hz}"].agg(["mean", "count"])
        ev = ev.join(xs, on="date")
        stat["xs_too_small"] = int((ev["count"].fillna(0) < MIN_XS).sum())
        ev = ev[ev["count"] >= MIN_XS].copy()
        ev["xs_mean"], ev["xs_n"] = ev.pop("mean"), ev.pop("count").astype(int)
        ev["y"] = ev[f"ret{hz}"] - ev["xs_mean"]
        stat["no_target"] = int((~np.isfinite(ev["y"].to_numpy(float))).sum())   # horizon 60: 청산이 경계를 넘는 탐색 행
        ev = ev[np.isfinite(ev["y"].to_numpy(float))].copy()
        feats = []
        for sym, g in ev.groupby("sym"):
            s = by_sym[sym]
            fb = stock_bar_features(s, stock_values(val_w, s), tf)
            feats.append(event_features(fb, g))
        F = pd.concat(feats).reindex(ev.index) if feats else pd.DataFrame(index=ev.index, columns=STOCK_FEATS + EVENT_FEATS)
        ev = pd.concat([ev, F], axis=1)
        ev["mcap_rank"] = lookup_wide(mrank, ev["date"].values, ev["sym"].values)
        if itab is not None:
            li = lookup_index(itab, ev["date"].values)
            for c in li.columns:
                ev[c] = li[c].to_numpy()
        else:
            ev["idx_above200"] = ev["idx_ret60"] = np.nan
        ev["market"], ev["timeframe"] = market, tf
        ev["is_kr"] = float(market == "kr_stock")
        ev["is_weekly"] = float(tf == "1w")
        ev["direction"] = ev["direction"].astype(float)
        ev[CAT_FEATURE] = ev["tool"].astype(str) + ":" + ev["variant"].astype(str)
        stat["rows"] = len(ev)
        info[f"{tf}_events"] = stat
        info["timing"][f"{tf}_build"] = round(time.time() - tb, 1)
        keep = ["market", "timeframe", "sym", "date", "k", "tool", "variant", "pattern_start", "height", "stop_level",
                "period", f"exit{hz}", f"ret{hz}", f"cost{hz}", f"delist{hz}", "xs_mean", "xs_n", "y"] + FEATURES
        parts.append(ev[keep].copy())
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    info["horizon"] = hz
    info["timing"]["total"] = round(time.time() - t0, 1)
    return df, info


def build_dataset(period: str = "explore", markets=MARKETS, null_seed: int | None = None, drift: str = "remove",
                  workers: int = 6, use_cache: bool = True, rebuild: bool = False, max_symbols: int | None = None,
                  horizon: int = H20) -> tuple[pd.DataFrame, dict]:
    """실제 데이터(또는 그 섞은 가짜 시장)로 데이터셋. period: explore(탐색만) | both(confirm 경로용, 탐색+확인)."""
    from autotrader.backtest import data as bdata
    if period not in ("explore", "both"):
        raise ValueError("period는 explore | both")
    tag = "real" if null_seed is None else f"null{null_seed}-{drift}"
    ms = "-".join(m.split("_")[0] for m in markets) + (f"-max{max_symbols}" if max_symbols else "")
    hs = "" if horizon == H20 else f"_h{horizon}"     # 20봉은 라운드 2 캐시 이름 그대로
    path = CACHE / f"dataset_v{DS_VERSION}_{period}_{tag}_{ms}{hs}_{bdata.dev_end().date()}.parquet"
    ipath = path.with_suffix(".json")
    if use_cache and not rebuild and path.exists():
        return pd.read_parquet(path), json.loads(ipath.read_text(encoding="utf-8"))
    parts, info = [], {"period": period, "null_seed": null_seed, "drift": drift, "horizon": horizon, "markets": {}}
    for m in markets:
        tl = time.time()
        wide = load_market(m, period, use_cache)
        if max_symbols and wide["close"].shape[1] > max_symbols:
            pick = np.random.default_rng(7).choice(wide["close"].shape[1], max_symbols, replace=False)
            cols = wide["close"].columns[np.sort(pick)]
            wide = {k: v[cols] for k, v in wide.items()}
        idx = load_index(m, period)
        load_s = round(time.time() - tl, 1)
        df, mi = build_market(m, wide, idx, PERIODS[period], null_seed, drift, workers, horizon=horizon)
        mi["timing"]["load"] = load_s
        info["markets"][m] = mi
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        df.to_parquet(path, index=False)
        ipath.write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return df, info


# ───────────────────────── 잘라내기 검사 ─────────────────────────
def _same(a, b, tol=1e-9) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol * np.maximum(1.0, np.abs(a)))


def check_truncation(market: str, wide: dict, idx: pd.Series | None, df: pd.DataFrame, n_events: int = 2000,
                     n_cuts: int = 3, seed: int = 0, null_seed: int | None = None, drift: str = "remove") -> dict:
    """미래참조 검사.
    (1) 신호 표본마다 그 종목 일봉을 신호일 t(포함)에서 잘라 탐지기를 다시 돌려 같은 신호(날짜·변형·시작·높이·손절)가 나오는지,
        종목·신호 특징을 다시 만들어 같은지 비교한다.
    (2) 날짜 특징(시총 백분위·지수)은 패널과 지수를 cut 날짜에서 잘라 다시 만들어 cut 이전 행 전체를 비교한다."""
    sub = df[df["market"] == market]
    rng = np.random.default_rng(seed)
    pick = sub.iloc[np.sort(rng.choice(len(sub), min(n_events, len(sub)), replace=False))] if len(sub) else sub
    stocks = {s.sym: s for s in H.stocks_from_wide(wide)}
    if null_seed is not None:
        stocks = {k: null_stock(s, null_seed, drift) for k, s in stocks.items()}
    val_w = daily_value(market, wide)
    res = {"events": 0, "detect_missing": 0, "detect_mismatch": 0, "feature_cells": 0, "feature_mismatch": 0,
           "mismatch_examples": []}
    feat_cols = STOCK_FEATS + EVENT_FEATS
    for _, r in pick.iterrows():
        s = stocks[r["sym"]]
        cut = int(np.searchsorted(s.dates, np.datetime64(r["date"]), side="right"))
        st = H.Stock(s.sym, s.dates[:cut], s.o[:cut], s.h[:cut], s.l[:cut], s.c[:cut])
        vt = stock_values(val_w, st)
        bars, _ = H.bars_of(st, r["timeframe"])
        d = P.detect(bars, r["tool"], r["timeframe"])
        hit = d[(pd.to_datetime(d["date"]).values == np.datetime64(r["date"])) & (d["variant"].astype(str) == r["variant"])]
        res["events"] += 1
        if hit.empty:
            res["detect_missing"] += 1
            continue
        h0 = hit.iloc[0]
        if not (pd.Timestamp(h0["pattern_start"]) == pd.Timestamp(r["pattern_start"])
                and _same(h0["height"], r["height"]) and _same(h0["stop_level"], r["stop_level"])):
            res["detect_mismatch"] += 1
        k = len(bars) - 1
        fb = stock_bar_features(st, vt, r["timeframe"])
        e = pd.DataFrame([{"k": k, "pattern_start": h0["pattern_start"], "height": h0["height"],
                           "stop_level": h0["stop_level"], "direction": r["direction"]}])
        f2 = event_features(fb, e).iloc[0]
        ok = _same(f2[feat_cols].to_numpy(float), r[feat_cols].to_numpy(float))
        res["feature_cells"] += len(feat_cols)
        res["feature_mismatch"] += int((~ok).sum())
        if (~ok).any() and len(res["mismatch_examples"]) < 5:
            res["mismatch_examples"].append({"sym": r["sym"], "date": str(r["date"]), "cols": list(np.array(feat_cols)[~ok])})
    # (2) 날짜 특징
    univ = H.universe_wide(market, wide)
    full_rank = mcap_rank_table(wide, univ)
    idx_dates = wide["close"].index
    cuts = [idx_dates[int(i)] for i in np.linspace(len(idx_dates) * 0.3, len(idx_dates) - 2, n_cuts)]
    res["date_cuts"], res["date_cells"], res["date_mismatch"] = [str(c.date()) for c in cuts], 0, 0
    for cut in cuts:
        w2 = H.truncate_wide(wide, cut)
        r2 = mcap_rank_table(w2, H.universe_wide(market, w2))
        rows = sub[sub["date"] <= cut]
        a = lookup_wide(full_rank, rows["date"].values, rows["sym"].values)
        b = lookup_wide(r2, rows["date"].values, rows["sym"].values)
        if idx is not None and len(idx):
            i2 = lookup_index(index_table(idx[idx.index <= cut]), rows["date"].values)
            i1 = lookup_index(index_table(idx), rows["date"].values)
            for c in i1.columns:
                res["date_cells"] += len(rows)
                res["date_mismatch"] += int((~_same(i1[c].to_numpy(), i2[c].to_numpy())).sum())
        res["date_cells"] += len(rows)
        res["date_mismatch"] += int((~_same(a, b)).sum())
    res["ok"] = (res["detect_missing"] == 0 and res["detect_mismatch"] == 0 and res["feature_mismatch"] == 0
                 and res["date_mismatch"] == 0)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="L003 R2 데이터셋 생성·잘라내기 검사 (모델 성과는 계산하지 않는다)")
    ap.add_argument("--period", choices=("explore",), default="explore")
    ap.add_argument("--markets", nargs="+", default=list(MARKETS))
    ap.add_argument("--null-seed", type=int)
    ap.add_argument("--drift", choices=("remove", "keep"), default="remove")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--horizon", type=int, choices=(20, 60), default=20)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--check-trunc", type=int, default=0, help="시장마다 잘라내기 검사 신호 수 (0 = 생략)")
    ap.add_argument("--out", help="검사 결과 JSON 경로")
    a = ap.parse_args(argv)
    t0 = time.time()
    df, info = build_dataset(a.period, a.markets, a.null_seed, a.drift, a.workers, rebuild=a.rebuild, horizon=a.horizon)
    t_build = round(time.time() - t0, 1)
    size = df.groupby(["market", "timeframe"]).size().rename("n").reset_index()
    by_tool = df.groupby(["market", "timeframe", "tool"]).size().unstack("tool").fillna(0).astype(int)
    print(f"데이터셋 {len(df)}행 ({t_build}초)\n{size.to_string(index=False)}\n{by_tool.to_string()}")
    print("특징 결측률:\n" + df[NUM_FEATURES].isna().mean().round(3).to_string())
    out = {"rows": len(df), "build_sec": t_build, "size": size.to_dict("records"),
           "by_tool": by_tool.reset_index().to_dict("records"), "info": info,
           "missing": df[NUM_FEATURES].isna().mean().round(4).to_dict()}
    if a.check_trunc:
        out["truncation"] = {}
        for m in a.markets:
            tc = time.time()
            wide = load_market(m, a.period)
            r = check_truncation(m, wide, load_index(m, a.period), df, a.check_trunc, null_seed=a.null_seed, drift=a.drift)
            r["sec"] = round(time.time() - tc, 1)
            out["truncation"][m] = r
            print(m, {k: v for k, v in r.items() if k != "mismatch_examples"})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
