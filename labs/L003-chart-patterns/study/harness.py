"""L003 A 단계 event study 하네스 (유효성 검정).

흐름
  1. 데이터: load()로 KR·US 일봉 (잠금 구간 자동 제외). --period explore면 경계(2018-03-12) 이전만 쓴다
     → 확인 구간 데이터는 탐지기·위약·헛발견 검사 어디에도 들어가지 않는다.
  2. 종목별로 일봉(또는 W-FRI 주봉)을 탐지기에 주고 신호(date, direction, tool, variant)를 받는다.
  3. 봉 표(bar table): 종목·봉마다 유니버스 여부, 직전 60봉 수익률·횡단면 5분위, 진입(t+1 일봉 시가)→h봉째 종가 수익률.
  4. 이벤트마다 같은 종목·같은 연도·같은 구간의 위약 날짜 10개 (추세 맞춤 = 같은 5분위 / 단순 = 분위 무시).
  5. 검정 단위 (tool, variant, direction, market, timeframe)별 초과수익, 날짜 묶음 부트스트랩, Holm, noise floor, 통과 판정.

수익률 정의
  - 신호 봉 t (일봉 = 그날, 주봉 = 그 주 마지막 거래일). 진입 = t 다음 일봉 시가.
  - 청산 = 진입 뒤 h봉째 종가. 진입 봉이 1봉째다: 일봉이면 t+h일 종가, 주봉이면 (신호 주 + h)주 마지막 거래일 종가.
  - h봉이 모자라면: 종목 데이터가 패널 끝까지 살아 있으면 '아직 모름'(NaN, 버림), 아니면 상장폐지 → 마지막 종가 청산(건수 기록).
  - 방향 부호(direction)를 곱한다. 비용은 왕복 (수수료+슬리피지)×2 + KR 매도세(청산일 기준 세율표).

초과수익 (검정 통계량)
  d_i = direction × (이벤트 총수익 − 그 이벤트 위약 총수익 평균).  위약도 같은 방향·같은 비용을 물리면 비용은 서로 상쇄되므로
  (KR 세율 연도 차이만 남는다) 검정은 비용 전 값으로 한다. '비용 후 초과수익' = 초과수익 − 이벤트 평균 왕복 비용
  (= 신호만 비용을 내고 위약은 '거래 안 함' 기준). 통과 기준 2 "초과수익 > 왕복 비용"은 '비용 후 초과수익 > 0'과 같다.
"""

from __future__ import annotations

import math
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.backtest import data as bdata

LAB = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / ".cache"
BOUNDARY = pd.Timestamp("2018-03-12")   # 탐색 / 확인 경계 (신호일 기준)
START = "2010-01-01"
HORIZONS = (5, 20, 60)
MAIN_H = 20                  # 주 지표 (사전등록)
N_PLACEBO = 10
MOM_BARS = 60                # 위약 분위 맞춤용 직전 수익률 길이 (봉)
N_QUINT = 5
MIN_XS = 25                  # 횡단면 분위를 매길 최소 종목 수 (이보다 적으면 분위 없음 → 추세 맞춘 위약 없음)
BLOCK_BARS = MAIN_H          # 같은 종목·같은 도구 이벤트 ±이 봉은 위약 후보에서 뺀다
KR_TOP_MCAP = 500
KR_MIN_VALUE = 1e9           # 20일 평균 거래대금 10억 원
KR_VALUE_WIN = 20
ALIVE_DAYS = 14              # 마지막 데이터가 패널 끝에서 이 안이면 '살아 있음'(상장폐지 아님)
N_BOOT = 2000
ALPHA = 0.05
MIN_N = 100
CONFIRM_RATIO = 0.5
MARKET_FIELDS = {"kr_stock": ("high", "low", "value", "market_cap"), "us_stock": ("high", "low")}
SURVIVORSHIP = {"kr_stock": "없음 (krx_1d 상장폐지 포함)", "us_stock": "있음 (us_1d 현재 구성종목만 → 결과 과대 가능, KR의 보조로 봄)"}


# ───────────────────────── 데이터 ─────────────────────────
def load_wide(market: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """load() 결과를 필드 → 날짜×종목 표로. 실행일의 dev_end를 캐시 키에 넣어 잠금 구간이 섞이지 않게 한다."""
    end = bdata.dev_end()
    path = CACHE / f"{market}_{START}_{end.date()}.pkl"
    if use_cache and path.exists():
        return pd.read_pickle(path)
    p = bdata.load(market, start=START, fields=MARKET_FIELDS[market])   # lockbox=False (기본) → 잠금 구간 제외
    wide = {k: v for k, v in p.items()}
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        pd.to_pickle(wide, path)
    return wide


def slice_wide(wide: dict, end=None) -> dict:
    """[처음, end) 구간 (end 미포함)."""
    if end is None:
        return wide
    idx = wide["close"].index
    m = idx < pd.Timestamp(end)
    return {k: v.loc[m] for k, v in wide.items()}


def truncate_wide(wide: dict, cut) -> dict:
    """[처음, cut] (cut 포함) — 잘라내기 검사용."""
    idx = wide["close"].index
    m = idx <= pd.Timestamp(cut)
    return {k: v.loc[m] for k, v in wide.items()}


def universe_wide(market: str, wide: dict) -> pd.DataFrame | None:
    """날짜×종목 bool. KR = 그날 시총 상위 500 ∧ 20일 평균 거래대금 ≥ 10억 ∧ 그날 거래 가능. US = None(전체).
    모두 그날 종가 시점까지의 값만 쓴다 (rank는 같은 날 횡단면, rolling은 과거 20행)."""
    if market != "kr_stock":
        return None
    mc = wide["market_cap"].where(wide["market_cap"] > 0)
    rank = mc.rank(axis=1, ascending=False, method="first")
    v20 = wide["value"].astype(float).rolling(KR_VALUE_WIN, min_periods=KR_VALUE_WIN).mean()
    trad = wide["tradable"].astype("boolean").fillna(False).astype(bool)
    return (rank <= KR_TOP_MCAP) & (v20 >= KR_MIN_VALUE) & trad


@dataclass
class Stock:
    sym: str
    dates: np.ndarray          # datetime64[ns], OHLC가 모두 있는 일봉만
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray


def stocks_from_wide(wide: dict) -> list[Stock]:
    idx = wide["close"].index.values
    O, H, L, C = (wide[f].to_numpy(dtype=float) for f in ("open", "high", "low", "close"))
    out = []
    for j, sym in enumerate(wide["close"].columns):
        ok = np.isfinite(O[:, j]) & np.isfinite(H[:, j]) & np.isfinite(L[:, j]) & np.isfinite(C[:, j])
        ok &= (O[:, j] > 0) & (C[:, j] > 0)
        if ok.sum() < 2:
            continue
        out.append(Stock(str(sym), idx[ok], O[ok, j], H[ok, j], L[ok, j], C[ok, j]))
    return out


def shuffle_stock(s: Stock, seed: int, boundary=BOUNDARY) -> Stock:
    """헛발견 검사용 가짜 시장: (시가/전일종가, 고가/종가, 저가/종가, 종가/전일종가) 묶음을 날마다 무작위로 섞는다.
    날짜·첫 종가는 그대로. 탐색/확인 구간 안에서만 섞는다 (확인 구간 수익이 탐색 구간으로 새지 않게)."""
    rng = np.random.default_rng([seed, zlib.crc32(s.sym.encode())])
    n = len(s.c)
    pc = s.c[:-1]
    tup = np.column_stack([s.o[1:] / pc, s.h[1:] / s.c[1:], s.l[1:] / s.c[1:], s.c[1:] / pc])
    seg = s.dates[1:] < np.datetime64(boundary)
    perm = np.arange(n - 1)
    for m in (seg, ~seg):
        pos = np.flatnonzero(m)
        perm[pos] = pos[rng.permutation(len(pos))]
    tup = tup[perm]
    c = np.empty(n)
    c[0] = s.c[0]
    c[1:] = s.c[0] * np.cumprod(tup[:, 3])
    o, h, l = s.o.copy(), s.h.copy(), s.l.copy()
    o[1:] = c[:-1] * tup[:, 0]
    h[1:] = c[1:] * tup[:, 1]
    l[1:] = c[1:] * tup[:, 2]
    return Stock(s.sym, s.dates, o, h, l, c)


# ───────────────────────── 봉 ─────────────────────────
def week_end_rows(dates: np.ndarray) -> np.ndarray:
    """W-FRI 주마다 마지막 거래일 행 번호."""
    wk = week_key(dates)
    last = np.flatnonzero(np.r_[wk[1:] != wk[:-1], True])
    return last


def week_key(dates) -> np.ndarray:
    """W-FRI 주 번호 (토요일 시작 주). 1970-01-03(토) 기준 7일 단위."""
    d = np.asarray(dates, dtype="datetime64[D]").astype(np.int64)
    return (d - 2) // 7


def bars_of(s: Stock, timeframe: str) -> tuple[pd.DataFrame, np.ndarray]:
    """탐지기에 줄 봉 표와, 봉마다 '그 봉 마지막 일봉 행 번호'(end_row)."""
    if timeframe == "1d":
        end = np.arange(len(s.c))
        df = pd.DataFrame({"open": s.o, "high": s.h, "low": s.l, "close": s.c}, index=pd.DatetimeIndex(s.dates))
        return df, end
    if timeframe != "1w":
        raise ValueError(timeframe)
    end = week_end_rows(s.dates)
    start = np.r_[0, end[:-1] + 1]
    hi = np.maximum.reduceat(s.h, start)
    lo = np.minimum.reduceat(s.l, start)
    df = pd.DataFrame({"open": s.o[start], "high": hi, "low": lo, "close": s.c[end]},
                      index=pd.DatetimeIndex(s.dates[end]))
    return df, end


# ───────────────────────── 비용 ─────────────────────────
class Costs:
    """config/settings.yaml에서 읽은 왕복 비용. KR 매도세는 청산일 기준 연도별 세율표."""

    def __init__(self, market: str):
        s = bdata._settings()
        m = s["markets"][market]
        self.fee = m["fee_pct"] / 100
        self.slip = s["backtest"]["slippage_pct"] / 100
        self.base = 2 * (self.fee + self.slip)
        tax = m.get("tax_pct_sell") or {}
        keys = sorted(tax)
        self.tax_dates = np.array([np.datetime64(pd.Timestamp(k)) for k in keys], dtype="datetime64[ns]")
        self.tax_vals = np.array([tax[k] / 100 for k in keys])

    def roundtrip(self, exit_dates) -> np.ndarray:
        d = np.asarray(exit_dates, dtype="datetime64[ns]")
        out = np.full(d.shape, self.base)
        if len(self.tax_vals):
            i = np.searchsorted(self.tax_dates, d, side="right") - 1
            out = out + np.where(i >= 0, self.tax_vals[np.clip(i, 0, None)], 0.0)
        return out

    def describe(self) -> str:
        t = ", ".join(f"{pd.Timestamp(d).date()}~ {v * 100:.2f}%" for d, v in zip(self.tax_dates, self.tax_vals))
        return (f"수수료 {self.fee * 100:.3f}%×2 + 슬리피지 {self.slip * 100:.3f}%×2"
                + (f" + 매도세(청산일 기준: {t})" if t else ""))


# ───────────────────────── 봉 표 ─────────────────────────
@dataclass
class BarTable:
    market: str
    timeframe: str
    df: pd.DataFrame                    # 유니버스 행만: sym, k, date, year, cl, mom, quint, period, ret_h, exit_h, delist_h, cost_h
    kdates: dict = field(default_factory=dict)   # sym → 봉 날짜 (이벤트 날짜 → 봉 번호)
    syms: list = field(default_factory=list)


def build_bars(market: str, timeframe: str, stocks: list[Stock], univ: pd.DataFrame | None,
               panel_last, costs: Costs) -> BarTable:
    panel_last = np.datetime64(pd.Timestamp(panel_last))
    alive_cut = panel_last - np.timedelta64(ALIVE_DAYS, "D")
    if univ is not None:
        u_idx = univ.index.values
        u_arr = univ.to_numpy()
        u_col = {c: j for j, c in enumerate(univ.columns)}
    cols = {k: [] for k in ("sym", "k", "date", "mom")}
    for h in HORIZONS:
        for p in ("ret", "exit", "delist"):
            cols[f"{p}{h}"] = []
    kdates, syms = {}, []
    for si, s in enumerate(stocks):
        syms.append(s.sym)
        _, end = bars_of(s, timeframe)
        bd = s.dates[end]
        kdates[s.sym] = bd
        K, n = len(end), len(s.c)
        if univ is None:
            keep = np.ones(K, bool)
        else:
            j = u_col.get(s.sym)
            if j is None:
                continue
            keep = u_arr[np.searchsorted(u_idx, bd), j]
        if not keep.any():
            continue
        bc = s.c[end]
        mom = np.full(K, np.nan)
        if K > MOM_BARS:
            mom[MOM_BARS:] = bc[MOM_BARS:] / bc[:-MOM_BARS] - 1
        entry = end + 1
        ok_entry = entry < n
        entry_c = np.minimum(entry, n - 1)
        alive = s.dates[-1] >= alive_cut
        kk = np.arange(K)
        for h in HORIZONS:
            jj = kk + h
            short = jj >= K
            exr = np.where(short, n - 1, end[np.minimum(jj, K - 1)])
            valid = ok_entry & (~short | (not alive))
            ret = s.c[exr] / s.o[entry_c] - 1
            ret = np.where(valid, ret, np.nan)
            cols[f"ret{h}"].append(ret[keep])
            cols[f"exit{h}"].append(s.dates[exr][keep])
            cols[f"delist{h}"].append((short & valid)[keep])
        cols["sym"].append(np.full(keep.sum(), si, dtype=np.int32))
        cols["k"].append(kk[keep])
        cols["date"].append(bd[keep])
        cols["mom"].append(mom[keep])
    df = pd.DataFrame({k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()})
    df["year"] = pd.DatetimeIndex(df["date"]).year.astype(np.int16)
    df["cl"] = week_key(df["date"].values) if timeframe == "1w" else df["date"].values.astype("datetime64[D]").astype(np.int64)
    # 보조(더 보수적) 묶음: 연속 MAIN_H봉 블록 (일봉 = 거래일 순번 // 20, 주봉 = 주 번호 // 20).
    # 보유 기간이 겹치는 이웃 날짜 신호의 상관까지 묶는다. 판정은 사양대로 cl로 하고 이 값은 참고 열로 낸다.
    if timeframe == "1w":
        df["blk"] = df["cl"].values // MAIN_H
    else:
        days = np.unique(df["cl"].values)
        df["blk"] = np.searchsorted(days, df["cl"].values) // MAIN_H
    # 횡단면 5분위: 같은 날짜(주봉은 같은 마지막 거래일)의 유니버스 종목끼리. 그날까지의 값만 쓴다.
    g = df.groupby("date")["mom"]
    pct = g.rank(pct=True, method="first")
    cnt = g.transform("count")
    q = np.ceil(pct * N_QUINT) - 1
    df["quint"] = q.where(cnt >= MIN_XS).astype("float32")
    # 구간: 0 = 탐색(신호일·20봉 청산일 모두 경계 전), 1 = 확인(신호일 ≥ 경계), -1 = 어느 쪽도 아님
    b = np.datetime64(BOUNDARY)
    d = df["date"].values
    v20 = np.isfinite(df[f"ret{MAIN_H}"].values)
    per = np.full(len(df), -1, np.int8)
    per[(d < b) & v20 & (df[f"exit{MAIN_H}"].values < b)] = 0
    per[(d >= b) & v20] = 1
    df["period"] = per
    for h in HORIZONS:   # 탐색 구간 행의 h봉 청산이 경계를 넘으면 그 h 값은 쓰지 않는다
        bad = (per == 0) & (df[f"exit{h}"].values >= b)
        df.loc[bad, f"ret{h}"] = np.nan
        df[f"cost{h}"] = costs.roundtrip(df[f"exit{h}"].values)
    return BarTable(market, timeframe, df, kdates, syms)


# ───────────────────────── 탐지 ─────────────────────────
_DET = {}


def _init_worker(mod_path: str, mod_name: str):
    import importlib
    import sys
    if mod_path not in sys.path:
        sys.path.insert(0, mod_path)
    _DET["detect"] = importlib.import_module(mod_name).detect


def _detect_chunk(args):
    stocks, tools, timeframes = args
    det = _DET["detect"]
    out = []
    for s in stocks:
        for tf in timeframes:
            bars, _ = bars_of(s, tf)
            for tool in tools:
                ev = det(bars, tool, tf)
                if ev is None or len(ev) == 0:
                    continue
                ev = pd.DataFrame(ev).copy()
                ev["sym"] = s.sym
                ev["timeframe"] = tf
                out.append(ev)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_detection(stocks: list[Stock], tools, timeframes, mod_path: str, mod_name: str,
                  workers: int = 1) -> pd.DataFrame:
    """종목 전체 이력으로 탐지 (유니버스 거르기는 나중). 탐지기는 mod_name 모듈의 detect(bars, tool, timeframe)."""
    if workers <= 1:
        _init_worker(mod_path, mod_name)
        res = [_detect_chunk((stocks, tools, timeframes))]
    else:
        n = max(1, len(stocks) // (workers * 4))
        chunks = [(stocks[i:i + n], tools, timeframes) for i in range(0, len(stocks), n)]
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(mod_path, mod_name)) as ex:
            res = list(ex.map(_detect_chunk, chunks))
    res = [r for r in res if len(r)]
    if not res:
        return pd.DataFrame(columns=["date", "direction", "tool", "variant", "sym", "timeframe"])
    ev = pd.concat(res, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["date"]).values.astype("datetime64[ns]")
    ev["direction"] = ev["direction"].astype(int)
    ev["variant"] = ev["variant"].astype(str)
    return ev


def attach_events(ev: pd.DataFrame, bt: BarTable, periods) -> tuple[pd.DataFrame, dict, dict]:
    """탐지 결과(한 market·timeframe) → 봉 표 행에 붙인 이벤트. 반환: (이벤트, 같은 도구 이벤트 봉 번호(차단용), 집계)."""
    stat = {"detected": len(ev)}
    sym_id = {s: i for i, s in enumerate(bt.syms)}
    ev = ev[ev["sym"].isin(sym_id)].copy()
    ev["sid"] = ev["sym"].map(sym_id).astype(np.int32)
    ks = np.full(len(ev), -1, np.int64)
    for sym, g in ev.groupby("sym").groups.items():
        bd = bt.kdates[sym]
        dv = ev.loc[g, "date"].values
        pos = np.searchsorted(bd, dv)
        pos_c = np.minimum(pos, len(bd) - 1)
        ks[ev.index.get_indexer(g)] = np.where(bd[pos_c] == dv, pos_c, -1)
    ev["k"] = ks
    stat["not_on_bar"] = int((ks < 0).sum())
    ev = ev[ev["k"] >= 0]
    blocked = {key: np.unique(g.values) for key, g in ev.groupby(["sid", "tool"])["k"]}   # 유니버스와 상관없이 모든 신호
    bars = bt.df.rename(columns={"sym": "sid"})
    m = ev.merge(bars, on=["sid", "k"], how="left", suffixes=("", "_bar"), indicator=True)
    stat["outside_universe"] = int((m["_merge"] == "left_only").sum())
    m = m[m["_merge"] == "both"].drop(columns=["_merge", "date_bar"], errors="ignore")
    stat["no_return_or_gap"] = int((~m["period"].isin(periods)).sum())
    m = m[m["period"].isin(periods)].reset_index(drop=True)
    stat["in_period"] = len(m)
    return m, blocked, stat


# ───────────────────────── 위약 ─────────────────────────
def sample_placebos(ev: pd.DataFrame, bt: BarTable, blocked: dict, seed: int, matched: bool) -> np.ndarray:
    """이벤트마다 위약 봉 표 행 번호 N_PLACEBO개 (없으면 -1). 같은 종목·같은 연도·같은 구간·유니버스·20봉 수익 있음,
    같은 도구 이벤트 ±BLOCK_BARS 제외. matched면 직전 60봉 수익률 횡단면 5분위도 같게."""
    rng = np.random.default_rng(seed)
    out = np.full((len(ev), N_PLACEBO), -1, np.int64)
    df = bt.df
    pool_ok = (df["period"].values >= 0)
    if matched:
        pool_ok &= np.isfinite(df["quint"].values)
    pidx = np.flatnonzero(pool_ok)
    psym = df["sym"].values[pidx]
    order = np.argsort(psym, kind="stable")
    pidx, psym = pidx[order], psym[order]
    pk, pyear = df["k"].values[pidx], df["year"].values[pidx].astype(np.int64)
    pper = df["period"].values[pidx].astype(np.int64)
    pq = np.nan_to_num(df["quint"].values[pidx], nan=-1).astype(np.int64) if matched else np.zeros(len(pidx), np.int64)
    pkey = pper * 1_000_000 + pyear * 10 + pq
    ekey = (ev["period"].values.astype(np.int64) * 1_000_000 + ev["year"].values.astype(np.int64) * 10
            + (np.nan_to_num(ev["quint"].values, nan=-9).astype(np.int64) if matched else 0))
    off = np.arange(-BLOCK_BARS, BLOCK_BARS + 1)
    for (sid, tool), rows in ev.groupby(["sid", "tool"]).indices.items():
        lo, hi = np.searchsorted(psym, sid), np.searchsorted(psym, sid, side="right")
        if hi == lo:
            continue
        bk = blocked.get((sid, tool))
        cand_ok = ~np.isin(pk[lo:hi], (bk[:, None] + off).ravel()) if bk is not None else np.ones(hi - lo, bool)
        ck, ci = pkey[lo:hi][cand_ok], pidx[lo:hi][cand_ok]
        er = ekey[rows]
        for key in np.unique(er):
            if key < 0:
                continue
            cand = ci[ck == key]
            P = len(cand)
            if P == 0:
                continue
            tgt = rows[er == key]
            m = len(tgt)
            if P >= N_PLACEBO:
                pick = np.argpartition(rng.random((m, P)), N_PLACEBO - 1, axis=1)[:, :N_PLACEBO]
            else:
                pick = rng.integers(P, size=(m, N_PLACEBO))
            out[tgt] = cand[pick]
    return out


def placebo_means(idx: np.ndarray, bt: BarTable, col: str) -> np.ndarray:
    v = bt.df[col].values[np.clip(idx, 0, None)].astype(float)
    v[idx < 0] = np.nan
    with np.errstate(all="ignore"):
        cnt = np.isfinite(v).sum(1)
        s = np.nansum(v, 1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


# ───────────────────────── 통계 ─────────────────────────
def norm_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def cluster_bootstrap(d: np.ndarray, cl: np.ndarray, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """날짜 묶음 부트스트랩. p = 부트스트랩 표준오차로 만든 양측 정규 p (Holm에 필요한 작은 p를 2000회로도 낼 수 있게),
    p_emp = 영가설 중심화 경험 p ((1+#|m*−m|≥|m|)/(B+1), 최소 1/2001)."""
    ok = np.isfinite(d)
    d, cl = d[ok], cl[ok]
    if len(d) == 0:
        return {"mean": np.nan, "se": np.nan, "p": np.nan, "p_emp": np.nan, "clusters": 0}
    _, inv = np.unique(cl, return_inverse=True)
    K = inv.max() + 1
    s = np.bincount(inv, weights=d, minlength=K)
    c = np.bincount(inv, minlength=K).astype(float)
    est = s.sum() / c.sum()
    if K < 2:
        return {"mean": est, "se": np.nan, "p": np.nan, "p_emp": np.nan, "clusters": int(K)}
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    step = max(1, 4_000_000 // K)
    for i in range(0, n_boot, step):
        b = min(step, n_boot - i)
        ix = rng.integers(K, size=(b, K))
        bs[i:i + b] = s[ix].sum(1) / c[ix].sum(1)
    se = bs.std(ddof=1)
    p = norm_p(est / se) if se > 0 else np.nan
    p_emp = (1 + np.sum(np.abs(bs - est) >= abs(est))) / (n_boot + 1)
    return {"mean": est, "se": se, "p": p, "p_emp": p_emp, "clusters": int(K)}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    adj = np.minimum(1, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


UNIT = ["tool", "variant", "direction", "market", "timeframe"]


def unit_stats(ev: pd.DataFrame, boot_seed: int = 0) -> pd.DataFrame:
    """검정 단위 × 구간별 요약. ev에는 d{h}_m (추세 맞춘 위약 대비), d{h}_s (단순 위약 대비), r{h}, cost{h} 열이 있다."""
    rows = []
    for key, g in ev.groupby(UNIT + ["period"], sort=True):
        r = dict(zip(UNIT + ["period"], key))
        use = g[np.isfinite(g[f"d{MAIN_H}_m"])]
        r["n_events"] = len(g)
        r["n"] = len(use)
        r["n_no_matched_placebo"] = int(len(g) - len(use))
        r["n_delisted_exit20"] = int(use[f"delist{MAIN_H}"].sum())
        for h in HORIZONS:
            r[f"ret{h}_gross"] = use[f"r{h}"].mean()
            r[f"ret{h}_net"] = (use[f"r{h}"] - use[f"cost{h}"]).mean()
            r[f"exc{h}_m"] = use[f"d{h}_m"].mean()
            r[f"exc{h}_s"] = use[f"d{h}_s"].mean()
        r[f"cost{MAIN_H}"] = use[f"cost{MAIN_H}"].mean()
        r[f"exc{MAIN_H}_m_after_cost"] = r[f"exc{MAIN_H}_m"] - r[f"cost{MAIN_H}"]
        bm = cluster_bootstrap(use[f"d{MAIN_H}_m"].values, use["cl"].values, boot_seed)
        bs_ = cluster_bootstrap(use[f"d{MAIN_H}_s"].values, use["cl"].values, boot_seed)
        bb = cluster_bootstrap(use[f"d{MAIN_H}_m"].values, use["blk"].values, boot_seed)
        r |= {"se_m": bm["se"], "p_m": bm["p"], "p_m_emp": bm["p_emp"], "clusters": bm["clusters"],
              "p_s": bs_["p"], "p_m_blk": bb["p"], "blocks": bb["clusters"]}
        rows.append(r)
    return pd.DataFrame(rows)


def judge(summary: pd.DataFrame) -> pd.DataFrame:
    """README 1절 A 통과 기준. Holm은 이번 실행의 모든 검정 단위(탐색 구간 20봉, 추세 맞춘 위약)."""
    s = summary.copy()
    ex = s[s["period"] == 0].copy()
    ex["p_holm"] = holm(ex["p_m"].values)
    ex["p_holm_blk"] = holm(ex["p_m_blk"].values)   # 참고: 20봉 블록 묶음 (더 보수적)
    cf = s[s["period"] == 1].set_index(UNIT)
    out = ex.set_index(UNIT)
    out["c1_holm"] = out["p_holm"] < ALPHA
    out["c2_cost"] = out[f"exc{MAIN_H}_m_after_cost"] > 0          # = 초과수익 > 왕복 비용
    out["c4_n"] = out["n"] >= MIN_N
    if len(cf):
        ce = cf[f"exc{MAIN_H}_m"].reindex(out.index)
        out["confirm_n"] = cf["n"].reindex(out.index)
        out["confirm_exc20"] = ce
        same = np.sign(ce) == np.sign(out[f"exc{MAIN_H}_m"])
        out["c3_confirm"] = (same & (ce.abs() >= CONFIRM_RATIO * out[f"exc{MAIN_H}_m"].abs())).where(ce.notna())
    else:
        out["confirm_n"] = np.nan
        out["confirm_exc20"] = np.nan
        out["c3_confirm"] = np.nan
    base = out["c1_holm"] & out["c2_cost"] & out["c4_n"]
    out["verdict"] = np.where(~base, "불통과",
                              np.where(out["c3_confirm"].isna(), "탐색 통과(확인 대기)",
                                       np.where(out["c3_confirm"].astype("boolean").fillna(False), "통과", "불통과(확인)")))
    # 참고 판정: Holm을 20봉 블록 묶음 p로 바꾼 것 (사양 판정보다 엄격하거나 같음)
    out["verdict_blk"] = np.where((out["verdict"] != "불통과") & (out["p_holm_blk"] < ALPHA), out["verdict"], "불통과")
    return out.reset_index()


# ───────────────────────── 잘라내기 검사 ─────────────────────────
def check_universe_causality(market: str, wide: dict, timeframe: str, cuts, costs: Costs) -> dict:
    """유니버스·직전 60봉 수익률·5분위를 데이터를 cut에서 잘라 다시 계산해 원래 값(cut 이전 행)과 비교."""
    full = _feat(market, wide, timeframe, costs)
    res = {"cuts": [], "compared": 0, "mismatch": 0, "missing_in_cut": 0}
    for cut in cuts:
        w2 = truncate_wide(wide, cut)
        part = _feat(market, w2, timeframe, costs)
        a = full[full["date"] <= np.datetime64(pd.Timestamp(cut))]   # cut은 주의 마지막 거래일로 고른다 (주봉 미완성 방지)
        m =a.merge(part, on=["sym", "date"], how="left", suffixes=("", "_cut"), indicator=True)
        miss = int((m["_merge"] == "left_only").sum())
        both = m[m["_merge"] == "both"]
        mm = 0
        for c in ("mom", "quint"):
            x, y = both[c].values.astype(float), both[f"{c}_cut"].values.astype(float)
            mm += int((~((x == y) | (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-12))).sum())
        # 유니버스: full에서 유니버스였던 행이 cut 데이터에서도 유니버스인가 (left_only = cut 데이터에서 유니버스 아님)
        res["cuts"].append({"cut": str(pd.Timestamp(cut).date()), "rows": int(len(a)), "mismatch": mm,
                            "universe_missing": miss, "extra_universe_in_cut": int(len(part[part["date"] <= np.datetime64(pd.Timestamp(cut))]) - len(both))})
        res["compared"] += len(both)
        res["mismatch"] += mm
        res["missing_in_cut"] += miss
    res["ok"] = res["mismatch"] == 0 and res["missing_in_cut"] == 0 and all(c["extra_universe_in_cut"] == 0 for c in res["cuts"])
    return res


def _feat(market, wide, timeframe, costs):
    stocks = stocks_from_wide(wide)
    bt = build_bars(market, timeframe, stocks, universe_wide(market, wide), wide["close"].index[-1], costs)
    df = bt.df[["sym", "date", "mom", "quint"]].copy()
    df["sym"] = np.array(bt.syms)[df["sym"].values]
    return df
