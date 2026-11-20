"""L015 — 거래량 동반 돌파 (전고점·이평), 실패 시 바로 손절, 다음 저항대 앞 익절, type: trade.

봉 t 종가까지 보고:
  1. 유니버스: 20일 평균 거래대금 ≥ min_value, 원본 종가 ≥ min_price, 가장 긴 이평이 계산될 만큼 이력.
  2. 돌파: 오늘 처음 넘은 저항선 중 가장 높은 것을 돌파선으로 쓴다.
     저항선 = 직전 brk_n봉(t 제외) 고가 최대(전고점), 이평 brk_mas(예: 60·120·240일).
     "처음 넘음" = 종가 > 선 이고 전일 종가 ≤ 그 선(전고점은 전일 기준 같은 창의 값과 비교).
  3. 거래량 확인: 거래량 ≥ vol_mult × 직전 20봉 평균 거래량, 양봉(종가 > 시가).
  4. 손절 (돌파 실패 = 선 아래로 되밀림) = 돌파선 × (1 − fail_buf).
  5. 목표 = 종가 위 가장 가까운 다음 저항 × (1 − tgt_buf). 저항 후보 = 종가 위의 이평(모든 ma_periods),
     직전 res_n봉 고가 최대. 위에 저항이 없으면(신고가 영역) 목표 = 종가 + rr_min × (종가 − 손절).
  6. 손익비 (목표 − 종가) / (종가 − 손절) ≥ rr_min. 같은 날 거래대금 순 상위 max_new개.
청산은 엔진이 손절·목표가로만 한다 (시간 청산 없음). 학습(fit) 없음.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rmean(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """rolling(k, min_periods=k).mean()과 같은 값 (창 안 k개가 모두 있어야 함). numpy 누적합으로 빠르게."""
    a = df.to_numpy(float)
    v = np.isfinite(a)
    cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(np.where(v, a, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(v, axis=0)])
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= k:
        s = cs[k:] - cs[:-k]
        n = cn[k:] - cn[:-k]
        out[k - 1:] = np.where(n == k, s / k, np.nan)
    return pd.DataFrame(out, df.index, df.columns)


def _rows(ok: pd.DataFrame, **cols) -> pd.DataFrame:
    """True인 칸마다 한 행 (ts, symbol, 그리고 같은 모양 표들의 값)."""
    ti, si = np.nonzero(ok.to_numpy(bool))
    out = pd.DataFrame({"ts": ok.index[ti], "symbol": ok.columns[si]})
    for name, f in cols.items():
        out[name] = f.to_numpy(float)[ti, si]
    return out


def _dollar_value(data):
    if "value" in data:
        return data.value
    return data.raw_close * data.volume if "raw_close" in data else data.close * data.volume


def generate_trades(data, params: dict):
    p = params
    o, h, l, c, v = data.open, data.high, data.low, data.close, data.volume
    periods = [int(k) for k in p["ma_periods"]]
    mmax = max(periods)
    val = _rmean(_dollar_value(data), 20)
    px = data.raw_close if "raw_close" in data else c
    hist = c.notna().rolling(mmax + 5, min_periods=1).sum() >= mmax + 5
    elig = (val >= p["min_value"]) & (px >= p["min_price"]) & hist
    mas = {k: _rmean(c, k) for k in periods}

    n = int(p["brk_n"])
    hi_n = h.rolling(n, min_periods=n).max().shift(1)
    C = c.to_numpy(float)
    C1 = c.shift(1).to_numpy(float)
    level = np.full(C.shape, -np.inf)
    lines = [(hi_n.to_numpy(float), hi_n.to_numpy(float))]          # 오늘 비교값, 어제 비교값(같은 창)
    lines += [(mas[int(k)].to_numpy(float), mas[int(k)].shift(1).to_numpy(float)) for k in p["brk_mas"]]
    with np.errstate(invalid="ignore"):
        for today, yday in lines:
            crossed = (C > today) & (C1 <= yday) & np.isfinite(today)
            level = np.where(crossed, np.maximum(level, today), level)
    level = pd.DataFrame(np.where(np.isfinite(level), level, np.nan), c.index, c.columns)

    vol_ok = v >= p["vol_mult"] * _rmean(v, 20).shift(1)
    stop = level * (1 - p["fail_buf"])

    cand = [m.to_numpy(float) for m in mas.values()]
    rn = int(p["res_n"])
    cand.append(h.rolling(rn, min_periods=rn).max().shift(1).to_numpy(float))
    res = np.full(C.shape, np.inf)
    with np.errstate(invalid="ignore"):
        for a in cand:
            res = np.where((a > C * 1.001) & np.isfinite(a), np.minimum(res, a), res)
    measured = C + p["rr_min"] * (C - stop.to_numpy(float))
    target = np.where(np.isfinite(res), res * (1 - p["tgt_buf"]), measured)
    target = pd.DataFrame(target, c.index, c.columns)
    rr = (target - c) / (c - stop)
    ok = elig & level.notna() & vol_ok & (c > o) & (rr >= p["rr_min"] - 1e-9) & (stop < c) & (target > c)

    df = _rows(ok, stop=stop, target=target, rr=rr, val=val)
    if df.empty:
        return []
    df = df.sort_values(["ts", "val", "symbol"], ascending=[True, False, True])
    df = df.groupby("ts", sort=False).head(int(p["max_new"]))
    df["side"] = "long"
    df["reason"] = "거래량 돌파 RR " + df["rr"].round(2).astype(str)
    return df[["ts", "symbol", "side", "stop", "target", "reason"]].reset_index(drop=True)
