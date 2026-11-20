"""L014 — 박스권 하단 지지 매수 · 상단 저항 앞 매도 (손익비 3:1 이상만), type: trade.

봉 t 종가까지 보고:
  1. 유니버스: 20일 평균 거래대금 ≥ min_value, 원본 종가 ≥ min_price, 250봉 이상 이력.
  2. 박스: 직전 box_n봉(t 제외)의 고가 최대 = 상단, 저가 최대 = 하단.
     폭 (상단/하단 − 1) ≤ max_width. 상단·하단 둘 다 최근 quiet봉보다 먼저 만들어졌다
     (최근 quiet봉 안에 새 고점·새 저점이 없음 = 경계가 이미 확인된 옆걸음. 하락 추세가 지금 새 저점을 만드는 중이면 제외).
  3. 하단 지지·반등 확인: 저가 ≤ 하단 × (1 + zone), 종가 > 하단, 양봉(종가 > 시가), 종가 > 전일 종가.
  4. 손절 = min(당일 저가, 하단) × (1 − stop_buf),  목표 = 상단 × (1 − tgt_buf) (상단 저항 앞에서 판다).
  5. 손익비 (목표 − 종가) / (종가 − 손절) ≥ rr_min. 같은 날 거래대금 순 상위 max_new개.
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
    n, q = int(p["box_n"]), int(p["quiet"])
    o, h, l, c = data.open, data.high, data.low, data.close
    val = _rmean(_dollar_value(data), 20)
    px = data.raw_close if "raw_close" in data else c
    hist = c.notna().rolling(255, min_periods=1).sum() >= 255
    elig = (val >= p["min_value"]) & (px >= p["min_price"]) & hist

    top = h.rolling(n, min_periods=n).max().shift(1)
    bot = l.rolling(n, min_periods=n).min().shift(1)
    top_old = h.rolling(n - q, min_periods=n - q).max().shift(1 + q)    # t-n .. t-q-1
    bot_old = l.rolling(n - q, min_periods=n - q).min().shift(1 + q)
    boxed = (top / bot - 1 <= p["max_width"]) & (top_old >= top) & (bot_old <= bot)

    bounce = (l <= bot * (1 + p["zone"])) & (c > bot) & (c > o) & (c > c.shift(1))
    stop = l.where(l < bot, bot) * (1 - p["stop_buf"])
    target = top * (1 - p["tgt_buf"])
    rr = (target - c) / (c - stop)
    ok = elig & boxed & bounce & (rr >= p["rr_min"]) & (stop < c) & (target > c)

    df = _rows(ok, stop=stop, target=target, rr=rr, val=val)
    if df.empty:
        return []
    df = df.sort_values(["ts", "val", "symbol"], ascending=[True, False, True])
    df = df.groupby("ts", sort=False).head(int(p["max_new"]))
    df["side"] = "long"
    df["reason"] = "박스 하단 반등 RR " + df["rr"].round(2).astype(str)
    return df[["ts", "symbol", "side", "stop", "target", "reason"]].reset_index(drop=True)
