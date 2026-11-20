"""L013 — 이동평균 지지 반등 (손익비 3:1 이상만 진입), type: trade.

봉 t 종가까지 보고:
  1. 유니버스: 20일 평균 거래대금 ≥ min_value, 원본 종가 ≥ min_price, 가장 긴 이평이 계산될 만큼 이력.
  2. 이평 k(상승 중: MA_k > 5봉 전 MA_k)마다 "지지 터치": 저가 ≤ MA_k × (1 + touch_pct), 종가 > MA_k,
     전일 종가 > 전일 MA_k (위에서 내려와 닿은 것 — 아래에서 뚫고 올라온 돌파는 L003·L015 영역), 양봉(종가 > 시가).
  3. 점수 = 터치한 이평들의 가중값 합 (여러 이평이 한곳에 모이면 지지가 강하다). 가중값은 fit()이 학습 창에서 정한다.
     가중값 0인 이평은 신호를 만들지 않는다.
  4. 손절 = min(당일 저가, 터치한 이평 중 가장 낮은 값) × (1 − stop_buf)
  5. 목표 = 종가 위 가장 가까운 저항 × (1 − tgt_buf). 저항 후보 = 종가 위에 있는 모든 이평, 직전 res_n봉 고가 최대,
     직전 250봉 고가 최대. 위에 저항이 하나도 없으면 목표를 정할 수 없어 진입하지 않는다.
  6. 손익비 (목표 − 종가) / (종가 − 손절) ≥ rr_min 인 것만. 같은 날 점수 → 거래대금 순 상위 max_new개.
청산은 엔진이 손절·목표가로만 한다 (시간 청산 없음).
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

DEFAULT_W = {5: 0.5, 10: 0.6, 20: 0.8, 60: 1.0, 120: 1.2, 240: 1.5}   # fit 전 사전값: 긴 이평일수록 강함 (사용자 가정)


def _mas(data, periods):
    c = data.close
    return {k: _rmean(c, k) for k in periods}


def _dollar_value(data):
    if "value" in data:
        return data.value
    return data.raw_close * data.volume if "raw_close" in data else data.close * data.volume


def _eligible(data, p, ma_max):
    val = _rmean(_dollar_value(data), 20)
    px = data.raw_close if "raw_close" in data else data.close
    hist = data.close.notna().rolling(ma_max + 5, min_periods=1).sum() >= ma_max + 5
    return (val >= p["min_value"]) & (px >= p["min_price"]) & hist, val


def _features(data, p):
    """신호 계산에 필요한 배열을 한 번에. 모두 봉 t 종가까지의 값."""
    periods = [int(k) for k in p["ma_periods"]]
    o, h, l, c = data.open, data.high, data.low, data.close
    mas = _mas(data, periods)
    elig, val = _eligible(data, p, max(periods))
    bull = c > o
    touch = {}
    for k, m in mas.items():
        rising = m > m.shift(5)
        touch[k] = (l <= m * (1 + p["touch_pct"])) & (c > m) & (c.shift(1) > m.shift(1)) & rising & bull & elig
    # 저항: 종가 위 가장 가까운 것
    C = c.to_numpy(float)
    cand = [m.to_numpy(float) for m in mas.values()]
    cand.append(h.rolling(int(p["res_n"]), min_periods=int(p["res_n"])).max().shift(1).to_numpy(float))
    cand.append(h.rolling(250, min_periods=250).max().shift(1).to_numpy(float))
    res = np.full(C.shape, np.inf)
    for a in cand:
        with np.errstate(invalid="ignore"):
            res = np.where((a > C * 1.001) & np.isfinite(a), np.minimum(res, a), res)
    res = np.where(np.isfinite(res), res, np.nan)
    return periods, mas, touch, pd.DataFrame(res, c.index, c.columns), val


def _signals(data, p, weights):
    periods, mas, touch, res, val = _features(data, p)
    c, l = data.close, data.low
    score = sum(touch[k].astype(float) * float(weights.get(k, weights.get(str(k), 0.0))) for k in periods)
    support = l.copy()
    for k in periods:
        w = float(weights.get(k, weights.get(str(k), 0.0)))
        if w > 0:
            support = support.where(~touch[k], np.fmin(support, mas[k]))
    stop = support * (1 - p["stop_buf"])
    target = res * (1 - p["tgt_buf"])
    rr = (target - c) / (c - stop)
    ok = (score > 0) & (rr >= p["rr_min"]) & (stop < c) & (target > c)
    return ok, score, stop, target, rr, val


def generate_trades(data, params: dict):
    w = params.get("weights") or DEFAULT_W
    ok, score, stop, target, rr, val = _signals(data, params, w)
    df = _rows(ok, score=score, stop=stop, target=target, rr=rr, val=val)
    if df.empty:
        return []
    df = df.sort_values(["ts", "score", "val", "symbol"], ascending=[True, False, False, True])
    df = df.groupby("ts", sort=False).head(int(params["max_new"]))
    df["side"] = "long"
    df["reason"] = "MA지지 점수 " + df["score"].round(2).astype(str) + " RR " + df["rr"].round(2).astype(str)
    return df[["ts", "symbol", "side", "stop", "target", "reason"]].reset_index(drop=True)


# ── 학습: 이평별 가중값 ──
def _outcome_r(data, ts_pos, sym_pos, stop, target, horizon, cost):
    """이벤트마다 t+1 시가 진입 → 손절·목표가 먼저 닿는 쪽 (같은 봉이면 손절, 시가 갭은 시가) → 아니면 horizon봉 종가.
    반환: 비용 뺀 R 배수. 학습 창 끝을 넘는 이벤트는 NaN (창 밖 데이터를 쓰지 않음)."""
    O, H, L, C = (data[f].to_numpy(float) for f in ("open", "high", "low", "close"))
    n = O.shape[0]
    m = len(ts_pos)
    r = np.full(m, np.nan)
    ent = np.full(m, np.nan)
    ok = ts_pos + horizon < n
    e = np.where(ok, ts_pos + 1, 0)
    ent[ok] = O[e[ok], sym_pos[ok]]
    alive = ok & np.isfinite(ent) & (ent > stop)
    exitp = np.full(m, np.nan)
    for hh in range(1, horizon + 1):
        if not alive.any():
            break
        i = np.where(alive)[0]
        t = ts_pos[i] + hh
        o, hi, lo, cl = O[t, sym_pos[i]], H[t, sym_pos[i]], L[t, sym_pos[i]], C[t, sym_pos[i]]
        s, g = stop[i], target[i]
        x = np.full(len(i), np.nan)
        x = np.where(np.isfinite(o) & (o <= s), o, x)
        x = np.where(np.isnan(x) & np.isfinite(o) & (o >= g), o, x)
        x = np.where(np.isnan(x) & (lo <= s), s, x)
        x = np.where(np.isnan(x) & (hi >= g), g, x)
        if hh == horizon:
            x = np.where(np.isnan(x), cl, x)
        done = np.isfinite(x)
        exitp[i[done]] = x[done]
        alive[i[done]] = False
    risk = (ent - stop) / ent
    r = ((exitp / ent - 1) - cost) / risk
    return r


def fit(train, params: dict) -> dict:
    """학습 창에서 이평마다 '그 이평 하나만 지지로 쓴 신호'의 비용 뒤 평균 R을 재고,
    표본 수로 줄인(shrink) 값을 가중값으로 쓴다: w_k = max(0, mean_R × n / (n + shrink_n)).
    가중값이 모두 0이면 그 시험 창에서는 거래하지 않는다."""
    p = dict(params)
    periods, mas, touch, res, val = _features(train, p)
    c, l = train.close, train.low
    target = (res * (1 - p["tgt_buf"])).to_numpy(float)
    C = c.to_numpy(float)
    learned, info = {}, {}
    for k in periods:
        stop = (np.fmin(l, mas[k]) * (1 - p["stop_buf"])).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = (target - C) / (C - stop)
        ev = touch[k].to_numpy(bool) & (rr >= p["rr_min"]) & (stop < C) & np.isfinite(target)
        tp, sp = np.nonzero(ev)
        r = _outcome_r(train, tp, sp, stop[tp, sp], target[tp, sp], int(p["fit_horizon"]), float(p["fit_cost"]))
        r = r[np.isfinite(r)]
        n = len(r)
        mr = float(np.mean(r)) if n else 0.0
        learned[k] = max(0.0, mr * n / (n + float(p["shrink_n"])))
        info[k] = {"n": n, "mean_r": round(mr, 4)}
    p["weights"] = learned
    p["fit_info"] = info
    return p
