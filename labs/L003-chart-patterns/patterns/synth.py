"""합성 가격 생성기 (A0 verifier용).

- random_walk: 일봉 변동성 vol의 로그 무작위 보행 OHLC.
- plant: 무작위 보행 뒤에 패턴 꼭짓점을 잇는 경로(작은 잡음)를 심고, 다시 무작위 보행을 잇는다.
  심은 패턴의 "완성 봉"(정의상 신호가 나야 할 봉)을 실제 생성된 가격으로 계산해 함께 돌려준다.
- 하락형(천장·HS·숏 되돌림·상승선 하향 돌파)은 상승형을 로그 가격에서 뒤집어(mirror) 만든다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import FIB_HI, FIB_LO, params


def _ohlc(logc: np.ndarray, vol_bar: np.ndarray, rng, start_logc: float | None = None) -> pd.DataFrame:
    """로그 종가 경로 → OHLC. 시가 = 전일 종가 ± 작은 갭, 고가·저가 = 몸통 밖 반정규 꼬리."""
    n = len(logc)
    prev = np.r_[logc[0] if start_logc is None else start_logc, logc[:-1]]
    lo_ = prev + rng.normal(0, 1, n) * vol_bar * 0.25
    hi_tail = np.abs(rng.normal(0, 1, n)) * vol_bar * 0.5
    lo_tail = np.abs(rng.normal(0, 1, n)) * vol_bar * 0.5
    lh = np.maximum(lo_, logc) + hi_tail
    ll = np.minimum(lo_, logc) - lo_tail
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"open": np.exp(lo_), "high": np.exp(lh), "low": np.exp(ll),
                         "close": np.exp(logc)}, index=idx)


def random_walk(n: int, vol: float = 0.02, seed=None, rng=None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(seed)
    logc = np.log(100) + np.cumsum(rng.normal(0, vol, n))
    return _ohlc(logc, np.full(n, vol), rng)


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """로그 가격 뒤집기: 상승형 패턴 → 하락형. 1/가격 × 상수."""
    k = 100.0 * 100.0
    return pd.DataFrame({"open": k / df["open"], "high": k / df["low"], "low": k / df["high"],
                         "close": k / df["close"]}, index=df.index)


def _path(levels, lengths, noise, rng):
    """꼭짓점 levels(로그)를 lengths 봉으로 잇는 선형 경로 + 잡음. 꼭짓점 봉 번호도 돌려준다."""
    seg, at, pos = [], [], 0
    for k in range(1, len(levels)):
        L = int(lengths[k - 1])
        x = np.linspace(levels[k - 1], levels[k], L + 1)[1:]
        seg.append(x); pos += L; at.append(pos - 1)
    p = np.concatenate(seg)
    eps = rng.normal(0, noise, len(p))
    eps[np.array(at)] = 0.0       # 꼭짓점은 정확히
    return p + eps, at


def _first_cross(c, line, start, up=True):
    m = (c[start:] > line[start:]) if up else (c[start:] < line[start:])
    w = np.flatnonzero(m)
    return int(start + w[0]) if len(w) else None


def plant(kind: str, timeframe: str = "1d", noise: float = 0.004, seed=None, pre: int = 120,
          post: int = 40, for_mirror: bool = False):
    """kind: 'ihs' | 'double_bottom' | 'triple_bottom' | 'bull_retrace' | 'up_break'
    for_mirror: 뒤집어 쓸 것(하락형)이면 True. 피보나치 비율은 가격 기준이라 뒤집은 가격에서 r이 되게 심는다.
    반환: (bars, 완성 봉 번호, 심은 꼭짓점 봉 번호들). 실패(정의상 완성 불가)면 None."""
    rng = np.random.default_rng(seed)
    pr = params(timeframe)
    # 주봉은 봉 하나가 일봉 5개 → 변동성·패턴 크기·잡음을 √5배 (봉 단위로 보면 같은 모양)
    sc = 1.0 if timeframe == "1d" else float(np.sqrt(5))
    vol = 0.02 * sc
    noise = noise * sc
    lo_len, hi_len = (8, 22) if timeframe == "1d" else (5, 12)
    Lr = lambda: int(rng.integers(lo_len, hi_len + 1))  # noqa: E731
    h = float(rng.uniform(0.12, 0.30) if timeframe == "1d" else rng.uniform(0.20, 0.50))  # 패턴 높이 (로그)
    e = lambda: float(rng.uniform(-0.004, 0.004))  # noqa: E731  같은 높이 꼭짓점의 어긋남 (≤ 0.8% 차)

    # 앞: 무작위 보행 후 진입 다리
    base = np.log(100) + np.cumsum(rng.normal(0, vol, pre))
    b0 = base[-1]

    if kind == "ihs":
        neck = b0
        S = neck - h * rng.uniform(0.6, 1.0)          # 어깨
        Hd = S - h * rng.uniform(0.3, 0.8)            # 머리 (더 낮음)
        lv = [b0 + h * 0.6, S + e(), neck + e(), Hd, neck + e(), S + e(), neck + h * 0.8]
    elif kind == "double_bottom":
        lv = [b0 + h * 0.3, b0 - h + e(), b0, b0 - h + e(), b0 + h * 0.7]
    elif kind == "triple_bottom":
        lv = [b0 + h * 0.3, b0 - h + e(), b0 + e(), b0 - h + e(), b0 + e(), b0 - h + e(),
              b0 + h * 0.7]
    elif kind == "bull_retrace":
        R = float(rng.uniform(0.20, 0.40) if timeframe == "1d" else rng.uniform(0.30, 0.70))
        r = float(rng.uniform(FIB_LO + 0.04, FIB_HI - 0.04))
        Lo_, Hi_ = b0 - 0.6 * R, b0 + 0.4 * R
        if not for_mirror:   # 되돌림 비율은 가격 기준
            pull = np.log(np.exp(Hi_) - r * (np.exp(Hi_) - np.exp(Lo_)))
        else:                # 뒤집은 가격 K/p 에서 비율 r
            pull = -np.log(np.exp(-Hi_) + r * (np.exp(-Lo_) - np.exp(-Hi_)))
        lv = [Lo_, Hi_, pull]
    elif kind == "up_break":
        drop = h * rng.uniform(0.3, 0.8)          # 첫 고점 → 마지막 고점 하락
        dep = h * rng.uniform(0.8, 1.2) + drop * 0.5   # 선 아래 골 깊이
        P1 = b0 + h * 0.5
        lens = [Lr() for _ in range(8)]
        tp = np.cumsum([0, lens[2], lens[3], lens[4], lens[5], lens[6]])  # P1,T1,P2,T2,P3,T3 (P1 기준)
        line = lambda t: P1 - drop * t / tp[4]      # noqa: E731
        lv = [b0 - h * 0.3, line(tp[0]), line(tp[1]) - dep, line(tp[2]) + float(rng.uniform(-0.004, 0.0)),
              line(tp[3]) - dep, line(tp[4]), line(tp[5]) - dep * 0.7,
              line(tp[5] + lens[7]) + dep * 0.4]
    else:
        raise ValueError(kind)

    if kind == "up_break":
        lengths = [lens[0] + 10] + lens[1:]
    else:
        lengths = [Lr() + 10] + [Lr() for _ in range(len(lv) - 1)]
        if kind in ("double_bottom", "triple_bottom"):  # 바닥 간격 ≥ min_sep 보장
            need = pr["min_sep"] + 2
            nb = 2 if kind == "double_bottom" else 3
            span = sum(lengths[2:2 + 2 * (nb - 1)])
            if span < need:
                lengths[2] += need - span
    lvl = [b0] + lv
    path, at = _path(lvl, lengths, noise, rng)
    off = pre
    vb = np.r_[np.full(pre, vol), np.full(len(path), max(noise, 0.002))]

    npath = len(path)
    if kind == "bull_retrace":
        # 반등 봉: 직전 고가 위로 마감하는 양봉, 이후 상승
        jump = max(0.03 * sc, 2.5 * abs(path[-1] - path[-2]))
        bounce = np.array([path[-1] + jump])
        rest = path[-1] + jump + np.cumsum(np.full(10, 0.01 * sc) + rng.normal(0, noise, 10))
        path = np.r_[path, bounce, rest]
        vb = np.r_[vb, np.full(11, max(noise, 0.002))]
    else:
        # 돌파 다리 연장: 마지막 꼭짓점에서 계속 같은 기울기로 조금 더
        last = path[-1]
        step = (lvl[-1] - lvl[-2]) / lengths[-1]
        ext = last + step * np.arange(1, 6) + rng.normal(0, noise, 5)
        path = np.r_[path, ext]
        vb = np.r_[vb, np.full(5, max(noise, 0.002))]
    after = path[-1] + np.cumsum(rng.normal(0, vol, post))
    logc = np.r_[base, path, after]
    vb = np.r_[vb, np.full(post, vol)]
    df = _ohlc(logc, vb, rng)
    # 꼭짓점 봉의 고가·저가를 정확히 꼭짓점 값으로 (극값이 그 봉에 있게)
    hh, ll, cc, oo = (df[x].to_numpy().copy() for x in ("high", "low", "close", "open"))
    verts = [off + a for a in at]
    for k, v in enumerate(verts):
        val = np.exp(lvl[k + 1])
        up_vertex = lvl[k + 1] > lvl[k] and (k + 2 >= len(lvl) or lvl[k + 2] < lvl[k + 1])
        dn_vertex = lvl[k + 1] < lvl[k] and (k + 2 >= len(lvl) or lvl[k + 2] > lvl[k + 1])
        if up_vertex:
            hh[v] = val; cc[v] = min(cc[v], val); oo[v] = min(oo[v], val)
            win = slice(max(v - 2, 0), v + 3)
            hh[win] = np.minimum(hh[win], val); hh[v] = val
        elif dn_vertex:
            ll[v] = val; cc[v] = max(cc[v], val); oo[v] = max(oo[v], val)
            win = slice(max(v - 2, 0), v + 3)
            ll[win] = np.maximum(ll[win], val); ll[v] = val
    hh = np.maximum(hh, np.maximum(oo, cc)); ll = np.minimum(ll, np.minimum(oo, cc))
    df = pd.DataFrame({"open": oo, "high": hh, "low": ll, "close": cc}, index=df.index)

    c = cc
    n = len(c)
    tt = np.arange(n)
    if kind == "ihs":
        i2, i4 = verts[2], verts[4]
        # 넥라인 = 두 겨드랑이(고점)를 잇는 선
        line = hh[i2] + (hh[i4] - hh[i2]) / (i4 - i2) * (tt - i2)
        comp = _first_cross(c, line, verts[5] + 1, up=True)
    elif kind in ("double_bottom", "triple_bottom"):
        mids = [verts[2]] if kind == "double_bottom" else [verts[2], verts[4]]
        neck = max(hh[m] for m in mids)
        comp = _first_cross(c, np.full(n, neck), verts[-2] + 1, up=True)
    elif kind == "bull_retrace":
        comp = off + npath  # 반등 봉
        # 반등 봉이 정의(양봉, 종가 > 전일 고가)를 만족하게 보정
        oo[comp] = min(oo[comp], cc[comp - 1]); ll[comp] = min(ll[comp], oo[comp])
        hh[comp - 1] = min(hh[comp - 1], cc[comp] * 0.995)
        hh[comp - 1] = max(hh[comp - 1], oo[comp - 1], cc[comp - 1])
        df = pd.DataFrame({"open": oo, "high": hh, "low": ll, "close": cc}, index=df.index)
        if not (cc[comp] > oo[comp] and cc[comp] > hh[comp - 1]):
            return None
    elif kind == "up_break":
        i1, i3 = verts[1], verts[5]   # 추세선은 로그 가격에서 직선
        line = np.exp(np.log(hh[i1]) + (np.log(hh[i3]) - np.log(hh[i1])) / (i3 - i1) * (tt - i1))
        comp = _first_cross(c, line, verts[6] + 1, up=True)
    if comp is None:
        return None
    return df, comp, verts


def plant_ma(kind: str, timeframe: str = "1d", seed=None):
    """이동평균 교차용: 하락 추세 뒤 상승 추세 (변동성 2%). 기대 신호 = 독립 구현으로 계산한 모든 교차."""
    rng = np.random.default_rng(seed)
    pr = params(timeframe)
    n_dn = pr["ma_long"] + int(rng.integers(50, 150))
    n_up = int(rng.integers(100, 250)) if timeframe == "1d" else int(rng.integers(30, 60))
    mu_dn = -float(rng.uniform(0.001, 0.004)); mu_up = float(rng.uniform(0.002, 0.006))
    vol = 0.02 if timeframe == "1d" else 0.045
    r = np.r_[rng.normal(mu_dn, vol, n_dn), rng.normal(mu_up, vol, n_up)]
    df = _ohlc(np.log(100) + np.cumsum(r), np.full(len(r), vol), rng)
    return df


def naive_ma_events(c: np.ndarray, fast_n: int, slow_n: int):
    """검사용 독립 구현 (순수 파이썬 반복). 반환: [(t, +1/-1)]."""
    def ma(t, k):
        return sum(c[t - k + 1:t + 1]) / k if t >= k - 1 else None
    out = []
    for t in range(1, len(c)):
        s0, s1 = ma(t - 1, slow_n), ma(t, slow_n)
        if s0 is None:
            continue
        f0 = c[t - 1] if fast_n == 1 else ma(t - 1, fast_n)
        f1 = c[t] if fast_n == 1 else ma(t, fast_n)
        if f1 > s1 and f0 <= s0:
            out.append((t, 1))
        elif f1 < s1 and f0 >= s0:
            out.append((t, -1))
    return out
