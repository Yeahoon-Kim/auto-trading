"""L003 차트 도구·패턴 탐지기 (A0).

인터페이스 (다른 subagent와 약속):
    TOOLS: 도구 이름 튜플
    detect(bars, tool, timeframe) -> DataFrame[date, direction, tool, variant, pattern_start, height, stop_level]

정의와 세부 수치의 근거는 같은 폴더의 DEFINITIONS.md.

인과성: 모든 계산은 앞에서 뒤로 한 번 도는 방식이라 t 이후 봉을 쓰지 않는다.
  - 극값(고점·저점) = ATR 배수 지그재그. 극값은 "확정 봉(conf)"부터만 패턴에 쓴다.
  - 패턴의 신호 구간 = [마지막 극값 확정 봉, 다음 극값 확정 봉). 다음 극값이 확정되면 그 패턴은 끝난다.
  - 잘라내기 검사(verify.py 3번)로 확인한다.
순수 numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOOLS: tuple[str, ...] = ("ma200_cross", "ma50_200_cross", "fib_retrace", "head_shoulders",
                          "double_triple", "trendline")

COLUMNS = ["date", "direction", "tool", "variant", "pattern_start", "height", "stop_level"]

# --- 공통 세부 수치 (DEFINITIONS.md에 근거) ---
ATR_N = 14          # Wilder ATR 기간 (봉 단위, 일봉·주봉 공통)
ZZ_K = 3.0          # 지그재그 반전 문턱 = 3 × ATR
TOL = 0.015         # "같은 높이" 허용 오차 1.5% (LMW 2000, 브리프 표)
FIB_LO, FIB_HI = 0.382, 0.618


def params(timeframe: str) -> dict:
    """이동평균 길이는 주봉에서 ÷5 (일봉 5개 = 주봉 1개).
    극값 기반 패턴의 길이 제약(min_sep, max_span)은 봉 단위로 같게 둔다: ATR(14)·3배 지그재그의 다리 길이가
    일봉·주봉에서 봉 단위로 거의 같기 때문 (무작위 보행 측정: 다리 중앙값 일봉 23봉, 주봉 21봉. DEFINITIONS.md 0절)."""
    if timeframe == "1d":
        return dict(ma_long=200, ma_short=50, min_sep=22, max_span=250)
    if timeframe == "1w":
        return dict(ma_long=40, ma_short=10, min_sep=22, max_span=250)
    raise ValueError(f"timeframe은 '1d' 또는 '1w': {timeframe!r}")


def to_weekly(bars: pd.DataFrame) -> pd.DataFrame:
    """일봉 → 주봉 (W-FRI). open 첫 값, high 최대, low 최소, close 마지막, index = 그 주 마지막 거래일."""
    key = bars.index.to_period("W-FRI")
    g = bars.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    out.index = pd.DatetimeIndex(pd.Series(bars.index, index=key).groupby(level=0).last().to_numpy())
    return out


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = ATR_N) -> np.ndarray:
    """Wilder ATR. 앞에서부터 재귀 계산 → 인과적."""
    pc = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def zigzag(h: np.ndarray, l: np.ndarray, c: np.ndarray, k: float = ZZ_K, n: int = ATR_N):
    """ATR 배수 지그재그. 반환: (idx 극값 봉, price 극값 가격, kind +1 고점/-1 저점, conf 확정 봉).

    오름 구간: 최고 high를 따라가다 종가 ≤ 최고 high − k·ATR_t 이면 그 최고점을 고점으로 확정 (conf = t).
    내림 구간: 대칭. 처음(방향 미정)에는 양쪽을 다 본다. 고점·저점은 번갈아 나온다.
    ATR_t는 t 봉까지의 값 → 확정은 t까지의 데이터로만 정해진다.
    """
    a = atr(h, l, c, n)
    H = h.tolist(); L = l.tolist(); C = c.tolist(); A = a.tolist()
    idx, px, kind, conf = [], [], [], []
    m = len(C)
    if m == 0:
        return (np.array([], int), np.array([], float), np.array([], int), np.array([], int))
    d = 0
    hi_i, hi_v, lo_i, lo_v = 0, H[0], 0, L[0]
    for t in range(1, m):
        if d >= 0 and H[t] > hi_v:
            hi_i, hi_v = t, H[t]
        if d <= 0 and L[t] < lo_v:
            lo_i, lo_v = t, L[t]
        at = A[t]
        if at != at:  # ATR 준비 전
            continue
        thr = k * at
        if d >= 0 and C[t] <= hi_v - thr:
            idx.append(hi_i); px.append(hi_v); kind.append(1); conf.append(t)
            d = -1
            j = hi_i + 1 + int(np.argmin(l[hi_i + 1:t + 1])) if hi_i < t else t
            lo_i, lo_v = j, L[j]
        elif d <= 0 and C[t] >= lo_v + thr:
            idx.append(lo_i); px.append(lo_v); kind.append(-1); conf.append(t)
            d = 1
            j = lo_i + 1 + int(np.argmax(h[lo_i + 1:t + 1])) if lo_i < t else t
            hi_i, hi_v = j, H[j]
    return (np.array(idx, int), np.array(px, float), np.array(kind, int), np.array(conf, int))


def pivots(bars: pd.DataFrame) -> pd.DataFrame:
    """지그재그 극값 표 (차트·디버깅용). 열: date, price, kind, conf_date."""
    h, l, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    idx, px, kind, conf = zigzag(h, l, c)
    return pd.DataFrame({"date": bars.index[idx], "price": px, "kind": kind,
                         "conf_date": bars.index[conf]})


# ---------------------------------------------------------------- 도구별 탐지
def _first(mask: np.ndarray, start: int) -> int | None:
    w = np.flatnonzero(mask)
    return int(start + w[0]) if len(w) else None


def _ev(t, direction, variant, start, height, stop, info=None):
    return dict(t=t, direction=direction, variant=variant, start=start, height=height,
                stop=stop, info=info or {})


def _ma_cross(c, fast_n, slow_n, names):
    """fast_n=1이면 종가 자체(종가 vs 이동평균)."""
    s = pd.Series(c)
    slow = s.rolling(slow_n, min_periods=slow_n).mean().to_numpy()
    fast = c if fast_n == 1 else s.rolling(fast_n, min_periods=fast_n).mean().to_numpy()
    ok = ~np.isnan(slow) & ~np.isnan(fast)
    ok[1:] &= ok[:-1]; ok[0] = False
    diff = fast - slow
    prev = np.r_[np.nan, diff[:-1]]
    up = ok & (diff > 0) & (prev <= 0)
    dn = ok & (diff < 0) & (prev >= 0)
    out = []
    for t in np.flatnonzero(up):
        out.append(_ev(int(t), 1, names[0], int(t), np.nan, float(slow[t])))
    for t in np.flatnonzero(dn):
        out.append(_ev(int(t), -1, names[1], int(t), np.nan, float(slow[t])))
    return out


def _window(conf, j, n):
    """패턴 신호를 찾는 봉 구간 [conf_j, 다음 극값 확정 봉)."""
    end = conf[j + 1] if j + 1 < len(conf) else n
    return int(conf[j]), int(end)


def _head_shoulders(o, h, l, c, pv, p):
    idx, px, kind, conf = pv
    n = len(c); out = []
    for j in range(4, len(idx)):
        e = px[j - 4:j + 1]; ii = idx[j - 4:j + 1]
        if ii[4] - ii[0] > p["max_span"]:
            continue
        s1, n2, hd, n4, s5 = e
        if abs(s1 - s5) > TOL * (s1 + s5) / 2 or abs(n2 - n4) > TOL * (n2 + n4) / 2:
            continue
        top = kind[j] == 1
        if top and not (hd > s1 and hd > s5):
            continue
        if not top and not (hd < s1 and hd < s5):
            continue
        i2, i4 = ii[1], ii[3]
        slope = (n4 - n2) / (i4 - i2)
        a, b = _window(conf, j, n)
        ts = np.arange(a, b)
        neck = n2 + slope * (ts - i2)
        t = _first(c[a:b] < neck, a) if top else _first(c[a:b] > neck, a)
        if t is None:
            continue
        neck_head = n2 + slope * (ii[2] - i2)
        info = dict(points=list(zip(ii.tolist(), e.tolist())),
                    line=(int(i2), float(n2), int(t), float(n2 + slope * (t - i2))))
        if top:
            out.append(_ev(t, -1, "hs", int(ii[0]), float(hd - neck_head), float(hd), info))
        else:
            out.append(_ev(t, 1, "ihs", int(ii[0]), float(neck_head - hd), float(hd), info))
    return out


def _double_triple(o, h, l, c, pv, p):
    idx, px, kind, conf = pv
    n = len(c); out = []

    def same(vals):
        return (max(vals) - min(vals)) <= TOL * (sum(vals) / len(vals))

    for j in range(2, len(idx)):
        bottom = kind[j] == -1
        chosen = None
        for m in (3, 2):  # 삼중 우선
            first = j - 2 * (m - 1)
            if first < 0:
                continue
            ext = px[first:j + 1:2].tolist()
            mids = px[first + 1:j:2].tolist()
            span = idx[j] - idx[first]
            if span < p["min_sep"] or span > p["max_span"] or not same(ext):
                continue
            chosen = (m, first, ext, mids)
            break
        if chosen is None:
            continue
        m, first, ext, mids = chosen
        neck = max(mids) if bottom else min(mids)
        a, b = _window(conf, j, n)
        t = _first(c[a:b] > neck, a) if bottom else _first(c[a:b] < neck, a)
        if t is None:
            continue
        info = dict(points=list(zip(idx[first:j + 1].tolist(), px[first:j + 1].tolist())),
                    hline=(int(idx[first]), float(neck), int(t)))
        name = {2: "double", 3: "triple"}[m]
        if bottom:
            out.append(_ev(t, 1, f"{name}_bottom", int(idx[first]), float(neck - min(ext)),
                           float(min(ext)), info))
        else:
            out.append(_ev(t, -1, f"{name}_top", int(idx[first]), float(max(ext) - neck),
                           float(max(ext)), info))
    return out


def _fib(o, h, l, c, pv, p):
    idx, px, kind, conf = pv
    n = len(c); out = []
    ph = np.r_[np.nan, h[:-1]]; pl = np.r_[np.nan, l[:-1]]
    for j in range(1, len(idx)):
        a, b = _window(conf, j, n)
        e0, e1 = px[j - 1], px[j]
        R = abs(e1 - e0)
        s = idx[j] + 1  # 극값 다음 봉부터 되돌림 극값을 잰다
        lo_a = max(a, s)
        if lo_a >= b or R <= 0:
            continue
        if kind[j] == 1:   # 상승 스윙 (저점 e0 → 고점 e1), 롱
            zl, zh = e1 - FIB_HI * R, e1 - FIB_LO * R
            run = np.minimum.accumulate(l[s:b])[lo_a - s:]
            cond = (run >= zl) & (run <= zh) & (c[lo_a:b] > o[lo_a:b]) & (c[lo_a:b] > ph[lo_a:b])
            t = _first(cond, lo_a)
            if t is not None:
                out.append(_ev(t, 1, "bull_retrace", int(idx[j - 1]), float(R), float(e0),
                               dict(points=[(int(idx[j - 1]), float(e0)), (int(idx[j]), float(e1))],
                                    zone=(int(idx[j]), float(zl), float(zh), int(t)))))
        else:              # 하락 스윙 (고점 e0 → 저점 e1), 숏
            zl, zh = e1 + FIB_LO * R, e1 + FIB_HI * R
            run = np.maximum.accumulate(h[s:b])[lo_a - s:]
            cond = (run >= zl) & (run <= zh) & (c[lo_a:b] < o[lo_a:b]) & (c[lo_a:b] < pl[lo_a:b])
            t = _first(cond, lo_a)
            if t is not None:
                out.append(_ev(t, -1, "bear_retrace", int(idx[j - 1]), float(R), float(e0),
                               dict(points=[(int(idx[j - 1]), float(e0)), (int(idx[j]), float(e1))],
                                    zone=(int(idx[j]), float(zl), float(zh), int(t)))))
    return out


def _trendline(o, h, l, c, pv, p):
    """하락 추세선(고점 ≥3 접점) 상향 돌파 / 상승 추세선(저점 ≥3) 하향 돌파. 선은 로그 가격에서 직선.

    고점 j가 확정될 때마다, j와 그 전 고점 i(사이에 고점 ≥1개)를 잇는 선 중
    (1) 기울기 음수, (2) i~j 사이 모든 high ≤ 선 + TOL, (3) 선에서 TOL 안의 고점(양 끝 포함) ≥ 3개
    인 것 중 가장 긴 선(가장 이른 i)을 후보로 삼는다 (TOL = log(1.015)).
    후보는 conf_j부터 j+(j−i)봉까지 살아 있고, 그 안에서 처음 종가 > 선인 봉이 돌파.
    같은 방향 돌파가 나면 그때 살아 있던 같은 방향 후보는 모두 소진된다 (한 추세 = 이벤트 1개).
    """
    idx, px, kind, conf = pv
    n = len(c)
    lh, ll, lc = np.log(h), np.log(l), np.log(c)
    tol = np.log1p(TOL)
    out = []
    for sign in (1, -1):  # +1: 고점·하락선·상향돌파, -1: 저점·상승선·하향돌파
        sel = np.flatnonzero(kind == sign)
        pi, pp = idx[sel], np.log(px[sel])
        cands = []  # (break_t, act, ev)
        for q in range(2, len(sel)):
            tj, vj = int(pi[q]), float(pp[q])
            best = None
            for r in range(q - 2, -1, -1):
                ti, vi = int(pi[r]), float(pp[r])
                if tj - ti > p["max_span"]:
                    break
                if (sign == 1 and vi <= vj) or (sign == -1 and vi >= vj):
                    continue
                slope = (vj - vi) / (tj - ti)
                line = vi + slope * (np.arange(ti, tj + 1) - ti)
                if sign == 1:
                    if np.any(lh[ti:tj + 1] > line + tol):
                        continue
                else:
                    if np.any(ll[ti:tj + 1] < line - tol):
                        continue
                lv = vi + slope * (pi[r:q + 1] - ti)
                touch = np.abs(pp[r:q + 1] - lv) <= tol
                if touch.sum() >= 3:
                    best = (r, ti, vi, slope, touch)
            if best is None:
                continue
            r, ti, vi, slope, touch = best
            act = int(conf[sel[q]])
            end = min(n, tj + (tj - ti) + 1)
            if act >= end:
                continue
            line = vi + slope * (np.arange(act, end) - ti)
            t = _first(lc[act:end] > line, act) if sign == 1 else _first(lc[act:end] < line, act)
            if t is None:
                continue
            lt = float(np.exp(vi + slope * (t - ti)))
            tps = [(int(x), float(np.exp(y))) for x, y in zip(pi[r:q + 1][touch], pp[r:q + 1][touch])]
            info = dict(points=tps, line=(ti, float(np.exp(vi)), int(t), lt))
            if sign == 1:
                ev = _ev(t, 1, "up_break", ti, float(lt - l[ti:t + 1].min()),
                         float(l[tj:t + 1].min()), info)
            else:
                ev = _ev(t, -1, "down_break", ti, float(h[ti:t + 1].max() - lt),
                         float(h[tj:t + 1].max()), info)
            cands.append((t, act, ev))
        cands.sort(key=lambda x: (x[0], x[1]))
        fired: list[int] = []
        for t, act, ev in cands:
            if any(act <= f <= t for f in fired):
                continue  # 이미 같은 방향 돌파로 소진된 선
            fired.append(t)
            out.append(ev)
    return out


_PIVOT_TOOLS = {"fib_retrace": _fib, "head_shoulders": _head_shoulders,
                "double_triple": _double_triple, "trendline": _trendline}


def _detect(bars: pd.DataFrame, tool: str, timeframe: str) -> pd.DataFrame:
    """detect + 차트용 info 열(봉 번호 기준 점·선)."""
    if tool not in TOOLS:
        raise ValueError(f"알 수 없는 도구: {tool!r}")
    p = params(timeframe)
    o, h, l, c = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    if tool == "ma200_cross":
        evs = _ma_cross(c, 1, p["ma_long"], ("up_break", "down_break"))
    elif tool == "ma50_200_cross":
        evs = _ma_cross(c, p["ma_short"], p["ma_long"], ("golden", "dead"))
    else:
        pv = zigzag(h, l, c)
        evs = _PIVOT_TOOLS[tool](o, h, l, c, pv, p)
    if not evs:
        df = pd.DataFrame({k: pd.Series(dtype=t) for k, t in
                           zip(COLUMNS + ["info"], ["datetime64[ns]", int, object, object,
                                                    "datetime64[ns]", float, float, object])})
        return df
    ix = bars.index
    df = pd.DataFrame({
        "date": ix[[e["t"] for e in evs]],
        "direction": [e["direction"] for e in evs],
        "tool": tool,
        "variant": [e["variant"] for e in evs],
        "pattern_start": ix[[e["start"] for e in evs]],
        "height": [e["height"] for e in evs],
        "stop_level": [e["stop"] for e in evs],
        "info": [e["info"] for e in evs],
    })
    df = df.drop_duplicates(subset=["date", "variant"]).sort_values(["date", "variant"])
    return df.reset_index(drop=True)


def detect(bars: pd.DataFrame, tool: str, timeframe: str) -> pd.DataFrame:
    """한 종목 bars(오름차순 DatetimeIndex, open/high/low/close)에서 tool 신호를 찾는다.

    반환 열: date(신호 확정 봉), direction(+1/-1), tool, variant, pattern_start, height, stop_level.
    이벤트가 없으면 같은 열의 빈 DataFrame.
    """
    return _detect(bars, tool, timeframe).drop(columns="info")
