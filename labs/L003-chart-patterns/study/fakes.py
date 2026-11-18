"""하네스 자체 검증용 가짜 탐지기 (실제 탐지기와 같은 인터페이스: detect(bars, tool, timeframe)).

fakeA  무작위 신호 (봉의 1%, 방향 ±1 무작위)            → 통과 0건, p값 대략 균등이어야 한다
fakeB  미래를 엿보는 신호 (다음 20봉 수익 > 0인 봉의 2%)   → 반드시 강하게 통과해야 한다 (검출력 확인, 하네스 시험 전용)
fakeC  모멘텀 신호 (직전 60봉 수익이 큰 봉의 일부)         → 단순 위약 대비보다 추세 맞춘 위약 대비 초과수익이 크게 줄어야 한다
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

# fakeA는 p값 분포를 보려고 서로 독립인 10개(fakeA0~9)로 둔다 → 검정 단위가 많아진다
TOOLS = tuple(f"fakeA{i}" for i in range(10)) + ("fakeB", "fakeC")
H = 20
MOM = 60
MOM_TH = {"1d": 0.30, "1w": 0.60}


def _rng(bars: pd.DataFrame, tool: str, timeframe: str) -> np.random.Generator:
    key = f"{tool}|{timeframe}|{len(bars)}|{bars.index[0]}|{bars['close'].iloc[0]:.6g}|{bars['close'].iloc[-1]:.6g}"
    return np.random.default_rng(zlib.crc32(key.encode()))


def detect(bars: pd.DataFrame, tool: str, timeframe: str) -> pd.DataFrame:
    n = len(bars)
    rng = _rng(bars, tool, timeframe)
    o, c = bars["open"].to_numpy(), bars["close"].to_numpy()
    u = rng.random(n)
    if tool.startswith("fakeA"):
        hit = u < 0.01
        dirs = np.where(rng.random(n) < 0.5, 1, -1)
        var = np.full(n, "rand")
    elif tool == "fakeB":
        fwd = np.full(n, np.nan)
        if n > H:   # 진입 = 다음 봉 시가 (일봉이면 하네스와 같다. 주봉은 다음 주 첫 일봉 시가 = 다음 주봉 시가)
            fwd[: n - H] = c[H:] / o[1: n - H + 1] - 1
        up, dn = fwd > 0, fwd < 0
        hit = (up | dn) & (u < 0.02)
        dirs = np.where(up, 1, -1)
        var = np.where(up, "peek_up", "peek_down")
    elif tool == "fakeC":
        mom = np.full(n, np.nan)
        if n > MOM:
            mom[MOM:] = c[MOM:] / c[:-MOM] - 1
        hit = (mom > MOM_TH[timeframe]) & (u < 0.05)
        dirs = np.ones(n, int)
        var = np.full(n, "mom_up")
    else:
        raise ValueError(tool)
    i = np.flatnonzero(hit)
    return pd.DataFrame({"date": bars.index[i], "direction": dirs[i], "tool": tool, "variant": var[i]})
