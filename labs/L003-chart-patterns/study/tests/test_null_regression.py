"""헛발견 회귀 검사: 가격 경로에 의존하는(인과적인) 탐지기를 무작위 보행 시장에 돌리면 초과수익이 0이어야 한다.

2026-11-18 실패 사례: 초기 사양 위약(같은 종목·같은 연도, 같은 도구 신호 ±20봉 제외)이 섞은 시장에서도 실제 탐지기 신호에
+1.5~2.5% '초과수익'을 만들었다. 신호 시점은 직전 긴 경로(200봉 이동평균 등)의 함수라, 신호 근처 날의 앞으로 20봉 창은
"신호가 바로 그때 처음 났다"는 조건 때문에 무조건부 분포와 다르다. 같은 연도로 좁히면 그런 날의 비중이 커지고, ±20봉 제외
(앞으로 20봉 안에 신호가 '나올' 날을 뺌)는 미래참조 선택이라 편향을 더 키운다. 무작위 신호(fakeA)는 가격과 무관해서 못 잡았다.

    uv run pytest labs/L003-chart-patterns/study/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H  # noqa: E402


def _rw_wide(n_sym=300, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2011-01-03", "2017-12-29")
    syms = [f"R{i:03d}" for i in range(n_sym)]
    r = rng.normal(0.0, 0.02, (len(idx), n_sym))
    c = 50 * np.exp(np.cumsum(r, 0))
    o = c / np.exp(r) * np.exp(rng.normal(0, 0.003, c.shape))   # 시가 ≈ 전일 종가
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.005, c.shape)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.005, c.shape)))
    return {k: pd.DataFrame(v, idx, syms) for k, v in (("open", o), ("high", hi), ("low", lo), ("close", c))}


def ma_cross(bars: pd.DataFrame, tool: str, timeframe: str) -> pd.DataFrame:
    """인과적 이동평균 교차 (50/200): 골든 +1, 데드 −1. 실제 ma50_200 탐지기와 같은 성격의 신호."""
    c = bars["close"]
    f, s = c.rolling(50).mean(), c.rolling(200).mean()
    up = (f > s) & (f.shift() <= s.shift())
    dn = (f < s) & (f.shift() >= s.shift())
    ev = pd.concat([pd.DataFrame({"date": c.index[up.values], "direction": 1, "variant": "golden"}),
                    pd.DataFrame({"date": c.index[dn.values], "direction": -1, "variant": "dead"})])
    return ev.assign(tool=tool)


def _excess(detect, block: str = "past", pool: str | None = None, tools=("ma",), timeframe="1d"):
    w = _rw_wide()
    stocks = H.stocks_from_wide(w)
    bt = H.build_bars("us_stock", timeframe, stocks, None, w["close"].index[-1], H.Costs("us_stock"))
    parts = []
    for s in stocks:
        bars, _ = H.bars_of(s, timeframe)
        for t in tools:
            e = detect(bars, t, timeframe)
            if len(e):
                parts.append(pd.DataFrame(e).assign(sym=s.sym))
    ev = pd.concat(parts, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["date"]).values.astype("datetime64[ns]")
    ev, blocked, _ = H.attach_events(ev, bt, (0, 1))
    res = {}
    for kind, matched in (("m", True), ("s", False)):
        idx = H.sample_placebos(ev, bt, blocked, seed=0, matched=matched, block=block, pool=pool)
        d = ev["direction"].values * (ev["ret20"].values - H.placebo_means(idx, bt, "ret20"))
        for key, g in pd.DataFrame({"d": d, "cl": ev["cl"].values, "v": ev["tool"] + "/" + ev["variant"].astype(str)}).groupby("v"):
            b = H.cluster_bootstrap(g["d"].values, g["cl"].values, n_boot=300)
            res[(key, kind)] = (b["mean"], b["p"], len(g))
    return res


def test_null_ma_cross_no_bias_default_placebo():
    res = _excess(ma_cross)
    for (key, kind), (m, p, n) in res.items():
        assert n > 300
        assert abs(m) < 0.006, (key, kind, m)      # |초과| < 0.6%p (편향이 있던 방식은 +1~2%)
        assert p > 0.01, (key, kind, m, p)


def test_null_ma_cross_old_block_is_biased():
    """검사의 검사: 초기 사양(같은 연도, ±20봉 차단)은 같은 데이터에서 편향을 보여야 한다 → 위 검사가 이 실패를 잡는다."""
    res = _excess(ma_cross, "both", pool="year")
    for v in ("ma/golden", "ma/dead"):
        m, p, _ = res[(v, "m")]
        assert m > 0.008 and p < 0.01, (v, m, p)


@pytest.mark.parametrize("pool", ["period", "xsec"])
def test_null_real_detectors_no_bias(pool):
    """실제 탐지기(patterns/)를 무작위 보행 시장에 돌린 회귀 검사. 탐지기가 없으면 건너뛴다."""
    lab = str(H.LAB)
    if lab not in sys.path:
        sys.path.insert(0, lab)
    patterns = pytest.importorskip("patterns")
    res = _excess(patterns.detect, pool=pool, tools=tuple(patterns.TOOLS))
    big = {k: v for k, v in res.items() if v[2] >= 200}
    assert big
    ps = np.array([v[1] for v in big.values()])
    ms = np.array([v[0] for v in big.values()])
    # 원 p<0.05 비율이 대략 명목값 근처, 0.1% 수준 유의는 없어야 한다
    assert (ps < 0.05).mean() <= 0.25, sorted(zip(ps.round(4), ms.round(4), big))[:5]
    assert (ps < 0.001).sum() == 0, sorted(zip(ps.round(5), ms.round(4), big))[:5]


def mom_weekly(bars: pd.DataFrame, tool: str, timeframe: str) -> pd.DataFrame:
    """인과적 모멘텀 신호 (직전 60봉 수익 상위 일부) — 가짜 탐지기 C와 같은 성격."""
    c = bars["close"].to_numpy()
    mom = np.full(len(c), np.nan)
    mom[60:] = c[60:] / c[:-60] - 1
    i = np.flatnonzero(mom > 0.5)[::3]          # 고정 문턱 (종목 전체 분위로 정하면 그 자체가 미래참조)
    return pd.DataFrame({"date": bars.index[i], "direction": 1, "variant": "mom", "tool": tool})


def test_null_weekly_momentum_default_unbiased():
    """같은 종목 위약(period)은 주봉·직전 수익 극단 신호에서 경로 공유 편향이 남았다. 기본(xsec)은 없어야 한다."""
    res = _excess(mom_weekly, timeframe="1w")
    m, p, n = res[("ma/mom", "m")]
    assert n > 300 and abs(m) < 0.02 and p > 0.01, (m, p, n)


def test_null_weekly_momentum_same_stock_pool_biased():
    """검사의 검사: 같은 종목·구간 전체 위약은 같은 데이터에서 음(−)의 편향을 보인다."""
    res = _excess(mom_weekly, pool="period", timeframe="1w")
    m, p, _ = res[("ma/mom", "m")]
    assert m < -0.01 and p < 0.01, (m, p)
