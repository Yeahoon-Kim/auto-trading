"""공용 엔진 sizing 옵션 (next_open 기본 / prev_close = paper 데스크처럼 신호 봉 종가로 수량). 합성 데이터.
실행: uv run pytest -q tests/test_engine_sizing.py"""

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import engine
from autotrader.backtest.data import Panel
from autotrader.risk.guards import load_config
from autotrader.runner.paper import _equity, _no_limits, replay

NO_COST = engine.Costs()


def _panel(close: dict, open_: dict) -> Panel:
    idx = pd.bdate_range("2024-01-01", periods=len(next(iter(close.values()))))
    return Panel({"open": pd.DataFrame(open_, index=idx, dtype=float),
                  "close": pd.DataFrame(close, index=idx, dtype=float)}, "kr_stock")


def _w(p, rows: dict):
    return pd.DataFrame(rows, index=p.index[: len(next(iter(rows.values())))])


@pytest.mark.parametrize(("gap_open", "weight"), [(110.0, 1 / 0.9), (90.0, 1 / 1.1)])
def test_prev_close_keeps_open_gap_in_weight(gap_open, weight):
    # 첫날 종가 100에 비중 1 → 수량 = 자산/100. 다음 날 시가 110이면 시가에 비중 1.1(현금 -0.1)로 사고,
    # 종가 100에서 자산 0.9 → 종가 비중 1/0.9. 시가 90이면 비중 0.9(현금 0.1), 종가 자산 1.1 → 1/1.1
    p = _panel({"A": [100, 100, 100]}, {"A": [100, gap_open, 100]})
    w = _w(p, {"A": [1.0]})
    nxt = engine.run(p, w, NO_COST, start=p.index[1])
    prv = engine.run(p, w, NO_COST, start=p.index[1], sizing="prev_close")
    assert nxt.weights.iloc[0]["A"] == pytest.approx(1.0)
    assert prv.weights.iloc[0]["A"] == pytest.approx(weight)
    # 장중 시가 → 종가 100: 수량이 같으니 손익 = 수량 × (100 - 시가)
    assert prv.returns.iloc[0] == pytest.approx(100 / 100 * (100 - gap_open) / 100)
    assert prv.meta["sizing"] == "prev_close"


def test_prev_close_whole_shares_reduces_buy_when_cash_short():
    p = _panel({"A": [100, 110, 110]}, {"A": [100, 110, 110]})
    w = _w(p, {"A": [1.0]})
    res = engine.run(p, w, NO_COST, start=p.index[1], capital=1000, whole_shares=True, sizing="prev_close")
    assert res.shares.iloc[0]["A"] == 9                 # 종가 100 기준 10주, 시가 110이라 현금 1000으로는 9주
    nxt = engine.run(p, w, NO_COST, start=p.index[1], capital=1000, whole_shares=True)
    assert nxt.shares.iloc[0]["A"] == 9                 # next_open도 시가 110 기준 9주 (같은 결과가 나오는 경우)
    p2 = _panel({"A": [100, 90, 90]}, {"A": [100, 90, 90]})
    res2 = engine.run(p2, w, NO_COST, start=p2.index[1], capital=1000, whole_shares=True, sizing="prev_close")
    nxt2 = engine.run(p2, w, NO_COST, start=p2.index[1], capital=1000, whole_shares=True)
    assert res2.shares.iloc[0]["A"] == 10 and nxt2.shares.iloc[0]["A"] == 11    # 시가 하락: 데스크는 10주만, 엔진 기본은 11주
    assert res2.equity.iloc[-1] == pytest.approx(1000)


def test_default_is_unchanged():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (60, 3)), axis=0))
    o = c * (1 + rng.normal(0, 0.01, c.shape))
    p = _panel({s: c[:, j] for j, s in enumerate("ABC")}, {s: o[:, j] for j, s in enumerate("ABC")})
    w = pd.DataFrame(rng.dirichlet([1, 1, 1], 60) * 0.9, index=p.index, columns=list("ABC"))
    a = engine.run(p, w, engine.Costs(fee_pct=0.1), capital=1e6, whole_shares=True)
    b = engine.run(p, w, engine.Costs(fee_pct=0.1), capital=1e6, whole_shares=True, sizing="next_open")
    pd.testing.assert_series_equal(a.equity, b.equity)
    with pytest.raises(ValueError, match="sizing"):
        engine.run(p, w, NO_COST, sizing="close")


def test_execution_block_passes_sizing():
    idx = pd.bdate_range("2024-01-01", periods=5)
    assert engine.execution_kw({"sizing": "prev_close", "capital": 100}, idx)["sizing"] == "prev_close"
    assert "sizing" not in engine.execution_kw({"capital": 100}, idx)


@pytest.mark.parametrize("rebalance", ["on_signal", "always"])
def test_prev_close_whole_shares_matches_paper_desk(rebalance):
    """같은 신호·가격·비용이면 paper 데스크 재생과 엔진(prev_close, 정수 주식)의 자산 곡선이 같아야 한다."""
    rng = np.random.default_rng(7)
    n, syms = 120, list("ABCDE")
    c = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (n, len(syms))), axis=0))
    o = np.vstack([c[:1], c[:-1]]) * (1 + rng.normal(0, 0.015, c.shape))      # 밤사이 갭
    p = _panel({s: c[:, j] for j, s in enumerate(syms)}, {s: o[:, j] for j, s in enumerate(syms)})
    month = p.index.to_series().dt.month
    first = p.index[(month != month.shift()).to_numpy()]
    w = pd.DataFrame(rng.dirichlet(np.ones(len(syms)), len(first)) * 0.95, index=first, columns=syms)
    W = engine.align_weights(w, p)
    costs = engine.Costs(fee_pct=0.015, slippage_pct=0.05, tax_pct_sell=0.2)
    cap = 10_000_000
    _, ev = replay(p, W, 0, cap, costs, _no_limits(load_config()), lab="T", market="kr_stock",
                      rebalance=rebalance, lot=1.0, limit_band_pct=None)
    res = engine.run(p, W, costs, start=p.index[1], capital=cap, whole_shares=True, rebalance=rebalance,
                     sizing="prev_close")
    desk = _equity(ev).reindex(res.equity.index)
    gap = (desk / res.equity - 1).abs()
    assert gap.max() < 1e-9, f"최대 차이 {gap.max():.2e} ({gap.idxmax().date()})"
