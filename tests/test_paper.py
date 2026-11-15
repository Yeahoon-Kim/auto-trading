"""paper 운용데스크 테스트 (합성 데이터, 네트워크·저장소 없음)."""

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import engine
from autotrader.backtest.data import Panel
from autotrader.execution.paper import Book, bars, step
from autotrader.risk.guards import load_config
from autotrader.runner.paper import _no_limits, compare_engine, replay

NO_COST = engine.Costs()
COST = engine.Costs(fee_pct=0.015, slippage_pct=0.05, tax_pct_sell=0.2)
KW = dict(lab="T", market="kr_stock", lot=1.0, limit_band_pct=None)


def _panel(close, open_=None, raw_close=None, tradable=None, start="2024-01-01"):
    c = pd.DataFrame(close, index=pd.bdate_range(start, periods=len(next(iter(close.values())))), dtype=float)
    f = {"open": pd.DataFrame(open_, index=c.index, dtype=float) if open_ else c.copy(), "close": c}
    if raw_close:
        f["raw_close"] = pd.DataFrame(raw_close, index=c.index, dtype=float)
    if tradable is not None:
        f["tradable"] = pd.DataFrame(tradable, index=c.index, dtype=object)
    return Panel(f, "kr_stock")


def _free():
    return _no_limits(load_config())


def _run(p, w, risk=None, capital=1_000_000, costs=NO_COST, rebalance="on_signal", start=0, **kw):
    W = engine.align_weights(w, p)
    return replay(p, W, start, capital, costs, risk or _free(), rebalance=rebalance, **{**KW, **kw})


def test_buy_next_open_and_costs():
    p = _panel({"A": [100, 100, 110, 110]}, {"A": [100, 100, 105, 110]})
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])
    book, ev = _run(p, w, costs=COST)
    fills = [e for e in ev if e["kind"] == "fill"]
    assert len(fills) == 1 and fills[0]["bar"] == str(p.index[1].date())    # 다음 봉 시가 체결
    q = fills[0]["qty"]
    assert q == np.floor(1_000_000 / 100 / (1 + COST.per_trade))
    assert book.cash == pytest.approx(1_000_000 - q * 100 * (1 + COST.per_trade))
    assert book.equity == pytest.approx(book.cash + q * 110)


def test_sell_pays_tax():
    p = _panel({"A": [100] * 4})
    w = pd.DataFrame({"A": [1.0, 0.0]}, index=p.index[[0, 1]])
    book, ev = _run(p, w, costs=COST)
    buy, sell = [e for e in ev if e["kind"] == "fill"]
    assert sell["side"] == "sell" and sell["qty"] == buy["qty"]
    assert sell["tax"] == pytest.approx(sell["qty"] * 100 * 0.002)
    assert not book.shares


def test_same_bar_not_processed_twice():
    p = _panel({"A": [100] * 3})
    W = engine.align_weights(pd.DataFrame({"A": [1.0]}, index=p.index[:1]), p)
    book = Book(cash=1000.0)
    raw = bars(p)
    step(book, p, raw, W, 0, NO_COST, _free(), rebalance="on_signal", **KW)
    ev = step(book, p, raw, W, 0, NO_COST, _free(), rebalance="on_signal", **KW)
    assert ev[0]["kind"] == "skip_bar" and len(book.pending) == 1


def test_limit_band_cancels_gap_up():
    p = _panel({"A": [100, 110, 110]}, {"A": [100, 108, 110]})
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])
    risk = _free()
    risk["orders"]["require_limit_orders"] = True
    book, ev = _run(p, w, risk=risk, limit_band_pct=5.0)
    assert [e["kind"] for e in ev if e["kind"] in ("fill", "cancel")] == ["cancel"]
    assert not book.shares and book.cash == 1_000_000


def test_untradable_cancels():
    p = _panel({"A": [100, 100, 100]}, tradable={"A": [True, False, True]})
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])
    book, ev = _run(p, w)
    assert any(e["kind"] == "cancel" for e in ev) and not book.shares


def test_split_adjusts_shares():
    # 2:1 분할: 원본가 100 → 50, 수정주가는 과거가 50으로 소급 → 수익 0
    p = _panel({"A": [50, 50, 50, 50]}, raw_close={"A": [100, 100, 50, 50]})
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])
    book, ev = _run(p, w)
    q0 = next(e for e in ev if e["kind"] == "fill")["qty"]
    assert any(e["kind"] == "corporate_action" for e in ev)
    assert book.shares["A"] == pytest.approx(q0 * 2)
    assert book.equity == pytest.approx(1_000_000)


def test_missing_quotes_exit():
    p = _panel({"A": [100, 100, 100, np.nan, np.nan, np.nan], "B": [10] * 6},
               tradable={"A": [True] * 3 + [None] * 3, "B": [True] * 6})
    w = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=p.index[:1])
    risk = _free()
    risk["data_staleness"]["unmeasurable_retries"] = 3
    book, ev = _run(p, w, risk=risk)
    assert "A" not in book.shares
    assert sum(e["kind"] == "unmeasurable_exit" for e in ev) == 1


def test_strategy_shape_not_cut_by_risk_yaml():
    """종목 수·종목 비중은 전략이 정한다 — risk.yaml 그대로 써도 20종목 × 5%, 3종목 × 33%가 그대로 들어간다."""
    risk = load_config()
    risk["orders"].update(max_order_value_krw=1e18, require_limit_orders=False)
    syms = [f"S{i:02d}" for i in range(20)]
    p = _panel({s: [100] * 3 for s in syms})
    book, _ = _run(p, pd.DataFrame({s: [0.05] for s in syms}, index=p.index[:1]), risk=risk)
    assert len(book.shares) == 20
    p = _panel({s: [100] * 3 for s in "ABC"})
    book, _ = _run(p, pd.DataFrame({s: [0.33] for s in "ABC"}, index=p.index[:1]), risk=risk)
    assert all(q * 100 > 0.3 * 1_000_000 for q in book.shares.values())


def test_invalid_weights_hold_all_orders():
    p = _panel({"A": [100] * 3, "B": [100] * 3})
    book, ev = _run(p, pd.DataFrame({"A": [0.7], "B": [0.7]}, index=p.index[:1]))
    assert not book.shares and not any(e["kind"] == "order" for e in ev)
    assert any(e["kind"] == "risk_block" and e["rule"] == "weights_invalid" for e in ev)


def test_kill_switch_blocks_buys_allows_sells():
    p = _panel({s: [100] * 5 for s in "ABC"})
    w = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=p.index[:1])
    risk = _free()
    book, _ = _run(p, w, risk=risk)
    assert set(book.shares) == {"A", "B"}

    # A 청산 + C 신규 → 매도만 나가고 C는 막힘
    risk["global"]["kill_switch"] = True
    book.last_bar = str(p.index[3].date())
    W = engine.align_weights(pd.DataFrame({"A": [0.0], "B": [0.5], "C": [0.5]}, index=p.index[4:5]), p)
    ev = step(book, p, bars(p), W, 4, NO_COST, risk, rebalance="on_signal", **KW)
    assert [(e["symbol"], e["side"]) for e in ev if e["kind"] == "order"] == [("A", "sell")]
    assert any(e["kind"] == "risk_block" and e["rule"] == "buy_halt" and e["symbol"] == "C" for e in ev)


def test_cash_shortfall_reduces_buy():
    # 종가 100 기준 전액 매수 주문 → 시가 102에 체결하려면 현금 부족 → 줄여서 체결
    p = _panel({"A": [100, 102, 102]}, {"A": [100, 102, 102]})
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])
    book, ev = _run(p, w)
    assert any(e["kind"] == "reduce" for e in ev)
    assert book.cash >= 0


@pytest.mark.parametrize("rebalance", ["on_signal", "always"])
def test_matches_engine_within_gap(rebalance):
    """같은 신호·비용이면 엔진(정수 주식)과 자산 곡선이 가깝다. 차이는 수량을 정하는 시점(종가 vs 시가)뿐."""
    rng = np.random.default_rng(0)
    n, syms = 260, list("ABCDE")
    r = rng.normal(0.0005, 0.015, (n, len(syms)))
    c = 10_000 * np.exp(np.cumsum(r, axis=0))
    o = c * np.exp(rng.normal(0, 0.004, c.shape))
    p = _panel({s: c[:, j] for j, s in enumerate(syms)}, {s: o[:, j] for j, s in enumerate(syms)})
    # 월초마다 직전 20일 수익 상위 2종목에 반반
    rows = {}
    mom = pd.DataFrame(c, index=p.index, columns=syms).pct_change(20)
    for d in pd.Series(p.index, index=p.index).groupby(p.index.to_period("M")).min():
        top = mom.loc[d].nlargest(2).index if mom.loc[d].notna().all() else []
        rows[d] = pd.Series({s: (0.5 if s in top else 0.0) for s in syms})
    w = pd.DataFrame(rows).T
    W = engine.align_weights(w, p)
    cap = 100_000_000
    book, ev = replay(p, W, 0, cap, COST, _free(), rebalance=rebalance, **KW)
    eq = pd.Series({pd.Timestamp(e["bar"]): e["equity"] for e in ev if e["kind"] == "equity"})
    cmp = compare_engine(p, W, 0, cap, COST, rebalance, eq)
    assert cmp["bars"] > 200
    assert cmp["max_abs_gap_pct"] < 1.0, cmp
