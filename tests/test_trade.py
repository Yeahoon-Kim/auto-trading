"""신호형(type: trade) 백테스트 재생기 테스트 (합성 봉, 네트워크·저장소 없음). 실행: uv run pytest -q tests/test_trade.py"""

import copy

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import engine, trade
from autotrader.backtest.data import Panel
from autotrader.risk import guards

NO_COST = engine.Costs()
ACCT = trade.Account(capital=1_000_000)
RISK = copy.deepcopy(guards.load_config())
RISK["orders"]["max_order_value_krw"] = 1e12     # 주문 상한이 수량을 묶지 않게
TRADE = {"max_position_pct": 100}                 # 위험 기준(1%)이 수량을 정하게


def _panel(bars: list[tuple], symbol="A", market="kr_stock") -> Panel:
    """bars: [(open, high, low, close), ...] 일봉."""
    idx = pd.bdate_range("2020-01-01", periods=len(bars))
    f = {k: pd.DataFrame({symbol: [b[i] for b in bars]}, index=idx, dtype=float)
         for i, k in enumerate(("open", "high", "low", "close"))}
    return Panel(f, market)


def _run(panel, sigs, costs=NO_COST, **kw):
    return trade.run(panel, sigs, costs, ACCT, trade_cfg=TRADE, risk_cfg=RISK, **kw)


def _sig(panel, i=0, **kw):
    return {"ts": panel.index[i], "symbol": "A", "side": "long", "stop": 95.0, "target": 110.0, **kw}


def _only(res):
    tl = res.meta["trade_log"]
    assert len(tl) == 1
    return tl.iloc[0]


FLAT = (100, 100, 100, 100)


def test_stop_loss_exits_at_stop_and_sizes_by_risk():
    p = _panel([FLAT, (100, 101, 99, 100), (100, 100, 94, 96), FLAT])
    res = _run(p, [_sig(p)])
    t = _only(res)
    assert t["exit_kind"] == "stop_loss" and t["exit"] == 95.0
    assert t["qty"] == 2000                            # 자본 1% = 10,000원 / (100-95)
    assert t["binding"] == "risk"
    assert t["r"] == pytest.approx(-1.0)
    assert res.equity.iloc[-1] == pytest.approx(1_000_000 - 10_000)
    assert res.trades == 1


def test_take_profit_exits_at_target():
    p = _panel([FLAT, (100, 101, 99, 100), (101, 112, 100, 111), FLAT])
    t = _only(_run(p, [_sig(p)]))
    assert t["exit_kind"] == "take_profit" and t["exit"] == 110.0
    assert t["r"] == pytest.approx(2.0)


def test_same_bar_stop_and_target_counts_as_stop():
    p = _panel([FLAT, (100, 101, 99, 100), (100, 115, 90, 105), FLAT])
    t = _only(_run(p, [_sig(p)]))
    assert t["exit_kind"] == "stop_loss" and t["exit"] == 95.0


def test_gap_through_stop_fills_at_open():
    p = _panel([FLAT, (100, 101, 99, 100), (90, 92, 88, 91), FLAT])
    t = _only(_run(p, [_sig(p)]))
    assert t["exit_kind"] == "stop_loss" and t["exit"] == 90.0
    assert t["r"] == pytest.approx(-2.0)              # 갭은 -1R보다 나쁘다


def test_intraday_flattens_at_close_of_entry_bar():
    p = _panel([FLAT, (100, 103, 98, 102), (105, 106, 104, 105)])
    res = _run(p, [_sig(p, intraday=True)])
    t = _only(res)
    assert t["exit_kind"] == "flat_before_close" and t["exit"] == 102.0
    assert t["entry_ts"] == t["exit_ts"] == p.index[1]
    assert res.weights.iloc[-1].abs().sum() == 0     # 다음 날엔 보유 없음


def test_intraday_not_flattened_in_24h_market():
    p = _panel([FLAT, (100, 103, 98, 102), (102, 104, 101, 103)], market="crypto_spot")
    acct = trade.Account(capital=1_000_000, currency="USDT", krw_per_unit=1.0, lot=1e-6)
    res = trade.run(p, [_sig(p, intraday=True)], NO_COST, acct, trade_cfg=TRADE, risk_cfg=RISK)
    assert _only(res)["exit_kind"] == "open"


def test_entry_bar_stop_after_open_fill():
    p = _panel([FLAT, (100, 101, 94, 97), FLAT])
    t = _only(_run(p, [_sig(p)]))
    assert t["exit_kind"] == "stop_loss" and t["entry_ts"] == t["exit_ts"]


def test_limit_entry_not_filled_is_dropped():
    p = _panel([FLAT, (100, 101, 99, 100), FLAT])
    res = _run(p, [_sig(p, entry=98.0)])
    assert res.trades == 0 and res.meta["blocked"] == {"미체결": 1}


def test_limit_fill_intrabar_ignores_target_on_fill_bar():
    # 지정가 98이 장중에 체결된 봉에서는 고가 111이 체결 전일 수 있어 목표가를 인정하지 않는다
    p = _panel([FLAT, (100, 111, 97, 105), (105, 106, 104, 105)])
    t = _only(_run(p, [_sig(p, entry=98.0)]))
    assert t["entry"] == 98.0 and t["exit_kind"] == "open"


def test_costs_fee_and_sell_tax_and_scaling():
    p = _panel([FLAT, (100, 101, 99, 100), (101, 112, 100, 111), FLAT])
    c = engine.Costs(fee_pct=0.1, slippage_pct=0.1, tax_pct_sell=0.2)
    res = _run(p, [_sig(p)], costs=c)
    t = _only(res)
    expect = 2000 * 100 * 0.002 + 2000 * 110 * (0.002 + 0.002)
    assert t["cost"] == pytest.approx(expect)
    assert res.equity.iloc[-1] == pytest.approx(1_000_000 + 2000 * 10 - expect)
    assert res.cost.sum() > 0 and res.gross.sum() > res.returns.sum()
    t2 = _only(_run(p, [_sig(p)], costs=c.scaled(2)))
    assert t2["cost"] == pytest.approx(2 * expect)


def test_short_stop_and_target():
    p = _panel([FLAT, (100, 101, 99, 100), (99, 100, 89, 90), FLAT])
    t = _only(_run(p, [_sig(p, side="short", stop=105.0, target=90.0)]))
    assert t["exit_kind"] == "take_profit" and t["exit"] == 90.0 and t["r"] == pytest.approx(2.0)
    p = _panel([FLAT, (100, 101, 99, 100), (100, 106, 89, 100), FLAT])
    t = _only(_run(p, [_sig(p, side="short", stop=105.0, target=90.0)]))
    assert t["exit_kind"] == "stop_loss" and t["exit"] == 105.0


def test_max_open_positions_and_duplicate_symbol_blocked():
    idx = pd.bdate_range("2020-01-01", periods=4)
    f = {k: pd.DataFrame(100.0, index=idx, columns=["A", "B", "C"]) for k in ("open", "high", "low", "close")}
    p = Panel(f, "kr_stock")
    sigs = [{"ts": idx[0], "symbol": s, "side": "long", "stop": 95.0} for s in ("A", "B", "C")]
    sigs.append({"ts": idx[1], "symbol": "A", "side": "long", "stop": 95.0})
    res = trade.run(p, sigs, NO_COST, ACCT, trade_cfg={**TRADE, "max_open_positions": 2}, risk_cfg=RISK)
    assert res.trades == 2
    assert res.meta["blocked"] == {"보유 종목 2 ≥ 한도 2": 1, "이미 보유 중인 종목": 1}


def test_kill_switch_ignored_in_backtest():
    r = copy.deepcopy(RISK)
    r["global"]["kill_switch"] = True
    p = _panel([FLAT, (100, 101, 99, 100), FLAT])
    assert trade.run(p, [_sig(p)], NO_COST, ACCT, trade_cfg=TRADE, risk_cfg=r).trades == 1


def test_start_uses_signal_from_previous_bar_only():
    p = _panel([FLAT, FLAT, (100, 101, 99, 100), FLAT])
    res = _run(p, [_sig(p, 0), _sig(p, 1)], start=p.index[2])
    t = _only(res)
    assert t["signal_ts"] == p.index[1] and t["entry_ts"] == p.index[2]


def _breakout(data, params):
    c = data.close
    hi = c.rolling(params["n"]).max().shift(1)
    return [{"ts": ts, "symbol": sym, "side": "long", "stop": c.at[ts, sym] * 0.95, "target": c.at[ts, sym] * 1.1}
            for (ts, sym), v in (c > hi).stack().items() if v]


def _cheat(data, params):
    c = data.close
    fut = c.shift(-1) > c          # 내일 오를 날만 진입 → 미래참조
    return [{"ts": ts, "symbol": sym, "side": "long", "stop": c.at[ts, sym] * 0.95}
            for (ts, sym), v in fut.stack().items() if v]


def _walk(n=600, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))
    return _panel(list(zip(o, hi, lo, c)))


def test_causality_check_passes_honest_and_catches_lookahead():
    p = _walk()
    assert trade.check_causality(_breakout, p, {"n": 20})["passed"]
    assert not trade.check_causality(_cheat, p, {})["passed"]


def test_walk_forward_stitches_and_matches_rerun():
    p = _walk(900)

    class S:
        generate_trades = staticmethod(_breakout)

    out = trade.walk_forward(S, p, {"n": 20}, NO_COST, ACCT, train="12M", test="6M", trade_cfg=TRADE, risk_cfg=RISK)
    test = out["test"]
    assert len(out["windows"]) >= 2 and test.trades > 0
    t0, t1 = test.returns.index[0], test.returns.index[-1] + pd.Timedelta(days=1)
    again = trade.run(p, out["trades"], NO_COST, ACCT, start=t0, end=t1, trade_cfg=TRADE, risk_cfg=RISK)
    assert again.returns.to_numpy() == pytest.approx(test.returns.to_numpy())
    st = trade.stats(test)
    assert st["closed"] + st["open_at_end"] == test.trades
    assert set(st["exit_kinds"]) <= {"stop_loss", "take_profit", "flat_before_close", "delisted"}
    m = test.metrics(p.bars_per_year)
    assert np.isfinite(m["sharpe"])
