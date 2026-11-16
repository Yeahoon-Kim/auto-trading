"""신호형(type: trade) paper 데스크 테스트 (합성 봉, 네트워크·저장소·알림 없음).
핵심: paper 데스크와 백테스트 재생이 같은 규칙(execution/trade_desk.py)으로 같은 결과를 내는지.
실행: uv run pytest -q tests/test_paper_trade.py"""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import engine
from autotrader.backtest import trade as bt
from autotrader.backtest.data import Panel
from autotrader.execution import trade_desk as desk
from autotrader.risk import guards
from autotrader.runner.paper import replay_trade

RISK = copy.deepcopy(guards.load_config())
RISK["global"]["kill_switch"] = False
RISK["orders"]["max_order_value_krw"] = 1e12
TRADE = {"max_position_pct": 30, "max_open_positions": 4}
CAP = 10_000_000
COSTS = engine.Costs(fee_pct=0.015, slippage_pct=0.05, tax_pct_sell=0.2)
FLAT = (100, 100, 100, 100)


def _spec(market="kr_stock", **kw):
    return desk.DeskSpec(lab="T", market=market, trade_cfg=TRADE, **kw)


def _random_panel(n=400, syms=("A", "B", "C"), seed=5, timeframe="1d", market="kr_stock") -> Panel:
    rng = np.random.default_rng(seed)
    idx = (pd.bdate_range("2021-01-01", periods=n) if timeframe == "1d"
           else pd.DatetimeIndex([pd.Timestamp("2021-01-04 14:00") + pd.Timedelta(days=d // 6, hours=d % 6)
                                  for d in range(n)]))
    f = {k: {} for k in ("open", "high", "low", "close")}
    for s in syms:
        c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
        o = np.concatenate([[100], c[:-1]]) * (1 + rng.normal(0, 0.006, n))
        f["open"][s], f["close"][s] = o, c
        f["high"][s] = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
        f["low"][s] = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))
    return Panel({k: pd.DataFrame(v, index=idx) for k, v in f.items()}, market, timeframe=timeframe)


def _signals(panel, n=15, intraday=False, limit=False):
    c = panel.close
    hi = c.rolling(n).max().shift(1)
    rows = []
    for (ts, s), v in (c > hi).stack().items():
        if v:
            px = c.at[ts, s]
            rows.append({"ts": ts, "symbol": s, "side": "long", "stop": px * 0.96, "target": px * 1.06,
                         "entry": px * 0.995 if limit else None, "intraday": intraday})
    return rows


def _by_bar(panel, sig):
    out = {}
    for i, r in zip(panel.index.searchsorted(pd.DatetimeIndex(sig["ts"])), sig.itertuples(index=False)):
        out.setdefault(int(i), []).append(r)
    return out


def _both(panel, rows, costs=COSTS, risk=RISK,
          start_i=1):
    sig = bt.normalize(rows, panel)
    spec = _spec(market=panel.market)
    book, ev = replay_trade(panel, _by_bar(panel, sig), start_i - 1, costs, risk, spec, capital=CAP)
    res = bt.run(panel, sig, costs, bt.Account(capital=CAP), start=panel.index[start_i], trade_cfg=TRADE,
                 risk_cfg=risk, lab="T")
    return book, ev, res


def _paper_equity(ev):
    return pd.Series({pd.Timestamp(e["bar"]): e["equity"] for e in ev if e["kind"] == "equity"})


@pytest.mark.parametrize("kw", [{}, {"limit": True}, {"intraday": True}])
def test_paper_matches_backtest_replay(kw):
    """같은 신호·가격·계좌면 paper 데스크와 백테스트 재생의 자산 곡선·거래가 같아야 한다 (규칙이 한 곳)."""
    p = _random_panel()
    book, ev, res = _both(p, _signals(p, **kw), start_i=30)
    eq = _paper_equity(ev).reindex(res.equity.index)
    assert eq.to_numpy() == pytest.approx(res.equity.to_numpy(), rel=1e-12)
    tl = res.meta["trade_log"]
    closed = tl[tl["exit_kind"] != "open"]
    assert len(closed) > 20
    cols = ["symbol", "exit_kind", "entry", "exit", "qty", "pnl"]
    got = pd.DataFrame(book.closed)[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, closed[cols].reset_index(drop=True), check_dtype=False)
    # 마지막 봉 신호는 paper만 주문으로 처리한다 (백테스트는 체결할 다음 봉이 없어 버림) → 그 봉은 빼고 비교
    last = str(p.index[-1])
    paper_blocked = pd.Series([e["rule"] for e in ev if e["kind"] == "risk_block" and e["bar"] != last])
    assert paper_blocked.value_counts().to_dict() == res.meta["blocked"]


def test_paper_matches_backtest_on_hourly_intraday():
    p = _random_panel(n=360, timeframe="1h", market="us_stock")
    rows = _signals(p, n=6, intraday=True)
    sig = bt.normalize(rows, p)
    spec = desk.DeskSpec(lab="T", market="us_stock", currency="USD", krw_per_unit=1000.0, trade_cfg=TRADE)
    book, ev = replay_trade(p, _by_bar(p, sig), 0, engine.Costs(), RISK, spec, capital=CAP)
    res = bt.run(p, sig, engine.Costs(), bt.Account(capital=CAP, currency="USD", krw_per_unit=1000.0),
                 start=p.index[1], trade_cfg=TRADE, risk_cfg=RISK)
    assert _paper_equity(ev).reindex(res.equity.index).to_numpy() == pytest.approx(res.equity.to_numpy())
    kinds = {r["exit_kind"] for r in book.closed}
    assert "flat_before_close" in kinds


def test_daily_runs_with_saved_book_equal_one_replay():
    """하루씩 따로 돌리며 장부를 JSON으로 저장·복원해도 한 번에 재생한 결과와 같다 (운영 방식)."""
    p = _random_panel(n=120)
    sig = bt.normalize(_signals(p), p)
    by_bar, costs, spec = _by_bar(p, sig), engine.Costs(fee_pct=0.1), _spec()
    one, _ = replay_trade(p, by_bar, 0, costs, RISK, spec, capital=CAP)
    book = None
    for i in range(len(p.index)):
        book, _ = replay_trade(p, by_bar, i, costs, RISK, spec, capital=CAP, end_i=i, book=book)
        # runner/paper.py _save와 같은 방식 (default 없이 json.dumps)
        book = desk.TradeBook.from_dict(json.loads(json.dumps(book.to_dict(), ensure_ascii=False)))
    assert book.equity == pytest.approx(one.equity)
    assert len(book.closed) == len(one.closed) and book.positions.keys() == one.positions.keys()


def test_same_bar_twice_is_skipped():
    p = _random_panel(n=40)
    sig = bt.normalize(_signals(p), p)
    book, _ = replay_trade(p, _by_bar(p, sig), 0, engine.Costs(), RISK, _spec(), capital=CAP)
    eq = book.equity
    book, ev = replay_trade(p, _by_bar(p, sig), 39, engine.Costs(), RISK, _spec(), capital=CAP, book=book)
    assert ev[0]["kind"] == "skip_bar" and book.equity == eq


def test_kill_switch_blocks_new_entries_in_paper_only():
    p = _random_panel(n=200)
    r = copy.deepcopy(RISK)
    r["global"]["kill_switch"] = True
    book, ev, res = _both(p, _signals(p), risk=r)
    assert res.trades > 0                                   # 백테스트 재생은 kill_switch를 끈다
    assert not [e for e in ev if e["kind"] == "fill"]
    assert book.blocked.get("kill_switch 켜짐 — 신규 주문 전면 중단", 0) > 0


def _kr(bars_adj, ratio, tradable=None) -> Panel:
    """수정주가 봉과 원본가/수정주가 비율(봉마다)로 kr_stock panel."""
    idx = pd.bdate_range("2024-01-01", periods=len(bars_adj))
    f = {k: pd.DataFrame({"A": [b[i] for b in bars_adj]}, index=idx, dtype=float)
         for i, k in enumerate(("open", "high", "low", "close"))}
    f["raw_close"] = f["close"] * pd.DataFrame({"A": ratio}, index=idx)
    if tradable is not None:
        f["tradable"] = pd.DataFrame({"A": tradable}, index=idx, dtype=object)
    return Panel(f, "kr_stock")


def test_signal_prices_converted_to_raw_and_split_adjusts_position():
    # 원본가 = 수정주가 × 2 였다가 셋째 봉에 2:1 분할 (원본가가 반으로) — 수정주가는 평평
    p = _kr([FLAT, FLAT, FLAT, FLAT, FLAT], [2, 2, 1, 1, 1])
    rows = [{"ts": p.index[0], "symbol": "A", "side": "long", "stop": 95.0, "target": 110.0}]
    book, ev = replay_trade(p, _by_bar(p, bt.normalize(rows, p)), 0, engine.Costs(), RISK, _spec(), capital=CAP)
    order = next(e for e in ev if e["kind"] == "order")
    assert order["stop"] == 190.0                              # 수정주가 95 → 원본가 190
    assert [e for e in ev if e["kind"] == "corporate_action"]
    pos = book.positions["A"]                                 # 분할 뒤에도 손절에 걸리지 않고 보유
    assert pos["stop"] == pytest.approx(95.0) and pos["entry"] == pytest.approx(100.0)
    assert pos["qty"] == pytest.approx(2 * order["qty"])
    assert book.equity == pytest.approx(CAP)


def test_unmeasurable_position_closed_after_missing_quotes():
    bars = [FLAT, FLAT] + [FLAT] * 5
    p = _kr(bars, [1] * 7, tradable=[True, True, None, None, None, None, None])
    rows = [{"ts": p.index[0], "symbol": "A", "side": "long", "stop": 95.0}]
    book, _ = replay_trade(p, _by_bar(p, bt.normalize(rows, p)), 0, engine.Costs(), RISK, _spec(), capital=CAP)
    out = [r for r in book.closed if r["exit_kind"] == "unmeasurable"]
    n = RISK["data_staleness"]["unmeasurable_retries"]
    assert len(out) == 1 and out[0]["exit_ts"] == p.index[1 + n] and out[0]["exit"] == 100.0
    assert not book.positions


def test_retirement_alert_only():
    p = _kr([FLAT, (100, 100, 100, 100), (100, 100, 96, 96), (96, 96, 90, 90), (90, 90, 90, 90)], [1] * 5)
    rows = [{"ts": p.index[0], "symbol": "A", "side": "long", "stop": 80.0}]
    book, ev = replay_trade(p, _by_bar(p, bt.normalize(rows, p)), 0, engine.Costs(), RISK, _spec(), capital=CAP,
                            retirement={"max_losing_days": 2})
    assert [e for e in ev if e["kind"] == "retirement"]
    assert "A" in book.positions                              # 매매는 그대로 (퇴출은 사람이 정함)


def test_ex_dividend_credits_cash_keeps_qty_and_moves_stop():
    # 수정주가는 평평(배당 재투자), 원본가는 셋째 봉 배당락에 102 → 100 (세후 배당 2.04원 = 100 × (1/f - 1))
    p = _kr([FLAT] * 5, [1.02, 1.02, 1, 1, 1])
    rows = [{"ts": p.index[0], "symbol": "A", "side": "long", "stop": 95.0, "target": 110.0}]
    book, ev = replay_trade(p, _by_bar(p, bt.normalize(rows, p)), 0, engine.Costs(), RISK, _spec(), capital=CAP)
    order = next(e for e in ev if e["kind"] == "order")
    div = [e for e in ev if e["kind"] == "dividend"]
    assert len(div) == 1 and not [e for e in ev if e["kind"] == "corporate_action"]
    pos = book.positions["A"]
    assert pos["qty"] == order["qty"]                          # 수량 그대로
    assert div[0]["cash"] == pytest.approx(order["qty"] * 2.0)  # 원본가 하락분만큼 현금
    assert pos["stop"] == pytest.approx(95.0)                  # 96.9 × (100/102): 손절이 배당락으로 걸리지 않음
    assert book.equity == pytest.approx(CAP)                   # 평가액은 백테스트의 배당 재투자와 같음


def test_float_noise_between_raw_and_adjusted_is_ignored():
    p = _kr([FLAT] * 4, [1.0, 1.0, 1.000004, 1.0])     # 실데이터 068270에서 본 크기의 오차
    rows = [{"ts": p.index[0], "symbol": "A", "side": "long", "stop": 95.0}]
    book, ev = replay_trade(p, _by_bar(p, bt.normalize(rows, p)), 0, engine.Costs(), RISK, _spec(), capital=CAP)
    assert not [e for e in ev if e["kind"] in ("dividend", "corporate_action")]
    assert float(book.positions["A"]["qty"]).is_integer()
