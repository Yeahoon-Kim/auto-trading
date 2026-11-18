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


# ── 빠졌던 경우 ──
def _multi(cols: dict, market="kr_stock", tradable=None) -> Panel:
    """cols: {종목: [(o, h, l, c) 또는 None(가격 없음)]}"""
    n = len(next(iter(cols.values())))
    idx = pd.bdate_range("2020-01-01", periods=n)
    f = {}
    for i, k in enumerate(("open", "high", "low", "close")):
        f[k] = pd.DataFrame({s: [np.nan if b is None else b[i] for b in bars] for s, bars in cols.items()},
                            index=idx, dtype=float)
    if tradable is not None:
        f["tradable"] = pd.DataFrame(tradable, index=idx)
    return Panel(f, market)


def test_delisted_position_force_closed_at_last_price():
    p = _multi({"A": [FLAT, (100, 101, 99, 100), (97, 98, 96, 97), None, None],
                "B": [FLAT] * 5})
    res = _run(p, [_sig(p, stop=90.0, target=None)])
    t = _only(res)
    assert t["exit_kind"] == "delisted" and t["exit"] == 97.0 and t["exit_ts"] == p.index[3]
    assert res.forced_exits == 1
    assert res.equity.iloc[-1] == pytest.approx(1_000_000 - 1000 * 3)   # 10,000 / (100-90) = 1000주


def test_untradable_bar_holds_and_blocks_entry():
    # 거래 없는 날(셋째 봉)에는 저가가 손절 밑이어도 판정 보류
    p = _multi({"A": [FLAT, (100, 101, 99, 100), (100, 100, 90, 100), (100, 101, 99, 100)]},
               tradable={"A": [True, True, False, True]})
    assert _only(_run(p, [_sig(p)]))["exit_kind"] == "open"
    # 체결 봉이 거래 없는 날이면 미체결
    res = _run(p, [_sig(p, 1)])
    assert res.trades == 0 and res.meta["blocked"] == {"미체결": 1}


def _hourly(days: list[list[tuple]], market="us_stock") -> Panel:
    """하루 여러 봉 (UTC 14:00부터 1시간 간격)."""
    ts = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=d, hours=14 + h)
          for d, bars in enumerate(days) for h in range(len(bars))]
    flat = [b for bars in days for b in bars]
    f = {k: pd.DataFrame({"A": [b[i] for b in flat]}, index=pd.DatetimeIndex(ts), dtype=float)
         for i, k in enumerate(("open", "high", "low", "close"))}
    return Panel(f, market, timeframe="1h")


def test_intraday_hourly_flattens_only_on_last_bar_of_day():
    p = _hourly([[FLAT, (100, 101, 99, 100.5), (100.5, 102, 100, 101), (101, 103, 100, 102)],
                 [(102, 103, 101, 102)] * 2])
    t = _only(_run(p, [_sig(p, 0, intraday=True)]))
    assert t["exit_kind"] == "flat_before_close"
    assert t["exit_ts"] == p.index[3] and t["exit"] == 102.0     # 그날 마지막 봉 종가, 중간 봉에선 안 함


CAP_KRW = guards.load_config()["orders"]["max_order_value_krw"]


def test_order_cap_not_applied_but_recorded():
    """1회 주문 상한은 백테스트에서 수량을 자르지 않고, 잘렸을 주문만 기록한다 (사용자 결정 2026-11-18)."""
    p = _panel([FLAT, (100, 101, 99, 100), (100, 100, 94, 96), FLAT])
    acct = trade.Account(capital=10 * CAP_KRW)
    res = trade.run(p, [_sig(p)], NO_COST, acct, trade_cfg=TRADE)    # 기본 risk.yaml 상한 그대로 넘김
    t = _only(res)
    want = int(acct.capital * 0.01 / 5)                               # 위험 1% / (100-95)
    assert want * 100 > CAP_KRW                                       # 상한보다 큰 주문
    assert t["qty"] == want and t["binding"] == "risk"                # 잘리지 않음
    oc = res.meta["order_cap"]
    assert oc["cap_krw"] == CAP_KRW and oc["would_cap"] == 1 and oc["would_block"] == 0
    row = oc["orders"].iloc[0]
    assert row["symbol"] == "A" and row["ts"] == p.index[0]
    assert row["qty"] == want and row["capped_qty"] == CAP_KRW // 100
    assert trade.stats(res)["order_cap"] == {"cap_krw": CAP_KRW, "would_cap": 1, "would_block": 0}


def test_order_under_cap_not_recorded():
    p = _panel([FLAT, (100, 101, 99, 100), FLAT])
    res = trade.run(p, [_sig(p)], NO_COST, trade.Account(capital=CAP_KRW), trade_cfg=TRADE)
    assert _only(res)["qty"] * 100 <= CAP_KRW
    assert res.meta["order_cap"]["would_cap"] == 0 and res.meta["order_cap"]["orders"].empty


def test_order_cap_would_block_when_capped_qty_zero():
    big = (CAP_KRW * 2, CAP_KRW * 2, CAP_KRW * 2, CAP_KRW * 2)        # 한 주 가격이 상한의 두 배
    p = _panel([big, big, big])
    res = trade.run(p, [_sig(p, stop=CAP_KRW * 1.9, target=None)], NO_COST,
                    trade.Account(capital=CAP_KRW * 100), trade_cfg=TRADE)
    assert res.trades == 1                                            # 백테스트는 진입
    oc = res.meta["order_cap"]
    assert oc["would_cap"] == 1 and oc["would_block"] == 1 and oc["orders"].iloc[0]["capped_qty"] == 0


def test_desk_still_applies_order_cap():
    """paper 데스크(같은 trade_desk.plan)는 상한을 그대로 적용한다 — 백테스트만 푼다."""
    from autotrader.execution import trade_desk as desk
    from autotrader.runner.paper import replay_trade
    p = _panel([FLAT, (100, 101, 99, 100), (100, 100, 94, 96), FLAT])
    risk = copy.deepcopy(guards.load_config())
    risk["global"]["kill_switch"] = False
    sig = trade.normalize([_sig(p)], p)
    by_bar = {0: list(sig.itertuples(index=False))}
    book, ev = replay_trade(p, by_bar, 0, NO_COST, risk, desk.DeskSpec(lab="T", market="kr_stock", trade_cfg=TRADE),
                            capital=10 * CAP_KRW)
    order = next(e for e in ev if e["kind"] == "order")
    assert order["qty"] == CAP_KRW // 100 and order["binding"] == "max_order_value"
    # 넘긴 risk 설정은 바꾸지 않는다
    assert risk["orders"]["max_order_value_krw"] == CAP_KRW


def test_foreign_account_converts_order_cap():
    p = _panel([FLAT, (100, 101, 99, 100), FLAT], market="us_stock")
    acct = trade.Account(capital=100_000, currency="USD", krw_per_unit=1000.0)
    res = trade.run(p, [_sig(p)], NO_COST, acct, trade_cfg=TRADE)    # 기본 risk.yaml: 주문 상한 100만 원
    t = _only(res)
    assert t["qty"] == 200 and t["binding"] == "risk"                 # 자르지 않음: $1,000 위험 / $5
    row = res.meta["order_cap"]["orders"].iloc[0]
    assert row["capped_qty"] == CAP_KRW / 1000 / 100                  # 100만 원 / 1000 = $1,000 / $100 = 10주


def test_foreign_account_without_fx_fails_loudly():
    p = _panel([FLAT, FLAT], market="us_stock")
    with pytest.raises(ValueError, match="krw_per_unit"):
        trade.run(p, [_sig(p)], NO_COST, trade.Account(capital=100_000, currency="USD"))


def test_atr_only_signal_uses_two_atr_stop():
    p = _panel([FLAT, (100, 101, 99, 100), (99, 100, 95, 96), FLAT])
    t = _only(_run(p, [_sig(p, stop=None, atr=2.0)]))
    assert t["stop"] == 96.0 and t["qty"] == 2500                    # 100 - 2×2, 10,000 / 4
    assert t["exit_kind"] == "stop_loss" and t["exit"] == 96.0


def test_wrong_side_stop_is_blocked_not_crash():
    p = _panel([FLAT, FLAT, FLAT])
    res = _run(p, [_sig(p, stop=105.0)])
    assert res.trades == 0 and list(res.meta["blocked"]) == ["신호 오류"]


def test_walk_forward_fit_sees_only_train_window():
    p = _walk(900)
    seen = []

    class S:
        generate_trades = staticmethod(_breakout)

        @staticmethod
        def fit(train, params):
            seen.append(train.index[-1])
            return {**params, "n": 10 + len(seen)}

    out = trade.walk_forward(S, p, {"n": 20}, NO_COST, ACCT, train="12M", test="6M", trade_cfg=TRADE, risk_cfg=RISK)
    for w, last_train in zip(out["windows"], seen):
        assert last_train < pd.Timestamp(w["test"][0])
        assert w["params"]["n"] == 10 + seen.index(last_train) + 1


def test_equity_equals_capital_plus_trade_pnl():
    p = _walk(700, seed=11)
    c = engine.Costs(fee_pct=0.015, slippage_pct=0.05, tax_pct_sell=0.2)
    res = _run(p, _breakout(p, {"n": 15}), costs=c)
    tl = res.meta["trade_log"]
    closed = tl[tl["exit_kind"] != "open"]
    assert len(closed) > 20
    open_ = tl[tl["exit_kind"] == "open"]
    mtm = sum(r.qty * (p.close["A"].iloc[-1] - r.entry) for r in open_.itertuples())
    in_cost = sum(r.qty * r.entry * c.per_trade for r in open_.itertuples())
    assert res.equity.iloc[-1] == pytest.approx(ACCT.capital + closed["pnl"].sum() + mtm - in_cost)


# ── 독립 대조: replay.history (같은 규칙을 다르게 구현한 기존 코드) ──
def test_matches_replay_history_on_random_sessions():
    """하루 7봉 × 250일 무작위 장중 데이터. 매일 둘째 봉 종가에 판단 → 셋째 봉 시가(= 둘째 봉 종가) 진입,
    당일 청산. replay.history도 같은 판정(손절 먼저, 미도달은 마감가)을 하므로 결과·R이 같아야 한다."""
    from autotrader.backtest import replay

    rng = np.random.default_rng(7)
    days, plans = [], {}
    for d in range(250):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, 7)))
        o = np.concatenate([[c[0] * (1 + rng.normal(0, 0.003))], c[:-1]])    # 장중 시가 = 직전 종가
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, 7)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, 7)))
        days.append(list(zip(o, hi, lo, c)))
        side = "long" if rng.random() < 0.5 else "short"
        e = c[1]
        stop, tgt = (e * 0.995, e * 1.008) if side == "long" else (e * 1.005, e * 0.992)
        plans[d] = (side, e, stop, tgt)
    p = _hourly(days)
    sigs = [{"ts": p.index[d * 7 + 1], "symbol": "A", "side": s, "stop": st, "target": tg, "intraday": True}
            for d, (s, e, st, tg) in plans.items()]
    res = _run(p, sigs)
    tl = res.meta["trade_log"].reset_index(drop=True)
    assert len(tl) == 250 and set(tl["exit_kind"]) <= {"stop_loss", "take_profit", "flat_before_close"}

    sess, dates = {}, sorted({t.date() for t in p.index})
    for d, date in enumerate(dates):
        g = pd.DataFrame(days[d], columns=["open", "high", "low", "close"])
        g.insert(0, "ts", p.index[d * 7:(d + 1) * 7])
        sess[date] = g
    order = {date: d for d, date in enumerate(dates)}
    hist = replay.history(sess, lambda g: (1, *[plans[order[g["ts"].iloc[0].date()]][i] for i in (1, 2, 3)],
                                            plans[order[g["ts"].iloc[0].date()]][0]))
    kind = {"stop_loss": "stop", "take_profit": "target", "flat_before_close": "neither"}
    assert [kind[k] for k in tl["exit_kind"]] == hist["outcome"].tolist()
    assert tl["r"].to_numpy() == pytest.approx(hist["r"].to_numpy(), abs=1e-9)
    assert (hist["outcome"] == "stop").sum() > 30 and (hist["outcome"] == "target").sum() > 30


# ── 1분봉: 장 마감 N분 전 청산 (risk.yaml flat_before_close_min, us_stock 10분) ──
def _minutes(sessions: list[tuple[str, str, dict]], symbols=("A",)) -> Panel:
    """sessions: [(첫 봉 시각 UTC, 마지막 봉 시각 UTC, {봉 시각: (o, h, l, c)} 덮어쓸 봉)]. 나머지 봉은 100 평평."""
    idx, rows = [], []
    for first, last, over in sessions:
        for ts in pd.date_range(first, last, freq="1min"):
            idx.append(ts)
            rows.append(over.get(ts.strftime("%H:%M"), FLAT))
    f = {k: pd.DataFrame({s: [r[i] for r in rows] for s in symbols}, index=pd.DatetimeIndex(idx), dtype=float)
         for i, k in enumerate(("open", "high", "low", "close"))}
    return Panel(f, "us_stock", timeframe="1m")


def test_minute_bars_flatten_n_minutes_before_close():
    # 마감 21:00 → 20:49 봉(끝 20:50, 남은 10분)에서 정리. 20:55 폭락은 이미 청산 뒤라 무관
    p = _minutes([("2020-01-02 20:00", "2020-01-02 20:59", {"20:49": (100, 101, 99, 100.7),
                                                           "20:55": (100, 100, 80, 81)}),
                  ("2020-01-03 20:00", "2020-01-03 20:59", {})])
    t = _only(_run(p, [_sig(p, 0, intraday=True)]))
    assert t["exit_kind"] == "flat_before_close"
    assert t["exit_ts"] == pd.Timestamp("2020-01-02 20:49") and t["exit"] == 100.7
    assert "10분 전" in t["exit_reason"]


def test_minute_bars_stop_before_flat_window_wins():
    p = _minutes([("2020-01-02 20:00", "2020-01-02 20:59", {"20:30": (100, 100, 94, 95)})])
    t = _only(_run(p, [_sig(p, 0, intraday=True)]))
    assert t["exit_kind"] == "stop_loss" and t["exit_ts"] == pd.Timestamp("2020-01-02 20:30")


def test_minute_bars_early_close_day_and_overnight_position():
    # 반일장: 그날 마지막 봉이 17:59 → 마감 18:00으로 보고 17:49 봉에서 정리. 당일 청산이 아닌 포지션은 넘어간다
    p = _minutes([("2020-01-02 17:00", "2020-01-02 17:59", {}),
                  ("2020-01-03 20:00", "2020-01-03 20:59", {})], symbols=("A", "B"))
    res = _run(p, [_sig(p, 0, intraday=True), {**_sig(p, 0), "symbol": "B"}])
    tl = res.meta["trade_log"].set_index("symbol")
    assert tl.loc["A", "exit_kind"] == "flat_before_close"
    assert tl.loc["A", "exit_ts"] == pd.Timestamp("2020-01-02 17:49")
    assert tl.loc["B", "exit_kind"] == "open"
