"""공용 백테스트 엔진 테스트 (합성 데이터, 네트워크·저장소 없음). 실행: uv run pytest -q"""

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import engine
from autotrader.backtest.data import Panel, dev_end
from autotrader.backtest.lab import _mark_lockbox

NO_COST = engine.Costs()


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    from autotrader.data import store
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    return tmp_path


def _panel(close: dict, open_: dict | None = None, start="2020-01-01") -> Panel:
    c = pd.DataFrame(close, index=pd.bdate_range(start, periods=len(next(iter(close.values())))), dtype=float)
    o = pd.DataFrame(open_, index=c.index, dtype=float) if open_ else c.copy()
    return Panel({"open": o, "close": c}, "us_stock")


def test_next_close_holds_from_next_bar():
    p = _panel({"A": [100, 110, 121, 133.1]})
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])          # 첫날 종가에 결정 → 다음 날 종가 체결
    r = engine.run(p, w, NO_COST, fill="next_close").returns
    assert r.iloc[0] == pytest.approx(0.0)                     # 체결 봉의 수익은 아직 없음
    assert r.iloc[1:].to_numpy() == pytest.approx([0.1, 0.1])


def test_next_open_splits_overnight_and_intraday():
    p = _panel({"A": [100, 110, 120]}, {"A": [100, 105, 125]})
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=p.index)
    r = engine.run(p, w, NO_COST, fill="next_open").returns
    assert r.iloc[0] == pytest.approx(110 / 105 - 1)          # 둘째 날 시가 매수 → 종가
    assert r.iloc[1] == pytest.approx(120 / 110 - 1)          # 이후는 종가→종가


def test_costs_turnover_and_sell_tax():
    p = _panel({"A": [100.0] * 5})
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=p.index)
    c = engine.Costs(fee_pct=0.1, slippage_pct=0.1, tax_pct_sell=0.2)
    res = engine.run(p, w, c, fill="next_close")
    assert res.turnover.sum() == pytest.approx(2.0)
    assert res.cost.sum() == pytest.approx(2 * 0.002 + 0.002)
    assert res.trades == 1
    assert engine.run(p, w, c.scaled(2), fill="next_close").cost.sum() == pytest.approx(2 * res.cost.sum())


def test_drift_is_rebalanced_and_charged():
    p = _panel({"A": [100, 100, 200, 200], "B": [100, 100, 100, 100]})
    w = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4}, index=p.index)
    res = engine.run(p, w, engine.Costs(fee_pct=1.0), fill="next_close")
    assert res.turnover.iloc[0] == pytest.approx(1.0)          # 둘째 날 종가에 첫 진입
    # 셋째 날 A가 두 배 → 비중 2/3:1/3으로 흘러감 → 0.5:0.5로 되돌리는 회전율 1/3
    assert res.turnover.iloc[1] == pytest.approx(1 / 3)


def test_delisted_position_is_force_closed_at_last_price():
    p = _panel({"A": [100, 100, 50, np.nan, np.nan], "B": [10.0] * 5})
    w = pd.DataFrame({"A": [1.0] * 5, "B": [0.0] * 5}, index=p.index)
    res = engine.run(p, w, NO_COST, fill="next_close")
    assert res.forced_exits == 1
    assert (1 + res.returns).prod() == pytest.approx(0.5)      # 마지막 가격 50에 청산
    assert res.weights["A"].iloc[-1] == 0


def test_halted_symbol_is_not_traded():
    p = _panel({"A": [100, np.nan, 100, 100]})
    w = pd.DataFrame({"A": [1.0] * 4}, index=p.index)
    res = engine.run(p, w, NO_COST, fill="next_close")
    assert res.weights["A"].iloc[0] == 0                        # 정지일에는 체결 안 됨
    assert res.weights["A"].iloc[1] == 1 and res.forced_exits == 0


def _rand_panel(n=600, k=3, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0))
    return _panel({f"S{i}": c[:, i] for i in range(k)})


def _momentum(data, params):
    return (data.close.pct_change(params["lookback"]) > 0).astype(float) / data.close.shape[1]


def _peek(data, params):
    return (data.close.shift(-1) > data.close).astype(float)   # 내일 종가를 봄 → 미래참조


def test_causality_check_catches_lookahead():
    p = _rand_panel()
    assert engine.check_causality(_momentum, p, {"lookback": 20})["passed"]
    assert not engine.check_causality(_peek, p, {})["passed"]


def test_walk_forward_windows_and_stitching():
    p = _rand_panel(n=1300)
    wins = engine.windows(p.index, train="24M", test="6M")
    assert len(wins) >= 4 and wins[1]["test"][0] == wins[0]["test"][1]

    class S:
        generate_signals = staticmethod(_momentum)
    out = engine.walk_forward(S, p, {"lookback": 20}, NO_COST, train="24M", test="6M")
    r = out["test"].returns
    assert r.index[0] >= wins[0]["test"][0] and r.index.is_monotonic_increasing
    # 이어 붙인 결과 = 전체 데이터로 한 번에 돌린 결과 (신호가 인과적이고 fit이 없으면 같아야 함)
    once = engine.run(p, _momentum(p, {"lookback": 20}), NO_COST, start=r.index[0], end=r.index[-1] + pd.Timedelta(days=1))
    assert r.to_numpy() == pytest.approx(once.returns.to_numpy())


def test_perturb_params():
    q = engine.perturb({"n": 10, "x": 0.5, "flag": True, "name": "a", "one": 1}, 0.8)
    assert q == {"n": 8, "x": 0.4, "flag": True, "name": "a", "one": 1}


def test_dev_end_is_six_months_back():
    assert dev_end("2026-11-15") == pd.Timestamp("2026-05-15")


def test_mark_lockbox_keeps_comments(tmp_path):
    f = tmp_path / "lab.yaml"
    f.write_text('id: L1\nlockbox:   # 판정 시\n  start: ""\n  end: ""\n  opened: false   # 한 번\ncosts:\n  fee_pct:\n',
                 encoding="utf-8")
    _mark_lockbox(f, "2026-05-15", "2026-11-13")
    s = f.read_text(encoding="utf-8")
    assert 'start: "2026-05-15"' in s and "opened: true   # 한 번" in s and "fee_pct:" in s


# ── 적립금·정수 주식·리밸런싱 방식·세율 표·원화 환산·배당세 ──
def test_monthly_contributions_flat_price():
    p = _panel({"A": [100.0] * 70}, start="2024-01-01")
    w = pd.DataFrame({"A": 1.0}, index=p.index)
    fl = engine.monthly(p.index, 100.0)
    res = engine.run(p, w, engine.Costs(fee_pct=1.0), fill="next_close", flows=fl)
    assert res.invested.iloc[-1] == 400.0                       # 1~4월 (첫 봉 적립은 다음 봉으로)
    assert res.equity.iloc[-1] == pytest.approx(400 * 0.99)
    assert res.money()["return_on_invested"] == pytest.approx(-0.01)


def test_whole_shares_keep_cash_and_pay_fee_from_cash():
    p = _panel({"A": [300, 300, 400, 400]})
    w = pd.DataFrame({"A": 1.0}, index=p.index)
    res = engine.run(p, w, engine.Costs(fee_pct=1.0), fill="next_close", capital=1000, whole_shares=True)
    assert res.shares["A"].iloc[-1] == 3                        # 1000 // (300 × 1.01)
    cash = 1000 - 3 * 300 * 1.01
    assert res.equity.iloc[-1] == pytest.approx(cash + 3 * 400)
    assert res.money()["cash_end"] == pytest.approx(cash)


def test_on_change_does_not_trade_drift():
    p = _panel({"A": [100, 100, 200, 200], "B": [100, 100, 100, 100]})
    w = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4}, index=p.index)
    res = engine.run(p, w, engine.Costs(fee_pct=1.0), fill="next_close", rebalance="on_change")
    assert res.turnover.iloc[0] == pytest.approx(1.0) and res.turnover.iloc[1:].sum() == 0
    assert res.weights["A"].iloc[-1] == pytest.approx(2 / 3)


def test_tax_schedule_by_date():
    c = engine.Costs(tax_pct_sell={"2020-01-01": 0.2, "2020-01-08": 0.1})
    idx = pd.DatetimeIndex(["2019-12-31", "2020-01-07", "2020-01-08", "2021-01-01"])
    assert c.tax_rates(idx) == pytest.approx([0.002, 0.002, 0.001, 0.001])
    assert c.scaled(2).tax_rates(idx[-1:]) == pytest.approx([0.002])


def test_xirr_one_year():
    t0 = pd.Timestamp("2024-01-01")
    assert engine.xirr([(t0, -100.0), (t0 + pd.Timedelta(days=365.25), 110.0)]) == pytest.approx(0.10, abs=1e-6)


def test_us_loader_dividend_tax_and_krw(tmp_store):
    from autotrader.backtest import data as bdata
    from autotrader.data import store
    ts = pd.date_range("2024-01-02 14:30", periods=4, freq="1D", tz="UTC")
    # 3번째 날 배당락: 가격은 그대로, adj_close 쪽 수익 +1%
    us = pd.DataFrame({"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0,
                       "adj_close": [100 / 1.01, 100 / 1.01, 100.0, 100.0]})
    store.write("us_1d", "SPY", us)
    store.write("ecos", "usdkrw_ecos", pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, tz="UTC"),
                                                    "value": [1000.0, 1000.0, 1000.0, 1000.0, 1100.0]}))
    full = bdata.load("us_stock", ["SPY"], lockbox=True, dividend_tax=0)
    net = bdata.load("us_stock", ["SPY"], lockbox=True, dividend_tax=0.15)
    assert full.close["SPY"].iloc[2] / full.close["SPY"].iloc[1] - 1 == pytest.approx(0.01)
    assert net.close["SPY"].iloc[2] / net.close["SPY"].iloc[1] - 1 == pytest.approx(0.0085)
    krw = bdata.load("us_stock", ["SPY"], lockbox=True, dividend_tax=0, currency="KRW")
    assert krw.meta["currency"] == "KRW" and krw.fx["SPY"].iloc[-1] == 1100
    assert krw.close["SPY"].iloc[-1] / krw.close["SPY"].iloc[-2] - 1 == pytest.approx(0.10)
