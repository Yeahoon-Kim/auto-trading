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


def test_on_signal_rebalances_all_only_when_target_changes():
    # A가 2배 → 비중 2/3로 불어남. 그 뒤 B를 빼고 C를 넣으면(A 목표는 그대로 0.5)
    # on_change는 A를 2/3로 둔 채 C를 0.5 사서 합이 1을 넘고, on_signal은 A까지 0.5로 맞춘다
    p = _panel({"A": [100, 100, 200, 200, 200], "B": [100] * 5, "C": [100] * 5})
    w = pd.DataFrame({"A": [0.5] * 5, "B": [0.5, 0.5, 0.5, 0, 0], "C": [0, 0, 0, 0.5, 0.5]}, index=p.index)
    sig = engine.run(p, w, engine.Costs(), fill="next_close", rebalance="on_signal")
    assert sig.turnover.iloc[1] == 0                        # 가격만 변한 봉은 매매 없음
    assert sig.weights.iloc[-1].sum() == pytest.approx(1.0)
    assert sig.weights["A"].iloc[-1] == pytest.approx(0.5)
    chg = engine.run(p, w, engine.Costs(), fill="next_close", rebalance="on_change")
    assert chg.weights.iloc[-1].sum() > 1.0                 # on_change의 알려진 성질 (문서화된 레버리지)


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


# ── 공매도·레버리지, 정수 주식 현금 부족, fit() ──
@pytest.mark.parametrize("wv", [-1.0, -0.5, 2.0])
def test_short_and_leverage_returns(wv):
    p = _panel({"A": [100, 100, 110, 99]})
    r = engine.run(p, pd.DataFrame({"A": wv}, index=p.index), NO_COST, fill="next_close").returns
    assert r.iloc[1] == pytest.approx(wv * 0.10) and r.iloc[2] == pytest.approx(wv * -0.10)


def test_short_whole_shares_pnl():
    p = _panel({"A": [100, 100, 110, 99]})
    r = engine.run(p, pd.DataFrame({"A": -1.0}, index=p.index), NO_COST, fill="next_close",
                   capital=1000, whole_shares=True, rebalance="on_change")
    assert r.shares["A"].tolist() == [-10, -10, -10]
    assert r.equity.tolist() == pytest.approx([1000, 900, 1010])


def test_whole_shares_swap_never_goes_negative_cash():
    c = pd.DataFrame({"A": [100.0] * 5, "B": [100.0] * 5}, index=pd.bdate_range("2024-01-01", periods=5))
    p = Panel({"open": c, "close": c}, "kr_stock")
    w = pd.DataFrame({"A": [1, 1, 0, 0, 0], "B": [0, 0, 1, 1, 1]}, index=c.index, dtype=float)
    r = engine.run(p, w, engine.Costs(tax_pct_sell=0.2), fill="next_close", capital=1000, whole_shares=True)
    cash = r.equity * (1 - r.weights.sum(axis=1))
    assert (cash >= -1e-9).all() and r.shares["B"].iloc[-1] == 9      # 매도세 2원 때문에 10주는 못 삼


def test_walk_forward_fit_sees_only_train_data():
    p = _rand_panel(n=1300)
    seen = []

    class S:
        generate_signals = staticmethod(_momentum)

        @staticmethod
        def fit(train, params):
            seen.append((train.index[0], train.index[-1]))
            best = max((5, 20, 60), key=lambda lb: engine.sharpe(
                engine.run(train, _momentum(train, {"lookback": lb}), NO_COST).returns, 252))
            return {**params, "lookback": best}

    out = engine.walk_forward(S, p, {"lookback": 20}, NO_COST, train="24M", test="6M")
    for (lo, hi), w in zip(seen, out["windows"]):
        assert hi < pd.Timestamp(w["test"][0]) and lo >= pd.Timestamp(w["train"][0])
    # 각 시험 구간 비중 = 그 창에서 고른 파라미터로 만든 신호
    w0 = out["windows"][1]
    c0, c1 = pd.Timestamp(w0["test"][0]), pd.Timestamp(w0["test"][1])
    sig = engine.align_weights(_momentum(p.slice(end=c1), w0["params"]), p.slice(end=c1))
    got = out["weights"].loc[(out["weights"].index >= c0) & (out["weights"].index < c1)]
    assert np.allclose(got.to_numpy(), sig.loc[got.index].to_numpy())


def test_kr_jump_neutralized_but_delisting_tail_kept():
    from autotrader.backtest.data import _neutralize_jumps
    d = pd.bdate_range("2024-01-01", periods=30)
    a = pd.DataFrame({"symbol": "A", "date": d, "open": 100.0, "close": 100.0})
    a.loc[10:, ["open", "close"]] = 2000.0                     # 정지 뒤 재상장: 20배 (계수 누락)
    a.loc[5:9, ["open", "close"]] = np.nan                     # 정지 기간 (가격 0 → 결측)
    b = pd.DataFrame({"symbol": "B", "date": d[:20], "open": 100.0, "close": 100.0})
    b.loc[15:, ["open", "close"]] = 10.0                       # 상장폐지 직전 정리매매 -90%
    px = pd.concat([a, b], ignore_index=True)
    st = _neutralize_jumps(px)
    assert st["neutralized"] == 1 and st["kept_delisting_tail"] == 1
    ca = px[px.symbol == "A"]["close"]
    assert ca.iloc[0] == pytest.approx(2000.0) and ca.iloc[-1] == 2000.0     # 이전 가격을 옮겨 이어 붙임
    assert px[px.symbol == "B"]["close"].iloc[-1] == 10.0


def test_kr_reused_code_is_split_into_listings():
    from autotrader.backtest.data import _split_reused_codes
    d1 = pd.bdate_range("2015-01-01", periods=10)
    d2 = pd.bdate_range("2024-03-01", periods=10)              # 8년 뒤 같은 코드로 다른 회사 상장
    px = pd.DataFrame({"symbol": "036220", "date": d1.append(d2), "close": [100.0] * 10 + [5000.0] * 10})
    st = _split_reused_codes(px)
    assert st["split"] == 1
    assert set(px["symbol"]) == {"036220", f"036220@{d1[-1]:%Y%m%d}"}
    assert (px.loc[px.symbol == "036220", "date"] >= d2[0]).all()   # 최근 상장이 원래 코드를 갖는다
    one = pd.DataFrame({"symbol": "A", "date": d1, "close": 1.0})
    assert _split_reused_codes(one) == {"split": 0}


def test_kr_share_event_fixes_missing_consolidation():
    from autotrader.backtest.data import _fix_share_events
    d = pd.bdate_range("2026-04-01", periods=20)
    a = pd.DataFrame({"symbol": "A", "date": d, "open": 203.0, "close": 203.0, "shares": 3_000_000.0})
    a.loc[10:, ["open", "close"]] = 6090.0                      # 30:1 병합인데 수정계수 1.0
    a.loc[10:, "shares"] = 100_000.0
    a.loc[15:, ["open", "close"]] = 6500.0                      # 이후 정상 변동은 그대로
    b = pd.DataFrame({"symbol": "B", "date": d, "open": 100.0, "close": 100.0, "shares": 1e6})
    b.loc[10:, ["open", "close"]] = 200.0                       # 주식수 변동 없는 급등은 건드리지 않음
    c = pd.DataFrame({"symbol": "C", "date": d, "open": 50.0, "close": 50.0, "shares": 1e6})
    c.loc[10:, "shares"] = 2e6                                  # 분할이 수정계수에 이미 반영됨 → 가격 연속, 건드리지 않음
    px = pd.concat([a, b, c], ignore_index=True)
    st = _fix_share_events(px)
    assert st["fixed"] == 1 and st["events"] == [f"A:{d[10]:%Y-%m-%d}"]
    ca = px[px.symbol == "A"]["close"].to_numpy()
    assert ca[0] == pytest.approx(6090.0) and ca[10] == 6090.0 and ca[-1] == 6500.0
    assert px[px.symbol == "B"]["close"].iloc[0] == 100.0


def test_untradable_day_blocks_next_close_fill():
    p = _panel({"A": [100, 100, 100, 100]})                     # 정지일에도 종가는 직전 값 복사 (KRX)
    p["tradable"] = pd.DataFrame({"A": [True, False, True, True]}, index=p.index)
    w = pd.DataFrame({"A": [1.0] * 4}, index=p.index)
    res = engine.run(p, w, NO_COST, fill="next_close")
    assert res.weights["A"].iloc[0] == 0 and res.weights["A"].iloc[1] == 1


def test_us_bad_adj_step_removed():
    from autotrader.backtest.data import _fix_adj_steps
    d = pd.bdate_range("2016-06-27", periods=6)
    px = pd.DataFrame({"symbol": "DHR", "date": d, "close": [66.0, 67.3, 67.9, 68.8, 71.3, 71.5],
                       "adj_close": [40.2, 41.0, 41.4, 41.9, 67.6, 67.8]})   # 분사 이중 반영
    st = _fix_adj_steps(px)
    assert st["fixed"] == 1
    r = px["adj_close"].pct_change()
    assert r.abs().max() < 0.05 and px["adj_close"].iloc[-1] == 67.8
    ok = pd.DataFrame({"symbol": "X", "date": d, "close": 100.0, "adj_close": [97.0] * 3 + [98.0] * 3})  # 배당 ~1%
    assert _fix_adj_steps(ok) == {"fixed": 0}


def test_kr_share_event_lagged_bonus_issue():
    from autotrader.backtest.data import _fix_share_events
    d = pd.bdate_range("2010-04-01", periods=60)
    a = pd.DataFrame({"symbol": "A", "date": d, "open": 2035.0, "close": 2035.0, "shares": 48e6})
    a.loc[20:, ["open", "close"]] = 1700.0                      # 무상증자 권리락 (계수 없음)
    a.loc[40:, "shares"] = 57.6e6                               # 신주는 20거래일 뒤 상장
    b = pd.DataFrame({"symbol": "B", "date": d, "open": 100.0, "close": 100.0, "shares": 1e6})
    b.loc[20:, ["open", "close"]] = 60.0                        # 실제 급락 + 나중 유상증자: 시총 변동이 0이 아니면 그대로
    b.loc[40:, "shares"] = 1.3e6
    px = pd.concat([a, b], ignore_index=True)
    st = _fix_share_events(px)
    assert st["lagged"] == 1 and st["events"] == [f"A:{d[20]:%Y-%m-%d}"]
    ra = px[px.symbol == "A"]["close"].pct_change()
    assert ra.abs().max() < 0.01
    assert px[px.symbol == "B"]["close"].iloc[0] == 100.0


def test_kr_idle_day_base_price_change_neutralized():
    from autotrader.backtest.data import _neutralize_jumps
    d = pd.bdate_range("2025-07-28", periods=10)
    px = pd.DataFrame({"symbol": "A", "date": d, "open": np.nan, "close": 2745.0, "volume": 0.0})
    px.loc[5:, "close"] = 3685.0                                # 정지 중 기준가 +34% (35% 미만)
    st = _neutralize_jumps(px)
    assert st["neutralized"] == 1 and px["close"].iloc[0] == pytest.approx(3685.0)


# ── on_signal: 신호 날 기준, 상장폐지는 그 종목만 ──
def _monthly_ew(idx, cols):
    m = idx.to_series().dt.month.diff() != 0
    return pd.DataFrame(1 / len(cols), index=idx[m], columns=cols)


def test_on_signal_rebalances_on_repeated_same_target():
    idx = pd.bdate_range("2024-01-01", periods=60)
    c = pd.DataFrame({"A": np.linspace(100, 150, 60), "B": 100.0, "C": np.linspace(100, 80, 60)}, idx)
    p = Panel({"open": c, "close": c}, "us_stock")
    r = engine.run(p, _monthly_ew(idx, list(c.columns)), engine.Costs(fee_pct=0.1), fill="next_close",
                   rebalance="on_signal")
    trade_days = r.turnover[r.turnover > 1e-9].index
    assert [d.month for d in trade_days] == [1, 2, 3]            # 매달 같은 1/3이어도 매달 맞춤
    for d in trade_days:
        assert r.weights.loc[d].to_numpy() == pytest.approx([1 / 3] * 3, abs=1e-3)


def test_on_signal_delisting_liquidates_only_that_symbol():
    idx = pd.bdate_range("2024-01-01", periods=40)
    c = pd.DataFrame({"A": np.linspace(100, 150, 40), "B": 100.0, "C": 100.0}, idx)
    c.loc[idx[15]:, "C"] = np.nan                                 # 1월 중순 상장폐지
    p = Panel({"open": c, "close": c}, "kr_stock")
    r = engine.run(p, _monthly_ew(idx, list(c.columns)), engine.Costs(fee_pct=0.1), fill="next_close",
                   rebalance="on_signal")
    d = idx[15]
    assert r.forced_exits == 1
    assert r.turnover.loc[d] == pytest.approx(r.weights.shift().loc[d, "C"], rel=0.05)   # C만 팔림
    assert r.weights.loc[d, "C"] == 0 and r.weights.loc[d, ["A", "B"]].sum() < 0.8        # 남은 돈은 현금
    feb = r.turnover[r.turnover.index.month == 2]
    assert (feb > 1e-9).sum() == 1                                # 2월 신호에 다시 투자


def test_on_signal_dense_frame_trades_only_on_change():
    idx = pd.bdate_range("2024-01-01", periods=30)
    c = pd.DataFrame({"A": np.linspace(100, 130, 30), "B": 100.0}, idx)
    p = Panel({"open": c, "close": c}, "us_stock")
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)            # 매일 행 (매일 계산하는 전략)
    w.iloc[20:] = [0.7, 0.3]
    r = engine.run(p, w, NO_COST, fill="next_close", rebalance="on_signal")
    assert list(r.turnover[r.turnover > 1e-9].index) == [idx[1], idx[21]]


def test_on_signal_matches_independent_monthly_rebalance():
    rng = np.random.default_rng(3)
    n, k = 300, 4
    idx = pd.bdate_range("2023-01-02", periods=n)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, k)), 0)), idx, list("ABCD"))
    p = Panel({"open": c, "close": c}, "us_stock")
    m = (idx.to_series().dt.month.diff() != 0).to_numpy()
    tw = pd.DataFrame(rng.dirichlet(np.ones(k), m.sum()), idx[m], c.columns)
    tw.iloc[3] = tw.iloc[2]                                       # 같은 목표를 다시 보낸 달도 섞음
    res = engine.run(p, tw, engine.Costs(fee_pct=0.1), fill="next_close", rebalance="on_signal")
    W, C, fee = engine.align_weights(tw, p).to_numpy(), c.to_numpy(), 0.001
    V, sh, cash, vals = 1.0, np.zeros(k), 1.0, []
    for t in range(1, n):
        V = sh @ C[t] + cash
        if m[t - 1]:                                              # 신호 다음 봉 종가에 전 종목 맞춤
            turn = np.abs(W[t - 1] * V - sh * C[t]).sum() / V
            V *= 1 - turn * fee
            sh = W[t - 1] * V / C[t]
            cash = V - sh @ C[t]
        vals.append(V)
    assert np.allclose((1 + res.returns).cumprod().to_numpy(), vals, rtol=1e-12)


def test_walk_forward_keeps_monthly_signal_dates():
    p = _rand_panel(n=1300)

    class S:
        @staticmethod
        def generate_signals(data, params):
            return _monthly_ew(data.index, list(data.close.columns))

    out = engine.walk_forward(S, p, {}, engine.Costs(fee_pct=0.1), fill="next_close",
                              train="24M", test="6M", execution={"rebalance": "on_signal"})
    r = out["test"]
    months = r.returns.index.to_period("M").unique()
    traded = r.turnover[r.turnover > 1e-9].index.to_period("M").unique()
    assert len(traded) >= len(months) - 1                         # 시험 구간 매달 리밸런싱


# ── 워크포워드 이음매: 신호 날 기준 리밸런싱에서도 한 번에 돌린 결과와 같아야 함 ──
@pytest.mark.parametrize("rebalance", ["on_signal", "on_change", "always"])
def test_walk_forward_equals_single_run_for_sparse_rebalance(rebalance):
    p = _rand_panel(n=1300, k=4)

    class S:
        @staticmethod
        def generate_signals(data, params):          # 월초마다 20일 모멘텀 상위 2종목 동일가중
            c = data.close
            m = (c.index.to_series().dt.month.diff() != 0).to_numpy()
            top = c.pct_change(20).rank(axis=1, ascending=False) <= 2
            return (top.astype(float) / 2).loc[m]

    costs = engine.Costs(fee_pct=0.1, tax_pct_sell=0.2)
    out = engine.walk_forward(S, p, {}, costs, train="24M", test="6M", execution={"rebalance": rebalance})
    t = out["test"]
    r0, r1 = t.returns.index[0], t.returns.index[-1] + pd.Timedelta(days=1)
    one = engine.run(p, S.generate_signals(p, {}), costs, start=r0, end=r1, rebalance=rebalance)
    assert np.allclose(t.returns.to_numpy(), one.returns.to_numpy(), rtol=0, atol=1e-12)
    assert t.turnover.iloc[0] == pytest.approx(one.turnover.iloc[0])   # 첫 봉 가짜 리밸런싱 없음


def test_criterion2_fails_when_train_sharpe_not_positive():
    from autotrader.backtest.lab import holds_up
    assert holds_up(1.0, 1.5) and not holds_up(0.5, 1.5)
    assert not holds_up(-0.3, -1.0)          # 예전에는 -0.3 ≥ -0.5라 통과했음
    assert not holds_up(0.8, 0.0)
