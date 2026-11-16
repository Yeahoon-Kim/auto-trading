"""코인 선물 펀딩비 반영 테스트 (합성 데이터, 임시 저장소)."""

import copy

import pandas as pd
import pytest

from autotrader.backtest import data as bdata
from autotrader.backtest import engine, lab, trade
from autotrader.backtest.data import Panel
from autotrader.data import store
from autotrader.risk import guards

NO_COST = engine.Costs()


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    return tmp_path


def _flat(n=4, funding=None) -> Panel:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    px = pd.DataFrame({"A": 100.0}, index=idx)
    f = {"open": px, "close": px.copy(), "high": px.copy(), "low": px.copy()}
    if funding is not None:
        f["funding"] = pd.DataFrame({"A": funding}, index=idx, dtype=float)
    return Panel(f, "crypto_futures")


def test_long_pays_short_receives_next_open():
    rate = [0.0, 0.001, 0.002, -0.001]
    p = _flat(funding=rate)
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])          # 봉 0 종가 결정 → 봉 1 시가부터 보유
    long = engine.run(p, w, NO_COST)
    assert long.returns.to_numpy() == pytest.approx([-0.001, -0.002, 0.001])   # 체결 봉(1)의 정산부터
    assert long.funding.to_numpy() == pytest.approx([-0.001, -0.002, 0.001])
    short = engine.run(p, -w, NO_COST)
    assert short.returns.to_numpy() == pytest.approx([0.001, 0.002, -0.001])
    m = long.metrics(365)
    assert m["funding_total"] == pytest.approx(0.999 * 0.998 * 1.001 - 1)
    assert m["gross_total_return"] == pytest.approx(m["total_return"])         # 펀딩은 비용이 아니라 수익에 포함


def test_next_close_fill_skips_funding_of_fill_bar():
    p = _flat(funding=[0.0, 0.001, 0.002, 0.0])
    w = pd.DataFrame({"A": [1.0]}, index=p.index[:1])          # 봉 1 종가에 체결 → 봉 1 정산은 안 냄
    r = engine.run(p, w, NO_COST, fill="next_close")
    assert r.returns.to_numpy() == pytest.approx([0.0, -0.002, 0.0])


def test_no_funding_field_means_no_funding():
    p = _flat()
    r = engine.run(p, pd.DataFrame({"A": [1.0]}, index=p.index[:1]), NO_COST)
    assert r.funding is None and "funding_per_year" not in r.metrics(365)


def test_trade_replay_charges_funding_in_cash():
    p = _flat(n=4, funding=[0.0, 0.0, 0.01, 0.0])
    risk = copy.deepcopy(guards.load_config())
    risk["orders"]["max_order_value_krw"] = 1e12
    acct = trade.Account(capital=1_000_000, currency="USDT", krw_per_unit=1.0, lot=1e-3)
    sig = {"ts": p.index[0], "symbol": "A", "side": "long", "stop": 95.0}
    res = trade.run(p, [sig], NO_COST, acct, trade_cfg={"max_position_pct": 100}, risk_cfg=risk)
    pos = res.weights["A"].iloc[0]                              # 봉 1 끝 비중 (결과 0번째 행 = 봉 1)
    assert pos > 0
    assert res.funding.iloc[1] == pytest.approx(-pos * 0.01, rel=1e-9)
    assert res.returns.iloc[1] == pytest.approx(res.funding.iloc[1])
    res_s = trade.run(p, [{**sig, "side": "short", "stop": 105.0}], NO_COST, acct,
                      trade_cfg={"max_position_pct": 100}, risk_cfg=risk)
    assert res_s.funding.iloc[1] > 0                            # 숏은 받는다


def _write_1m(sym, start, minutes):
    ts = pd.date_range(start, periods=minutes, freq="min", tz="UTC")
    store.write("binance_um_1m", sym, pd.DataFrame({"ts": ts, "open": 100.0, "high": 100.0, "low": 100.0,
                                                    "close": 100.0, "volume": 1.0}))


def test_loader_buckets_funding_per_bar(tmp_store):
    _write_1m("BTCUSDT", "2024-01-01", 3 * 24 * 60)             # 3일치 (마지막 날은 다음 날 00:00 전까지)
    _write_1m("NEWUSDT", "2024-01-01", 3 * 24 * 60)             # 펀딩 데이터 없는 종목
    ev = pd.to_datetime(["2024-01-01 00:00:00.001", "2024-01-01 08:00", "2024-01-01 16:00",
                         "2024-01-02 00:00", "2024-01-02 08:00:00.001", "2024-01-03 15:59:59.999"], utc=True, format="mixed")
    store.write("binance_um_funding", "BTCUSDT",
                pd.DataFrame({"ts": ev, "funding_rate": [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4], "mark_price": 100.0}))
    p = bdata.load("crypto_futures", ["BTCUSDT", "NEWUSDT"], timeframe="1d", lockbox=True)
    f = p.funding["BTCUSDT"]
    assert f.to_numpy() == pytest.approx([6e-4, 9e-4, 6e-4])    # 15:59:59.999 → 15:59 → 1/3 봉
    assert (p.funding["NEWUSDT"] == 0).all()
    meta = p.meta["funding"]
    assert meta["events"] == 6 and meta["symbols_without_funding"] == ["NEWUSDT"]
    h8 = bdata.load("crypto_futures", ["BTCUSDT"], timeframe="8h", lockbox=True).funding["BTCUSDT"]
    assert h8.loc["2024-01-01 08:00"] == pytest.approx(2e-4)     # 08:00 정산은 08:00에 시작하는 봉
    assert h8.loc["2024-01-01 00:00"] == pytest.approx(1e-4)
    off = bdata.load("crypto_futures", ["BTCUSDT"], timeframe="1d", lockbox=True, funding=False)
    assert "funding" not in off and off.meta["note"] == "선물 펀딩비 미반영"


def test_report_lines():
    on = {"funding": {"source": "binance_um_funding (봉 안 정산 합)", "events": 6, "range": "a~b",
                      "symbols_without_funding": ["NEWUSDT"]}}
    line = lab._funding_lines(on)[0]
    assert "반영: 예" in line and "NEWUSDT" in line
    assert "아니오" in lab._funding_lines({"note": "선물 펀딩비 미반영"})[0]
    assert lab._funding_lines({}) == []
