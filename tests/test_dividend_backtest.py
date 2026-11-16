"""KR 배당 재투자(총수익)와 배당 신호 필드(dps, div_yield) 시점 테스트. 합성 데이터, 임시 저장소."""

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import data as bdata
from autotrader.data import store

KST = "Asia/Seoul"
DAYS = pd.bdate_range("2024-03-07", periods=10)    # 03-07 … 03-20, 5번째(위치 4)가 03-13


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    return tmp_path


def _kst(d) -> pd.Timestamp:
    return pd.Timestamp(d).tz_localize(KST).tz_convert("UTC")


def _write_kr(close, factor=None, sym="900001"):
    ts = DAYS.tz_localize("UTC")
    store.write("krx_1d", "ALL", pd.DataFrame({"ts": ts, "symbol": sym, "open": close, "close": close,
                                               "shares": 1e6, "volume": 1000.0, "market": "KOSPI"}),
                key=["ts", "symbol"])
    store.write("kr_adj_factor", sym, pd.DataFrame({"ts": ts, "factor": factor if factor is not None else 1.0}))


def _write_schedule(ex, dps, sym="900001"):
    store.write("kr_dividend_schedule", "ALL",
                pd.DataFrame({"ts": [_kst(pd.Timestamp(ex) + pd.Timedelta(days=1))], "symbol": [sym],
                              "kind": ["결산"], "stock_kind": ["보통"], "dps": [dps],
                              "ex_date": [pd.Timestamp(ex)]}), key=["ts", "symbol"])


def _kr(**kw):
    return bdata.load("kr_stock", ["900001"], lockbox=True, neutralize_jumps=False, **kw)


def test_kr_ex_date_return_includes_net_dividend(tmp_store):
    close = [1000.0] * 3 + [900.0] * 7            # 03-12 배당락: 100원 배당만큼 하락
    _write_kr(close)
    _write_schedule(DAYS[3], 100)
    r = lambda p: p.close["900001"].pct_change().iloc[3]
    taxed, gross, off = _kr(fields=("raw_close",)), _kr(dividend_tax=0), _kr(reinvest_dividends=False)
    assert r(off) == pytest.approx(-0.10)                         # 예전: 배당락 하락만 손실
    assert r(gross) == pytest.approx(0.0)                         # (900 + 100) / 1000 - 1
    assert r(taxed) == pytest.approx((900 + 100 * (1 - 0.154)) / 1000 - 1)   # 배당소득세 15.4%
    assert taxed.close["900001"].iloc[-1] == 900.0                # 최근 가격은 실제 가격 그대로
    assert taxed.raw_close["900001"].iloc[0] == 1000.0
    d = taxed.meta["dividends"]
    assert d["reinvested"] and d["events"] == 1 and d["tax"] == pytest.approx(0.154)
    assert not off.meta["dividends"]["reinvested"]


def test_kr_dividend_uses_raw_basis_across_split(tmp_store):
    # 배당락 다음 날 2:1 분할 (수정계수 0.5 → 1.0). 배당 수익은 원본 기준으로 계산돼 분할과 무관해야 한다.
    close = [1000.0] * 3 + [900.0] * 2 + [450.0] * 5
    _write_kr(close, factor=[0.5] * 5 + [1.0] * 5)
    _write_schedule(DAYS[3], 100)
    p = _kr(dividend_tax=0)
    ret = p.close["900001"].pct_change()
    assert ret.iloc[3] == pytest.approx(0.0) and ret.iloc[5] == pytest.approx(0.0)


def _write_dart(first_filed=DAYS[4], filed="2025-01-10", dps=500.0):
    store.write("dart_dividend", "ALL", pd.DataFrame({
        "ts": [_kst("2023-12-31")], "corp_code": ["00999999"], "symbol": ["900001"], "bsns_year": [2023],
        "reprt_code": ["11011"], "rcept_no": ["20250110000001"], "filed": [_kst(filed)], "has_data": [True],
        "dps_common": [dps]}), key=["corp_code", "bsns_year", "reprt_code"])
    if first_filed is not None:
        store.write("dart_periodic_filings", "ALL", pd.DataFrame({
            "ts": [_kst(first_filed)], "rcept_no": ["20240313000001"], "corp_code": ["00999999"],
            "symbol": ["900001"], "kind": ["사업보고서"], "period": ["2023.12"], "report_nm": ["사업보고서 (2023.12)"],
            "correction": [False]}), key=["rcept_no"])


def test_kr_dps_visible_only_from_day_after_first_filing(tmp_store):
    # 최초 접수 03-13, 정정(filed) 2025-01-10 → 03-14부터 보인다. 03-20 2:1 분할 → 그날부터 주당 250원.
    close = [10000.0] * 9 + [5000.0]
    _write_kr(close, factor=[0.5] * 9 + [1.0])
    _write_dart()
    p = _kr(fields=("dps", "div_yield"))
    dps = p.dps["900001"]
    assert dps.loc[:"2024-03-13"].isna().all()                     # 최초 접수일 당일까지는 안 보임
    assert dps.loc["2024-03-14"] == 500.0                          # 다음 날부터 (정정일 2025-01-10을 기다리지 않음)
    assert dps.loc["2024-03-20"] == pytest.approx(250.0)           # 분할 뒤 주식 기준
    assert p.div_yield["900001"].loc["2024-03-14"] == pytest.approx(0.05)
    assert p.div_yield["900001"].loc["2024-03-20"] == pytest.approx(0.05)
    assert bdata.check_dividend_causality(p)["passed"]


def test_kr_preferred_share_gets_dps_pref(tmp_store):
    # 우선주 900005(보통주 900000)는 같은 사업보고서의 dps_pref, 같은 시점 규칙
    _write_kr([10000.0] * 10, sym="900000")
    _write_kr([8000.0] * 10, sym="900005")
    store.write("kr_dividend_schedule", "ALL", pd.DataFrame({
        "ts": [_kst("2023-12-31")], "symbol": ["900005"], "kind": ["결산"], "stock_kind": ["우선"], "dps": [505],
        "ex_date": [pd.Timestamp("2023-12-27")]}), key=["ts", "symbol"])
    _write_dart()                                                  # 최초 접수일 기록 (회사 00999999)
    store.write("dart_dividend", "ALL", pd.DataFrame({
        "ts": [_kst("2023-12-31")], "corp_code": ["00999999"], "symbol": ["900000"], "bsns_year": [2023],
        "reprt_code": ["11011"], "filed": [_kst("2025-01-10")], "has_data": [True],
        "dps_common": [500.0], "dps_pref": [505.0]}), key=["corp_code", "bsns_year", "reprt_code"])
    p = bdata.load("kr_stock", ["900000", "900005"], lockbox=True, fields=("dps", "div_yield"))
    assert p.dps["900005"].loc[:"2024-03-13"].isna().all()
    assert p.dps["900005"].loc["2024-03-14"] == 505.0 and p.dps["900000"].loc["2024-03-14"] == 500.0
    assert p.div_yield["900005"].loc["2024-03-14"] == pytest.approx(505 / 8000)


def test_kr_dps_parse_gap_is_unknown_not_zero(tmp_store):
    # 배당성향은 있는데 주당배당금이 빈 행(파싱 누락) → 0이 아니라 NaN. 배당 흔적이 없으면 0(무배당).
    _write_kr([10000.0] * 10)
    _write_dart()

    def dart(payout):
        store.write("dart_dividend", "ALL", pd.DataFrame({
            "ts": [_kst("2023-12-31")], "corp_code": ["00999999"], "symbol": ["900001"], "bsns_year": [2023],
            "reprt_code": ["11011"], "filed": [_kst("2025-01-10")], "has_data": [True],
            "dps_common": [np.nan], "payout_ratio": [payout]}), key=["corp_code", "bsns_year", "reprt_code"])

    dart(14.3)
    assert _kr(fields=("dps",)).dps["900001"].isna().all()
    store.write("dart_dividend", "ALL", pd.DataFrame({                 # 우선주에만 배당 → 보통주는 0
        "corp_code": ["00999999"], "bsns_year": [2023], "reprt_code": ["11011"], "ts": [_kst("2023-12-31")],
        "symbol": ["900001"], "filed": [_kst("2025-01-10")], "has_data": [True], "dps_common": [np.nan],
        "payout_ratio": [14.3], "dps_pref": [100.0]}), key=["corp_code", "bsns_year", "reprt_code"])
    assert _kr(fields=("dps",)).dps["900001"].loc["2024-03-14"] == 0.0
    dart(np.nan)
    assert _kr(fields=("dps",)).dps["900001"].loc["2024-03-14"] == 0.0


def test_kr_dps_falls_back_to_filed_when_first_filing_unknown(tmp_store):
    _write_kr([10000.0] * 10)
    _write_dart(first_filed=None, filed=DAYS[6])                   # 최초 접수일 없음 → filed(더 늦음) 다음 날
    dps = _kr(fields=("dps",)).dps["900001"]
    assert dps.loc[:DAYS[6]].isna().all() and dps.loc[DAYS[7]] == 500.0


def test_dividend_causality_check_catches_early_timing(tmp_store, monkeypatch):
    _write_kr([10000.0] * 10)
    _write_dart()
    real = bdata.dividends

    def leaky(market, symbols=None, asof=None):                    # 접수일 당일부터 보이게 하는 잘못된 규칙
        ev = real(market, symbols, asof)
        return ev.assign(known=ev["src_date"])

    monkeypatch.setattr(bdata, "dividends", leaky)
    p = _kr(fields=("dps",))
    assert p.dps["900001"].loc["2024-03-13"] == 500.0
    chk = bdata.check_dividend_causality(p)
    assert not chk["passed"] and chk["cuts"][0]["mismatch_bars"] > 0


def test_us_dps_trailing_from_ex_date(tmp_store):
    ts = pd.date_range("2024-01-02 14:30", periods=5, freq="1D", tz="UTC")
    store.write("us_1d", "SPY", pd.DataFrame({
        "ts": ts, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0,
        "adj_close": [100 / 1.01, 100 / 1.01, 100.0, 100.0, 100.0]}))
    store.write("us_dividends", "ALL", pd.DataFrame({
        "ts": [pd.Timestamp("2024-01-04", tz="America/New_York").tz_convert("UTC")] * 2, "id": ["a", "b"],
        "symbol": ["SPY", "SPY"], "kind": ["cash_dividend", "cash_dividend"], "rate": [4.0, 9.0],
        "special": [False, True]}), key=["id"])
    p = bdata.load("us_stock", ["SPY"], lockbox=True, dividend_tax=0, fields=("dps", "div_yield"))
    dps = p.dps["SPY"]
    assert dps.iloc[1] == 0.0                                       # 배당락 전날까지 없음
    assert dps.iloc[2] == pytest.approx(1.0) and dps.iloc[-1] == pytest.approx(1.0)   # close 기준 금액 (특별배당 제외)
    assert p.div_yield["SPY"].iloc[2] == pytest.approx(0.01)
    assert bdata.check_dividend_causality(p)["passed"]


def test_report_states_kr_dividend_reinvestment():
    from autotrader.backtest.lab import _dividend_line
    on = {"market": "kr_stock", "data": {"dividends": {"reinvested": True, "tax": 0.154, "events": 3, "symbols": 1}}}
    off = {"market": "kr_stock", "data": {"dividends": {"reinvested": False}}}
    assert "KR 배당 재투자 반영: 예" in _dividend_line(on) and "15.4%" in _dividend_line(on)
    assert "**아니오**" in _dividend_line(off) and "**아니오**" in _dividend_line({"market": "kr_stock", "data": {}})


def test_report_states_dividend_data_range(tmp_store):
    from autotrader.backtest.lab import report
    _write_kr([10000.0] * 10)
    _write_dart(first_filed=None)
    p = _kr(fields=("dps",))
    cov = p.meta["dividend_data"]
    assert cov["years"] == {2023: 1} and cov["filed_fallback"] == 1
    s = {"lab": "L", "run_at": "-", "market": "kr_stock", "data": p.meta, "checks": [],
         "test": {}, "costs": {"fee_pct": 0, "slippage_pct": 0, "tax_pct_sell": 0}}
    assert "실행 시점 배당 데이터 범위: 사업연도별 행 수 2023 1" in report(s)
