"""KR 지배주주 순이익·EPS, US EDGAR 재무 필드 시점 테스트. 합성 데이터, 임시 저장소."""

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import data as bdata
from autotrader.data import store
from autotrader.data.collectors import dividends as dcol
from tests.test_fin_backtest import DAYS, _row, _write_fin

COMMON, PREF = "900010", "900015"
SPLIT = pd.Timestamp("2024-07-01")      # 2:1 액면분할
NY = "America/New_York"


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(dcol, "first_filed", lambda: pd.DataFrame())
    return tmp_path


# ── KR ──
def _write_kr_px():
    ts = DAYS.tz_localize("UTC")
    after = DAYS >= SPLIT
    rows = []
    for sym, sh in ((COMMON, 1000.0), (PREF, 200.0)):
        rows.append(pd.DataFrame({"ts": ts, "symbol": sym, "open": np.where(after, 500.0, 1000.0),
                                  "close": np.where(after, 500.0, 1000.0), "shares": np.where(after, 2 * sh, sh),
                                  "volume": 1000.0, "market": "KOSPI", "market_cap": 1e9}))
        store.write("kr_adj_factor", sym, pd.DataFrame({"ts": ts, "factor": np.where(after, 1.0, 0.5)}))
    store.write("krx_1d", "ALL", pd.concat(rows), key=["ts", "symbol"])


def _kr_rows():
    rows = [_row(y, p, sym=COMMON) for y in (2023, 2024) for p in ("Q1", "H1", "Q3", "FY")]
    for r in rows:   # 지배주주 몫 = 순이익의 80%
        r["net_income_controlling"], r["net_income_controlling_cum"] = 0.8 * r["net_income"], 0.8 * r["net_income_cum"]
        r["nic_source"] = "cfs"
    return rows


def _kr_load(**kw):
    return bdata.load("kr_stock", [COMMON, PREF], lockbox=True, neutralize_jumps=False, reinvest_dividends=False,
                      fields=("raw_close", "net_income_controlling_ttm", "net_income_ttm", "eps_ttm"), **kw)


def test_kr_controlling_ttm_and_eps(tmp_store):
    _write_kr_px()
    _write_fin(_kr_rows())
    p = _kr_load()
    nic, ni, eps = p.net_income_controlling_ttm[COMMON], p.net_income_ttm[COMMON], p.eps_ttm[COMMON]
    # 2024 Q1 공개(05-15 → 05-16): 순이익 TTM 10.1, 지배 80%
    assert ni.loc["2024-05-16"] == pytest.approx(10.1)
    assert nic.loc["2024-05-16"] == pytest.approx(8.08)
    assert np.isnan(eps.loc["2024-03-15"])
    # 분모 = 보통주 1000 + 우선주 200 (자사주 미차감)
    assert eps.loc["2024-05-16"] == pytest.approx(8.08 / 1200)
    # 2:1 분할 뒤에는 그날 주식 기준 (보고서는 그대로 → 수정계수로 환산): raw_close / eps가 이어진다
    assert eps.loc["2024-07-01"] == pytest.approx(8.08 / 2400)
    per = p.raw_close[COMMON] / eps
    assert per.loc["2024-06-28"] == pytest.approx(per.loc["2024-07-01"])
    # H1 공개(08-14 → 08-15) 뒤에는 그 시점 주식수(분할 후 2400)로
    assert eps.loc["2024-08-15"] == pytest.approx(0.8 * 10.2 / 2400)
    assert p.eps_ttm[PREF].isna().all()                       # 우선주는 EPS를 두지 않는다
    assert p.meta["fin_data"]["nic_source"] == {"cfs": 8} and p.meta["fin_data"]["nic_missing"] == 0
    assert "자사주 미차감" in p.meta["fin_data"]["eps"]
    assert bdata.check_fin_causality(p, n_cuts=6)["passed"]


def test_kr_controlling_missing_is_nan_not_net_income(tmp_store):
    _write_kr_px()
    rows = _kr_rows()
    rows[4]["net_income_controlling"] = np.nan            # 2024 Q1 지배주주 순이익 아직 없음 (백필 전)
    _write_fin(rows)
    p = _kr_load()
    assert np.isnan(p.net_income_controlling_ttm[COMMON].loc["2024-05-16"])
    assert np.isnan(p.eps_ttm[COMMON].loc["2024-05-16"])
    assert p.net_income_ttm[COMMON].loc["2024-05-16"] == pytest.approx(10.1)
    assert p.meta["fin_data"]["nic_missing"] == 1


# ── US ──
def _ny(d):
    return pd.Timestamp(d).tz_localize(NY).tz_convert("UTC") if d else pd.NaT


def _fact(field, start, end, val, filed, form="10-Q", rank=0):
    return {"ts": _ny(end), "symbol": "AAA", "cik": 1, "field": field, "tag": f"T{rank}", "tag_rank": rank,
            "unit": "shares" if field == "shares_outstanding" else "USD", "start": _ny(start), "end": _ny(end),
            "val": float(val), "accn": f"{filed}-{form}", "fy": 2023, "fp": "Q", "form": form, "filed": _ny(filed),
            "frame": None}


def _write_us():
    ts = pd.bdate_range("2024-01-02", "2024-07-31").tz_localize("UTC") + pd.Timedelta(hours=14, minutes=30)
    store.write("us_1d", "AAA", pd.DataFrame({"ts": ts, "open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0,
                                              "volume": 1.0, "adj_close": 50.0}))
    f = [
        _fact("revenue", "2023-01-01", "2023-03-31", 10, "2023-05-01"),
        _fact("revenue", "2023-04-01", "2023-06-30", 20, "2023-08-01"),
        _fact("revenue", "2023-01-01", "2023-06-30", 30, "2023-08-01"),          # 6개월 누계 (무시)
        _fact("revenue", "2023-07-01", "2023-09-30", 30, "2023-11-01"),
        _fact("revenue", "2023-01-01", "2023-09-30", 60, "2023-11-01"),          # 9개월 누계
        _fact("revenue", "2023-01-01", "2023-12-31", 100, "2024-02-15", "10-K"),  # 4분기 = 100 − 60
        _fact("revenue", "2024-01-01", "2024-03-31", 11, "2024-05-01"),
        _fact("revenue", "2024-01-01", "2024-03-31", 999, "2024-05-01", rank=1),  # 같은 공시 후순위 태그 (무시)
        _fact("revenue", "2023-01-01", "2023-12-31", 104, "2024-06-03", "10-K/A"),  # 정정: 4분기 44
        _fact("equity", None, "2023-12-31", 500, "2024-02-15", "10-K"),
        _fact("equity", None, "2024-03-31", 520, "2024-05-01"),
        _fact("equity", None, "2023-12-31", 1, "2024-06-03", "10-K/A"),          # 옛 기간 정정은 새 기간을 덮지 않음
        _fact("net_income", "2023-01-01", "2023-12-31", 7, "2024-02-15", "10-K"),
    ]
    store.write("edgar_facts", "ALL", pd.DataFrame(f), key=["symbol", "tag", "unit", "start", "end", "accn", "field"])


def _us_load(timing):
    return bdata.load("us_stock", ["AAA"], lockbox=True, fin_timing=timing,
                      fields=("equity", "revenue_ttm", "net_income_controlling_ttm"))


def test_us_first_vs_last_and_q4(tmp_store):
    _write_us()
    first, last = _us_load("first"), _us_load("last")
    for p in (first, last):
        rev, eq = p.revenue_ttm["AAA"], p.equity["AAA"]
        assert np.isnan(rev.loc["2024-02-15"])                     # 공시 당일은 안 보임
        assert rev.loc["2024-02-16"] == pytest.approx(100.0)       # 10+20+30+(100−60)
        assert rev.loc["2024-05-01"] == pytest.approx(100.0)
        assert rev.loc["2024-05-02"] == pytest.approx(101.0)       # 후순위 태그 999는 무시
        assert eq.loc["2024-02-16"] == 500 and eq.loc["2024-05-02"] == 520
        assert eq.loc["2024-06-04"] == 520                         # 옛 기간 정정은 최신 기간을 덮지 않음
    assert first.revenue_ttm["AAA"].loc["2024-06-04"] == pytest.approx(101.0)   # 최초 공시값 유지
    assert last.revenue_ttm["AAA"].loc["2024-06-03"] == pytest.approx(101.0)
    assert last.revenue_ttm["AAA"].loc["2024-06-04"] == pytest.approx(105.0)    # 10-K/A 다음 거래일부터 44
    assert np.isnan(first.net_income_controlling_ttm["AAA"].loc["2024-07-31"])   # 분기 값이 없으면 NaN
    d = first.meta["fin_data"]
    assert d["amendments"] == 2 and d["symbols"] == 1 and "생존편향" in d["note"]
    for p in (first, last):
        assert bdata.check_fin_causality(p, n_cuts=8)["passed"]


def test_us_causality_catches_leak(tmp_store):
    _write_us()
    p = _us_load("last")
    p.fin_reports = p.fin_reports.assign(filed_d=p.fin_reports["filed_d"] + pd.Timedelta(days=1))
    assert not bdata.check_fin_causality(p, n_cuts=12)["passed"]


def test_us_empty_and_bad_fields(tmp_store):
    ts = pd.bdate_range("2024-01-02", periods=5).tz_localize("UTC")
    store.write("us_1d", "AAA", pd.DataFrame({"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                                              "volume": 1.0, "adj_close": 1.0}))
    p = bdata.load("us_stock", ["AAA"], lockbox=True, fields=("equity",))   # edgar_facts 아직 없음
    assert p.equity["AAA"].isna().all() and p.meta["fin_data"]["rows"] == 0
    with pytest.raises(ValueError, match="us_stock"):
        bdata.load("us_stock", ["AAA"], lockbox=True, fields=("total_assets",))
    with pytest.raises(ValueError):
        bdata.load("crypto_spot", ["BTCUSDT"], lockbox=True, fields=("equity",))


def test_report_lines_kr_and_us(tmp_store):
    from autotrader.backtest import lab
    _write_us()
    us = lab._fin_lines(_us_load("first").meta)
    assert "edgar_facts" in us[1] and "정정 공시" in us[1] and "생존편향" in us[-1]
    _write_kr_px()
    _write_fin(_kr_rows())
    kr = lab._fin_lines(_kr_load().meta)
    assert "지배주주 순이익 없는 분기 0개" in kr[1] and "자사주 미차감" in kr[2]
