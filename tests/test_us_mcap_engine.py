"""백테스트 엔진 US 시가총액 필드(market_cap·shares·cik) 테스트. 합성 데이터, 임시 저장소.
분할 전후 연속성, 다중 클래스(cik), 결측, 시점(공시 다음 거래일), 미래참조 검사, report 문구, 템플릿 top_by_mcap."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import data as bdata
from autotrader.backtest import lab
from autotrader.data import store

NY = "America/New_York"
DEI, BS = "EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"
DAYS = pd.bdate_range("2020-04-01", "2021-03-31")
SPLIT = "2020-08-31"   # AAA 4:1 분할


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    return tmp_path


def _ny(d: str) -> pd.Timestamp:
    return pd.Timestamp(d).tz_localize(NY).tz_convert("UTC")


def _bars(sym: str, close) -> None:
    ts = (DAYS.tz_localize(NY) + pd.Timedelta(hours=16)).tz_convert("UTC")
    c = np.broadcast_to(np.asarray(close, dtype=float), (len(DAYS),))
    store.write("us_1d", sym, pd.DataFrame({"ts": ts, "open": c, "high": c, "low": c, "close": c,
                                            "volume": 1e6, "adj_close": c}))


def _fact(sym, cik, tag, end, filed, val):
    return {"ts": _ny(end), "symbol": sym, "cik": cik, "field": "shares_outstanding" if tag == BS else "shares_dei",
            "tag": tag, "tag_rank": 0, "unit": "shares", "start": pd.NaT, "end": _ny(end), "val": float(val),
            "accn": f"{sym}-{filed}", "fy": 2020, "fp": "Q", "form": "10-Q", "filed": _ny(filed), "frame": None}


def _write(splits: bool = True):
    """AAA: 2020-08-31 4:1 분할. 저장 종가는 오늘 기준(분할 소급) 100 고정 → 실제 시총은 분할 전후 4만×1만 그대로.
    GOOG·GOOGL: 같은 회사(cik 9), 회사 전체 주식수. CCC: 주식수 공시 없음."""
    _bars("AAA", 100.0)
    _bars("GOOG", 50.0)
    _bars("GOOGL", np.where(DAYS < "2020-10-01", 50.5, 51.0))
    _bars("CCC", 10.0)
    f = [
        _fact("AAA", 1, DEI, "2020-04-20", "2020-05-01", 1e8),       # 분할 전 공시 → ×4
        _fact("AAA", 1, DEI, "2020-07-20", "2020-07-31", 1e8),       # 금요일 공시 → 월요일 08-03부터
        _fact("AAA", 1, DEI, "2020-10-20", "2020-10-30", 4e8),       # 분할 뒤 → 그대로
        _fact("AAA", 1, DEI, "2021-01-20", "2021-02-01", 4.2e8),
    ]
    for s in ("GOOG", "GOOGL"):
        f += [_fact(s, 9, DEI, "2020-04-20", "2020-05-01", 6.8e8), _fact(s, 9, DEI, "2020-10-20", "2020-10-30", 6.7e8)]
    store.write("edgar_facts", "ALL", pd.DataFrame(f), key=["symbol", "tag", "unit", "start", "end", "accn", "field"])
    if splits:
        store.write("us_splits", "ALL", pd.DataFrame({"ts": [_ny(SPLIT)], "symbol": ["AAA"], "ratio": [4.0]}),
                    key=["ts", "symbol"])


def _load(fields=("market_cap", "shares", "cik"), **kw):
    return bdata.load("us_stock", ["AAA", "GOOG", "GOOGL", "CCC"], lockbox=True, fields=fields, **kw)


def test_split_continuity_and_timing(tmp_store):
    _write()
    p = _load()
    mc, sh = p.market_cap["AAA"], p.shares["AAA"]
    assert np.isnan(mc.loc["2020-04-30"]) and np.isnan(mc.loc["2020-05-01"])   # 공시 당일까지 안 보임
    assert mc.loc["2020-05-04"] == pytest.approx(100 * 4e8)                    # 다음 거래일부터, 분할 보정 ×4
    assert mc.loc["2020-07-31"] == pytest.approx(4e10) and sh.loc["2020-08-03"] == pytest.approx(4e8)
    # 분할 전후 시총 연속 (저장 종가가 분할 소급이라 주식수도 같은 기준)
    assert mc.loc["2020-08-28"] == pytest.approx(mc.loc["2020-08-31"])
    assert mc.loc["2021-02-01"] == pytest.approx(4e10) and mc.loc["2021-02-02"] == pytest.approx(4.2e10)
    assert p.cik["AAA"].dropna().eq(1).all()
    m = p.meta["mcap_data"]
    assert m["split_rows"] == 1 and "warning" not in m and m["events"] == 8
    assert 0 < m["missing"] < 0.5 and m["symbols_without_shares"] == ["CCC"]


def test_multi_class_same_cik_kept(tmp_store):
    _write()
    p = _load()
    d = "2020-11-02"
    assert p.cik.loc[d, "GOOG"] == p.cik.loc[d, "GOOGL"] == 9                  # 같은 회사 표시
    assert p.market_cap.loc[d, "GOOG"] == pytest.approx(50 * 6.7e8)            # 둘 다 회사 전체 시총 (엔진은 안 뺌)
    assert p.market_cap.loc[d, "GOOGL"] == pytest.approx(51 * 6.7e8)
    assert p.meta["mcap_data"]["multi_class"] == {"9": ["GOOG", "GOOGL"]}


def test_missing_and_no_data(tmp_store):
    _write()
    p = _load()
    assert p.market_cap["CCC"].isna().all() and p.cik["CCC"].isna().all()   # 공시 없는 종목은 NaN (0 아님)
    assert p.market_cap["AAA"].loc["2020-04-01":"2020-05-01"].isna().all()   # 첫 공시 전 결측
    # 주식수 원천 자체가 없으면 전부 NaN, report에 없음 표시
    _bars("ZZZ", 1.0)
    z = bdata.load("us_stock", ["ZZZ"], lockbox=True, fields=("market_cap",))
    assert z.market_cap["ZZZ"].isna().all() and z.meta["mcap_data"]["events"] == 0
    # kr_stock market_cap(krx_1d)은 이 경로를 타지 않는다
    assert "mcap_fields" not in bdata.load("us_stock", ["AAA"], lockbox=True).meta


def test_no_split_history_warns(tmp_store):
    _write(splits=False)
    p = _load()
    assert "us_splits 없음" in p.meta["mcap_data"]["warning"]
    assert p.market_cap["AAA"].loc["2020-08-03"] == pytest.approx(1e10)   # 분할 보정 못 해 1/4로 작게


def test_causality_check_passes_and_catches_leak(tmp_store):
    _write()
    p = _load()
    r = bdata.check_mcap_causality(p, n_cuts=12)
    assert r["passed"] and r["fields"] == ["market_cap", "shares", "cik"]
    # 공시 당일부터 보이게 새면(값을 하루 앞당김) 잡아야 한다
    p["market_cap"] = p["market_cap"].shift(-1)
    assert not bdata.check_mcap_causality(p, n_cuts=12)["passed"]
    # 시가총액 필드가 없으면 검사 대상 없음 → 통과
    assert bdata.check_mcap_causality(bdata.load("us_stock", ["AAA"], lockbox=True))["fields"] == []


def test_report_lines(tmp_store):
    _write()
    lines = lab._mcap_lines(_load().meta)
    text = "\n".join(lines)
    assert "공시일 다음 거래일" in lines[0] and "결측률" in text
    assert "GOOG·GOOGL" in text and "cik로 묶어" in text
    assert "V:" in text and "생존편향" in text and "BE" in text and "CCC" in text
    assert lab._mcap_lines({}) == []


def test_template_top_by_mcap_dedups_cik(tmp_store):
    _write()
    spec = importlib.util.spec_from_file_location(
        "tmpl_strategy", Path(__file__).resolve().parents[1] / "labs" / "_template" / "strategy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = _load()
    top = mod.top_by_mcap(p, n=2)
    d = "2020-11-02"
    assert top.loc[d].sum() == 2 and top.loc[d, "AAA"] and top.loc[d, "GOOG"] and not top.loc[d, "GOOGL"]
    pref = p.close.copy()      # 가격이 높은 클래스를 남기는 방침 예
    top2 = mod.top_by_mcap(p, n=2, prefer=pref)
    assert top2.loc[d, "GOOGL"] and not top2.loc[d, "GOOG"]
    assert not top.loc["2020-04-01"].any()   # 시총이 아직 없으면 아무것도 안 뽑음


def test_same_day_filings_pick_latest_asof(tmp_store):
    """같은 날 10-Q와 옛 기준일 8-K가 같이 공시되면 기준일이 최근인 값 하나 (입력 크기와 무관하게 같은 결과, KO 2016)."""
    _write()
    old = dict(_fact("AAA", 1, DEI, "2020-10-20", "2021-02-01", 4.1e8), accn="AAA-2021-02-01-8K", form="8-K")
    store.write("edgar_facts", "ALL", pd.DataFrame([old]), key=["symbol", "tag", "unit", "start", "end", "accn", "field"])
    p = _load()
    assert p.market_cap["AAA"].loc["2021-02-02"] == pytest.approx(4.2e10)
    assert bdata.check_mcap_causality(p, n_cuts=12)["passed"]
