"""KR 재무 필드(dart_fin) 시점·TTM 테스트. 합성 데이터, 임시 저장소."""

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import data as bdata
from autotrader.data import store
from autotrader.data.collectors import dividends as dcol

KST = "Asia/Seoul"
DAYS = pd.bdate_range("2023-01-02", "2025-06-30")
SYM = "900001"
REPRT = {"Q1": "11013", "H1": "11012", "Q3": "11014", "FY": "11011"}
QEND = {"Q1": "03-31", "H1": "06-30", "Q3": "09-30", "FY": "12-31"}
FILED = {"Q1": "05-15", "H1": "08-14", "Q3": "11-14", "FY": "03-15"}   # FY는 다음 해
# 분기 3개월 매출: 2023 = 10, 20, 30, 40 (연 100), 2024 = 11, 21, 31, 41 (연 104)
QREV = {2023: [10.0, 20.0, 30.0, 40.0], 2024: [11.0, 21.0, 31.0, 41.0]}


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(dcol, "first_filed", lambda: pd.DataFrame())   # 기본: 최초 접수일 모름 → filed
    return tmp_path


def _kst(d) -> pd.Timestamp:
    return pd.Timestamp(d).tz_localize(KST).tz_convert("UTC")


def _write_px(sym=SYM, days=DAYS):
    ts = days.tz_localize("UTC")
    store.write("krx_1d", "ALL", pd.DataFrame({"ts": ts, "symbol": sym, "open": 1000.0, "close": 1000.0,
                                               "shares": 1e6, "volume": 1000.0, "market": "KOSPI",
                                               "market_cap": 1e9}), key=["ts", "symbol"])
    store.write("kr_adj_factor", sym, pd.DataFrame({"ts": ts, "factor": 1.0}))


def _row(year, period, sym=SYM, filed=None, fs_div="CFS", **over):
    q = ["Q1", "H1", "Q3", "FY"].index(period)
    rev = QREV[year]
    v3 = sum(rev) if period == "FY" else rev[q]
    cum = sum(rev[:q + 1])
    fy_next = period == "FY"
    filed = filed or f"{year + fy_next}-{FILED[period]}"
    r = {"corp_code": "C" + sym, "symbol": sym, "rcept_no": pd.Timestamp(filed).strftime("%Y%m%d") + "000001",
         "fs_div": fs_div, "currency": "KRW", "ts": _kst(f"{year}-{QEND[period]}"), "filed": _kst(filed),
         "bsns_year": year, "reprt_code": REPRT[period], "period": period,
         "revenue": v3, "revenue_cum": cum, "net_income": v3 / 10, "net_income_cum": cum / 10,
         "total_equity": 1000.0 + year * 10 + q, "total_liabilities": 500.0 + q}
    return r | over


def _write_fin(rows):
    store.write("dart_fin", "ALL", pd.DataFrame(rows), key=["symbol", "bsns_year", "reprt_code"])


def _all_rows(**over):
    return [_row(y, p, **over) for y in (2023, 2024) for p in ("Q1", "H1", "Q3", "FY")]


def _load(fields=("equity", "revenue_ttm", "net_income_ttm"), **kw):
    return bdata.load("kr_stock", [SYM], lockbox=True, neutralize_jumps=False, reinvest_dividends=False,
                      fields=fields, **kw)


def test_visible_only_from_next_trading_day(tmp_store):
    _write_px()
    _write_fin(_all_rows())
    p = _load()
    eq = p.equity[SYM]
    # 2023 Q1 접수 2023-05-15(월) → 05-16(화)부터
    assert np.isnan(eq.loc["2023-05-15"])
    assert eq.loc["2023-05-16"] == 1000.0 + 20230
    # 2024 FY 접수 2025-03-15(토) → 다음 거래일 03-17(월)부터. 금요일 03-14엔 3분기 값
    assert eq.loc["2025-03-14"] == 1000.0 + 20240 + 2
    assert eq.loc["2025-03-17"] == 1000.0 + 20240 + 3


def test_ttm_is_four_quarter_sum_and_matches_cum(tmp_store):
    _write_px()
    _write_fin(_all_rows())
    ttm = _load().revenue_ttm[SYM]
    # 4분기가 처음 다 모이는 건 2023 사업보고서 (2024-03-15 금 접수 → 03-18 월부터): 10+20+30+40
    assert np.isnan(ttm.loc["2024-03-15"])
    assert ttm.loc["2024-03-18"] == 100.0
    # 2024 Q1 (05-15 수 → 05-16): 11+20+30+40 = Q1누적24 + FY23 − Q1누적23
    assert ttm.loc["2024-05-16"] == 101.0 == 11 + 100 - 10
    assert ttm.loc["2024-08-15"] == 102.0 == 32 + 100 - 30         # H1 누적 대조
    assert ttm.loc["2024-11-15"] == 103.0 == 63 + 100 - 60         # Q3 누적 대조
    assert ttm.loc["2025-03-17"] == 104.0                          # FY − Q3 누적으로 만든 4분기 41 포함


def test_ttm_nan_when_consolidation_basis_changes(tmp_store):
    _write_px()
    rows = _all_rows()
    rows[4] = _row(2024, "Q1", fs_div="OFS")      # 2024 Q1만 별도
    _write_fin(rows)
    ttm = _load().revenue_ttm[SYM]
    assert ttm.loc["2024-03-18"] == 100.0
    assert np.isnan(ttm.loc["2024-05-16"])        # 연결 3개 분기 + 별도 1개 분기는 더하지 않는다
    assert _load().equity[SYM].loc["2024-05-16"] == 1000.0 + 20240   # 재무상태는 그대로 최신 분기


def test_q4_nan_when_fy_and_q3_basis_differ(tmp_store):
    """4분기 = FY − Q3 누적은 두 보고서의 fs_div가 같을 때만. 2023 Q3만 별도(OFS), FY는 연결(CFS)이면 2023 4분기 손익은
    NaN이고, 그 4분기를 포함하되 2023 Q3은 포함하지 않는 TTM(2024 Q3 기준: 23Q4~24Q3)도 NaN이어야 한다
    (예전에는 연결 연간 − 별도 누적이 섞인 값이 TTM에 들어갔다, validator 보고 약 280행)."""
    _write_px()
    rows = _all_rows()
    rows[2] = _row(2023, "Q3", fs_div="OFS")
    _write_fin(rows)
    rep = bdata.fin_reports()
    q4 = rep[(rep["period"] == "FY") & (rep["qidx"] == 2023 * 4 + 3)].iloc[0]
    assert np.isnan(q4["revenue"]) and np.isnan(q4["net_income"])
    assert q4["total_equity"] == 1000.0 + 20230 + 3                # 재무상태는 그대로
    q4_24 = rep[(rep["period"] == "FY") & (rep["qidx"] == 2024 * 4 + 3)].iloc[0]
    assert q4_24["revenue"] == 41.0                                # 같은 기준인 해는 그대로
    ttm = _load().revenue_ttm[SYM]
    assert np.isnan(ttm.loc["2024-11-15"])                         # 23Q4(혼재)~24Q3
    assert ttm.loc["2025-03-17"] == 104.0                          # 24Q1~24Q4는 모두 연결


def test_company_report_q4_same_rule(tmp_store):
    """intel/company.py 기업 리포트의 4분기(FY−Q3 누적)도 같은 규칙: 기준이 다르면 손익 없음."""
    from autotrader.intel import company
    rows = _all_rows()
    rows[2] = _row(2023, "Q3", fs_div="OFS")
    for r in rows:  # 리포트는 영업이익 열도 쓴다
        r["operating_income"], r["operating_income_cum"] = r["revenue"] / 2, r["revenue_cum"] / 2
    _write_fin(rows)
    f = company._fund_dart(SYM, pd.Timestamp("2025-06-30")).set_index("period_label")
    assert np.isnan(f.loc["2023Q4*", "revenue"]) and "다름" in f.loc["2023Q4*", "flag"]
    assert f.loc["2023Q4*", "total_equity"] == 1000.0 + 20230 + 3
    assert f.loc["2024Q4*", "revenue"] == 41.0 and f.loc["2024Q4*", "flag"] == "FY−Q3누적"


def test_missing_quarter_keeps_last_complete_ttm(tmp_store):
    _write_px()
    _write_fin([r for r in _all_rows() if not (r["bsns_year"] == 2024 and r["period"] == "Q1")])
    ttm = _load().revenue_ttm[SYM]
    # 2024 Q1 보고서가 없으면 그 뒤 TTM(Q1~Q3 2024)은 만들 수 없다 → 마지막으로 다 모인 2023 TTM 유지
    assert ttm.loc["2024-11-15"] == 100.0
    # 2024 연간 TTM(2024 Q1~Q4)도 2024 Q1이 필요 → 여전히 2023 값
    assert ttm.loc["2025-03-17"] == 100.0


def test_correction_timing_first_vs_last(tmp_store, monkeypatch):
    _write_px()
    rows = _all_rows()
    rows[4] = _row(2024, "Q1", filed="2024-09-02", total_equity=7.0)   # 정정 보고서: filed = 정정일
    _write_fin(rows)
    ff = pd.DataFrame({"corp_code": ["C" + SYM], "kind": ["분기보고서"], "period": ["2024.03"],
                       "first_filed": [_kst("2024-05-15")]})
    monkeypatch.setattr(dcol, "first_filed", lambda: ff)
    first, last = _load(fin_timing="first"), _load(fin_timing="last")
    # first: 최초 접수 다음 날부터 (값은 정정 값 — 원래 값이 저장돼 있지 않음)
    assert first.equity[SYM].loc["2024-05-16"] == 7.0
    # last: 정정일 다음 날까지는 이전 분기(2023 FY) 값
    assert last.equity[SYM].loc["2024-05-16"] == 1000.0 + 20230 + 3
    assert last.equity[SYM].loc["2024-09-03"] == 1000.0 + 20240 + 1   # 그때는 이미 더 새 H1이 알려져 있음
    # 2024 H1 TTM은 Q1이 필요 → last에서는 정정일 다음 날부터 (그 전엔 마지막으로 계산된 2023 TTM),
    # first에서는 H1 공개 다음 날부터
    assert last.revenue_ttm[SYM].loc["2024-08-15"] == 100.0
    assert last.revenue_ttm[SYM].loc["2024-09-03"] == 102.0
    assert first.revenue_ttm[SYM].loc["2024-08-15"] == 102.0
    assert first.meta["fin_data"]["corrected"] == 1
    assert first.meta["fin_data"]["fallback"] == len(rows) - 1      # 나머지는 최초 접수일 모름 → filed
    assert last.meta["fin_data"]["timing"] == "last"


def test_late_old_report_does_not_override_newer(tmp_store):
    _write_px()
    rows = _all_rows()
    rows[2] = _row(2023, "Q3", filed="2024-06-03", total_equity=5.0)   # 2023 Q3를 늦게 처음 접수
    _write_fin(rows)
    eq = _load().equity[SYM]
    assert eq.loc["2024-06-04"] == 1000.0 + 20240        # 이미 알려진 2024 Q1 값 유지
    ttm = _load().revenue_ttm[SYM]
    # 2023 4분기 손익은 Q3 누적이 필요 → 2023 Q3 공개(06-03) 다음 날 전에는 2024 Q1 TTM도 계산 못 함
    assert np.isnan(ttm.loc["2024-05-16"])
    assert ttm.loc["2024-06-04"] == 101.0


def test_delisted_symbol_stops_with_prices(tmp_store):
    _write_px()
    gone = "900002"
    _write_px(gone, DAYS[DAYS < "2024-06-01"])      # 2024-05-31 상장폐지
    _write_fin(_all_rows() + [_row(y, p, sym=gone) for y in (2023, 2024) for p in ("Q1", "H1", "Q3", "FY")])
    p = bdata.load("kr_stock", [SYM, gone], lockbox=True, neutralize_jumps=False, reinvest_dividends=False,
                   fields=("equity",))
    eq = p.equity[gone]
    assert eq.loc["2024-05-31"] == 1000.0 + 20240     # 폐지 전 마지막 날까지는 값
    assert eq.loc["2024-06-03":].isna().all()          # 그 뒤 보고서가 있어도 가격이 없으면 비어 있음
    assert p.close[gone].loc["2024-06-03":].isna().all()


def test_causality_check_passes_and_catches_leak(tmp_store):
    _write_px()
    _write_fin(_all_rows())
    p = _load()
    ok = bdata.check_fin_causality(p)
    assert ok["passed"] and ok["fields"] == ["equity", "revenue_ttm", "net_income_ttm"]
    # 원천 표의 공개일을 하루 늦추면(= panel은 하루 일찍 본 것과 같음) 검사가 잡아야 한다
    p.fin_reports = p.fin_reports.assign(src_date=p.fin_reports["src_date"] + pd.Timedelta(days=1))
    p.fin_reports.loc[:, "known"] = p.fin_reports["known"] + pd.Timedelta(days=1)
    p.fin_reports.loc[:, "flow_known"] = p.fin_reports["flow_known"] + pd.Timedelta(days=1)
    assert not bdata.check_fin_causality(p, n_cuts=12)["passed"]


def test_coverage_and_errors(tmp_store):
    _write_px()
    _write_fin(_all_rows() + [_row(2024, "Q1", sym="900009", currency="USD")])
    p = _load()
    d = p.meta["fin_data"]
    assert d["symbols"] == 1 and d["non_krw_skipped"] == 1 and d["first"] == "2023-05-15"
    assert 0 < d["missing"]["equity"] < d["missing"]["revenue_ttm"] < 1
    assert "최초 접수일" in p.meta["fin_fields"]
    from autotrader.backtest import lab
    lines = lab._fin_lines(p.meta)
    assert lines[0].startswith("- 재무 필드 시점") and "결측률" in lines[1] and "2023-05-15" in lines[1]
    with pytest.raises(ValueError):
        _load(fin_timing="bogus")
    with pytest.raises(ValueError, match="kr_stock"):
        bdata.load("us_stock", ["AAPL"], fields=("eps_ttm",), lockbox=True)   # US 주당 값은 없음
