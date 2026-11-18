"""DART 기업 연고표 수집기 테스트 (네트워크 없이, 모의 응답)."""

import pandas as pd
import pytest

from autotrader.data import store
from autotrader.data.collectors import dart_profile as dp

KST = "Asia/Seoul"
# 2026-11-18 실제 응답 형태 (exctvSttus, 값은 가공)
EXEC = [
    {"rcept_no": "20250311001085", "corp_cls": "Y", "corp_code": "00000001", "corp_name": "가나전자", "nm": "홍길동",
     "sexdstn": "남", "birth_ym": "1962년 03월", "ofcps": "부회장", "rgist_exctv_at": "사내이사", "fte_at": "상근",
     "chrg_job": "대표이사", "main_career": "서울대 전자공학\n가나전자 사업부장", "mxmm_shrholdr_relate": "계열회사 임원",
     "hffc_pd": " 58개월", "tenure_end_on": "2026년 03월 17일", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "corp_cls": "Y", "corp_code": "00000001", "corp_name": "가나전자", "nm": "김철수",
     "sexdstn": "여", "birth_ym": "1970년 01월", "ofcps": "사외이사", "rgist_exctv_at": "사외이사", "fte_at": "비상근",
     "chrg_job": "-", "main_career": "청주고\n前 국회의원 보좌관", "mxmm_shrholdr_relate": "-",
     "hffc_pd": "12개월", "tenure_end_on": "2027년 03월 17일", "stlm_dt": "2024-12-31"},
]


def test_report_rows_keeps_raw_text_and_point_in_time():
    now = pd.Timestamp("2026-11-18 08:00", tz="UTC")
    first = pd.Timestamp("2025-03-11", tz=KST).tz_convert("UTC")
    d = dp.report_rows("00000001", "000001", 2024, "2024.12", first, EXEC, now)
    assert len(d) == 2 and d["seq"].tolist() == [0, 1]
    assert d["main_career"].iloc[0] == "서울대 전자공학\n가나전자 사업부장"  # 줄바꿈까지 원문 그대로
    assert d["hffc_pd"].iloc[0] == " 58개월"  # 앞 공백도 그대로 (파싱은 lab에서)
    assert (d["ts"] == pd.Timestamp("2024-12-31", tz=KST)).all()
    assert (d["filed"] == pd.Timestamp("2025-03-11", tz=KST)).all()
    assert (d["first_filed"] == first).all()
    assert (d["corp_code"] == "00000001").all() and (d["symbol"] == "000001").all()
    assert dp.report_rows("00000001", "000001", 2024, "2024.12", None, [], now).empty


def test_report_rows_without_stlm_uses_period_end():
    rows = [{k: v for k, v in EXEC[0].items() if k != "stlm_dt"}]
    d = dp.report_rows("00000001", "000001", 2016, "2016.03", None, rows, pd.Timestamp("2026-11-18", tz="UTC"))
    assert d["ts"].iloc[0] == pd.Timestamp("2016-03-31", tz=KST)
    assert pd.isna(d["first_filed"].iloc[0])


def test_report_pairs_only_years_with_annual_report():
    codes = pd.DataFrame({"corp_code": ["A", "B", "C"], "symbol": ["1", "2", "3"]})
    dividend = pd.DataFrame({
        "corp_code": ["A", "A", "B", "Z"], "bsns_year": [2014, 2016, 2016, 2016], "has_data": [True, True, False, True],
        "ts": [pd.Timestamp(f"{y}-12-31", tz=KST).tz_convert("UTC") for y in (2014, 2016, 2016, 2016)]})
    filings = pd.DataFrame({  # dividends.first_filed() 형태
        "corp_code": ["A", "C", "C"], "kind": ["사업보고서", "사업보고서", "반기보고서"],
        "period": ["2016.12", "2017.03", "2017.06"],
        "first_filed": [pd.Timestamp("2017-03-30", tz="UTC"), pd.Timestamp("2017-06-29", tz="UTC"),
                        pd.Timestamp("2017-08-14", tz="UTC")]})
    p = dp.report_pairs(codes, dividend, filings, 2015)
    got = {(r.corp_code, r.bsns_year): (r.period, r.first_filed) for r in p.itertuples()}
    # A 2014는 시작 연도 전, B 2016은 배당 '없음'뿐이고 사업보고서 목록에도 없음, Z는 상장 종목이 아님
    assert set(got) == {("A", 2016), ("C", 2017)}
    assert got[("A", 2016)] == ("2016.12", pd.Timestamp("2017-03-30", tz="UTC"))
    assert got[("C", 2017)][0] == "2017.03"  # 3월 결산 → 사업연도 = 결산 연도


def test_todo_calls_resume_and_recheck():
    pairs = pd.DataFrame({"corp_code": ["A", "B", "A"], "symbol": ["1", "2", "1"], "bsns_year": [2024, 2024, 2025],
                          "period": ["2024.12", "2024.12", "2025.12"], "first_filed": [pd.NaT] * 3})
    now = pd.Timestamp("2026-04-20", tz="UTC")
    calls = pd.DataFrame({
        "api": ["exctvSttus", "otrCprInvstmntSttus", "hyslrSttus", "exctvSttus", "otrCprInvstmntSttus"],
        "corp_code": ["A", "A", "A", "A", "A"], "bsns_year": [2024, 2024, 2024, 2025, 2025],
        "status": ["000", "013", "013", "013", "013"],
        "fetched": pd.to_datetime(["2025-04-01", "2025-08-01", "2025-04-01", "2026-04-15", "2026-03-01"], utc=True)})
    todo = [(ds, c, y) for ds, c, _, y, _, _ in dp.todo_calls(pairs, calls, now)]
    assert todo == [
        # 2025 먼저(최근 연도부터). A: 임원은 5일 전 '없음' → 기다림, 출자는 '없음' 뒤 14일 지남 → 다시, 최대주주는 처음
        ("dart_investments", "A", 2025), ("dart_major_holders", "A", 2025),
        # 2024 A: 임원 받음, 출자 마감 뒤 '없음' → 끝, 최대주주 마감 전 '없음' → 다시. B는 3종 모두 처음
        ("dart_major_holders", "A", 2024),
        ("dart_executives", "B", 2024), ("dart_investments", "B", 2024), ("dart_major_holders", "B", 2024),
    ]


def test_company_todo_refresh():
    codes = pd.DataFrame({"corp_code": ["A", "B", "C"], "symbol": ["1", "2", "3"]})
    now = pd.Timestamp("2026-11-18", tz="UTC")
    company = pd.DataFrame({"corp_code": ["A", "B"],
                            "fetched": pd.to_datetime(["2026-11-01", "2026-07-01"], utc=True)})
    assert dp.company_todo(codes, company, now, 90) == ["B", "C"]
    assert dp.company_todo(codes, pd.DataFrame(), now, 90) == ["A", "B", "C"]


@pytest.fixture
def fake_dart(monkeypatch):
    """_corp_codes·_call을 모의로 바꾼다. 호출 기록을 돌려준다."""
    codes = pd.DataFrame({"corp_code": ["00000001", "00000002"], "symbol": ["000001", "000002"]})
    monkeypatch.setattr(dp, "_corp_codes", lambda: codes)
    monkeypatch.setattr(dp._dart_limit, "wait", lambda: None)
    log: list[tuple] = []
    state = {"limit_after": None}

    def call(ep, params):
        log.append((ep, params.get("corp_code"), params.get("bsns_year")))
        if state["limit_after"] is not None and len(log) > state["limit_after"]:
            raise dp.DartLimit(ep)
        if ep == "company":
            return {"status": "000", "message": "정상", "corp_code": params["corp_code"], "adres": "충청북도 청주시 흥덕구",
                    "stock_code": "000001", "est_dt": "19900101"}
        if ep == "exctvSttus" and params["corp_code"] == "00000001":
            return {"status": "000", "list": EXEC}
        return {"status": "013", "message": "조회된 데이타가 없습니다."}

    monkeypatch.setattr(dp, "_call", call)
    # 사업보고서가 있다고 확인된 회사·연도: 1번 2023·2024, 2번 2024 (dart_dividend로)
    store.write("dart_dividend", "ALL", pd.DataFrame({
        "ts": [pd.Timestamp(f"{y}-12-31", tz=KST) for y in (2023, 2024, 2024)],
        "corp_code": ["00000001", "00000001", "00000002"], "symbol": ["000001", "000001", "000002"],
        "bsns_year": [2023, 2024, 2024], "reprt_code": ["11011"] * 3, "has_data": [True] * 3}),
        key=["corp_code", "bsns_year", "reprt_code"])
    return log, state


def test_collect_budget_and_resume(fake_dart):
    log, _ = fake_dart
    budget = [5]
    dp.collect_dart_profile(budget)
    # 기업개황 2회 → 보고서 3회 (2024 1번 회사 3종). 예산을 넘지 않는다
    assert budget[0] == 0 and len(log) == 5
    assert [e for e, *_ in log] == ["company", "company", "exctvSttus", "otrCprInvstmntSttus", "hyslrSttus"]
    assert log[2][1:] == ("00000001", "2024")
    ex = store.read("dart_executives", "ALL")
    assert len(ex) == 2 and ex["main_career"].iloc[1] == "청주고\n前 국회의원 보좌관"
    co = store.read("dart_company", "ALL")
    assert set(co["corp_code"]) == {"00000001", "00000002"} and co["adres"].iloc[0] == "충청북도 청주시 흥덕구"
    assert not store.get_state(dp.CALLS_DS, "ALL")["backfill_done"]

    # 다음 실행: 받은 것은 건너뛰고 남은 6회(2024 2번 회사 3종, 2023 1번 회사 3종)만, 기업개황은 다시 묻지 않는다
    log.clear()
    budget = [100]
    dp.collect_dart_profile(budget)
    assert budget[0] == 94 and "company" not in [e for e, *_ in log]
    assert [(c, y) for _, c, y in log] == [("00000002", "2024")] * 3 + [("00000001", "2023")] * 3
    assert store.get_state(dp.CALLS_DS, "ALL")["backfill_done"]
    calls = store.read(dp.CALLS_DS, "ALL")
    assert len(calls) == 9 and calls.duplicated(dp.CALLS_KEY).sum() == 0

    # 그다음 실행: 할 것 없음 (013 '없음'은 14일 안이라 다시 묻지 않는다)
    log.clear()
    dp.collect_dart_profile([100])
    assert log == []


def test_collect_stops_on_daily_limit(fake_dart):
    _, state = fake_dart
    state["limit_after"] = 3  # 4번째 호출에서 020
    budget = [100]
    dp.collect_dart_profile(budget)
    assert budget[0] == 0  # 뒤에 남은 단계도 이번 실행은 쉰다
    assert store.get_state(dp.CALLS_DS, "ALL")["status"] != "error"
    calls = store.read(dp.CALLS_DS, "ALL")
    assert len(calls) == 1 and calls["api"].iloc[0] == "exctvSttus"  # 한도 전까지 받은 것은 저장됨
    assert len(store.read("dart_executives", "ALL")) == 2


def test_no_budget_no_calls(fake_dart):
    log, _ = fake_dart
    dp.collect_dart_profile([0])
    assert log == []


def test_first_filed_dtype_stable():
    now = pd.Timestamp("2026-11-18", tz="UTC")
    a = dp.report_rows("00000001", "000001", 2024, "2024.12", None, EXEC, now)
    b = dp.report_rows("00000001", "000001", 2024, "2024.12", pd.Timestamp("2025-03-11", tz="UTC"), EXEC, now)
    assert str(a["first_filed"].dtype) == str(b["first_filed"].dtype) == "datetime64[ns, UTC]"
