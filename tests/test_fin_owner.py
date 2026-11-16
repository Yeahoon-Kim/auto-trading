"""KR 지배주주 순이익(dart_fin_owner → dart_fin.net_income_controlling) 파싱·진행 규칙 테스트 (네트워크 없이)."""

import pandas as pd
import pytest

from autotrader.data import store
from autotrader.data.collectors import kr_public as kp


def _r(sj, aid, nm, v, cum=""):
    return {"sj_div": sj, "account_id": aid, "account_nm": nm, "thstrm_amount": v, "thstrm_add_amount": cum,
            "rcept_no": "20240814004071"}


# SK 2024 반기 (실제 응답 일부): 지배 180,968 + 비지배 273,164 = 전체 454,132 (억 원 단위 생략)
SK_H1 = [
    _r("CIS", "ifrs-full_ComprehensiveIncomeAttributableToOwnersOfParent", "지배기업소유주지분", "552864", "1391770"),
    _r("CIS", "ifrs-full_ProfitLoss", "반기순이익(손실)", "454132", "1179384"),
    _r("CIS", "ifrs-full_ProfitLossAttributableToNoncontrollingInterests", "비지배지분", "273164", "840993"),
    _r("CIS", "ifrs-full_ProfitLossAttributableToOwnersOfParent", "지배기업소유주지분", "180968", "338391"),
]


def test_parse_owner_standard_id():
    assert kp.parse_owner(SK_H1, "11012") == (180968, 338391, "cfs")


def test_parse_owner_nonstandard_name_skips_comprehensive():
    rows = [  # SK 2016 1분기: '-표준계정코드 미사용-'. 포괄이익 지배분(328,400)은 고르면 안 된다
        _r("CIS", "-표준계정코드 미사용-", "연결당기순이익", "1062677", "1062677"),
        _r("CIS", "ifrs_ComprehensiveIncomeAttributableToOwnersOfParent", "지배기업소유주지분", "328400", "328400"),
        _r("CIS", "-표준계정코드 미사용-", "지배기업소유주지분순이익(손실)", "387051", "387051"),
        _r("CIS", "-표준계정코드 미사용-", "비지배지분순이익", "675626", "675626"),
    ]
    assert kp.parse_owner(rows, "11013") == (387051, 387051, "cfs")


def test_parse_owner_is_before_cis_and_fy_cum():
    rows = [  # 삼성전자 2015 사업보고서: 누적 칸이 비어 있다 → 누적 = 연간
        _r("CIS", "ifrs_ProfitLoss", "당기순이익(손실)", "19060144"),
        _r("IS", "ifrs_ProfitLoss", "당기순이익(손실)", "19060144"),
        _r("IS", "ifrs_ProfitLossAttributableToOwnersOfParent", "지배기업의 소유주에게 귀속되는 당기순이익(손실)", "18694628"),
        _r("IS", "ifrs_ProfitLossAttributableToNoncontrollingInterests", "비지배지분에 귀속되는 당기순이익(손실)", "365516"),
    ]
    assert kp.parse_owner(rows, "11011") == (18694628, 18694628, "cfs")


def test_parse_owner_sign_fix():
    rows = [  # 073540 2016 1분기: 순손실인데 지배주주 항목이 양수로 제출됨, 비지배 칸은 빈 값
        _r("CIS", "ifrs_ProfitLoss", "당기순이익", "-1090425060", "-1090425060"),
        _r("CIS", "ifrs_ProfitLossAttributableToOwnersOfParent", "지배기업 소유주지분", "1090425060", "1090425060"),
        _r("CIS", "ifrs_ProfitLossAttributableToNoncontrollingInterests", "비지배지분", ""),
    ]
    assert kp.parse_owner(rows, "11013") == (-1090425060, -1090425060, "cfs_signfix")


def test_parse_owner_no_nci_and_missing():
    no_split = [_r("CIS", "ifrs-full_ProfitLossBeforeTax", "법인세비용차감전순이익", "1431"),
                _r("CIS", "ifrs-full_ProfitLoss", "당기순이익", "2007")]
    assert kp.parse_owner(no_split, "11011") == (2007, 2007, "cfs_no_nci")
    comp_only = no_split + [  # 비지배는 있는데 포괄이익만 나눔 → 모름
        _r("CIS", "ifrs-full_ComprehensiveIncomeAttributableToOwnersOfParent", "지배기업의 소유주지분", "2100"),
        _r("CIS", "ifrs-full_ComprehensiveIncomeAttributableToNoncontrollingInterests", "비지배지분", "-90")]
    assert kp.parse_owner(comp_only, "11014") is None
    assert kp.parse_owner([], "11014") is None


def _fin():
    ts = pd.Timestamp("2024-06-29 15:00", tz="UTC")
    return pd.DataFrame({
        "ts": [ts] * 3, "symbol": ["034730", "005930", "999990"], "corp_code": ["00181712", "00126380", "00999999"],
        "bsns_year": [2024] * 3, "reprt_code": ["11012"] * 3, "rcept_no": ["R1", "R2", "R3"],
        "fs_div": ["CFS", "CFS", "OFS"], "net_income": [454132.0, 9841.0, 50.0],
        "net_income_cum": [1179384.0, 16596.0, 80.0]})


def _owner_rows(rows):
    return pd.DataFrame([{
        "ts": pd.Timestamp("2024-06-29 15:00", tz="UTC"), "bsns_year": 2024, "reprt_code": "11012", "rcept_no": r[1],
        "status": "000", "fetched": pd.Timestamp("2026-11-16", tz="UTC"), "symbol": r[0], "fin_rcept_no": r[1],
        "has_data": r[2] is not None, "net_income_controlling": r[2] if r[2] is not None else float("nan"),
        "net_income_controlling_cum": r[3], "source": r[4]} for r in rows])


def test_owner_todo_cfs_only_and_refetch_on_correction():
    fin = _fin()
    assert kp.owner_todo(fin, pd.DataFrame())["symbol"].tolist() == ["005930", "034730"]  # 별도(OFS)는 안 받는다
    owner = _owner_rows([("034730", "R1", 180968.0, 338391.0, "cfs"), ("005930", "R2", None, float("nan"), None)])
    assert kp.owner_todo(fin, owner).empty  # 항목 없음(has_data=False)도 다시 묻지 않는다
    fin.loc[fin["symbol"] == "034730", "rcept_no"] = "R1b"  # 정정 보고서 → 다시 받기
    assert kp.owner_todo(fin, owner)["symbol"].tolist() == ["034730"]


def test_attach_controlling():
    fin = _fin()
    owner = _owner_rows([("034730", "R1", 180968.0, 338391.0, "cfs")])
    out = kp.attach_controlling(fin, owner).set_index("symbol")
    assert out.loc["034730", ["net_income_controlling", "net_income_controlling_cum"]].tolist() == [180968, 338391]
    assert out.loc["034730", "nic_source"] == "cfs"
    assert out.loc["999990", ["net_income_controlling", "net_income_controlling_cum"]].tolist() == [50, 80]  # 별도
    assert out.loc["999990", "nic_source"] == "ofs"
    assert pd.isna(out.loc["005930", "net_income_controlling"]) and out.loc["005930", "nic_source"] == "cfs_missing"
    fin.loc[fin["symbol"] == "034730", "rcept_no"] = "R1b"  # 정정 뒤 아직 못 받은 값은 쓰지 않는다
    again = kp.attach_controlling(kp.attach_controlling(fin, owner), owner).set_index("symbol")  # 두 번 붙여도 같다
    assert again.loc["034730", "nic_source"] == "cfs_missing" and len(again) == 3
    assert kp.attach_controlling(fin, pd.DataFrame())["nic_source"].tolist() == ["cfs_missing", "cfs_missing", "ofs"]


def test_collect_owner_budget_limit_and_patch(monkeypatch):
    store.write("dart_fin", "ALL", _fin(), key=["symbol", "bsns_year", "reprt_code"])
    calls = []

    def fake(corp, year, reprt):
        calls.append(corp)
        if len(calls) == 2:
            raise kp.DartLimitReached("한도 초과")
        return "000", SK_H1

    monkeypatch.setattr(kp, "_owner", fake)
    budget = [10]
    kp.collect_dart_owner(budget)
    assert budget == [9] and len(store.read(kp.OWNER_DS, "ALL")) == 1  # 한도에 걸리면 멈추고 받은 것만 저장
    assert not store.get_state(kp.OWNER_DS, "ALL")["backfill_done"]

    monkeypatch.setattr(kp, "_owner", lambda c, y, r: ("013", []))
    kp.collect_dart_owner([10])
    owner = store.read(kp.OWNER_DS, "ALL")
    assert len(owner) == 2 and owner["has_data"].sum() == 1
    assert store.get_state(kp.OWNER_DS, "ALL")["backfill_done"]

    kp.patch_dart_fin()
    fin = store.read("dart_fin", "ALL").set_index("symbol")
    assert set(fin["nic_source"]) == {"cfs", "ofs", "cfs_missing"} and len(fin) == 3
    # collect_dart_fin이 같은 키를 다시 쓰면(열 없음) 그 행은 비지만, patch가 다시 채운다
    store.write("dart_fin", "ALL", _fin().iloc[[1]], key=["symbol", "bsns_year", "reprt_code"])
    kp.patch_dart_fin()
    fin = store.read("dart_fin", "ALL").set_index("symbol")
    assert fin.loc["005930", "net_income_controlling"] == pytest.approx(180968)  # 첫 호출(005930)이 SK_H1을 받음
