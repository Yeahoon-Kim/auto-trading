"""DART 사업보고서 '사업의 내용' 본문 수집기 테스트 (네트워크 없이, 모의 zip)."""

import io
import zipfile

import pandas as pd
import pytest

from autotrader.data import store
from autotrader.data.collectors import dart_business as db

KST = "Asia/Seoul"
# 실제 서식을 줄인 것: 목차 표에도 'II. 사업의 내용' 글자가 있지만 TITLE이 아니다. 머리말은 utf-8, 실제는 cp949.
XML = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY>
<TABLE><TR><TD>II. 사업의 내용</TD><TD>11</TD></TR></TABLE>
<SECTION-1><TITLE ATOC="Y">I. 회사의 개요</TITLE><P>연혁: 마스크 공장 준공</P></SECTION-1>
<SECTION-1><TITLE ATOC="Y" AASSOCNOTE="D-0-2-0-0">II.  사업의   내용</TITLE>
<P>◆click◆『수주상황』 삽입</P><P>11011#*_수주상황.dsl</P>
<SECTION-2><TITLE ATOC="Y">1. 사업의 개요</TITLE><P>당사는 위생용지와&cr;KF94 마스크를 만듭니다.&amp;</P></SECTION-2>
<SECTION-2><TITLE ATOC="Y">2. 주요 제품 및 서비스</TITLE>
<TABLE><TR><TH>제품</TH><TH>비중</TH></TR><TR><TD>화장지</TD><TD>70%</TD></TR></TABLE></SECTION-2>
<SECTION-2><TITLE ATOC="Y">3. 원재료 및 생산설비</TITLE><P>펄프</P></SECTION-2>
</SECTION-1>
<SECTION-1><TITLE ATOC="Y" AASSOCNOTE="D-0-3-0-0">III. 재무에 관한 사항</TITLE><P>매출액</P></SECTION-1>
</BODY></DOCUMENT>"""


def _zip(rcept: str, xml: str = XML, enc: str = "cp949", attach: bool = True) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        if attach:  # 더 큰 첨부(감사보고서)가 앞에 있어도 본문 파일을 먼저 본다
            z.writestr(f"{rcept}_00760.xml", ("<DOCUMENT>" + "감사" * 5000 + "</DOCUMENT>").encode(enc))
        z.writestr(f"{rcept}.xml", xml.encode(enc))
    return b.getvalue()


def test_section_extract_and_cleanup():
    r = db.parse_zip(_zip("20190401002993"), "20190401002993")
    t = r["business_text"]
    assert r["status"] == "ok" and r["encoding"] == "cp949" and r["n_files"] == 2
    assert "위생용지와\nKF94 마스크를 만듭니다.&" in t        # &cr; → 줄바꿈, HTML 엔티티 풀기
    assert "연혁" not in t and "매출액" not in t               # I절·III절은 빠진다
    assert "펄프" in t and "◆click◆" not in t and ".dsl" not in t
    assert r["products_text"].splitlines() == ["제품\t비중", "화장지\t70%"]  # 표는 칸=탭, 줄=줄바꿈


def test_utf8_document_and_missing_section():
    r = db.parse_zip(_zip("1", enc="utf-8"), "1")
    assert r["status"] == "ok" and r["encoding"] == "utf-8"
    no = XML.replace("사업의   내용", "기타 사항")
    r = db.parse_zip(_zip("2", no, attach=False), "2")
    assert r["status"] == "no_section" and r["business_text"] == ""


def _filings():
    ts = lambda d: pd.Timestamp(d, tz=KST).tz_convert("UTC")
    return pd.DataFrame({
        "ts": [ts("2025-03-20"), ts("2025-05-01"), ts("2026-03-23"), ts("2026-03-27"), ts("2015-03-30"),
               ts("2026-03-15"), ts("2014-03-31")],
        "rcept_no": ["R24A", "R24A2", "R25A", "R25A2", "R14A", "R25B", "R13A"],
        "corp_code": ["A", "A", "A", "A", "A", "B", "A"],
        "symbol": ["1"] * 5 + ["2", "1"],
        "kind": ["사업보고서"] * 7,
        "period": ["2024.12", "2024.12", "2025.12", "2025.12", "2014.12", "2025.12", "2013.12"],
        "correction": [False, True, True, True, False, False, False],
    })


def test_targets_original_first_and_order():
    codes = pd.DataFrame({"corp_code": ["A", "B"], "symbol": ["1", "2"]})
    t = db.targets(_filings(), codes, 2014)
    assert list(zip(t["corp_code"], t["period"])) == [("A", "2025.12"), ("B", "2025.12"), ("A", "2024.12"),
                                                      ("A", "2014.12")]  # 최근 연도부터, 2013은 범위 밖
    a24 = t[(t.corp_code == "A") & (t.period == "2024.12")].iloc[0]
    assert a24.rcept_no == "R24A" and a24.original_found and a24.last_rcept_no == "R24A2"
    assert a24.correction_rcept_nos == "R24A2" and a24.n_filings == 2
    assert a24.first_filed == pd.Timestamp("2025-03-20", tz=KST)
    a25 = t[(t.corp_code == "A") & (t.period == "2025.12")].iloc[0]  # 정정본만 보임 → 가장 이른 것, 표시
    assert a25.rcept_no == "R25A" and not a25.original_found
    assert db.targets(_filings(), codes[codes.corp_code == "B"], 2014)["corp_code"].tolist() == ["B"]  # 상장 종목만


def test_todo_skip_done_and_retry():
    codes = pd.DataFrame({"corp_code": ["A", "B"], "symbol": ["1", "2"]})
    t = db.targets(_filings(), codes, 2014)
    now = pd.Timestamp("2026-11-18", tz="UTC")
    done = pd.DataFrame({
        "corp_code": ["A", "B", "A", "A"], "period": ["2025.12", "2025.12", "2024.12", "2014.12"],
        "rcept_no": ["R25A", "R25B", "R24A", "OLD"], "status": ["ok", "014", "parse_error", "ok"],
        "collected_at": pd.to_datetime(["2026-11-01", "2026-11-01", "2026-11-10", "2026-11-01"], utc=True)})
    left = db.todo(t, done, now)
    # A 2025 끝, B 2025 실패 뒤 14일 지남 → 다시, A 2024 실패 8일 전 → 기다림, A 2014 최초 접수본이 바뀜 → 다시
    assert list(zip(left["corp_code"], left["period"])) == [("B", "2025.12"), ("A", "2014.12")]


@pytest.fixture
def fake_dart(monkeypatch):
    codes = pd.DataFrame({"corp_code": ["A", "B"], "symbol": ["1", "2"]})
    monkeypatch.setattr(db, "_corp_codes", lambda: codes)
    store.write("dart_periodic_filings", "ALL", _filings(), key=["rcept_no"])
    log: list[str] = []
    state = {"limit_after": None}

    def download(rcept):
        log.append(rcept)
        if state["limit_after"] is not None and len(log) > state["limit_after"]:
            raise db.DartLimit(rcept)
        if rcept == "R25B":
            return "014", b""
        return "000", _zip(rcept)

    monkeypatch.setattr(db, "_download", download)
    return log, state


def test_collect_budget_resume(fake_dart):
    log, _ = fake_dart
    budget = [2]
    db.collect_dart_business(budget)
    assert budget == [0] and log == ["R25A", "R25B"]  # 2025 사업연도 먼저
    d = store.read(db.DS, "ALL")
    assert set(d["status"]) == {"ok", "014"}
    ok = d[d.status == "ok"].iloc[0]
    assert "KF94 마스크" in ok.business_text and ok.n_chars == len(ok.business_text) and ok.zip_bytes > 0
    assert ok.ts == pd.Timestamp("2025-12-31", tz=KST) and ok.first_filed == pd.Timestamp("2026-03-23", tz=KST)
    assert not store.get_state(db.DS, "ALL")["backfill_done"]

    log.clear()
    db.collect_dart_business([100])  # 받은 것·최근 실패는 건너뛰고 나머지만
    assert log == ["R24A", "R14A"]
    assert store.get_state(db.DS, "ALL")["backfill_done"]
    assert len(store.read(db.DS, "ALL")) == 4


def test_collect_stops_on_limit(fake_dart):
    _, state = fake_dart
    state["limit_after"] = 1
    budget = [100]
    db.collect_dart_business(budget)
    assert budget == [0] and len(store.read(db.DS, "ALL")) == 1  # 한도 전까지 받은 것은 저장
    assert store.get_state(db.DS, "ALL")["status"] != "error"


def test_no_budget(fake_dart):
    log, _ = fake_dart
    db.collect_dart_business([0])
    assert log == []
