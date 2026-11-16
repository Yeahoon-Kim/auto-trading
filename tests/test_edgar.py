"""SEC EDGAR 수집기 파싱·갱신 규칙 테스트 (네트워크 없이)."""

import httpx
import pandas as pd

from autotrader.data import store
from autotrader.data.collectors import edgar

# AAPL companyfacts 일부 (실제 응답 형태). 같은 기간 EPS가 다음 해 10-K 비교 재무제표에 다시 나온다.
FACTS = {"cik": 320193, "entityName": "Apple Inc.", "facts": {
    "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2024-10-18", "val": 15115823000, "accn": "0000320193-24-000123", "fy": 2024, "fp": "FY",
         "form": "10-K", "filed": "2024-11-01", "frame": "CY2024Q3I"}]}}},
    "us-gaap": {
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            {"start": "2023-10-01", "end": "2024-09-28", "val": 6.08, "accn": "0000320193-24-000123", "fy": 2024,
             "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
            {"start": "2023-10-01", "end": "2024-09-28", "val": 6.08, "accn": "0000320193-25-000079", "fy": 2025,
             "fp": "FY", "form": "10-K", "filed": "2025-10-31"}]}},
        "Revenues": {"units": {"USD": [
            {"start": "2015-09-27", "end": "2016-09-24", "val": 215639000000, "accn": "0000320193-18-000145",
             "fy": 2018, "fp": "FY", "form": "10-K", "filed": "2018-11-05", "frame": "CY2016"}]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"start": "2023-10-01", "end": "2024-09-28", "val": 391035000000, "accn": "0000320193-24-000123",
             "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"}]}},
        "StockholdersEquity": {"units": {"USD": [
            {"end": "2024-09-28", "val": 56950000000, "accn": "0000320193-24-000123", "fy": 2024, "fp": "FY",
             "form": "10-K", "filed": "2024-11-01"},
            {"end": "2024-09-28", "val": 56950000000, "accn": "0000320193-25-000001", "fy": 2024, "fp": "FY",
             "form": "10-K/A", "filed": "2025-01-15"}]}},
        "AccountsPayableCurrent": {"units": {"USD": [  # 받지 않는 태그
            {"end": "2024-09-28", "val": 1, "accn": "x", "form": "10-K", "filed": "2024-11-01"}]}},
    }}}

INDEX = """Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
1-A POS          Doroni Aerospace, Inc.                                        1890225     20261113    edgar/data/1890225/0001683168-26-008661.txt
10-K             Apple Inc.                                                    320193      20261113    edgar/data/320193/0000320193-26-000026.txt
10-Q/A           Some  Double  Spaced Corp                                     1234        20261113    edgar/data/1234/0000001234-26-000001.txt
8-K              NVIDIA CORP                                                   1045810     20261113    edgar/data/1045810/0001045810-26-000001.txt
"""


def test_parse_facts():
    d = edgar.parse_facts(FACTS, "AAPL")
    assert set(d["field"]) == {"eps_diluted", "revenue", "equity", "shares_outstanding"}
    eps = d[d["field"] == "eps_diluted"]
    assert len(eps) == 2 and eps["filed"].min() == pd.Timestamp("2024-11-01", tz=edgar.NY)  # 최초·재게재 모두
    eq = d[d["field"] == "equity"]
    assert sorted(eq["form"]) == ["10-K", "10-K/A"] and eq["start"].isna().all()  # 정정도 남는다, 시점 항목
    rev = d[d["field"] == "revenue"].set_index("tag")["tag_rank"]
    assert rev["Revenues"] == 0 and rev["RevenueFromContractWithCustomerExcludingAssessedTax"] == 1
    assert (d["ts"] == d["end"]).all() and (d["cik"] == 320193).all()
    assert edgar.parse_facts({"facts": {}}, "X").empty


def test_parse_index():
    assert edgar.parse_index(INDEX) == {320193, 1234}  # 8-K·기타 제외, 정정 포함


def test_todo():
    now = pd.Timestamp("2026-11-16", tz="UTC")
    comp = pd.DataFrame({"symbol": ["AAPL", "NVDA", "MSFT", "NEW", "NOCIK"],
                         "cik": pd.array([320193, 1045810, 789019, 1, None], dtype="Int64")})
    fetched = pd.DataFrame({"symbol": ["AAPL", "NVDA", "MSFT"],
                            "ts": [now - pd.Timedelta(days=d) for d in (1, 2, 40)]})
    assert edgar.todo(comp, fetched, {320193}, now, 30) == ["NEW", "MSFT", "AAPL"]


def test_run_without_user_agent_skips(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(edgar, "ROOT", store.DATA_ROOT)  # .env 없는 폴더
    edgar.run()
    st = store.get_state(edgar.DS, "ALL")
    assert st["status"] == "skipped" and "SEC_USER_AGENT" in st["error"]


def test_run_end_to_end(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "test-agent test@example.com")
    seen = []

    def fake_get(client, url):
        seen.append((client.headers["User-Agent"], url))
        req = httpx.Request("GET", url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                                             "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "Berkshire"}},
                                  request=req)
        if "CIK0000320193" in url:
            return httpx.Response(200, json=FACTS, request=req)
        return httpx.Response(404, request=req)

    monkeypatch.setattr(edgar, "_get", fake_get)
    edgar.run(["AAPL", "BRK-B"])
    facts, co = store.read(edgar.DS, "ALL"), store.read(edgar.CO_DS, "ALL")
    assert len(facts) == 7 and set(co["symbol"]) == {"AAPL", "BRK-B"}  # BRK.B → BRK-B 매핑, 사실 없는 회사도 기록
    assert all(ua == "test-agent test@example.com" for ua, _ in seen)
    assert store.get_state(edgar.DS, "ALL")["backfill_done"]
    edgar.run(["AAPL", "BRK-B"])  # 다시 돌려도 중복 없음 (색인은 어제까지 확인한 상태라 새 대상 없음)
    assert len(store.read(edgar.DS, "ALL")) == 7


def test_bulk_load_from_zip():
    import io
    import json
    import zipfile

    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("CIK0000320193.json", json.dumps(FACTS))
        z.writestr("CIK0000000001.json", json.dumps({"cik": 1, "facts": {}}))  # 유니버스 밖 회사
    body = raw.getvalue()

    def handler(req):
        assert req.url == httpx.URL(edgar.BULK_URL)
        return httpx.Response(200, content=body, headers={"last-modified": "Mon, 16 Nov 2026 08:00:00 GMT"})

    got = []
    targets = pd.DataFrame({"cik": [320193, 999]}, index=["AAPL", "NOZIP"])
    edgar.bulk_load(httpx.Client(transport=httpx.MockTransport(handler)), targets,
                    lambda s, df, ts: got.append((s, len(df), ts)))
    gen = pd.Timestamp("2026-11-16 08:00", tz="UTC")
    assert got == [("AAPL", 7, gen), ("NOZIP", 0, gen)]  # zip 생성 시각을 수집 시각으로
    assert not list(store.DATA_ROOT.glob("tmp*"))  # 임시 zip은 지운다


def test_run_uses_bulk_for_many_new(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "test-agent test@example.com")
    monkeypatch.setattr(edgar, "config", lambda: {"edgar": {"bulk_min_new": 2}})
    api = []

    def fake_get(client, url):
        req = httpx.Request("GET", url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                                             "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "Berkshire"}},
                                  request=req)
        api.append(url)
        return httpx.Response(404, request=req)

    def fake_bulk(client, targets, emit):
        for s in targets.index:
            emit(s, edgar.parse_facts(FACTS, s) if s == "AAPL" else pd.DataFrame(), pd.Timestamp.now(tz="UTC"))

    monkeypatch.setattr(edgar, "_get", fake_get)
    monkeypatch.setattr(edgar, "bulk_load", fake_bulk)
    edgar.run(["AAPL", "BRK-B"])
    assert api == [] and len(store.read(edgar.DS, "ALL")) == 7  # 새 종목은 회사별 API를 부르지 않는다
    assert set(store.read(edgar.CO_DS, "ALL")["symbol"]) == {"AAPL", "BRK-B"}
