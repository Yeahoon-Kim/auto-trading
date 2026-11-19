"""네이버 수집기 (collectors/naver.py) — 모의 응답으로 창 나누기·저장·이어받기·한도·과금 멈춤을 확인한다."""

from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from autotrader.data import store
from autotrader.data.collectors import naver

CFG = {"naver": {
    "trend": {"max_calls_per_day": 1000, "max_calls_per_month": 5000, "refresh_tail_days": 3, "extra_keywords": ["X 테마주"]},
    "news": {"max_calls_per_day": 1000, "max_calls_per_month": 5000, "theme_queries": ["테마주"],
             "theme_max_pages": 3, "surge_pct": 15, "max_stock_queries": 5, "stock_max_pages": 1},
}}
H = {"X-NCP-APIGW-API-KEY-ID": "id", "X-NCP-APIGW-API-KEY": "sec"}


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(naver, "config", lambda: CFG)
    monkeypatch.setattr(naver, "_sleep", lambda s: None)
    monkeypatch.setattr(naver._limit, "wait", lambda: None)


def _mock(monkeypatch, handler):
    log = []

    def h(req):
        log.append(req)
        return handler(req)
    monkeypatch.setattr(naver, "_http", httpx.Client(transport=httpx.MockTransport(h)))
    return log


def _today(monkeypatch, day: str):
    monkeypatch.setattr(naver, "_kst_today", lambda: pd.Timestamp(day))


# ── 창 ──
def test_windows_closed_and_one_open():
    w = naver.windows(pd.Timestamp("2017-03-10"))
    assert [(f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}", c) for s, e, c in w] == [
        ("2016-01-01", "2016-12-31", True), ("2016-07-01", "2017-03-09", False)]
    w = naver.windows(pd.Timestamp("2017-01-01"))  # 12-31에 끝난 창은 닫힘, 다음 창이 갱신 대상
    assert w[-1][2] is False and f"{w[-1][0]:%m-%d}" == "07-01" and f"{w[-1][1]:%Y-%m-%d}" == "2016-12-31"
    assert all(c for _, _, c in w[:-1])


# ── 검색어 트렌드 ──
def _trend_handler(req):
    b = json.loads(req.content)
    days = pd.date_range(b["startDate"], b["endDate"])
    res = [{"title": g["groupName"], "keywords": g["keywords"],
            "data": [{"period": f"{d:%Y-%m-%d}", "ratio": 1.0 + i} for i, d in enumerate(days)]}
           for g in b["keywordGroups"]]
    return httpx.Response(200, json={"startDate": b["startDate"], "endDate": b["endDate"], "timeUnit": "date",
                                     "results": res})


def _universe(n=6):
    u = pd.DataFrame({"code": [f"{i:05d}0" for i in range(n)] + ["000005"], "name": [f"종목{i}" for i in range(n)] + ["우선주"],
                      "market": "KOSPI", "delisted": pd.NaT})
    store.save_universe("kr_daily", u)


def test_trend_backfill_windows_and_resume(monkeypatch):
    _universe(6)
    _today(monkeypatch, "2017-03-10")
    log = _mock(monkeypatch, _trend_handler)
    naver.collect_trend(H)
    # 키워드 7개(보통주 6 + 추가 1, 우선주 제외) → 창마다 2요청, 창 2개
    assert len(log) == 4
    bodies = [json.loads(r.content) for r in log]
    assert all(len(b["keywordGroups"]) <= 5 for b in bodies)
    assert {(b["startDate"], b["endDate"]) for b in bodies} == {("2016-01-01", "2016-12-31"), ("2016-07-01", "2017-03-09")}
    assert log[0].headers["X-NCP-APIGW-API-KEY-ID"] == "id"
    df = store.read(naver.TREND_DS, "ALL")
    assert set(df.columns) >= {"ts", "keyword", "symbol", "ratio", "win_start", "win_end", "group", "collected_at"}
    k0 = df[df["keyword"] == "종목0"]
    assert set(k0["win_end"]) == {"2016-12-31", "2017-03-09"}
    # 행의 날짜는 늘 자기 창 안 (미래참조 없음)
    assert (k0["ts"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d") <= k0["win_end"]).all()
    assert df.loc[df["keyword"] == "X 테마주", "symbol"].eq("").all()
    assert df.loc[df["keyword"] == "종목1", "symbol"].eq("000010").all()
    # 다시 돌리면 아무것도 안 받는다
    naver.collect_trend(H)
    assert len(log) == 4
    # 다음 날: 갱신 창만 새 끝(어제)으로, 이미 받은 키워드라 끝 3일만 저장
    _today(monkeypatch, "2017-03-11")
    naver.collect_trend(H)
    assert len(log) == 6
    new = store.read(naver.TREND_DS, "ALL")
    new = new[(new["win_end"] == "2017-03-10") & (new["keyword"] == "종목0")]
    assert len(new) == 3


def test_trend_daily_limit_stops_and_resumes(monkeypatch):
    _universe(6)
    _today(monkeypatch, "2017-03-10")
    cfg = {"naver": {**CFG["naver"], "trend": {**CFG["naver"]["trend"], "max_calls_per_day": 1}}}
    monkeypatch.setattr(naver, "config", lambda: cfg)
    log = _mock(monkeypatch, _trend_handler)
    naver.collect_trend(H)  # 한도 1 → 1요청 후 저장하고 끝
    assert len(log) == 1
    assert naver.calls("trend") == (1, 1)
    assert len(store.read(naver.TREND_DS, "ALL")["keyword"].unique()) == 5
    monkeypatch.setattr(naver, "config", lambda: CFG)
    naver.collect_trend(H)  # 나머지만
    assert len(log) == 4


def test_trend_refresh_waits_until_yesterday_is_published(monkeypatch):
    _universe(2)
    _today(monkeypatch, "2017-03-10")

    def h(req):
        b = json.loads(req.content)
        if b["endDate"] == "2017-03-09":  # 어제 값이 아직 없음
            b["endDate"] = "2017-03-08"
            req = httpx.Request("POST", req.url, json=b)
        return _trend_handler(req)
    _mock(monkeypatch, h)
    naver.collect_trend(H)
    df = store.read(naver.TREND_DS, "ALL")
    assert set(df["win_end"]) == {"2016-12-31"}  # 갱신 창은 저장하지 않고 다음 실행에


@pytest.mark.parametrize("status,body", [
    (401, {"error": {"errorCode": "200", "message": "Authentication Failed"}}),
    (402, {"error": "x"}),
    (400, {"errMsg": "유료 전환 안내: 결제 수단 등록 필요"}),
    (429, {"error": "Too Many Requests"}),
])
def test_stop_on_billing_auth_or_quota(monkeypatch, status, body):
    _universe(2)
    _today(monkeypatch, "2017-03-10")
    log = _mock(monkeypatch, lambda req: httpx.Response(status, json=body))
    with pytest.raises(naver.NaverStop):
        naver.collect_trend(H)
    assert len(log) == (3 if status == 429 else 1)


def test_bad_request_is_code_error(monkeypatch):
    _mock(monkeypatch, lambda req: httpx.Response(400, json={"errMsg": "TypeError: Invalid Date Range"}))
    with pytest.raises(ValueError):
        naver._trend_call(["a"], pd.Timestamp("2015-01-01"), pd.Timestamp("2015-02-01"), H)


def test_keys_missing_skips(monkeypatch):
    monkeypatch.setattr(naver, "keys", lambda: None)
    log = _mock(monkeypatch, _trend_handler)
    naver.run_trend()
    naver.run_news()
    assert log == []


# ── 뉴스 ──
def _item(i: int, minute: pd.Timestamp) -> dict:
    return {"title": f"<b>테마주</b> 기사 {i} &amp; 특징", "originallink": f"https://press{i % 3}.co.kr/a/{i}",
            "link": f"https://n.news.naver.com/{i}", "description": f"요약 <b>{i}</b>",
            "pubDate": minute.strftime("%a, %d %b %Y %H:%M:%S +0900")}


def _news_handler(articles: list[dict]):
    def h(req):
        start, display = int(req.url.params["start"]), int(req.url.params["display"])
        assert req.url.params["sort"] == "date"
        if not 1 <= start <= 1000:  # API HUB 문서와 2026-11-19 실제 응답
            return httpx.Response(400, json={"errorMessage": "Invalid start value (부적절한 start 값입니다.)",
                                             "errorCode": "SE03"})
        if not 1 <= display <= 100:
            return httpx.Response(400, json={"errorMessage": "Invalid display value", "errorCode": "SE02"})
        page = articles[start - 1:start - 1 + 100]
        return httpx.Response(200, json={"lastBuildDate": "x", "total": len(articles), "start": start,
                                         "display": len(page), "items": page})
    return h


def test_news_paging_dedup_and_resume(monkeypatch):
    t0 = pd.Timestamp("2026-11-18 17:18")
    arts = [_item(i, t0 - pd.Timedelta(minutes=i)) for i in range(250)]  # 최신순
    log = _mock(monkeypatch, _news_handler(arts))
    naver.collect_news(H)
    assert len(log) == 3  # 100 + 100 + 50
    df = store.read(naver.NEWS_DS, "ALL")
    assert len(df) == 250 and df["link"].is_unique
    r = df.sort_values("ts").iloc[-1]
    assert r["title"] == "테마주 기사 0 & 특징" and r["summary"] == "요약 0"
    assert r["ts"] == pd.Timestamp("2026-11-18 08:18", tz="UTC")  # 17:18 KST, 분 단위 그대로
    assert r["domain"] == "press0.co.kr" and r["query"] == "테마주" and r["query_kind"] == "theme"
    # 새 기사 120건이 올라옴 → 지난번 마지막 기사보다 오래된 기사가 나오는 페이지에서 멈춘다
    arts2 = [_item(1000 + i, t0 + pd.Timedelta(minutes=120 - i)) for i in range(120)] + arts
    log2 = _mock(monkeypatch, _news_handler(arts2))
    naver.collect_news(H)
    assert len(log2) == 2
    df = store.read(naver.NEWS_DS, "ALL")
    assert len(df) == 370 and df["link"].is_unique


def test_page_starts_respect_api_cap():
    assert naver.page_starts(3) == [1, 101, 201]
    assert naver.page_starts(11) == [1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1000]
    assert naver.page_starts(20) == naver.page_starts(11)  # 상한에 닿으면 더 넘기지 않는다
    assert max(naver.page_starts(50)) <= 1000


def test_news_deep_paging_stays_within_start_cap(monkeypatch):
    """2026-11-19: 11쪽째 start=1001 → 400 SE03으로 전체 실행이 죽었다. 이제 1,099건까지 받고 400 없이 끝난다."""
    cfg = {"naver": {**CFG["naver"], "news": {**CFG["naver"]["news"], "theme_max_pages": 11}}}
    monkeypatch.setattr(naver, "config", lambda: cfg)
    t0 = pd.Timestamp("2026-11-18 17:18")
    arts = [_item(i, t0 - pd.Timedelta(minutes=i)) for i in range(1500)]
    log = _mock(monkeypatch, _news_handler(arts))
    n, failed = naver.collect_news(H)
    assert failed == [] and [int(r.url.params["start"]) for r in log][-2:] == [901, 1000]
    assert n == 1099 and len(store.read(naver.NEWS_DS, "ALL")) == 1099


def test_news_one_query_failure_does_not_kill_run(monkeypatch):
    """한 질의가 400이면 그 질의만 건너뛰고, 실행 기록은 partial, 실패 질의는 다음 실행이 처음부터 다시."""
    cfg = {"naver": {**CFG["naver"], "news": {**CFG["naver"]["news"], "theme_queries": ["나쁨", "좋음"]}}}
    monkeypatch.setattr(naver, "config", lambda: cfg)
    monkeypatch.setattr(naver, "keys", lambda: H)
    _today(monkeypatch, "2026-11-18")
    monkeypatch.setattr(naver, "datetime", _FixedDT("2026-11-18"))
    t0 = pd.Timestamp("2026-11-18 17:18")
    good = _news_handler([_item(i, t0 - pd.Timedelta(minutes=i)) for i in range(150)])

    def h(req):
        if req.url.params["query"] == "나쁨" and req.url.params["start"] != "1":  # 2쪽째에서 400
            return httpx.Response(400, json={"errorMessage": "Invalid start value", "errorCode": "SE03"})
        return good(req)
    log = _mock(monkeypatch, h)
    naver.run_news()  # 예외 없이 끝난다
    df = store.read(naver.NEWS_DS, "ALL")
    assert set(df["query"]) == {"나쁨", "좋음"}  # 실패 질의도 받은 쪽(100건)은 저장
    assert (df["query"] == "나쁨").sum() == 100 and (df["query"] == "좋음").sum() == 150
    r = naver.runs(naver.NEWS_DS)
    assert r["status"].tolist() == ["partial"] and r["rows"].tolist() == [250] and r["calls"].tolist() == [len(log)]
    assert "나쁨" in r["error"].iloc[0] and "SE03" in r["error"].iloc[0]
    g = naver.gaps(naver.NEWS_DS, until="2026-11-18")
    assert g["problem"].tolist() == ["partial"] and g["rows"].tolist() == [250]
    with naver._ledger() as con:
        marked = {q for (q,) in con.execute("SELECT query FROM naver_news_query")}
    assert marked == {"좋음"}  # 실패 질의는 마지막 기사 시각을 올리지 않는다


def test_news_all_queries_fail_is_error(monkeypatch):
    monkeypatch.setattr(naver, "keys", lambda: H)
    _today(monkeypatch, "2026-11-18")
    monkeypatch.setattr(naver, "datetime", _FixedDT("2026-11-18"))
    _mock(monkeypatch, lambda req: httpx.Response(400, json={"errorCode": "SE01"}))
    with pytest.raises(RuntimeError, match="모두 실패"):
        naver.run_news()
    assert naver.runs(naver.NEWS_DS)["status"].tolist() == ["error"]
    assert naver.gaps(naver.NEWS_DS, until="2026-11-18")["problem"].tolist() == ["failed"]


def test_news_surge_queries(monkeypatch):
    kst = lambda d: pd.Timestamp(d, tz="Asia/Seoul").tz_convert("UTC")
    krx = pd.DataFrame({"ts": [kst("2026-11-16")] * 3 + [kst("2026-11-17")] * 3,
                        "symbol": ["A", "B", "C"] * 2, "name": ["가나", "다라", "마바"] * 2,
                        "close": [100, 100, 100, 130, 116, 105]})
    store.write("krx_1d", "ALL", krx, key=["ts", "symbol"])
    q = naver.surge_queries(15, 5)
    assert q["query"].tolist() == ["가나", "다라"] and q["symbol"].tolist() == ["A", "B"]
    seen = []

    def h(req):
        seen.append(req.url.params["query"])
        return httpx.Response(200, json={"items": [_item(len(seen), pd.Timestamp("2026-11-18 09:00"))]})
    _mock(monkeypatch, h)
    naver.collect_news(H)
    assert seen == ["테마주", "가나", "다라"]
    df = store.read(naver.NEWS_DS, "ALL")
    assert df.loc[df["query"] == "가나", ["query_kind", "symbol"]].values.tolist() == [["stock", "A"]]


def test_news_billing_stop_keeps_fetched(monkeypatch):
    calls = []

    def h(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json={"items": [_item(1, pd.Timestamp("2026-11-18 09:00"))]})
        return httpx.Response(403, json={"error": {"message": "subscription required"}})
    cfg = {"naver": {**CFG["naver"], "news": {**CFG["naver"]["news"], "theme_queries": ["a", "b"]}}}
    monkeypatch.setattr(naver, "config", lambda: cfg)
    monkeypatch.setattr(naver, "keys", lambda: H)
    _mock(monkeypatch, h)
    with pytest.raises(naver.NaverStop):
        naver.run_news()
    assert len(store.read(naver.NEWS_DS, "ALL")) == 1  # 멈추기 전에 받은 것은 저장
    assert store.get_state(naver.NEWS_DS, "ALL")["status"] == "error"


# ── 수집 기록·빠진 날 ──
def test_run_log_and_gaps(monkeypatch):
    monkeypatch.setattr(naver, "keys", lambda: H)
    t0 = pd.Timestamp("2026-11-18 17:18")
    _mock(monkeypatch, _news_handler([_item(i, t0 - pd.Timedelta(minutes=i)) for i in range(50)]))
    for day in ["2026-11-16", "2026-11-18"]:  # 11-17(화) 빠짐
        _today(monkeypatch, day)
        monkeypatch.setattr(naver, "datetime", _FixedDT(day))
        naver.run_news()
    r = naver.runs(naver.NEWS_DS)
    assert r["kst_date"].tolist() == ["2026-11-16", "2026-11-18"] and r["status"].tolist() == ["ok", "ok"]
    assert r["rows"].tolist() == [50, 0] and r["calls"].tolist() == [1, 1]
    g = naver.gaps(naver.NEWS_DS, until="2026-11-18").set_index("date")
    assert g.loc["2026-11-17", "problem"] == "missing" and g.loc["2026-11-16", "problem"] == ""
    assert g.loc["2026-11-18", "problem"] == "empty"  # 성공했지만 새 기사 0건


def test_run_log_records_error(monkeypatch):
    monkeypatch.setattr(naver, "keys", lambda: H)
    _today(monkeypatch, "2026-11-18")
    monkeypatch.setattr(naver, "datetime", _FixedDT("2026-11-18"))
    _mock(monkeypatch, lambda req: httpx.Response(401, json={"error": {"message": "Authentication Failed"}}))
    with pytest.raises(naver.NaverStop):
        naver.run_news()
    r = naver.runs(naver.NEWS_DS)
    assert r["status"].tolist() == ["error"] and "NaverStop" in r["error"].iloc[0]
    assert naver.gaps(naver.NEWS_DS, until="2026-11-18")["problem"].tolist() == ["failed"]


class _FixedDT:
    """naver.datetime 대체: now()만 그날 09:00 KST로 고정."""

    def __init__(self, day: str):
        self._t = pd.Timestamp(f"{day} 09:00", tz="Asia/Seoul").tz_convert("UTC").to_pydatetime()

    def now(self, tz=None):
        return self._t


# ── 뉴스 과거 백필 ──
BF = {"enabled": True, "templates": ["{name} 특징주"], "max_calls_per_day": 1000, "max_calls_per_month": 5000,
      "search_day_ceiling": 2000, "max_attempts": 2}


def _krx(rows):
    """rows: (KST 날짜, 종목코드, 이름). krx_1d를 임시 data에 쓴다."""
    df = pd.DataFrame(rows, columns=["day", "symbol", "name"])
    df["ts"] = pd.to_datetime(df["day"]).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    store.write("krx_1d", "ALL", df.drop(columns="day").assign(close=1.0), key=["ts", "symbol"])


def _bf_cfg(monkeypatch, **kw):
    cfg = {"naver": {**CFG["naver"], "news_backfill": {**BF, **kw}}}
    monkeypatch.setattr(naver, "config", lambda: cfg)


def _by_query(arts: dict[str, int]):
    """질의별 기사 수 → 그 수만큼 최신순 기사 (total = 그 수)."""
    t0 = pd.Timestamp("2026-11-18 17:18")
    pools = {q: [_item(hash(q) % 1000 * 10_000 + i, t0 - pd.Timedelta(days=i)) for i in range(k)]
             for q, k in arts.items()}

    def h(req):
        return _news_handler(pools.get(req.url.params["query"], []))(req)
    return h


def test_name_history_old_names_and_shared_names():
    _krx([("2011-01-03", "000040", "옛이름"), ("2014-04-17", "000040", "새이름"), ("2020-01-02", "000040", "새이름"),
          ("2012-01-02", "123450", "같은이름"), ("2019-01-02", "999990", "같은이름"), ("2012-01-02", "000045", "새이름우")])
    h = naver.name_history().set_index("name")
    assert h.loc["옛이름", "symbols"] == "000040" and h.loc["옛이름", "last"] == "2011-01-03"
    assert h.loc["새이름", "first"] == "2014-04-17" and h.loc["새이름", "last"] == "2020-01-02"
    assert h.loc["같은이름", "symbols"] == "123450|999990"  # 같은 이름은 질의 하나
    assert "새이름우" in h.index  # 우선주도 대상


def test_backfill_all_pages_total_truncation_and_resume(monkeypatch):
    _bf_cfg(monkeypatch)
    _krx([("2011-01-03", "000040", "작은"), ("2015-01-05", "000040", "큰"), ("2012-01-02", "000050", "없음")])
    log = _mock(monkeypatch, _by_query({"작은 특징주": 250, "큰 특징주": 1500, "없음 특징주": 0}))
    n, failed = naver.collect_backfill(H)
    assert failed == []
    starts = {}
    for r in log:
        starts.setdefault(r.url.params["query"], []).append(int(r.url.params["start"]))
    assert starts["작은 특징주"] == [1, 101, 201] and starts["없음 특징주"] == [1]
    assert starts["큰 특징주"] == naver.page_starts(11)  # start ≤ 1000
    assert all(int(r.url.params["display"]) == 100 and r.url.params["sort"] == "date" for r in log)
    q = naver.backfill_queries().set_index("query")
    assert q.loc["작은 특징주", "total"] == 250 and q.loc["작은 특징주", "got"] == 250
    assert q.loc["작은 특징주", "truncated"] == 0 and q.loc["큰 특징주", "truncated"] == 1
    assert q.loc["큰 특징주", "got"] == 1100 and q.loc["없음 특징주", "status"] == "done"
    df = store.read(naver.BACKFILL_DS, "ALL")
    assert n == len(df) == 250 + 1099  # 1,000번째 쪽 1건 겹침 제거
    assert set(df["query_kind"]) == {"backfill"} and set(df.loc[df["query"] == "큰 특징주", "symbol"]) == {"000040"}
    assert {"ts", "title", "summary", "link", "originallink", "domain", "collected_at"} <= set(df.columns)
    assert naver.calls("news_backfill")[0] == len(log) and naver.calls("news")[0] == 0  # 장부 따로
    # 이어받기: 끝난 질의는 다시 부르지 않는다
    log2 = _mock(monkeypatch, _by_query({}))
    assert naver.collect_backfill(H) == (0, []) and log2 == []


def test_backfill_daily_cap_stops_and_resumes(monkeypatch):
    _bf_cfg(monkeypatch, max_calls_per_day=4)
    _today(monkeypatch, "2026-11-18")
    _krx([("2011-01-03", f"00{i}000", f"종목{i}") for i in range(3)])
    _mock(monkeypatch, _by_query({f"종목{i} 특징주": 250 for i in range(3)}))
    naver.collect_backfill(H)
    q = naver.backfill_queries()
    assert len(q) == 1 and naver.calls("news_backfill")[0] == 4  # 3쪽짜리 1개 끝, 2번째는 중간에 멈춤(저장 안 함)
    _today(monkeypatch, "2026-11-19")
    naver.collect_backfill(H)
    _today(monkeypatch, "2026-11-20")
    naver.collect_backfill(H)
    q = naver.backfill_queries()
    assert len(q) == 3 and (q["status"] == "done").all() and (q["got"] == 250).all()


def test_backfill_ceiling_protects_live(monkeypatch):
    """오늘 live가 이미 많이 썼으면 백필은 천장에서 멈춘다 (live 몫 보호)."""
    _bf_cfg(monkeypatch, search_day_ceiling=5)
    _today(monkeypatch, "2026-11-18")
    for _ in range(4):
        naver._count("news")
    _krx([("2011-01-03", "000040", "작은")])
    log = _mock(monkeypatch, _by_query({"작은 특징주": 250}))
    naver.collect_backfill(H)
    assert len(log) == 1 and naver.backfill_queries().empty


def test_backfill_failed_query_retries_up_to_max(monkeypatch):
    _bf_cfg(monkeypatch)
    _krx([("2011-01-03", "000040", "나쁨"), ("2011-01-03", "000050", "좋음")])
    good = _by_query({"좋음 특징주": 50})

    def h(req):
        if req.url.params["query"] == "나쁨 특징주":
            return httpx.Response(400, json={"errorMessage": "Invalid query", "errorCode": "SE01"})
        return good(req)
    log = _mock(monkeypatch, h)
    n, failed = naver.collect_backfill(H)
    assert n == 50 and len(failed) == 1 and failed[0].startswith("나쁨 특징주")
    naver.collect_backfill(H)
    naver.collect_backfill(H)  # max_attempts=2 → 세 번째에는 부르지 않는다
    assert sum(r.url.params["query"] == "나쁨 특징주" for r in log) == 2
    q = naver.backfill_queries().set_index("query")
    assert q.loc["나쁨 특징주", "status"] == "error" and q.loc["나쁨 특징주", "attempts"] == 2


def test_backfill_billing_stop(monkeypatch):
    _bf_cfg(monkeypatch)
    monkeypatch.setattr(naver, "keys", lambda: H)
    _krx([("2011-01-03", "000040", "작은")])
    _mock(monkeypatch, lambda req: httpx.Response(402, text="유료 전환 안내: 결제 수단을 등록하세요"))
    with pytest.raises(naver.NaverStop):
        naver.run_news_backfill()
    assert naver.runs(naver.BACKFILL_DS)["status"].tolist() == ["error"]
    assert store.get_state(naver.BACKFILL_DS, "ALL")["status"] == "error"


def test_run_news_does_live_first_then_backfill(monkeypatch):
    _bf_cfg(monkeypatch)
    monkeypatch.setattr(naver, "keys", lambda: H)
    _krx([("2011-01-03", "000040", "작은")])
    log = _mock(monkeypatch, _by_query({"테마주": 30, "작은 특징주": 20}))
    naver.run_news()
    assert [r.url.params["query"] for r in log] == ["테마주", "작은 특징주"]
    assert naver.runs()["dataset"].tolist() == [naver.NEWS_DS, naver.BACKFILL_DS]
    # live가 멈추면 백필은 하지 않는다
    log2 = _mock(monkeypatch, lambda req: httpx.Response(401, json={"error": {"message": "Authentication Failed"}}))
    with pytest.raises(naver.NaverStop):
        naver.run_news()
    assert all(r.url.params["query"] == "테마주" for r in log2)


def test_backfill_budget_in_real_config_protects_live():
    """실제 config: live 상한 + 백필 상한 ≤ 천장 < 공식 하루 한도 25,000."""
    from autotrader.data.collectors.common import config as real_config
    c = real_config()["naver"]
    bf = c["news_backfill"]
    assert c["news"]["max_calls_per_day"] + bf["max_calls_per_day"] <= bf["search_day_ceiling"]
    assert bf["search_day_ceiling"] + c["news"]["max_calls_per_day"] <= 25_000
