"""네이버 데이터 (NAVER API HUB, 네이버 클라우드 중개): 검색어 트렌드(`naver_trend`), 뉴스 앞으로 수집(`naver_news_live`).

인증: `.env`의 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET → 헤더 X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY.
      (옛 NAVER Developers의 X-Naver-Client-Id 방식이 아니다.) 키가 없으면 건너뛴다.
엔드포인트 (API HUB 문서 2026-09-17판):
  - 검색어 트렌드: POST https://naverapihub.apigw.ntruss.com/search-trend/v1/search  (월 5만 건)
  - 뉴스 검색:     GET  https://naverapihub.apigw.ntruss.com/search/v1/news           (문서마다 하루 2.5만 / 월 77.5만)
  키당 50회/초. "한시적 무료, 유료 전환 시 사전 공지" → 과금·한도 관련 응답이면 즉시 멈춘다(NaverStop).

호출 한도: `config/data.yaml`의 naver.{trend,news}.max_calls_per_day / max_calls_per_month.
  호출 수는 실제 HTTP 요청(재시도 포함)마다 state.sqlite의 api_calls 표에 KST 날짜별로 센다. 한도에 닿으면 저장하고
  끝낸다(다음 실행에 이어받기).

── naver_trend (symbol=ALL) ──
  행 = 키워드 × 날짜 × 요청 창. 열: ts(날짜 KST 0시 → UTC), keyword, symbol(종목코드, 추가 키워드는 ""), ratio,
  win_start·win_end(요청 창, 'YYYY-MM-DD'), group(같은 요청에 들어간 키워드, '|'로 연결), collected_at(UTC).
  - ratio는 **같은 요청의 모든 키워드·날짜 중 최댓값 = 100** 인 상대값이다 (2026-11-18 샘플: 그룹 단독/합동 비가 상수,
    겹치는 두 창의 비도 상수 1.0000 → 창이 달라도 값은 상수배 차이뿐, 과거 값 재계산 없음).
  - 창: 2016-01-01부터 1월 1일·7월 1일에 시작하는 1년 창 (반씩 겹침). 끝난 창은 한 번만 받는다.
  - 아직 안 끝난 창 중 가장 먼저 시작한 창은 매일 [창 시작, 어제]로 다시 받는다 (그날까지 끝난 창).
    이때 처음 받는 날은 창 전체, 그 뒤로는 마지막 `refresh_tail_days`일만 저장한다 (전날 받은 것과 겹쳐 이을 수 있게).
  - 키워드: kr_daily 유니버스의 보통주(코드 끝자리 0) 중 2016년 이후 상장 중이던 종목명 + naver.trend.extra_keywords.

── naver_news_live (symbol=ALL) ──
  행 = 기사 × 수집 질의. 열: ts(pubDate, 분 단위 → UTC), title·summary(태그 제거한 제목·요약), link, originallink,
  domain(originallink의 호스트 = 언론사 대신), query, query_kind(theme|stock), symbol(stock 질의의 종목코드),
  collected_at(UTC). 키 = (link, query). 본문은 받지 않는다. 저작권 대상이라 data/ 밖(git·외부)에 올리지 않는다.
  - 질의: naver.news.theme_queries + 최근 거래일 krx_1d에서 naver.news.surge_pct 이상 오른 종목명(상한가 포함).
  - 최신순(sort=date)으로 100건씩 넘기다가, 그 질의에서 지난번에 받은 가장 늦은 기사 시각보다 오래된 기사가 나오면 멈춘다.
    API HUB 문서(api.ncloud-docs.com/docs/naver-api-hub-search-news): start 1~1000, display 1~100 → 범위를 넘으면
    400 SE03/SE02. 그래서 쪽 시작 위치는 1, 101, …, 901, 1000 (마지막 쪽은 1,000번째부터, 1건 겹침) — 질의당
    최신 1,099건까지. 그 안에서 못 따라잡으면 로그에 '누락 가능'을 남긴다.
    (2026-11-19 첫 실행은 11번째 쪽을 start=1001로 불러 400으로 전체가 죽었다.)
  - 한 질의가 실패해도(400 등) 그 질의만 건너뛰고 다음 질의로 간다. 실패한 질의는 받은 쪽까지 저장하되 '마지막 기사
    시각'을 올리지 않아 다음 실행이 처음부터 다시 받는다. 실행 기록 status는 'partial'. 과금·인증(NaverStop)과
    호출 한도(DailyLimit)는 질의가 아니라 실행 전체를 멈춘다. 질의가 모두 실패하면 오류로 멈춘다(알림).
"""

from __future__ import annotations

import html
import os
import re
import sqlite3
import time
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
import pandas as pd
from dotenv import dotenv_values

from .. import store
from ..redact import redact
from .common import RateLimiter, config, logger

KST = "Asia/Seoul"
TREND_URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
TREND_DS, NEWS_DS, SYMBOL = "naver_trend", "naver_news_live", "ALL"
TREND_START = pd.Timestamp("2016-01-01")
# 200이 아닌 응답 본문에 이런 말이 있으면 과금 문제로 보고 즉시 멈춘다. 한도 초과(429)는 재시도 후 반복되면 멈춘다.
BILLING_WORDS = ("유료", "과금", "요금", "결제", "청구", "billing", "payment", "charge", "subscription", "credit")

_http = httpx.Client(timeout=30)
_limit = RateLimiter(10)  # 키당 50회/초 한도보다 크게 여유
_sleep = time.sleep  # 테스트에서 바꿔 끼우는 지점


class NaverStop(RuntimeError):
    """과금·인증·한도 관련 응답 → 수집을 멈추고 사람에게 알린다 (cli가 텔레그램으로 알림)."""


class DailyLimit(RuntimeError):
    """config의 하루·월 호출 한도에 닿음 → 저장하고 끝낸다 (다음 실행에 이어받기)."""


# ── 키 ──
def keys() -> dict | None:
    cid, sec = os.environ.get("NAVER_CLIENT_ID", "").strip(), os.environ.get("NAVER_CLIENT_SECRET", "").strip()
    main_env = store.ROOT.parents[1].parent / ".env"  # git worktree(<메인>/.claude/worktrees/<이름>)면 메인 .env
    if not (cid and sec) and store.ROOT.parent.name == "worktrees" and main_env.is_file():
        v = dotenv_values(main_env)
        cid, sec = (v.get("NAVER_CLIENT_ID") or "").strip(), (v.get("NAVER_CLIENT_SECRET") or "").strip()
    if not (cid and sec):
        return None
    return {"X-NCP-APIGW-API-KEY-ID": cid, "X-NCP-APIGW-API-KEY": sec}


# ── 호출 수 장부 (state.sqlite의 api_calls 표) ──
def _ledger() -> sqlite3.Connection:
    store.STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(store.STATE_DB, timeout=60)
    con.execute("CREATE TABLE IF NOT EXISTS api_calls (day TEXT, api TEXT, n INTEGER, PRIMARY KEY (day, api))")
    con.execute("CREATE TABLE IF NOT EXISTS naver_trend_done (keyword TEXT, win_start TEXT, win_end TEXT,"
                " PRIMARY KEY (keyword, win_start, win_end))")
    con.execute("CREATE TABLE IF NOT EXISTS naver_news_query (query TEXT PRIMARY KEY, last_pub TEXT, last_run TEXT)")
    return con


def _kst_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=KST).normalize().tz_localize(None)


def calls(api: str) -> tuple[int, int]:
    """(오늘, 이번 달) 호출 수. 날짜는 KST."""
    day = _kst_today()
    with _ledger() as con:
        d = con.execute("SELECT COALESCE(SUM(n),0) FROM api_calls WHERE day=? AND api=?",
                        (f"{day:%Y-%m-%d}", api)).fetchone()[0]
        m = con.execute("SELECT COALESCE(SUM(n),0) FROM api_calls WHERE day LIKE ? AND api=?",
                        (f"{day:%Y-%m}-%", api)).fetchone()[0]
    return int(d), int(m)


def _count(api: str) -> None:
    with _ledger() as con:
        con.execute("INSERT INTO api_calls VALUES (?,?,1) ON CONFLICT(day, api) DO UPDATE SET n = n + 1",
                    (f"{_kst_today():%Y-%m-%d}", api))


def _check_limit(api: str) -> None:
    c = config()["naver"][api]
    d, m = calls(api)
    if d >= c["max_calls_per_day"] or m >= c["max_calls_per_month"]:
        raise DailyLimit(f"naver {api} 호출 한도 도달 (오늘 {d:,}/{c['max_calls_per_day']:,},"
                         f" 이번 달 {m:,}/{c['max_calls_per_month']:,})")


def _request(api: str, method: str, url: str, headers: dict, **kw) -> dict:
    """한도 확인 → 호출 수 기록 → 요청. 429·5xx·네트워크 오류는 3번까지 재시도.
    과금·인증·권한 관련(401·402·403, 본문에 BILLING_WORDS, 429 반복)은 NaverStop."""
    last = ""
    for i in range(3):
        _check_limit(api)
        _limit.wait()
        _count(api)
        try:
            r = _http.request(method, url, headers=headers, **kw)
        except httpx.TransportError as e:
            last = f"{type(e).__name__}"
            _sleep(2 ** i)
            continue
        if r.status_code == 200:
            return r.json()
        body = r.text[:500]
        if r.status_code in (401, 402, 403) or any(w in body.lower() for w in BILLING_WORDS):
            raise NaverStop(f"naver {api} HTTP {r.status_code}: {body}")
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}: {body}"
            _sleep(2 ** i)
            continue
        raise ValueError(f"naver {api} HTTP {r.status_code}: {body}")  # 400: 요청 형식 오류 → 코드 문제
    if last.startswith("HTTP 429"):
        raise NaverStop(f"naver {api} 429 반복 (월 한도 소진 또는 과금 상태 의심): {last}")
    raise RuntimeError(f"naver {api} 요청 실패: {last}")


# ── 검색어 트렌드 ──
def windows(today: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, bool]]:
    """(시작, 끝, 끝났나). 1월 1일·7월 1일 시작 1년 창. 끝이 어제 이후인 창은 끝을 어제로 자른다.
    안 끝난 창은 가장 먼저 시작한 것 하나만 돌려준다 (매일 다시 받는 창)."""
    yday = today - pd.Timedelta(days=1)
    out, s = [], TREND_START
    while s <= yday:
        e = s + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        if e <= yday:
            out.append((s, e, True))
        else:
            out.append((s, yday, False))
            break  # 더 늦게 시작한 안 끝난 창은 받지 않는다
        s = s + pd.DateOffset(months=6)
    return out


def trend_keywords() -> pd.DataFrame:
    """keyword, symbol. 종목명(보통주, 2016년 이후 상장 중이던 것) + 추가 키워드."""
    u = store.load_universe("kr_daily", max_age_days=100_000)
    rows = []
    if not u.empty:
        u = u[u["code"].astype(str).str.endswith("0")]
        u = u[u["delisted"].isna() | (pd.to_datetime(u["delisted"]) >= TREND_START)]
        u = u.sort_values("code")
        rows = list(zip(u["name"].astype(str).str.strip(), u["code"].astype(str), strict=True))
    rows += [(k, "") for k in config()["naver"]["trend"].get("extra_keywords") or []]
    df = pd.DataFrame(rows, columns=["keyword", "symbol"])
    return df[df["keyword"] != ""].drop_duplicates("keyword").reset_index(drop=True)


def _trend_call(keywords: list[str], start: pd.Timestamp, end: pd.Timestamp, headers: dict) -> pd.DataFrame:
    body = {"startDate": f"{start:%Y-%m-%d}", "endDate": f"{end:%Y-%m-%d}", "timeUnit": "date",
            "keywordGroups": [{"groupName": k, "keywords": [k]} for k in keywords]}
    j = _request("trend", "POST", TREND_URL, headers, json=body)
    rows = [(r["title"], d["period"], d["ratio"]) for r in j.get("results") or [] for d in r.get("data") or []]
    return pd.DataFrame(rows, columns=["keyword", "date", "ratio"])


def _trend_window(kw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, closed: bool, headers: dict) -> int:
    """한 창에서 아직 안 받은 키워드를 5개씩 받는다. 200요청마다 저장하고 받은 키워드를 표시한다."""
    ws, we = f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"
    with _ledger() as con:
        done = {k for (k,) in con.execute("SELECT keyword FROM naver_trend_done WHERE win_start=? AND win_end=?",
                                          (ws, we))}
        seen = set() if closed else {k for (k,) in con.execute(
            "SELECT DISTINCT keyword FROM naver_trend_done WHERE win_start=?", (ws,))}
    todo = kw[~kw["keyword"].isin(done)]
    if todo.empty:
        return 0
    sym = dict(zip(kw["keyword"], kw["symbol"], strict=True))
    tail = int(config()["naver"]["trend"].get("refresh_tail_days", 14))
    logger.info(f"{TREND_DS} 창 {ws}~{we}{'' if closed else ' (매일 갱신)'}: {len(todo):,}개 키워드")
    names = todo["keyword"].tolist()
    buf, got, n = [], [], 0

    def flush() -> None:
        nonlocal buf, got, n
        if buf:
            n += store.write(TREND_DS, SYMBOL, pd.concat(buf), key=["ts", "keyword", "win_start", "win_end"])
        with _ledger() as con:
            con.executemany("INSERT OR IGNORE INTO naver_trend_done VALUES (?,?,?)", [(k, ws, we) for k in got])
        buf, got = [], []

    try:
        for i in range(0, len(names), 5):
            group = names[i:i + 5]
            df = _trend_call(group, start, end, headers)
            if not closed and not df.empty and df["date"].max() < we:
                logger.info(f"{TREND_DS}: {we} 값이 아직 없음 → 다음 실행에 다시")
                break
            if not df.empty:
                df["ts"] = pd.to_datetime(df["date"]).dt.tz_localize(KST).dt.tz_convert("UTC")
                if not closed:  # 이미 이 창을 받은 키워드는 끝부분만 (앞부분은 상수배 차이뿐)
                    cut = f"{end - pd.Timedelta(days=tail - 1):%Y-%m-%d}"
                    df = df[~df["keyword"].isin(seen) | (df["date"] >= cut)]
                df = df.assign(symbol=df["keyword"].map(sym).fillna(""), win_start=ws, win_end=we,
                               group="|".join(group), collected_at=pd.Timestamp.now(tz="UTC").floor("s"))
                buf.append(df[["ts", "keyword", "symbol", "ratio", "win_start", "win_end", "group", "collected_at"]])
            got += group
            if len(got) >= 1000:
                flush()
    finally:
        flush()
    return n


def collect_trend(headers: dict) -> int:
    kw = trend_keywords()
    if kw.empty:
        logger.info(f"{TREND_DS}: 키워드 없음 (kr_daily 유니버스 없음) → 건너뜀")
        return 0
    wins = windows(_kst_today())
    # 매일 갱신 창 먼저 (그날 값), 그다음 끝난 창 백필 (최근 창부터)
    order = [w for w in wins if not w[2]] + sorted([w for w in wins if w[2]], key=lambda w: w[0], reverse=True)
    n = 0
    try:
        for s, e, closed in order:
            n += _trend_window(kw, s, e, closed, headers)
        store.set_state(TREND_DS, SYMBOL, backfill_done=True)
    except DailyLimit as e:
        logger.info(f"{e} → 다음 실행에 이어받기")
    d, m = calls("trend")
    logger.info(f"{TREND_DS}: {n:,}행 저장, 호출 오늘 {d:,} / 이번 달 {m:,}")
    return n


# ── 뉴스 앞으로 수집 ──
_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return html.unescape(_TAG.sub("", s or "")).strip()


NEWS_PAGE, NEWS_MAX_START = 100, 1000  # API HUB 문서: display 1~100, start 1~1000


def page_starts(max_pages: int) -> list[int]:
    """쪽 시작 위치. 1, 101, …을 start 상한(1000)에서 자르고, 상한에 닿은 쪽 다음은 없다."""
    out: list[int] = []
    for p in range(max_pages):
        s = min(1 + NEWS_PAGE * p, NEWS_MAX_START)
        if out and s <= out[-1]:
            break
        out.append(s)
    return out


def _news_page(query: str, start: int, headers: dict) -> tuple[pd.DataFrame, int]:
    j = _request("news", "GET", NEWS_URL, headers,
                 params={"query": query, "display": NEWS_PAGE, "start": start, "sort": "date"})
    items = j.get("items") or []
    if not items:
        return pd.DataFrame(), 0
    d = pd.DataFrame(items)
    out = pd.DataFrame({
        "ts": pd.to_datetime(d["pubDate"], format="%a, %d %b %Y %H:%M:%S %z").dt.tz_convert("UTC"),
        "title": d["title"].map(_clean), "summary": d["description"].map(_clean),
        "link": d["link"].fillna(""), "originallink": d["originallink"].fillna(""),
    })
    out["domain"] = [urlparse(o or l).hostname or "" for o, l in zip(out["originallink"], out["link"], strict=True)]
    return out, len(items)


def surge_queries(pct: float, max_n: int) -> pd.DataFrame:
    """krx_1d 최근 거래일에 pct% 이상 오른 종목 (query=종목명, symbol). 상승률 큰 순으로 max_n개."""
    files = sorted((store.DATA_DIR / "krx_1d" / SYMBOL).glob("*.parquet"))[-2:]
    if not files:
        return pd.DataFrame(columns=["query", "symbol"])
    d = pd.concat([pd.read_parquet(f, columns=["ts", "symbol", "name", "close"]) for f in files])
    days = sorted(d["ts"].unique())[-2:]
    if len(days) < 2:
        return pd.DataFrame(columns=["query", "symbol"])
    prev = d[d["ts"] == days[0]].set_index("symbol")["close"]
    cur = d[d["ts"] == days[1]].set_index("symbol")
    chg = (cur["close"] / prev.reindex(cur.index) - 1) * 100
    hit = cur.assign(chg=chg)[chg >= pct].sort_values("chg", ascending=False).head(max_n)
    logger.info(f"{NEWS_DS}: {pd.Timestamp(days[1]).tz_convert(KST):%Y-%m-%d} 급등 {len(hit)}종목 질의")
    return pd.DataFrame({"query": hit["name"].astype(str).str.strip(), "symbol": hit.index.astype(str)})


def _news_query(query: str, kind: str, symbol: str, max_pages: int, headers: dict) -> tuple[pd.DataFrame, str]:
    """(받은 기사, 오류). 오류가 나면 그때까지 받은 쪽과 오류 문구를 돌려준다 (NaverStop·DailyLimit은 그대로 올린다)."""
    with _ledger() as con:
        row = con.execute("SELECT last_pub FROM naver_news_query WHERE query=?", (query,)).fetchone()
    last = pd.Timestamp(row[0]) if row and row[0] else None
    parts, caught_up, err, starts = [], last is None, "", page_starts(max_pages)
    for start in starts:
        try:
            df, cnt = _news_page(query, start, headers)
        except (NaverStop, DailyLimit):
            raise
        except Exception as e:  # noqa: BLE001 — 400(요청 형식)·재시도 소진 등 → 이 질의만 건너뛴다
            err = f"{type(e).__name__}: {e}"
            logger.error(f"{NEWS_DS} '{query}' start={start} 실패 → 이 질의 건너뜀 (다음 실행에 처음부터): {err}")
            break
        if not df.empty:
            parts.append(df)
        if cnt < NEWS_PAGE:
            caught_up = True
            break
        if last is not None and df["ts"].min() < last:
            caught_up = True
            break
    if not caught_up and not err:
        logger.warning(f"{NEWS_DS} '{query}': 최신 {starts[-1] + NEWS_PAGE - 1:,}건 안에서 지난 수집({last})까지 못 감"
                       " → 누락 가능")
    if not parts:
        return pd.DataFrame(), err
    out = pd.concat(parts, ignore_index=True)
    return out.assign(query=query, query_kind=kind, symbol=symbol,
                      collected_at=pd.Timestamp.now(tz="UTC").floor("s")), err


def _mark_queries(df: pd.DataFrame) -> None:
    """저장한 뒤에 질의별 가장 늦은 기사 시각을 올린다 (저장 실패 시 다음 실행이 다시 받게)."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with _ledger() as con:
        for q, ts in df.groupby("query")["ts"].max().items():
            row = con.execute("SELECT last_pub FROM naver_news_query WHERE query=?", (q,)).fetchone()
            last = max(ts, pd.Timestamp(row[0])) if row and row[0] else ts
            con.execute("INSERT OR REPLACE INTO naver_news_query VALUES (?,?,?)", (q, last.isoformat(), now))


def _new_rows(df: pd.DataFrame) -> int:
    """이미 저장된 (link, query)를 뺀 새 행 수 (수집 기록의 '받은 건수')."""
    folder = store.DATA_DIR / NEWS_DS / SYMBOL
    old = set()
    for y in df["ts"].dt.strftime("%Y").unique():
        f = folder / f"{y}.parquet"
        if f.exists():
            o = pd.read_parquet(f, columns=["link", "query"])
            old |= set(zip(o["link"], o["query"], strict=True))
    return sum((k, q) not in old for k, q in zip(df["link"], df["query"], strict=True))


def collect_news(headers: dict) -> tuple[int, list[str]]:
    """(새로 저장한 행 수, 실패한 질의의 오류 목록 '질의: 오류'). 질의가 모두 실패하면 RuntimeError."""
    c = config()["naver"]["news"]
    qs = [(q, "theme", "") for q in c.get("theme_queries") or []]
    sq = surge_queries(float(c.get("surge_pct", 15)), int(c.get("max_stock_queries", 60)))
    qs += [(q, "stock", s) for q, s in zip(sq["query"], sq["symbol"], strict=True)]
    buf, n, failed = [], 0, {}
    try:
        for q, kind, sym in qs:
            pages = int(c.get("theme_max_pages", 11) if kind == "theme" else c.get("stock_max_pages", 2))
            df, err = _news_query(q, kind, sym, pages, headers)
            if err:
                failed[q] = err
            if not df.empty:
                buf.append(df)
    except DailyLimit as e:
        logger.info(f"{e} → 다음 실행에 이어받기")
    finally:
        if buf:
            df = pd.concat(buf, ignore_index=True).drop_duplicates(["link", "query"])
            n = _new_rows(df)
            store.write(NEWS_DS, SYMBOL, df, key=["link", "query"])
            _mark_queries(df[~df["query"].isin(failed)])  # 실패한 질의는 다음 실행이 처음부터
    d, m = calls("news")
    logger.info(f"{NEWS_DS}: {len(qs)}개 질의 (실패 {len(failed)}), 새 기사 {n:,}행, 호출 오늘 {d:,} / 이번 달 {m:,}")
    errs = [f"{q}: {e}" for q, e in failed.items()]
    if qs and len(failed) == len(qs):
        raise RuntimeError(f"{NEWS_DS} 질의 {len(qs)}개 모두 실패 — " + " | ".join(errs)[:400])
    return n, errs


# ── 수집 기록 (state.sqlite의 naver_runs 표) — 빠진 날을 나중에 찾기 위해 실행마다 1행 ──
API_OF = {TREND_DS: "trend", NEWS_DS: "news"}


def _record(ds: str, started: datetime, status: str, rows: int, calls_before: int, error: str = "") -> None:
    fin = datetime.now(UTC)
    with _ledger() as con:
        con.execute("CREATE TABLE IF NOT EXISTS naver_runs (dataset TEXT, started_at TEXT, finished_at TEXT,"
                    " kst_date TEXT, status TEXT, rows INTEGER, calls INTEGER, error TEXT)")
        con.execute("INSERT INTO naver_runs VALUES (?,?,?,?,?,?,?,?)", (
            ds, started.isoformat(timespec="seconds"), fin.isoformat(timespec="seconds"),
            f"{pd.Timestamp(started).tz_convert(KST):%Y-%m-%d}", status, int(rows),
            calls(API_OF[ds])[0] - calls_before if status != "skip" else 0, redact(error)[:500]))


def runs(ds: str | None = None) -> pd.DataFrame:
    """수집 기록. 열: dataset, started_at·finished_at(UTC), kst_date, status(ok·limit·partial·error·skip), rows, calls,
    error. status: ok = 끝까지 받음, limit = 호출 한도에 닿아 일부만(다음 실행에 이어받음), partial = 일부 질의가
    실패(error에 '질의: 오류', 실패 질의는 다음 실행이 처음부터), error = 오류로 멈춤, skip = 키 없음."""
    with _ledger() as con:
        con.execute("CREATE TABLE IF NOT EXISTS naver_runs (dataset TEXT, started_at TEXT, finished_at TEXT,"
                    " kst_date TEXT, status TEXT, rows INTEGER, calls INTEGER, error TEXT)")
        df = pd.read_sql("SELECT * FROM naver_runs" + (" WHERE dataset=?" if ds else "") + " ORDER BY started_at",
                         con, params=(ds,) if ds else None)
    return df


def gaps(ds: str = NEWS_DS, since: str | None = None, until: str | None = None) -> pd.DataFrame:
    """since~until(KST 날짜, 기본: 첫 수집일~오늘) 하루씩 수집 상태. 열: date, weekday, trading_day, runs, ok_runs,
    rows, problem. problem: 'missing'(실행 기록 없음), 'failed'(실행은 했으나 ok·limit·partial 없음), 'partial'(일부
    질의만 성공한 실행뿐 — ok·limit 실행이 없음), 'empty'(성공했지만 0행 — 뉴스는 거래일이면 의심), ''(정상).
    ok_runs는 ok·limit 실행 수, rows는 ok·limit·partial 실행의 새 행 합.
    trading_day는 krx_1d에 그날 행이 있으면 True, 주말 False, 그 밖은 None(아직 모름)."""
    r = runs(ds)
    if since is None:
        if r.empty:
            return pd.DataFrame(columns=["date", "weekday", "trading_day", "runs", "ok_runs", "rows", "problem"])
        since = r["kst_date"].min()
    days = pd.date_range(since, until or f"{_kst_today():%Y-%m-%d}")
    krx = set()
    for f in sorted((store.DATA_DIR / "krx_1d" / SYMBOL).glob("*.parquet"))[-2:]:
        krx |= {f"{t:%Y-%m-%d}" for t in pd.read_parquet(f, columns=["ts"])["ts"].dt.tz_convert(KST).unique()}
    last_krx = max(krx) if krx else ""
    out = []
    for d in days:
        k = f"{d:%Y-%m-%d}"
        g = r[r["kst_date"] == k]
        ok = g[g["status"].isin(["ok", "limit"])]
        part = g[g["status"] == "partial"]
        got = int(ok["rows"].sum() + part["rows"].sum())
        td = False if d.weekday() >= 5 else (k in krx if k <= last_krx else None)
        problem = ("missing" if g.empty else "failed" if ok.empty and part.empty else "partial" if ok.empty
                   else "empty" if got == 0 else "")
        out.append((k, d.day_name()[:3], td, len(g), len(ok), got, problem))
    return pd.DataFrame(out, columns=["date", "weekday", "trading_day", "runs", "ok_runs", "rows", "problem"])


def _run(ds: str, fn) -> None:
    started = datetime.now(UTC)
    headers = keys()
    if headers is None:
        logger.info(f"{ds}: NAVER_CLIENT_ID/NAVER_CLIENT_SECRET 없음 → 건너뜀")
        _record(ds, started, "skip", 0, 0, "키 없음")
        return
    api = API_OF[ds]
    before = calls(api)[0]
    try:
        res = fn(headers)  # 트렌드: 행 수, 뉴스: (행 수, 실패한 질의 오류 목록)
    except Exception as e:
        _record(ds, started, "error", 0, before, f"{type(e).__name__}: {e}")
        if isinstance(e, NaverStop):
            store.set_state(ds, SYMBOL, status="error", error=str(e)[:500])
            logger.error(f"{ds} 중단 (과금·인증·한도 의심, HQ 보고 필요): {e}")
        raise  # cli가 텔레그램으로 알린다
    c = config()["naver"][api]
    d, m = calls(api)
    hit = d >= c["max_calls_per_day"] or m >= c["max_calls_per_month"]
    n, failed = res if isinstance(res, tuple) else (res, [])
    if failed:
        logger.error(f"{ds}: 질의 {len(failed)}개 실패 (나머지는 저장) — 수집 기록 status=partial")
    _record(ds, started, "partial" if failed else "limit" if hit else "ok", n, before, " | ".join(failed))


def run_trend(symbols: list[str] | None = None) -> None:
    _run(TREND_DS, collect_trend)


def run_news(symbols: list[str] | None = None) -> None:
    _run(NEWS_DS, collect_news)

