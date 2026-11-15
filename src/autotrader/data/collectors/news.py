"""미국 주식·코인 뉴스 (Alpaca News API, 무료 Basic 플랜, Benzinga 기사).

- news (symbol=ALL): 기사 하나가 한 행. 컬럼 id, ts(기사 작성 시각 created_at, UTC), updated_at, symbols(쉼표로
  이은 티커, 코인은 BTCUSD 형태), headline, summary, source, url. 본문(content)은 받지 않는다.
- 범위: 2015년 이후. 무료 플랜도 과거 전체를 준다(한 페이지 최대 50건, 분당 200회).
- 키: 환경변수 ALPACA_API_KEY / ALPACA_SECRET_KEY. 없으면 건너뛴다.
- 이어받기: 마지막으로 받은 기사 시각부터 다시 받는다(같은 초 기사가 여럿일 수 있어 그 시각을 포함, id로 중복 제거).

시점 규칙: 뉴스는 ts(기사 시각) 이후에만 쓸 수 있다. 분봉 전략은 ts가 속한 봉의 다음 봉부터, 일봉 전략은
          장 마감 뒤 기사면 다음 거래일부터 쓴다. headline·summary는 나중에 고쳐질 수 있고 저장되는 건 받을 때의
          최신 값이라, updated_at이 ts보다 늦은 기사는 원래 문구와 다를 수 있다.
"""

from __future__ import annotations

import httpx
import pandas as pd

from .. import store
from .common import RateLimiter, get_json, logger
from .us_alpaca import keys

DATASET, SYMBOL = "news", "ALL"
API = "https://data.alpaca.markets/v1beta1/news"
START = pd.Timestamp("2015-01-01", tz="UTC")
COLUMNS = ["id", "ts", "updated_at", "symbols", "headline", "summary", "source", "url"]
_limit = RateLimiter(3)  # 분당 180회 (한도 200)
_http = httpx.Client(timeout=60)


def _page(start: pd.Timestamp, end: pd.Timestamp, headers: dict, page_token: str | None) -> tuple:
    _limit.wait()
    params = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "limit": 50, "sort": "asc", "include_content": "false"}
    if page_token:
        params["page_token"] = page_token
    j = get_json(_http, API, params, headers=headers)
    rows = j.get("news") or []
    if not rows:
        return pd.DataFrame(columns=COLUMNS), j.get("next_page_token")
    df = pd.DataFrame(rows)
    df = pd.DataFrame({
        "id": df["id"].astype("int64"), "ts": pd.to_datetime(df["created_at"], utc=True, format="ISO8601"),
        "updated_at": pd.to_datetime(df["updated_at"], utc=True, format="ISO8601"),
        "symbols": df["symbols"].map(lambda s: ",".join(s or [])),
        "headline": df["headline"].fillna(""), "summary": df["summary"].fillna(""),
        "source": df["source"].fillna(""), "url": df["url"].fillna(""),
    })
    return df, j.get("next_page_token")


def collect(headers: dict) -> int:
    """마지막 기사 시각부터 (처음이면 2015년부터) 지금까지 앞으로 받는다. 끊겨도 이어받는다."""
    st = store.get_state(DATASET, SYMBOL)
    start = st["last_ts"] if st["last_ts"] is not None else START
    end = pd.Timestamp.now(tz="UTC").floor("s")
    buf, n, token = [], 0, None
    try:
        while True:
            df, token = _page(start, end, headers, token)
            if not df.empty:
                buf.append(df)
            if sum(map(len, buf)) >= 20_000 or (not token and buf):  # 2만 건마다 저장 → 끊겨도 여기부터
                n += store.write(DATASET, SYMBOL, pd.concat(buf), key=["id"])
                buf = []
                logger.info(f"{DATASET} {n:,}건 (~{store.get_state(DATASET, SYMBOL)['last_ts']:%Y-%m-%d})")
            if not token:
                break
        store.set_state(DATASET, SYMBOL, backfill_done=True)
    except Exception as e:
        if buf:
            n += store.write(DATASET, SYMBOL, pd.concat(buf), key=["id"])
        store.set_state(DATASET, SYMBOL, status="error", error=str(e)[:500])
        logger.error(f"{DATASET} 실패: {e}")
    return n


def run(symbols: list[str] | None = None) -> None:
    headers = keys()
    if headers is None:
        logger.info(f"{DATASET}: ALPACA_API_KEY/ALPACA_SECRET_KEY 없음 → 건너뜀")
        return
    collect(headers)
