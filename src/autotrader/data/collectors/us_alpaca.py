"""미국 주식 1분봉 장기 이력 (Alpaca Market Data, 무료 Basic 플랜).

- us_1m_sip: us 유니버스(S&P500 + Nasdaq100) 1분봉, 2016~. 전체 거래소 체결(SIP) 기준, 원본 가격(분할 미반영).
  프리·애프터마켓 봉도 들어 있다 (정규장만 쓰려면 09:30~16:00 동부시간으로 거른다).
  Yahoo 기반 us_1m(정규장, 분할 반영, 최근 30일부터 누적)과 소스·가격 기준이 달라 섞지 않는다.
- 무료 플랜 제약: 최근 15분 데이터는 못 받는다(그래서 16분 전까지만 요청), 분당 200회.
- 키: 환경변수 ALPACA_API_KEY / ALPACA_SECRET_KEY (모의투자 계정 키로 충분). 없으면 건너뛴다.
한계: 유니버스가 현재 구성종목이라 생존편향이 있다. 분할이 있으면 과거 가격 기준이 바뀌므로
      백테스트에서는 us_1d의 adj_close/close 비율로 맞춘다.
ts는 봉 시작 시각(UTC).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd

from .. import store
from .common import RateLimiter, config, get_json, logger
from .us_stock import universe

DATASET = "us_1m_sip"
API = "https://data.alpaca.markets/v2/stocks/bars"
_limit = RateLimiter(3)  # 분당 180회 (한도 200)
_http = httpx.Client(timeout=60)


def keys() -> dict | None:
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s} if k and s else None


def _alpaca_symbol(yahoo: str) -> str:
    return yahoo.replace("-", ".")  # BRK-B → BRK.B


def _bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp, headers: dict, page_token: str | None) -> tuple:
    _limit.wait()
    params = {"symbols": _alpaca_symbol(symbol), "timeframe": "1Min", "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000, "feed": "sip", "adjustment": "raw",
              "sort": "asc"}
    if page_token:
        params["page_token"] = page_token
    j = get_json(_http, API, params, headers=headers)
    rows = (j.get("bars") or {}).get(_alpaca_symbol(symbol), [])
    df = pd.DataFrame(rows)
    if not df.empty:
        df = pd.DataFrame({
            "ts": pd.to_datetime(df["t"], utc=True), "open": df["o"].astype(float), "high": df["h"].astype(float),
            "low": df["l"].astype(float), "close": df["c"].astype(float), "volume": df["v"].astype(float),
            "trades": df["n"].astype(float), "vwap": df["vw"].astype(float),
        })
    return df, j.get("next_page_token")


def collect(symbol: str, headers: dict) -> None:
    """마지막으로 받은 봉 다음부터 (처음이면 시작일부터) 16분 전까지 앞으로 받는다. 끊겨도 이어받는다."""
    st = store.get_state(DATASET, symbol)
    start = (st["last_ts"] + pd.Timedelta(minutes=1)) if st["last_ts"] is not None \
        else pd.Timestamp(config()["us"]["start_1m_sip"], tz="UTC")
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=16)
    if start >= end:
        return
    buf, n, token = [], 0, None
    try:
        while True:
            df, token = _bars(symbol, start, end, headers, token)
            if not df.empty:
                buf.append(df)
            if sum(map(len, buf)) >= 100_000 or (not token and buf):  # 10만 행마다 저장
                n += store.write(DATASET, symbol, pd.concat(buf))
                buf = []
            if not token:
                break
        store.set_state(DATASET, symbol, backfill_done=True)
        if n:
            logger.info(f"{DATASET} {symbol} {n:,}행")
    except Exception as e:
        if buf:
            store.write(DATASET, symbol, pd.concat(buf))
        store.set_state(DATASET, symbol, status="error", error=str(e)[:500])
        logger.error(f"{DATASET} {symbol} 실패: {e}")


def run(symbols: list[str] | None = None) -> None:
    headers = keys()
    if headers is None:
        logger.info(f"{DATASET}: ALPACA_API_KEY/ALPACA_SECRET_KEY 없음 → 건너뜀")
        return
    tickers = symbols or universe()["yahoo"].tolist()
    # 이미 시작한 종목(최신분 갱신)을 먼저, 처음 받는 종목(긴 백필)은 뒤에
    started = [t for t in tickers if store.get_state(DATASET, t)["last_ts"] is not None]
    for i, t in enumerate(started + [t for t in tickers if t not in started], 1):
        collect(t, headers)
        if i % 50 == 0:
            logger.info(f"{DATASET} {i}/{len(tickers)}")
