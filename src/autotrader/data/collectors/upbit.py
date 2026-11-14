"""Upbit KRW 마켓 1분봉 수집 (김치 프리미엄 비교용, 키 불필요).

Upbit API는 과거 방향으로만 페이지를 넘기므로(to=), 최신분은 지금부터 마지막 수집 시점까지,
백필은 가장 오래된 수집 시점부터 시작일까지 거꾸로 받는다.
거래가 없던 분은 봉이 없다 (빈 구간 = 무거래).
"""

from __future__ import annotations

import httpx
import pandas as pd

from .. import store
from . import binance
from .common import RateLimiter, config, get_json, logger

API = "https://api.upbit.com/v1"
DATASET = "upbit_1m"
_http = httpx.Client(timeout=30)
_limit = RateLimiter(8)  # 시세 조회 한도 초당 10회


def build_universe() -> pd.DataFrame:
    markets = pd.DataFrame(get_json(_http, f"{API}/market/all"))
    krw = markets[markets["market"].str.startswith("KRW-")].copy()
    krw["base"] = krw["market"].str[4:]
    u = krw[krw["base"].isin(set(binance.universe()["base"]))][["market", "base"]]
    store.save_universe("upbit", u)
    logger.info(f"Upbit 유니버스 {len(u)}개")
    return u


def universe() -> pd.DataFrame:
    u = store.load_universe("upbit")
    return u if not u.empty else build_universe()


def _page(market: str, to: pd.Timestamp | None) -> pd.DataFrame:
    _limit.wait()
    params = {"market": market, "count": 200}
    if to is not None:
        params["to"] = to.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = get_json(_http, f"{API}/candles/minutes/1", params)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    df = pd.DataFrame({
        "ts": pd.to_datetime(d["candle_date_time_utc"]).dt.tz_localize("UTC"),
        "open": d["opening_price"].astype(float),
        "high": d["high_price"].astype(float),
        "low": d["low_price"].astype(float),
        "close": d["trade_price"].astype(float),
        "volume": d["candle_acc_trade_volume"].astype(float),
        "quote_volume": d["candle_acc_trade_price"].astype(float),
    })
    return df[df["ts"] < pd.Timestamp.now(tz="UTC").floor("min")]  # 진행 중인 봉 제외


def _walk_back(market: str, to: pd.Timestamp | None, stop: pd.Timestamp) -> bool:
    """to부터 stop까지 거꾸로 수집. stop에 도달했거나 데이터가 끝났으면 True."""
    buf, n = [], 0
    while True:
        df = _page(market, to)
        if df.empty:
            reached = True
            break
        buf.append(df[df["ts"] >= stop])
        to = df["ts"].min()
        if to <= stop:
            reached = True
            break
        if sum(map(len, buf)) >= 20_000:  # 2만 행마다 저장 (중간에 끊겨도 이어받기 가능)
            n += store.write(DATASET, market, pd.concat(buf))
            buf = []
    if buf:
        n += store.write(DATASET, market, pd.concat(buf))
    logger.info(f"{DATASET} {market} {n}행")
    return reached


def collect(market: str) -> None:
    st = store.get_state(DATASET, market)
    start = pd.Timestamp(config()["upbit"]["start_1m"], tz="UTC")
    try:
        if st["last_ts"] is not None:  # 최신분
            _walk_back(market, None, st["last_ts"])
        if not st["backfill_done"]:  # 과거분
            done = _walk_back(market, st["first_ts"], start)
            store.set_state(DATASET, market, backfill_done=done)
    except Exception as e:
        store.set_state(DATASET, market, status="error", error=str(e)[:500])
        logger.error(f"{DATASET} {market} 실패: {e}")


def run(symbols: list[str] | None = None) -> None:
    u = universe()
    if symbols:
        u = u[u["market"].isin(symbols)]
    for m in u["market"]:
        collect(m)
