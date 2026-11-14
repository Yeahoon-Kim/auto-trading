"""Binance 현물·USDT-M 선물 1분봉과 펀딩비 수집.

- 지난 달까지: data.binance.vision 월별 압축파일 (대량 히스토리, 키 불필요)
- 이번 달: REST API
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pandas as pd

from .. import store
from .common import RateLimiter, config, get_json, logger

SPOT_API = "https://api.binance.com/api/v3"
UM_API = "https://fapi.binance.com/fapi/v1"
VISION = "https://data.binance.vision/data"
KLINE_COLS = ["ts", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
MARKETS = {
    "spot": {"dataset": "binance_spot_1m", "api": SPOT_API, "vision": f"{VISION}/spot/monthly/klines"},
    "um": {"dataset": "binance_um_1m", "api": UM_API, "vision": f"{VISION}/futures/um/monthly/klines"},
}

_http = httpx.Client(timeout=60, follow_redirects=True)
_limit = RateLimiter(8)


# ── 유니버스 ──
def build_universe() -> pd.DataFrame:
    cfg = config()["binance"]
    tickers = pd.DataFrame(get_json(_http, f"{SPOT_API}/ticker/24hr"))
    tickers = tickers[tickers["symbol"].str.endswith("USDT")].copy()
    tickers["base"] = tickers["symbol"].str[:-4]
    bad = tickers["base"].isin(cfg["exclude_bases"]) | tickers["base"].str.contains(r"(?:UP|DOWN|BULL|BEAR)$")
    tickers = tickers[~bad]
    tickers["quoteVolume"] = tickers["quoteVolume"].astype(float)
    info = get_json(_http, f"{UM_API}/exchangeInfo")
    perps = {s["symbol"] for s in info["symbols"]
             if s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"}
    # 무기한 선물이 있는 코인만 (토큰화 주식 등 현물 전용 상품 제외, 베이시스·펀딩비 전략에 필요)
    tickers = tickers[tickers["symbol"].isin(perps)]
    top = tickers.nlargest(cfg["top_n"], "quoteVolume")[["symbol", "base", "quoteVolume"]]
    top["has_perp"] = True
    store.save_universe("binance", top)
    logger.info(f"Binance 유니버스 {len(top)}개 (선물 있음 {top['has_perp'].sum()}개)")
    return top


def universe() -> pd.DataFrame:
    u = store.load_universe("binance")
    return u if not u.empty else build_universe()


# ── 파싱 ──
def _parse_klines(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.iloc[:, :12].copy()
    raw.columns = KLINE_COLS
    raw = raw[pd.to_numeric(raw["ts"], errors="coerce").notna()]  # 헤더 행 제거
    ts = raw["ts"].astype("int64")
    unit = "us" if ts.iloc[0] > 10**14 else "ms"  # 2025년부터 현물 파일은 마이크로초
    df = pd.DataFrame({"ts": pd.to_datetime(ts, unit=unit, utc=True)})
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]:
        df[c] = raw[c].astype("float64").values
    df["trades"] = raw["trades"].astype("int64").values
    return df


def _vision_month(market: str, symbol: str, month: pd.Timestamp) -> pd.DataFrame | None:
    url = f"{MARKETS[market]['vision']}/{symbol}/1m/{symbol}-1m-{month:%Y-%m}.zip"
    r = _http.get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str)
    return _parse_klines(raw)


def _api_klines(market: str, symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    out, cursor = [], int(start.timestamp() * 1000)
    end = int(pd.Timestamp.now(tz="UTC").floor("min").timestamp() * 1000) - 60_000  # 미완성 봉 제외
    while cursor <= end:
        _limit.wait()
        rows = get_json(_http, f"{MARKETS[market]['api']}/klines",
                        {"symbol": symbol, "interval": "1m", "startTime": cursor, "endTime": end, "limit": 1000})
        if not rows:
            break
        out.append(_parse_klines(pd.DataFrame(rows)))
        cursor = rows[-1][0] + 60_000
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── 수집 ──
def collect_klines(market: str, symbol: str) -> None:
    dataset = MARKETS[market]["dataset"]
    st = store.get_state(dataset, symbol)
    start = (st["last_ts"] + pd.Timedelta(minutes=1)) if st["last_ts"] is not None \
        else pd.Timestamp(config()["binance"]["start_1m"], tz="UTC")
    this_month = pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
    try:
        month = start.normalize().replace(day=1)
        while month < this_month:  # 지난 달까지는 월별 파일
            df = _vision_month(market, symbol, month)
            if df is not None:
                n = store.write(dataset, symbol, df[df["ts"] >= start])
                logger.info(f"{dataset} {symbol} {month:%Y-%m} {n}행 (파일)")
            month += pd.DateOffset(months=1)
            start = max(start, month)
        df = _api_klines(market, symbol, start)
        n = store.write(dataset, symbol, df)
        store.set_state(dataset, symbol, backfill_done=True)
        logger.info(f"{dataset} {symbol} 최신분 {n}행 (API)")
    except Exception as e:
        store.set_state(dataset, symbol, status="error", error=str(e)[:500])
        logger.error(f"{dataset} {symbol} 실패: {e}")


def collect_funding(symbol: str) -> None:
    dataset = "binance_um_funding"
    st = store.get_state(dataset, symbol)
    start = (st["last_ts"] + pd.Timedelta(seconds=1)) if st["last_ts"] is not None \
        else pd.Timestamp(config()["binance"]["start_funding"], tz="UTC")
    cursor, total = int(start.timestamp() * 1000), 0
    try:
        while True:
            _limit.wait()
            rows = get_json(_http, f"{UM_API}/fundingRate", {"symbol": symbol, "startTime": cursor, "limit": 1000})
            if not rows:
                break
            df = pd.DataFrame(rows)
            df = pd.DataFrame({
                "ts": pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True),
                "funding_rate": df["fundingRate"].astype(float),
                "mark_price": pd.to_numeric(df.get("markPrice"), errors="coerce"),
            })
            total += store.write(dataset, symbol, df)
            if len(rows) < 1000:
                break
            cursor = rows[-1]["fundingTime"] + 1
        store.set_state(dataset, symbol, backfill_done=True)
        logger.info(f"{dataset} {symbol} {total}행")
    except Exception as e:
        store.set_state(dataset, symbol, status="error", error=str(e)[:500])
        logger.error(f"{dataset} {symbol} 실패: {e}")


def collect_positioning(symbol: str) -> None:
    """미결제약정(OI)과 계정 기준 롱/숏 비율, 1시간 단위. Binance가 최근 30일만 주므로 매일 누적."""
    dataset, base = "binance_um_oi", "https://fapi.binance.com/futures/data"
    st = store.get_state(dataset, symbol)
    start = max(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=29),
                (st["last_ts"] + pd.Timedelta(hours=1)) if st["last_ts"] is not None else pd.Timestamp(0, tz="UTC"))
    try:
        frames = {}
        for name, path in [("oi", "openInterestHist"), ("ls", "globalLongShortAccountRatio")]:
            cursor, rows = int(start.timestamp() * 1000), []
            while True:
                _limit.wait()
                page = get_json(_http, f"{base}/{path}",
                                {"symbol": symbol, "period": "1h", "startTime": cursor, "limit": 500})
                if not page:
                    break
                rows += page
                if len(page) < 500:
                    break
                cursor = page[-1]["timestamp"] + 1
            frames[name] = pd.DataFrame(rows)
        oi, ls = frames["oi"], frames["ls"]
        if oi.empty:
            return
        df = pd.DataFrame({"ts": pd.to_datetime(oi["timestamp"].astype("int64"), unit="ms", utc=True),
                           "oi": oi["sumOpenInterest"].astype(float),
                           "oi_value": oi["sumOpenInterestValue"].astype(float)})
        if not ls.empty:
            ls = pd.DataFrame({"ts": pd.to_datetime(ls["timestamp"].astype("int64"), unit="ms", utc=True),
                               "long_short_ratio": ls["longShortRatio"].astype(float)})
            df = df.merge(ls, on="ts", how="left")
        store.write(dataset, symbol, df)
        store.set_state(dataset, symbol, backfill_done=True)
    except Exception as e:
        store.set_state(dataset, symbol, status="error", error=str(e)[:500])
        logger.error(f"{dataset} {symbol} 실패: {e}")


def run(symbols: list[str] | None = None) -> None:
    u = universe()
    if symbols:
        u = u[u["symbol"].isin(symbols)]
    for _, row in u.iterrows():
        collect_klines("spot", row["symbol"])
        if row["has_perp"]:
            collect_klines("um", row["symbol"])
            collect_funding(row["symbol"])
            collect_positioning(row["symbol"])
