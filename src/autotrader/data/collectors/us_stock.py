"""미국 주식 수집 (yfinance, 무료).

- us_1d: S&P500 + Nasdaq100 일봉 (2010~). 주의: Yahoo의 OHLC는 이미 분할 반영 값이고,
         adj_close는 분할+배당 반영. 분할이 생기면 과거 값이 바뀌므로 감지 시 전체 재수집.
- us_1m: 같은 유니버스 1분봉. Yahoo가 최근 30일만 주므로 매일 누적해야 쌓인다.
한계: 유니버스가 현재 구성종목 기준이라 생존편향이 있다 (무료 데이터 한계).
ts는 봉 시작 시각(UTC). 일봉은 09:30 미국 동부시간.
"""

from __future__ import annotations

import shutil

import FinanceDataReader as fdr
import httpx
import pandas as pd
import yfinance as yf

from .. import store
from .common import config, get_json, logger

NY = "America/New_York"
CHUNK = 50


def build_universe() -> pd.DataFrame:
    sp = fdr.StockListing("S&P500")[["Symbol", "Name"]].assign(index="SP500")
    rows = get_json(httpx.Client(timeout=30), "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})["data"]["data"]["rows"]
    ndx = pd.DataFrame(rows)[["symbol", "companyName"]].rename(columns={"symbol": "Symbol", "companyName": "Name"})
    ndx["index"] = "NASDAQ100"
    u = pd.concat([sp, ndx]).groupby("Symbol", as_index=False).agg({"Name": "first", "index": ",".join})
    # Yahoo 표기: 클래스 주식은 하이픈 (FDR은 BRKB, Nasdaq은 BRK.B 형태로 줌)
    fix = {"BRKB": "BRK-B", "BFB": "BF-B", "BRKA": "BRK-A"}
    u["yahoo"] = u["Symbol"].map(fix).fillna(u["Symbol"].str.replace(".", "-", regex=False))
    u.columns = [c.lower() for c in u.columns]
    store.save_universe("us", u)
    logger.info(f"US 유니버스 {len(u)}개")
    return u


def universe() -> pd.DataFrame:
    u = store.load_universe("us")
    return u if not u.empty else build_universe()


def _download(tickers: list[str], **kw) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, group_by="ticker", auto_adjust=False, progress=False, threads=True, **kw)
    out = {}
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        d = raw[t].dropna(subset=["Close"])
        if d.empty:
            continue
        idx = d.index
        idx = idx.tz_localize(NY) + pd.Timedelta(hours=9, minutes=30) if idx.tz is None else idx
        df = pd.DataFrame({
            "ts": idx.tz_convert("UTC"),
            "open": d["Open"].values, "high": d["High"].values, "low": d["Low"].values,
            "close": d["Close"].values, "volume": d["Volume"].values,
        })
        daily = kw.get("interval", "1d") == "1d"
        if "Adj Close" in d and daily:
            df["adj_close"] = d["Adj Close"].values
        # 아직 끝나지 않은 봉 제외 (일봉: 16:00 장 마감 전, 분봉: 진행 중인 1분)
        bar_len = pd.Timedelta(hours=6, minutes=30) if daily else pd.Timedelta(minutes=1)
        df = df[df["ts"] + bar_len <= pd.Timestamp.now(tz="UTC")]
        out[t] = df
    return out


def collect_daily(tickers: list[str]) -> None:
    start = config()["us"]["start_1d"]
    new = [t for t in tickers if not store.get_state("us_1d", t)["backfill_done"]]
    old = [t for t in tickers if t not in new]
    for i in range(0, len(new), CHUNK):  # 백필
        for t, df in _download(new[i:i + CHUNK], start=start, interval="1d").items():
            store.write("us_1d", t, df)
            store.set_state("us_1d", t, backfill_done=True)
        logger.info(f"us_1d 백필 {min(i + CHUNK, len(new))}/{len(new)}")
    for i in range(0, len(old), CHUNK):  # 최신분 + 분할 감지
        recent = _download(old[i:i + CHUNK], period="1mo", interval="1d")
        for t, df in recent.items():
            prev = store.read("us_1d", t, start=df["ts"].min())
            chk = prev.merge(df, on="ts", suffixes=("_old", "_new"))
            if not chk.empty and ((chk["close_old"] / chk["close_new"] - 1).abs() > 1e-4).any():
                logger.info(f"us_1d {t} 과거 가격 변경(분할 등) 감지 → 전체 재수집")
                shutil.rmtree(store.DATA_DIR / "us_1d" / t, ignore_errors=True)
                full = _download([t], start=start, interval="1d").get(t)
                if full is not None:
                    store.write("us_1d", t, full)
                continue
            store.write("us_1d", t, df)
        logger.info(f"us_1d 최신분 {min(i + CHUNK, len(old))}/{len(old)}")


def collect_minute(tickers: list[str]) -> None:
    """최근 29일을 7일 창으로 나눠 받는다 (이미 받은 구간은 중복 제거로 무시)."""
    now = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    windows = [(now - pd.Timedelta(days=d + 7), now - pd.Timedelta(days=d)) for d in range(0, 28, 7)]
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        for s, e in windows:
            try:
                for t, df in _download(chunk, start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"),
                                       interval="1m").items():
                    store.write("us_1m", t, df)
                    store.set_state("us_1m", t, backfill_done=True)  # 받을 수 있는 과거분(30일)은 매번 다 받음
            except Exception as ex:
                logger.error(f"us_1m {chunk[0]}.. {s:%m-%d} 실패: {ex}")
        logger.info(f"us_1m {min(i + CHUNK, len(tickers))}/{len(tickers)}")


def run(what: str = "all", symbols: list[str] | None = None) -> None:
    tickers = symbols or universe()["yahoo"].tolist()
    if what in ("all", "minute"):
        collect_minute(tickers)
    if what in ("all", "daily"):
        if not symbols:  # 벤치마크 ETF는 유니버스(구성종목)와 따로 일봉만 받는다
            tickers = tickers + [t for t in config()["us"].get("benchmarks_1d", []) if t not in tickers]
        collect_daily(tickers)
