"""거시·심리 지표 수집 (키 불필요).

- macro_1d          : Yahoo 일봉 — 변동성(VIX, VIX3M, MOVE), 지수, 환율, 달러, 금리, 원자재
- fred              : FRED 일간·주간 시계열 — 국채 금리, 장단기 금리차, 신용 스프레드, 금융여건
- crypto_fear_greed : alternative.me 코인 공포·탐욕 지수 (전체 이력)

시점 주의: FRED·Yahoo 값의 날짜는 "관측일"이다. 실제로 알 수 있는 건 보통 다음 영업일이므로,
전략에서 쓸 때는 하루 늦춰 쓴다 (미래참조 방지). ts는 관측일 00:00 UTC.
"""

from __future__ import annotations

import io

import httpx
import pandas as pd
import yfinance as yf

from .. import store
from .common import config, get_json, logger

YAHOO = {
    "^VIX": "VIX", "^VIX3M": "VIX3M", "^MOVE": "MOVE",
    "^GSPC": "SPX", "^IXIC": "NASDAQ", "^RUT": "RUSSELL2000", "^KS11": "KOSPI", "^KQ11": "KOSDAQ",
    "KRW=X": "USDKRW", "DX-Y.NYB": "DXY", "JPY=X": "USDJPY",
    "^TNX": "US10Y_YIELD", "CL=F": "WTI", "GC=F": "GOLD", "HG=F": "COPPER",
}
FRED = {
    "DGS2": "미 국채 2년", "DGS10": "미 국채 10년", "T10Y2Y": "10년-2년 금리차", "T10Y3M": "10년-3개월 금리차",
    "DFF": "연방기금금리", "BAMLH0A0HYM2": "하이일드 스프레드", "BAMLC0A0CM": "투자등급 스프레드",
    "T5YIE": "5년 기대인플레이션", "NFCI": "시카고 금융여건지수(주간)", "DTWEXBGS": "달러지수(광의)",
}
_http = httpx.Client(timeout=60, headers={"User-Agent": "Mozilla/5.0"})


def collect_yahoo() -> None:
    start = config()["us"]["start_1d"]
    raw = yf.download(list(YAHOO), start=start, interval="1d", group_by="ticker", auto_adjust=False,
                      progress=False, threads=True)
    ny_now = pd.Timestamp.now(tz="America/New_York")
    cutoff = ny_now.normalize() + (pd.Timedelta(days=1) if ny_now.hour >= 17 else pd.Timedelta(0))
    for ticker, name in YAHOO.items():
        try:
            d = raw[ticker].dropna(subset=["Close"])
            d = d[d.index < cutoff.tz_localize(None)]  # 아직 안 끝난 날 제외
            df = pd.DataFrame({
                "ts": pd.to_datetime(d.index).tz_localize("UTC"),
                "open": d["Open"].values, "high": d["High"].values, "low": d["Low"].values,
                "close": d["Close"].values, "volume": d["Volume"].values,
            })
            store.write("macro_1d", name, df)
            store.set_state("macro_1d", name, backfill_done=True)
        except Exception as e:
            store.set_state("macro_1d", name, status="error", error=str(e)[:500])
            logger.error(f"macro_1d {name} 실패: {e}")
    logger.info(f"macro_1d {len(YAHOO)}개 갱신")


def collect_fred() -> None:
    for sid in FRED:
        try:
            r = _http.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid})
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = ["date", "value"]
            d["value"] = pd.to_numeric(d["value"], errors="coerce")  # 휴일은 '.'
            d = d.dropna()
            df = pd.DataFrame({"ts": pd.to_datetime(d["date"]).dt.tz_localize("UTC"), "value": d["value"].values})
            store.write("fred", sid, df[df["ts"] >= pd.Timestamp("2005-01-01", tz="UTC")])
            store.set_state("fred", sid, backfill_done=True)
        except Exception as e:
            store.set_state("fred", sid, status="error", error=str(e)[:500])
            logger.error(f"fred {sid} 실패: {e}")
    logger.info(f"fred {len(FRED)}개 갱신")


def collect_fear_greed() -> None:
    try:
        rows = get_json(_http, "https://api.alternative.me/fng/", {"limit": 0, "format": "json"})["data"]
        d = pd.DataFrame(rows)
        df = pd.DataFrame({
            "ts": pd.to_datetime(d["timestamp"].astype("int64"), unit="s", utc=True),
            "value": d["value"].astype(float),
            "label": d["value_classification"],
        })
        store.write("crypto_fear_greed", "FNG", df)
        store.set_state("crypto_fear_greed", "FNG", backfill_done=True)
        logger.info(f"crypto_fear_greed {len(df)}행")
    except Exception as e:
        store.set_state("crypto_fear_greed", "FNG", status="error", error=str(e)[:500])
        logger.error(f"crypto_fear_greed 실패: {e}")


def run(symbols: list[str] | None = None) -> None:
    collect_yahoo()
    collect_fred()
    collect_fear_greed()
