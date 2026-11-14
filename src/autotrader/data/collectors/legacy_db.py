"""기존 stock-analysis-site(Docker Postgres)에 쌓인 데이터를 parquet로 가져온다.

컨테이너 stock-analysis-site-db-1 이 켜져 있어야 한다 (docker start stock-analysis-site-db-1).
한 번 가져오면 컨테이너 없이 쓸 수 있다. 다시 실행하면 전체를 새로 덮어쓴다.

저장 위치: data/processed/legacy/*.parquet  (종목×날짜 긴 표, 횡단면 분석용)
- daily_prices_kr / daily_prices_us : 2016-08~ 일봉 + 시가총액
    KR = 원본 가격 (KIS와 동일), US = 분할 반영 가격, 상장폐지 종목 없음(생존편향)
- instruments, index_constituents (실제 KOSPI200·KOSDAQ150·S&P500·NDX, 2026-08 기준)
- financial_lines (K-IFRS 349사·US GAAP 511사, 2021~), fundamentals
    시점 주의: period_label·effective_date는 회계기간 기준이라 실제 공개일이 아니다.
    전략에서는 공개 지연 규칙(분기 +45일, 연간 +90일)을 적용해서 쓴다.
- disclosures (SEC 1995~, DART 2024-08~), insider_events (SEC Form 4 2000~, DART 2024-08~)
"""

from __future__ import annotations

import io
import subprocess

import pandas as pd

from .. import store
from .common import logger

CONTAINER, DB_USER, DB_NAME = "stock-analysis-site-db-1", "stock", "stock"
OUT = store.DATA_DIR / "legacy"

QUERIES = {
    "instruments": "SELECT id, symbol, market, name, name_ko, currency, asset_type, exchange, corp_code, cik, "
                   "shares_outstanding, shares_as_of FROM instruments",
    "index_constituents": "SELECT 'KR' AS market, index_code, symbol, as_of FROM kr_index_constituents "
                          "UNION ALL SELECT 'US', index_code, symbol, as_of FROM us_index_constituents",
    "financial_lines": "SELECT i.symbol, i.market, f.period_label, f.period_type, f.statement_type, f.line_key, "
                       "f.line_label, f.value, f.currency, f.accounting FROM financial_statement_lines f "
                       "JOIN instruments i ON i.id = f.instrument_id",
    "fundamentals": "SELECT i.symbol, i.market, f.period_label, f.period_type, f.effective_date, f.revenue, "
                    "f.operating_income, f.net_income, f.eps FROM fundamentals_period f "
                    "JOIN instruments i ON i.id = f.instrument_id",
    "disclosures": "SELECT i.symbol, i.market, d.source, d.external_id, d.title, d.filed_at, d.report_type, "
                   "d.category, d.is_insider, d.filer_name, d.url FROM disclosures d "
                   "LEFT JOIN instruments i ON i.id = d.instrument_id",
    "insider_events": "SELECT i.symbol, i.market, e.event_date, e.insider_name, e.role, e.transaction, e.shares, "
                      "e.price, e.shares_total, e.source FROM insider_events e "
                      "JOIN instruments i ON i.id = e.instrument_id",
    "daily_prices_kr": "SELECT i.symbol, p.trade_date, p.open, p.high, p.low, p.close, p.volume, p.market_cap "
                       "FROM daily_prices p JOIN instruments i ON i.id = p.instrument_id WHERE i.market = 'KR'",
    "daily_prices_us": "SELECT i.symbol, p.trade_date, p.open, p.high, p.low, p.close, p.volume, p.market_cap "
                       "FROM daily_prices p JOIN instruments i ON i.id = p.instrument_id WHERE i.market = 'US'",
}
DATE_COLS = {"trade_date", "filed_at", "event_date", "effective_date", "as_of", "shares_as_of"}


def _copy(sql: str) -> pd.DataFrame:
    cmd = ["docker", "exec", CONTAINER, "psql", "-U", DB_USER, "-d", DB_NAME, "-c",
           f"COPY ({sql}) TO STDOUT WITH CSV HEADER"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    df = pd.read_csv(io.BytesIO(out), dtype={"symbol": str, "corp_code": str, "cik": str, "external_id": str},
                     low_memory=False)
    for c in DATE_COLS & set(df.columns):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def run(symbols: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sql in QUERIES.items():
        if symbols and name not in symbols:
            continue
        df = _copy(sql)
        sort = [c for c in ("symbol", "trade_date", "filed_at", "event_date", "period_label") if c in df.columns]
        if sort:
            df = df.sort_values(sort)
        df.to_parquet(OUT / f"{name}.parquet", compression="zstd", index=False)
        logger.info(f"legacy {name}: {len(df):,}행")
