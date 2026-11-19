"""데이터 수집 CLI.

    uv run python -m autotrader.data.cli collect binance     # 현물·선물 1분봉 + 펀딩비
    uv run python -m autotrader.data.cli collect upbit
    uv run python -m autotrader.data.cli collect kr          # 분봉 → 일봉 순서
    uv run python -m autotrader.data.cli collect kr-daily
    uv run python -m autotrader.data.cli collect kr-minute
    uv run python -m autotrader.data.cli collect us
    uv run python -m autotrader.data.cli collect us-sip      # Alpaca 1분봉 2016~ (키 필요)
    uv run python -m autotrader.data.cli collect news        # Alpaca 뉴스 2015~ (키 필요)
    uv run python -m autotrader.data.cli collect dividends   # 배당: DART 사업보고서(KR), Alpaca(US)
    uv run python -m autotrader.data.cli collect edgar       # US 재무: SEC EDGAR companyfacts (SEC_USER_AGENT 필요)
    uv run python -m autotrader.data.cli collect us-splits   # US 분할 이력 (Yahoo, 발표일은 Alpaca 2020~)
    uv run python -m autotrader.data.cli collect naver-trend # 네이버 검색어 트렌드 2016~ (NAVER API HUB 키 필요)
    uv run python -m autotrader.data.cli collect naver-news  # 네이버 뉴스 검색, 지금부터 쌓기 + 과거 백필 (같은 키)
    uv run python -m autotrader.data.cli collect naver-news-backfill  # 과거 백필만 (수동)
    uv run python -m autotrader.data.cli naver-gaps          # 네이버 수집이 빠진 날 (--dataset naver_trend, --since)
    uv run python -m autotrader.data.cli collect binance --symbols BTCUSDT ETHUSDT
    uv run python -m autotrader.data.cli status

백필과 갱신은 같은 명령이다. 수집 상태(data/state.sqlite)를 보고 비어 있는 구간만 받는다.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import store
from .redact import redact


def main() -> None:
    p = argparse.ArgumentParser(prog="autotrader.data.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("source", choices=["binance", "upbit", "kr", "kr-daily", "kr-minute", "us", "us-daily",
                                      "us-minute", "us-sip", "news", "macro", "legacy", "kr-public",
                                      "dividends", "edgar", "us-splits", "naver-trend", "naver-news", "naver-news-backfill"])
    c.add_argument("--symbols", nargs="*")
    sub.add_parser("status")
    g = sub.add_parser("naver-gaps", help="네이버 수집 기록에서 빠진 날 찾기")
    g.add_argument("--dataset", default="naver_news_live", choices=["naver_news_live", "naver_trend", "naver_news_backfill"])
    g.add_argument("--since", help="KST 날짜 YYYY-MM-DD (기본: 첫 수집일)")
    g.add_argument("--all", action="store_true", help="문제없는 날도 보이기")
    h = sub.add_parser("health")
    h.add_argument("--notify", action="store_true", help="문제가 있으면 텔레그램으로 알림")
    a = p.parse_args()

    if a.cmd == "status":
        pd.set_option("display.width", 200)
        print(store.summary().to_string(index=False))
        return

    if a.cmd == "naver-gaps":
        from .collectors import naver
        df = naver.gaps(a.dataset, a.since)
        bad = df[df["problem"] != ""]
        print((df if a.all else bad).to_string(index=False) if len(df if a.all else bad) else "빠진 날 없음")
        print(f"{a.dataset}: {len(df)}일 중 문제 {len(bad)}일"
              f" (거래일 문제 {int((bad['trading_day'] != False).sum())}일)")
        return 1 if (bad["trading_day"] != False).any() else 0

    from ..monitoring import health, notify

    if a.cmd == "health":
        issues = health.check()
        text = health.report(issues) + "\n" + health.progress()
        print(text)
        if a.notify and issues:
            notify.send(text)
        return

    from .collectors import (
        binance,
        dividends,
        edgar,
        kr_public,
        kr_stock,
        legacy_db,
        macro,
        naver,
        news,
        upbit,
        us_alpaca,
        us_splits,
        us_stock,
    )
    from .collectors.common import keep_awake
    keep_awake()
    src = a.source
    try:
        if src == "binance":
            binance.run(a.symbols)
        elif src == "upbit":
            upbit.run(a.symbols)
        elif src == "macro":
            macro.run(a.symbols)
        elif src == "legacy":
            legacy_db.run(a.symbols)
        elif src == "kr-public":
            kr_public.run(a.symbols)
        elif src == "dividends":
            dividends.run(a.symbols)
        elif src == "edgar":
            edgar.run(a.symbols)
        elif src == "naver-trend":
            naver.run_trend(a.symbols)
        elif src == "naver-news":
            naver.run_news(a.symbols)  # live 다음에 과거 백필 (config naver.news_backfill.enabled)
        elif src == "naver-news-backfill":
            naver.run_news_backfill(a.symbols)
        elif src.startswith("kr"):
            kr_stock.run({"kr": "all", "kr-daily": "daily", "kr-minute": "minute"}[src], a.symbols)
        elif src == "news":
            news.run(a.symbols)
        elif src == "us-splits":
            us_splits.run(a.symbols)
        elif src == "us-sip":
            us_alpaca.run(a.symbols)
        elif src.startswith("us"):
            us_stock.run({"us": "all", "us-daily": "daily", "us-minute": "minute"}[src], a.symbols)
    except Exception as e:  # 수집기 전체가 죽는 오류는 숨기지 않고 알린다
        notify.send(f"🔴 수집기 중단: {src}\n{type(e).__name__}: {redact(e)[:500]}")
        raise


if __name__ == "__main__":
    sys.exit(main())
