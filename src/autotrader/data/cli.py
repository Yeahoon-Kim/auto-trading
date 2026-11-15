"""데이터 수집 CLI.

    uv run python -m autotrader.data.cli collect binance     # 현물·선물 1분봉 + 펀딩비
    uv run python -m autotrader.data.cli collect upbit
    uv run python -m autotrader.data.cli collect kr          # 분봉 → 일봉 순서
    uv run python -m autotrader.data.cli collect kr-daily
    uv run python -m autotrader.data.cli collect kr-minute
    uv run python -m autotrader.data.cli collect us
    uv run python -m autotrader.data.cli collect us-sip      # Alpaca 1분봉 2016~ (키 필요)
    uv run python -m autotrader.data.cli collect binance --symbols BTCUSDT ETHUSDT
    uv run python -m autotrader.data.cli status

백필과 갱신은 같은 명령이다. 수집 상태(data/state.sqlite)를 보고 비어 있는 구간만 받는다.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import store


def main() -> None:
    p = argparse.ArgumentParser(prog="autotrader.data.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("source", choices=["binance", "upbit", "kr", "kr-daily", "kr-minute", "us", "us-daily",
                                      "us-minute", "us-sip", "macro", "legacy", "kr-public"])
    c.add_argument("--symbols", nargs="*")
    sub.add_parser("status")
    h = sub.add_parser("health")
    h.add_argument("--notify", action="store_true", help="문제가 있으면 텔레그램으로 알림")
    a = p.parse_args()

    if a.cmd == "status":
        pd.set_option("display.width", 200)
        print(store.summary().to_string(index=False))
        return

    from ..monitoring import health, notify

    if a.cmd == "health":
        issues = health.check()
        text = health.report(issues) + "\n" + health.progress()
        print(text)
        if a.notify and issues:
            notify.send(text)
        return

    from .collectors import binance, kr_public, kr_stock, legacy_db, macro, upbit, us_alpaca, us_stock
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
        elif src.startswith("kr"):
            kr_stock.run({"kr": "all", "kr-daily": "daily", "kr-minute": "minute"}[src], a.symbols)
        elif src == "us-sip":
            us_alpaca.run(a.symbols)
        elif src.startswith("us"):
            us_stock.run({"us": "all", "us-daily": "daily", "us-minute": "minute"}[src], a.symbols)
    except Exception as e:  # 수집기 전체가 죽는 오류는 숨기지 않고 알린다
        notify.send(f"🔴 수집기 중단: {src}\n{type(e).__name__}: {str(e)[:500]}")
        raise


if __name__ == "__main__":
    sys.exit(main())
