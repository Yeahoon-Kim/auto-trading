"""데이터 수집 CLI.

    uv run python -m autotrader.data.cli collect binance     # 현물·선물 1분봉 + 펀딩비
    uv run python -m autotrader.data.cli collect upbit
    uv run python -m autotrader.data.cli collect kr          # 분봉 → 일봉 순서
    uv run python -m autotrader.data.cli collect kr-daily
    uv run python -m autotrader.data.cli collect kr-minute
    uv run python -m autotrader.data.cli collect us
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
    c.add_argument("source", choices=["binance", "upbit", "kr", "kr-daily", "kr-minute", "us", "us-daily", "us-minute"])
    c.add_argument("--symbols", nargs="*")
    sub.add_parser("status")
    a = p.parse_args()

    if a.cmd == "status":
        pd.set_option("display.width", 200)
        print(store.summary().to_string(index=False))
        return

    from .collectors import binance, kr_stock, upbit, us_stock
    from .collectors.common import keep_awake
    keep_awake()
    src = a.source
    if src == "binance":
        binance.run(a.symbols)
    elif src == "upbit":
        upbit.run(a.symbols)
    elif src.startswith("kr"):
        kr_stock.run({"kr": "all", "kr-daily": "daily", "kr-minute": "minute"}[src], a.symbols)
    elif src.startswith("us"):
        us_stock.run({"us": "all", "us-daily": "daily", "us-minute": "minute"}[src], a.symbols)


if __name__ == "__main__":
    sys.exit(main())
