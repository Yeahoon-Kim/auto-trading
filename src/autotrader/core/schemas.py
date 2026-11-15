"""운용데스크 인수인계 스키마: Signal → Plan → Order → Fill (architecture.md §9-4).

역할마다 넘기는 물건의 모양을 고정한다.
- Signal: lab(전략)이 낸 진입 판단. "무엇을, 왜" 만 있고 수량은 없다.
- Plan:   리스크팀이 Signal을 받아 사이징·가드를 통과시킨 결과. 차단되면 qty=0과 사유.
- Order:  실행팀에 넘길 주문 (지정가 기본). 브로커 호출은 이 모듈 밖에서 한다.
- Fill:   체결 기록. 수수료·세금은 금액으로 남긴다. 비율 모델은 backtest.engine.Costs를 따른다
          (fee_pct·slippage_pct·tax_pct_sell) — 여기서 비용 모델을 따로 만들지 않는다.
- Position: 보유 포지션. 리스크팀 position_guard가 손절·목표·장 마감 전 청산을 판정한다.

규칙
- 모든 시각은 timezone-aware UTC. naive datetime은 받지 않는다 (시장별 현지 시각 혼동 방지).
- 각 단계는 앞 단계의 id를 들고 간다 (plan.signal_id, order.plan_id, fill.order_id) — 사후 대사용.
- 출처 lab과 사유(reason)는 끝까지 따라간다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from autotrader.risk.sizing import SizingResult

SIDES = ("long", "short")


def utc(ts: datetime) -> datetime:
    """aware datetime을 UTC로. naive면 거부한다."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"시각에 시간대가 없음 (UTC aware 필요): {ts!r}")
    return ts.astimezone(UTC)


def _id() -> str:
    return uuid.uuid4().hex[:12]


def _check_side(side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side는 {SIDES} 중 하나: {side!r}")


@dataclass
class Signal:
    lab: str                          # 출처 lab (예: "L001")
    symbol: str
    market: str                       # kr_stock / us_stock / crypto_spot / crypto_futures
    side: str                         # 진입 방향 long / short
    price: float                      # 판단 기준가 (진입 희망가)
    reason: str                       # 왜 들어가는가 (사람이 읽는 문장)
    ts: datetime                      # 판단 시각 (UTC)
    quote_ts: datetime | None = None  # 판단에 쓴 시세의 시각 (데이터 신선도 검사용)
    stop: float | None = None         # 전략이 정한 손절가. 없으면 atr로
    atr: float | None = None
    target: float | None = None       # 목표가 (선택)
    intraday: bool = False            # 당일 청산 전략이면 장 마감 전 정리 대상
    id: str = field(default_factory=_id)

    def __post_init__(self):
        _check_side(self.side)
        self.ts = utc(self.ts)
        if self.quote_ts is not None:
            self.quote_ts = utc(self.quote_ts)


@dataclass
class Plan:
    signal_id: str
    lab: str
    symbol: str
    market: str
    side: str
    entry: float
    stop: float | None
    target: float | None
    intraday: bool
    qty: float                        # 차단이면 0
    reason: str                       # Signal의 사유를 그대로 승계
    ts: datetime                      # 계획 시각 (UTC)
    sizing: SizingResult | None = None   # 사이징 결과 (어떤 한도가 묶었는지 포함). 사이징 전에 막히면 None
    blocked: str | None = None        # 차단 사유. None이면 통과
    checks: list[str] = field(default_factory=list)   # 통과/실패한 가드 기록
    id: str = field(default_factory=_id)

    def __post_init__(self):
        _check_side(self.side)
        self.ts = utc(self.ts)

    @property
    def approved(self) -> bool:
        return self.blocked is None and self.qty > 0


@dataclass
class Order:
    plan_id: str
    lab: str
    symbol: str
    market: str
    side: str                         # buy / sell (주문 방향. 포지션 방향과 다름)
    qty: float
    order_type: str                   # limit / market
    limit_price: float | None
    reason: str
    ts: datetime
    mode: str = "paper"               # paper / live — 기본은 paper
    id: str = field(default_factory=_id)

    def __post_init__(self):
        if self.side not in ("buy", "sell"):
            raise ValueError(f"주문 side는 buy/sell: {self.side!r}")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("지정가 주문에 limit_price가 없음")
        if self.qty <= 0:
            raise ValueError(f"주문 수량은 양수: {self.qty}")
        self.ts = utc(self.ts)


@dataclass
class Fill:
    order_id: str
    lab: str
    symbol: str
    side: str                         # buy / sell
    qty: float
    price: float                      # 실제 체결가 (슬리피지는 limit_price/기준가와의 차이로 드러난다)
    ts: datetime
    fee: float = 0.0                  # 수수료 금액
    tax: float = 0.0                  # 세금 금액 (매도세 등)
    id: str = field(default_factory=_id)

    def __post_init__(self):
        self.ts = utc(self.ts)

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Position:
    lab: str
    symbol: str
    market: str
    side: str                         # long / short
    qty: float
    entry: float
    stop: float | None = None
    target: float | None = None
    intraday: bool = False
    opened_ts: datetime | None = None

    def __post_init__(self):
        _check_side(self.side)
        if self.opened_ts is not None:
            self.opened_ts = utc(self.opened_ts)
