"""리스크팀: 포지션 사이징 (tradermonty/claude-trading-skills의 position-sizer에서 가져온 아이디어).

결정론적 코드다. LLM은 사이징에 관여하지 않는다.

원칙
- 생존 우선: 거래당 위험(손절까지 잃는 돈)을 자본의 일정 비율로 고정한다 (기본 1%, 2% 초과 금지).
- 내림만 한다: 수량은 절대 올림하지 않는다 (위험·비중 한도를 넘지 않게).
- 가장 엄한 한도가 이긴다: 위험 기준·종목 비중·섹터 비중·전체 열린 위험(heat)·주문 상한 중 최소값.
  어떤 한도가 수량을 묶었는지(binding)를 항상 결과에 남긴다.
- Kelly는 하프 Kelly만, 그리고 위험 비율의 상한으로만 쓴다 (승률·손익비는 백테스트 통계에서).
- 최소 주문 금액 미달이면 0이 아니라 BLOCK과 사유를 낸다 (소액 계좌에서 조용히 0주가 되는 문제 방지).
- 선물은 계약 승수가 다르므로 multiplier를 반드시 넘긴다 (주식 로직을 그대로 쓰면 크기가 틀어짐).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SizingInput:
    equity: float                     # 총자산 (계좌 통화)
    entry: float
    stop: float | None = None         # 손절가. 없으면 atr로 계산
    atr: float | None = None
    atr_mult: float = 2.0
    risk_pct: float = 1.0             # 거래당 위험 (자본 대비 %)
    side: str = "long"
    lot: float = 1.0                  # 수량 단위 (주식 1주, 코인은 거래소 stepSize)
    multiplier: float = 1.0           # 선물 계약 승수 (주식·현물 1)
    max_position_pct: float | None = None     # 종목 하나 최대 비중
    max_sector_pct: float | None = None
    sector_exposure_pct: float = 0.0          # 같은 섹터에 이미 들어간 비중
    max_heat_pct: float | None = None         # 열린 포지션 전체 위험 한도
    open_heat_pct: float = 0.0                # 이미 열린 위험 합계
    max_order_value: float | None = None      # 1회 주문 상한 (금액)
    min_notional: float = 0.0                 # 거래소 최소 주문 금액
    win_rate: float | None = None             # Kelly용 (백테스트 통계)
    payoff: float | None = None               # 평균이익/평균손실


@dataclass
class SizingResult:
    qty: float
    notional: float
    risk_amount: float
    risk_pct: float
    stop: float
    binding: str
    candidates: dict = field(default_factory=dict)
    blocked: str | None = None


MAX_RISK_PCT = 2.0


def half_kelly_pct(win_rate: float, payoff: float) -> float:
    """하프 Kelly (%). 음수면 0 → 기대값이 없는 전략에는 자본을 주지 않는다."""
    k = win_rate - (1 - win_rate) / payoff
    return max(0.0, k) * 100 / 2


def _floor(q: float, lot: float) -> float:
    n = math.floor(q / lot + 1e-9) * lot
    return round(n, 10)


def size(x: SizingInput) -> SizingResult:
    stop = x.stop
    if stop is None:
        if x.atr is None:
            raise ValueError("stop 또는 atr 중 하나는 필요")
        d = x.atr * x.atr_mult
        stop = x.entry - d if x.side == "long" else x.entry + d
    if (x.side == "long" and stop >= x.entry) or (x.side == "short" and stop <= x.entry) or stop <= 0:
        raise ValueError(f"손절가가 진입가의 잘못된 쪽에 있음: entry={x.entry} stop={stop} side={x.side}")

    risk_pct = min(x.risk_pct, MAX_RISK_PCT)
    if x.win_rate is not None and x.payoff:
        risk_pct = min(risk_pct, half_kelly_pct(x.win_rate, x.payoff))
    per_unit_risk = abs(x.entry - stop) * x.multiplier
    per_unit_value = x.entry * x.multiplier

    cand = {"risk": x.equity * risk_pct / 100 / per_unit_risk}
    if x.max_position_pct is not None:
        cand["max_position"] = x.equity * x.max_position_pct / 100 / per_unit_value
    if x.max_sector_pct is not None:
        cand["max_sector"] = max(0.0, x.equity * (x.max_sector_pct - x.sector_exposure_pct) / 100 / per_unit_value)
    if x.max_heat_pct is not None:
        cand["max_heat"] = max(0.0, x.equity * (x.max_heat_pct - x.open_heat_pct) / 100 / per_unit_risk)
    if x.max_order_value is not None:
        cand["max_order_value"] = x.max_order_value / per_unit_value

    binding = min(cand, key=cand.get)
    if risk_pct <= 0:
        binding = "kelly"
    qty = _floor(cand[binding], x.lot) if risk_pct > 0 else 0.0
    notional = qty * per_unit_value
    blocked = None
    if qty <= 0:
        blocked = f"수량 0 ({binding} 한도)"
    elif notional < x.min_notional:
        blocked = f"주문 금액 {notional:,.0f} < 최소 주문 {x.min_notional:,.0f}"
    risk_amount = qty * per_unit_risk
    return SizingResult(qty if not blocked else 0.0, notional, risk_amount,
                        risk_amount / x.equity * 100, stop, binding,
                        {k: round(v, 6) for k, v in cand.items()}, blocked)
