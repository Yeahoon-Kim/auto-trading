"""리스크팀: 결정적 가드 (architecture.md §9-3·5·7, config/risk.yaml).

판정만 한다. 브로커 호출·주문 실행은 하지 않는다.
- data_staleness: 시세가 오래됐거나 없으면 신규 주문 차단. 보유 포지션 시세를 N회 연속 못 받으면 "측정 불가" → 정리 판정.
- position_guard: 손절·목표가·장 마감 전 청산 판정 (전략과 독립적으로 리스크팀이 집행).
- orders.min_notional_check: 주문 금액이 거래소 최소 주문 금액 미만이면 차단하고 사유를 남긴다.
- make_plan: 신호형(type: trade) 전략의 Signal → Plan. 위 가드와 sizing.size()를 거치며, 막히면 qty=0과 사유를 남긴다.
  사이징 값은 lab.yaml trade 블록(trade_params). 비중형(type: portfolio)은 make_plan을 쓰지 않는다
  (목표 비중이 곧 사이징 — execution/paper.py가 계좌 보호 한도만 적용).

모두 순수 함수다. 현재 시각(now)·장 마감 시각·계좌 상태는 호출하는 쪽이 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from autotrader.core.schemas import Order, Plan, Position, Signal, utc
from autotrader.risk.sizing import SizingInput, size

RISK_YAML = Path(__file__).resolve().parents[3] / "config" / "risk.yaml"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or RISK_YAML).read_text(encoding="utf-8"))


def _market_key(market: str, table: dict) -> str | None:
    """risk.yaml 키로 바꾼다. crypto_spot·crypto_futures는 'crypto' 항목을 쓴다."""
    if market in table:
        return market
    if market.startswith("crypto") and "crypto" in table:
        return "crypto"
    return None


# --- §9-3 데이터 신선도 ---------------------------------------------------------

def check_staleness(market: str, quote_ts: datetime | None, now: datetime, cfg: dict) -> str | None:
    """신규 주문 가능 여부. 막히면 사유 문자열, 통과면 None.
    설정에 없는 시장은 기준을 모르므로 막는다 (옛 데이터로 판단하지 않는다)."""
    table = cfg["data_staleness"]["max_quote_age_sec"]
    key = _market_key(market, table)
    if key is None:
        return f"시세 신선도 기준 없음: {market}"
    if quote_ts is None:
        return "시세 시각 없음"
    age = (utc(now) - utc(quote_ts)).total_seconds()
    if age < 0:
        return f"시세 시각이 현재보다 미래 ({-age:.0f}초) — 시계 확인 필요"
    if age > table[key]:
        return f"시세 {age:.0f}초 경과 > 한도 {table[key]}초 ({key})"
    return None


def is_unmeasurable(missed_quotes: int, cfg: dict) -> bool:
    """보유 포지션 시세를 연속 missed_quotes회 못 받았으면 측정 불가."""
    return missed_quotes >= cfg["data_staleness"]["unmeasurable_retries"]


# --- §9-7 최소 주문 금액 ---------------------------------------------------------

def check_min_notional(notional: float, min_notional: float, cfg: dict) -> str | None:
    if not cfg["orders"].get("min_notional_check", True):
        return None
    if notional < min_notional:
        return f"주문 금액 {notional:,.0f} < 최소 주문 {min_notional:,.0f}"
    return None


# --- §9-4 Signal → Plan ----------------------------------------------------------

@dataclass
class Account:
    """사이징에 필요한 계좌 상태 (호출하는 쪽이 브로커/장부에서 채운다)."""
    equity: float                    # 총자산 (계좌 통화)
    currency: str = "KRW"
    krw_per_unit: float | None = None   # 계좌 통화 1단위 = 몇 원 (KRW 아니면 필수, 주문 상한 환산용)
    open_heat_pct: float = 0.0       # 이미 열린 위험 합계 (자본 대비 %)
    sector_exposure_pct: float = 0.0  # 같은 섹터 기존 비중 (%)
    open_positions: int = 0


TRADE_KEYS = ("risk_per_trade_pct", "max_heat_pct", "max_position_pct", "max_sector_pct",
              "max_open_positions", "kelly")


def trade_params(lab_trade: dict | None, cfg: dict) -> dict:
    """신호형 lab의 사이징 값 = risk.yaml trade.defaults 위에 lab.yaml trade 블록.
    caps를 넘으면 거부한다 (lab은 더 엄하게만 정할 수 있다). 백테스트·paper·live가 같은 값을 쓴다."""
    t = cfg["trade"]
    lab_trade = {k: v for k, v in (lab_trade or {}).items() if v is not None}
    unknown = set(lab_trade) - set(TRADE_KEYS)
    if unknown:
        raise ValueError(f"lab.yaml trade에 모르는 항목: {sorted(unknown)} (가능: {TRADE_KEYS})")
    p = {**t["defaults"], **lab_trade}
    for k, cap in (t.get("caps") or {}).items():
        if p.get(k) is not None and p[k] > cap:
            raise ValueError(f"lab.yaml trade.{k}={p[k]} > risk.yaml 상한 {cap}")
    return p


def make_plan(signal: Signal, account: Account, now: datetime, cfg: dict, *, trade: dict | None = None,
              lot: float = 1.0, multiplier: float = 1.0, min_notional: float = 0.0,
              win_rate: float | None = None, payoff: float | None = None) -> Plan:
    """신호형(type: trade) 전략의 Signal → Plan.
    trade: lab.yaml trade 블록 (없으면 risk.yaml trade.defaults). trade_params()로 합치고 상한을 검사한다.
    가드 순서: kill_switch → 데이터 신선도 → 보유 종목 수 → 사이징(가장 엄한 한도) → 최소 주문 금액.
    첫 번째로 걸린 사유를 blocked에 남기고, 지나온 검사는 checks에 남긴다."""
    checks: list[str] = []
    p = trade_params(trade, cfg)

    def plan(qty=0.0, sizing=None, blocked=None, stop=signal.stop) -> Plan:
        return Plan(signal_id=signal.id, lab=signal.lab, symbol=signal.symbol, market=signal.market,
                    side=signal.side, entry=signal.price, stop=stop, target=signal.target,
                    intraday=signal.intraday, qty=qty, reason=signal.reason, ts=now,
                    sizing=sizing, blocked=blocked, checks=checks)

    if cfg["global"].get("kill_switch"):
        return plan(blocked="kill_switch 켜짐 — 신규 주문 전면 중단")
    checks.append("kill_switch ok")

    if why := check_staleness(signal.market, signal.quote_ts, now, cfg):
        return plan(blocked=f"데이터 신선도: {why}")
    checks.append("data_staleness ok")

    max_open = p["max_open_positions"]
    if account.open_positions >= max_open:
        return plan(blocked=f"보유 종목 {account.open_positions} ≥ 한도 {max_open}")
    checks.append("max_open_positions ok")

    if account.currency == "KRW":
        krw = 1.0
    elif account.krw_per_unit:
        krw = account.krw_per_unit
    else:
        raise ValueError(f"계좌 통화 {account.currency}: krw_per_unit 없이는 주문 상한(원) 환산 불가")

    check_min = cfg["orders"].get("min_notional_check", True)
    res = size(SizingInput(
        equity=account.equity, entry=signal.price, stop=signal.stop, atr=signal.atr,
        risk_pct=p["risk_per_trade_pct"], side=signal.side, lot=lot, multiplier=multiplier,
        max_position_pct=p["max_position_pct"], max_sector_pct=p["max_sector_pct"],
        sector_exposure_pct=account.sector_exposure_pct, max_heat_pct=p["max_heat_pct"],
        open_heat_pct=account.open_heat_pct, max_order_value=cfg["orders"]["max_order_value_krw"] / krw,
        # min_notional 차단은 sizing이 사유와 함께 낸다. 설정에서 끄면 0으로 넘겨 검사 생략
        min_notional=min_notional if check_min else 0.0,
        win_rate=win_rate if p.get("kelly") == "half" else None, payoff=payoff))
    if res.blocked:
        return plan(sizing=res, blocked=f"사이징: {res.blocked}", stop=res.stop)
    checks.append(f"sizing ok (binding={res.binding})")
    if check_min:
        checks.append("min_notional ok")
    return plan(qty=res.qty, sizing=res, stop=res.stop)


def plan_to_order(plan: Plan, now: datetime, cfg: dict, mode: str = "paper",
                  limit_price: float | None = None) -> Order:
    """승인된 Plan을 진입 주문으로. require_limit_orders면 지정가(기본 진입가)."""
    if not plan.approved:
        raise ValueError(f"승인되지 않은 Plan은 주문으로 만들 수 없음: {plan.blocked}")
    limit = cfg["orders"].get("require_limit_orders", True)
    return Order(plan_id=plan.id, lab=plan.lab, symbol=plan.symbol, market=plan.market,
                 side="buy" if plan.side == "long" else "sell", qty=plan.qty,
                 order_type="limit" if limit else "market",
                 limit_price=(limit_price or plan.entry) if limit else limit_price,
                 reason=plan.reason, ts=now, mode=mode)


# --- §9-5 보유 포지션 보호 ---------------------------------------------------------

@dataclass
class ExitDecision:
    symbol: str
    kind: str          # unmeasurable / stop_loss / take_profit / flat_before_close
    reason: str
    price: float | None
    ts: datetime


def check_position(pos: Position, price: float | None, now: datetime, cfg: dict, *,
                   close_at: datetime | None = None, missed_quotes: int = 0) -> ExitDecision | None:
    """청산해야 하면 ExitDecision, 아니면 None.
    우선순위: 측정 불가 → 손절 → 목표가 → 장 마감 전 정리.
    close_at: 오늘 장 마감 시각 (UTC aware). 24시간 시장이거나 모르면 None."""
    g = cfg["position_guard"]
    now = utc(now)

    def exit_(kind, reason):
        return ExitDecision(pos.symbol, kind, reason, price, now)

    if is_unmeasurable(missed_quotes, cfg):
        return exit_("unmeasurable", f"시세 {missed_quotes}회 연속 수신 실패 — 측정 불가 포지션 정리")
    if price is None:
        return None   # 아직 재시도 한도 안. 이번 회차는 판정 보류

    long = pos.side == "long"
    if g.get("enforce_stop_loss", True) and pos.stop is not None and             ((long and price <= pos.stop) or (not long and price >= pos.stop)):
        return exit_("stop_loss", f"손절: 현재가 {price} {'≤' if long else '≥'} 손절가 {pos.stop}")
    if g.get("enforce_take_profit", True) and pos.target is not None and             ((long and price >= pos.target) or (not long and price <= pos.target)):
        return exit_("take_profit", f"목표가 도달: 현재가 {price} {'≥' if long else '≤'} 목표가 {pos.target}")

    if pos.intraday and close_at is not None:
        mins = (g.get("flat_before_close_min") or {}).get(pos.market)
        if mins is not None:
            left = utc(close_at) - now
            if left <= timedelta(minutes=mins):
                return exit_("flat_before_close",
                             f"장 마감 {max(0, left.total_seconds()) / 60:.0f}분 전 (기준 {mins}분) — 당일 청산")
    return None
