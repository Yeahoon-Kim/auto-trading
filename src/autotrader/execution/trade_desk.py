"""신호형(type: trade) 데스크 처리: 백테스트 재생(backtest/trade.py)과 paper 데스크가 같이 쓰는 봉 단위 규칙.

규칙이 여기 한 곳에만 있다. 백테스트와 paper가 다른 결과를 내면 입력(가격·신호·계좌)이 다른 것이지 규칙이 다른 것이 아니다.

봉 하나의 순서 (호출하는 쪽이 이 순서로 부른다)
  1. manage — 보유 포지션 청산 판정 (guards.check_position을 시가 → 불리한 극값 → 유리한 극값 → 종가 순으로)
  2. fill   — 직전 봉에 만든 계획 체결 (시장가: 시가 / 지정가: 하루짜리, 안 닿으면 버림) + 체결 봉 안 청산 판정
  3. plan   — 이 봉 종가 평가액으로 신호 → guards.make_plan (가드 + sizing.size) → 다음 봉 체결 대기

규칙 (보수적)
- 같은 봉에서 손절과 목표가가 둘 다 닿으면 손절 (봉 안의 순서를 모르므로 불리한 극값을 먼저 본다).
- 시가가 이미 손절/목표 너머면 시가에 청산 (갭은 손절가보다 나쁘게 체결).
- 지정가가 장중에 체결된 봉에서는 목표가를 인정하지 않는다 (체결 전에 고가가 나왔을 수 있음). 손절은 인정.
- 당일 청산(intraday): 봉 끝 시각과 그날 장 마감 시각으로 check_position의 장 마감 전 정리 판정 → 그 봉 종가.
- 같은 종목을 보유(또는 체결 대기) 중이면 새 신호는 버린다.
- heat·보유 종목 수에는 열린 포지션과 같은 봉의 체결 대기 계획을 모두 넣는다.
비용: 체결 금액 × fee(수수료+슬리피지), 롱 청산 금액 × tax(매도세).

이 모듈은 backtest/를 import하지 않는다 (순환 방지). 가격·시각·비용률은 호출하는 쪽이 넘긴다.
상장폐지(백테스트: 미래 데이터로 앎)·측정 불가(paper: 시세 연속 결측)는 호출하는 쪽이 close()로 처리한다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from autotrader.core.schemas import Position, Signal
from autotrader.risk import guards

# quote(symbol) → (open, high, low, close, 거래 가능) 또는 None(이 봉 시세 없음)
Quote = Callable[[str], "tuple[float, float, float, float, bool] | None"]


@dataclass
class DeskSpec:
    """계좌·사이징 설정 (실행 중 바뀌지 않음)."""
    lab: str
    market: str
    currency: str = "KRW"
    krw_per_unit: float | None = 1.0
    lot: float = 1.0
    min_notional: float = 0.0
    multiplier: float = 1.0
    trade_cfg: dict | None = None      # lab.yaml trade 블록

    def check(self) -> None:
        if self.currency != "KRW" and not self.krw_per_unit:
            # make_plan이 신호마다 ValueError를 내 전부 "신호 오류"로 조용히 막히는 것을 막는다
            raise ValueError(f"계좌 통화 {self.currency}: lab.yaml paper.krw_per_unit이 필요하다 (주문 상한 환산용)")


@dataclass
class TradeBook:
    """신호형 장부. paper는 JSON으로 저장한다 (시각은 문자열로 바뀌어도 된다).
    positions: 종목 → {side, qty, entry, stop, target, intraday, entry_ts, signal_ts, cost_in, risk, binding}
    pending:   다음 봉 체결 대기 [{symbol, side, qty, stop, target, intraday, limit, signal_ts, risk_pct, binding, ...}]
    closed:    청산된 거래 기록, blocked: 진입 못 한 신호 사유별 건수"""
    cash: float
    positions: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    blocked: dict = field(default_factory=dict)
    # paper 운영용 (백테스트는 안 씀). 필드 이름은 execution/paper.py Book과 같다 (check_retirement 공용)
    last_bar: str | None = None
    last_px: dict = field(default_factory=dict)
    missed: dict = field(default_factory=dict)
    equity: float = 0.0
    prev_equity: float = 0.0
    peak: float = 0.0
    losing_streak: int = 0
    retirement: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON으로 바로 쓸 수 있는 dict (시각은 문자열로)."""
        return json.loads(json.dumps(asdict(self), default=str))

    @classmethod
    def from_dict(cls, d: dict) -> TradeBook:
        return cls(**d)


def sgn(side: str) -> int:
    return 1 if side == "long" else -1


def aware(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def bar_length(timeframe: str) -> pd.Timedelta:
    n = int(timeframe[:-1] or 1)
    return {"d": pd.Timedelta(days=n), "h": pd.Timedelta(hours=n), "m": pd.Timedelta(minutes=n)}[timeframe[-1]]


def bar_times(ts, day_last_ts, blen: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(봉 시작, 봉 끝, 그날 장 마감 = 그날 마지막 봉의 끝). 모두 UTC aware."""
    bo = aware(ts)
    return bo, bo + blen, aware(day_last_ts) + blen


def position(p: dict, symbol: str, spec: DeskSpec) -> Position:
    return Position(lab=spec.lab, symbol=symbol, market=spec.market, side=p["side"], qty=p["qty"], entry=p["entry"],
                    stop=p["stop"], target=p["target"], intraday=p["intraday"])


def value(book: TradeBook, spec: DeskSpec, px: Callable[[str], float]) -> float:
    """현금 + 보유 평가 (px: 종목 → 평가 가격)."""
    m = spec.multiplier
    return book.cash + sum(sgn(p["side"]) * p["qty"] * m * px(s) for s, p in book.positions.items())


def block(book: TradeBook, why: str, ev: list | None = None, **info) -> None:
    # 사이징 차단은 어떤 한도인지까지 (숫자만 지움), 나머지는 사유 앞부분으로 묶는다
    k = re.sub(r"\d[\d,.]*", "N", why)[:60] if why.startswith("사이징") else why.split(":")[0].split(" (")[0][:40]
    book.blocked[k] = book.blocked.get(k, 0) + 1
    if ev is not None:
        ev.append({"kind": "risk_block", "rule": k, "reason": why, **info})


def check_exit(P: Position, o, h, l, c, bar_open, bar_end, day_close, cfg, *, allow_target: bool,
               check_open: bool):
    """한 봉 안의 청산 판정 (guards.check_position을 시가 → 불리한 극값 → 유리한 극값 → 종가 순으로).
    반환: (가격, kind, 사유) 또는 None. 같은 봉에서 손절·목표가 둘 다면 손절 (불리한 극값을 먼저 본다)."""
    long = P.side == "long"
    now0 = bar_open.to_pydatetime()
    if check_open and np.isfinite(o):
        d = guards.check_position(P, float(o), now0, cfg)
        if d and d.kind in ("stop_loss", "take_profit") and (allow_target or d.kind == "stop_loss"):
            return float(o), d.kind, d.reason + " (시가 갭)"
    adverse, favorable = (l, h) if long else (h, l)
    if np.isfinite(adverse):
        d = guards.check_position(P, float(adverse), now0, cfg)
        if d and d.kind == "stop_loss":
            return float(P.stop), d.kind, d.reason
    if allow_target and np.isfinite(favorable):
        d = guards.check_position(P, float(favorable), now0, cfg)
        if d and d.kind == "take_profit":
            return float(P.target), d.kind, d.reason
    if P.intraday and day_close is not None and np.isfinite(c):
        d = guards.check_position(P, float(c), bar_end.to_pydatetime(), cfg, close_at=day_close.to_pydatetime())
        if d and d.kind == "flat_before_close":
            return float(c), d.kind, d.reason
    return None


def close(book: TradeBook, spec: DeskSpec, symbol: str, ts, price: float, kind: str, reason: str,
          fee: float, tax: float, ev: list | None = None) -> tuple[float, float]:
    """포지션 청산. 반환: (청산 금액, 비용)."""
    p = book.positions.pop(symbol)
    m = spec.multiplier
    notional = p["qty"] * m * price
    c = notional * fee + (notional * tax if p["side"] == "long" else 0.0)
    book.cash += sgn(p["side"]) * notional - c
    pnl = sgn(p["side"]) * p["qty"] * m * (price - p["entry"]) - p["cost_in"] - c
    rec = {"symbol": symbol, "side": p["side"], "signal_ts": p["signal_ts"], "entry_ts": p["entry_ts"],
           "exit_ts": ts, "entry": p["entry"], "exit": price, "stop": p["stop"], "target": p["target"],
           "qty": p["qty"], "intraday": p["intraday"], "exit_kind": kind, "exit_reason": reason,
           "cost": p["cost_in"] + c, "pnl": pnl, "r": pnl / p["risk"] if p["risk"] > 0 else np.nan,
           "binding": p["binding"]}
    book.closed.append(rec)
    if ev is not None:
        ev.append({"kind": "fill", "symbol": symbol, "side": "sell" if p["side"] == "long" else "buy",
                   "qty": p["qty"], "price": price, "fee": notional * fee, "tax": c - notional * fee,
                   "exit_kind": kind, "pnl": pnl, "r": rec["r"], "reason": reason})
    return notional, c


def manage(book: TradeBook, spec: DeskSpec, quote: Quote, ts, bar_open, bar_end, day_close, fee: float,
           tax: float, cfg: dict, ev: list | None = None) -> tuple[float, float]:
    """1. 보유 포지션 청산 판정. 시세 없음·거래 없는 날은 판정 보류. 반환: (매매 금액, 비용)."""
    traded = cost = 0.0
    for s in list(book.positions):
        q = quote(s)
        if q is None or np.isnan(q[3]) or not q[4]:
            continue
        o, h, l, c, _ = q
        ex = check_exit(position(book.positions[s], s, spec), o, h, l, c, bar_open, bar_end, day_close, cfg,
                        allow_target=True, check_open=True)
        if ex:
            a, k = close(book, spec, s, ts, *ex, fee=fee, tax=tax, ev=ev)
            traded, cost = traded + a, cost + k
    return traded, cost


def fill(book: TradeBook, spec: DeskSpec, quote: Quote, ts, bar_open, bar_end, day_close, fee: float,
         tax: float, cfg: dict, ev: list | None = None) -> tuple[float, float, int]:
    """2. 직전 봉 계획 체결 (+ 체결 봉 안 청산 판정). 반환: (매매 금액, 비용, 체결 건수)."""
    traded = cost = 0.0
    n = 0
    m = spec.multiplier
    for pl in book.pending:
        s = pl["symbol"]
        q = quote(s)
        if q is None or np.isnan(q[0]) or np.isnan(q[3]) or not q[4]:
            block(book, "미체결: 체결 봉 가격 없음", ev, symbol=s, order_id=pl.get("plan_id"))
            continue
        o, h, l, c, _ = q
        long = pl["side"] == "long"
        lim = pl["limit"]
        if lim is None:
            px, at_open = float(o), True
        elif (long and l <= lim) or (not long and h >= lim):
            at_open = (long and o <= lim) or (not long and o >= lim)
            px = float(o) if at_open else float(lim)
        else:
            block(book, "미체결: 지정가 안 닿음", ev, symbol=s, order_id=pl.get("plan_id"))
            continue
        notional = pl["qty"] * m * px
        cin = notional * fee
        book.cash -= sgn(pl["side"]) * notional + cin
        traded, cost, n = traded + notional, cost + cin, n + 1
        book.positions[s] = {"side": pl["side"], "qty": pl["qty"], "entry": px, "stop": pl["stop"],
                             "target": pl["target"], "intraday": pl["intraday"], "entry_ts": ts,
                             "signal_ts": pl["signal_ts"], "cost_in": cin,
                             "risk": pl["qty"] * m * abs(px - pl["stop"]) if pl["stop"] is not None else np.nan,
                             "binding": pl.get("binding")}
        if ev is not None:
            ev.append({"kind": "fill", "symbol": s, "side": "buy" if long else "sell", "qty": pl["qty"],
                       "price": px, "fee": cin, "tax": 0.0, "order_id": pl.get("plan_id"),
                       "reason": "진입 (시가)" if at_open else "진입 (지정가 장중 체결)"})
        # 체결 봉 안에서도 손절·(시가 체결이면) 목표가·당일 청산 판정
        ex = check_exit(position(book.positions[s], s, spec), o, h, l, c, bar_open, bar_end, day_close, cfg,
                        allow_target=at_open, check_open=at_open)
        if ex:
            a, k = close(book, spec, s, ts, *ex, fee=fee, tax=tax, ev=ev)
            traded, cost = traded + a, cost + k
    book.pending = []
    return traded, cost, n


def _num(v) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def plan(book: TradeBook, spec: DeskSpec, rows: Iterable, equity: float, ref_px: Callable[[str], float],
         signal_ts, now, cfg: dict, ev: list | None = None) -> None:
    """3. 이 봉 신호 → make_plan → 체결 대기. rows: 신호 행 (symbol, side, stop, atr, target, entry, intraday, reason).
    now: 판단 시각 (봉 끝, UTC aware datetime). 시세 시각도 now로 넘긴다 (봉 종가가 판단에 쓴 시세)."""
    rows = list(rows)
    if not rows:
        return
    m = spec.multiplier
    heat = (sum(p["risk"] for p in book.positions.values() if np.isfinite(p["risk"])) / equity * 100
            if equity > 0 else 0.0)
    heat += sum(pl["risk_pct"] for pl in book.pending)
    busy = set(book.positions) | {pl["symbol"] for pl in book.pending}
    for r in rows:
        if r.symbol in busy:
            block(book, "이미 보유 중인 종목", ev, symbol=r.symbol)
            continue
        entry = _num(r.entry)
        ref = entry if entry is not None else ref_px(r.symbol)
        if ref is None or not np.isfinite(ref) or ref <= 0:
            block(book, "판단 봉 가격 없음", ev, symbol=r.symbol)
            continue
        sgl = Signal(lab=spec.lab, symbol=r.symbol, market=spec.market, side=r.side, price=ref,
                     reason=r.reason, ts=now, quote_ts=now, stop=_num(r.stop), atr=_num(r.atr),
                     target=_num(r.target), intraday=bool(r.intraday))
        acct = guards.Account(equity=equity, currency=spec.currency, krw_per_unit=spec.krw_per_unit,
                              open_heat_pct=heat, open_positions=len(book.positions) + len(book.pending))
        try:
            p = guards.make_plan(sgl, acct, now, cfg, trade=spec.trade_cfg, lot=spec.lot,
                                 multiplier=m, min_notional=spec.min_notional)
        except ValueError as e:
            block(book, f"신호 오류: {e}", ev, symbol=r.symbol)
            continue
        if not p.approved:
            block(book, p.blocked or "차단", ev, symbol=r.symbol)
            continue
        pl = {"symbol": r.symbol, "side": p.side, "qty": p.qty, "stop": p.stop, "target": p.target,
              "intraday": p.intraday, "limit": entry, "signal_ts": signal_ts, "risk_pct": p.sizing.risk_pct,
              "binding": p.sizing.binding if p.sizing else None, "plan_id": p.id, "reason": p.reason}
        book.pending.append(pl)
        if ev is not None:
            ev.append({"kind": "order", "plan_id": p.id, "symbol": r.symbol,
                       "side": "buy" if p.side == "long" else "sell", "qty": p.qty,
                       "order_type": "limit" if entry is not None else "market_on_open", "limit_price": entry,
                       "stop": p.stop, "target": p.target, "intraday": p.intraday, "binding": pl["binding"],
                       "risk_pct": pl["risk_pct"], "reason": p.reason})
        busy.add(r.symbol)
        heat += p.sizing.risk_pct


def open_rows(book: TradeBook) -> list[dict]:
    """끝까지 보유 중인 거래 (거래 기록용)."""
    return [{"symbol": s, "side": p["side"], "signal_ts": p["signal_ts"], "entry_ts": p["entry_ts"],
             "entry": p["entry"], "stop": p["stop"], "target": p["target"], "qty": p["qty"],
             "intraday": p["intraday"], "exit_kind": "open", "cost": p["cost_in"], "binding": p["binding"]}
            for s, p in book.positions.items()]
