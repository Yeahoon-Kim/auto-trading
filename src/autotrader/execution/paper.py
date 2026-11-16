"""운용데스크 (paper): 목표 비중 → 리스크 한도 → 주문 → 가상 체결 → 장부.

lab의 generate_signals()가 낸 목표 비중을 그대로 받아, 실제 주문처럼 **전날 종가로 수량을 정한 지정가 주문**을 내고
다음 봉 시가에 가상 체결한다. 백테스트 엔진(backtest/engine.py, whole_shares)과 같은 규칙을 따르되,
엔진은 체결 봉 시가로 수량을 정하고 데스크는 주문 시점(종가)에 정하므로 결과가 조금 다르다.
그 차이는 tests/test_paper.py가 잰다 (risk.yaml live_gate.max_paper_vs_backtest_gap_pct와 같은 개념).

봉 하나를 처리하는 순서 (step):
  (1~2는 체결·평가, 3은 퇴출 감시, 4는 주문)
  1. settle  — 이전 봉에 낸 주문을 이 봉 시가에 체결 (매도 먼저, 현금 부족하면 큰 매수부터 줄임).
               거래정지·시가 없음·지정가 밖이면 취소한다 (엔진도 그날 못 사면 다음 신호까지 기다린다).
  2. mark    — 기업행위(분할·병합) 반영: 원본가 수익과 수정주가 수익이 다르면 보유 수량을 그 비율로 고친다.
               보유 종목 시세가 없으면 missed를 세고, risk.yaml 한도에 닿으면 마지막 가격에 가상 청산(측정 불가).
               종가로 평가해 자산·고점을 갱신한다.
  3. retire  — lab.yaml retirement(낙폭·연속 손실)에 걸리면 알림만 낸다. 매매는 백테스트와 같게 계속하고
               퇴출은 사람이 status로 정한다 (전략 결과를 바꾸는 자동 차단은 두지 않는다).
  4. decide  — 신호가 나온 봉이면 목표 비중을 주문으로 바꾼다. risk.yaml의 계좌 보호 한도만 적용한다:
               kill_switch면 매수 중단(매도는 허용), 1회 주문 금액 상한, 최소 주문 금액,
               비중 합 상한·음수 비중(버그 의심 → 그날 주문 전부 보류).

주의
- 비중형(type: portfolio) 전용. 종목 수·종목 비중은 전략이 정하고 백테스트에 들어간 그대로 따른다.
  손절 기반 사이징(risk.guards.make_plan)은 신호형(type: trade) 전략 경로에서만 쓴다. Order.plan_id에는 리밸런싱 id.
- 상장폐지는 미래를 모르므로 엔진처럼 "마지막 가격 다음 봉에 청산"할 수 없다 → 시세가 연속으로 없으면 정리.
- 가격은 원본가(원화·달러 그대로)로 장부를 쓴다. kr_stock은 panel의 raw_close/close 비율로 원본 시가를 만든다.
- 순수 함수다. 파일 저장·알림·데이터 읽기는 runner/paper.py가 한다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from autotrader.backtest.engine import Costs
from autotrader.core.schemas import Fill, Order

EPS = 1e-9


@dataclass
class Book:
    """모의 계좌 장부. JSON으로 그대로 저장한다."""
    cash: float
    shares: dict[str, float] = field(default_factory=dict)
    pending: list[dict] = field(default_factory=list)     # 다음 봉에 체결할 주문 (Order 필드 + bar)
    last_bar: str | None = None                          # 마지막으로 처리한 봉 (같은 봉 두 번 처리 방지)
    last_px: dict[str, float] = field(default_factory=dict)   # 종목별 마지막 원본 종가
    missed: dict[str, int] = field(default_factory=dict)      # 보유 종목 시세 연속 결측 횟수
    equity: float = 0.0
    prev_equity: float = 0.0
    peak: float = 0.0
    losing_streak: int = 0        # 연속 손실 봉 수 (퇴출 조건 감시용)
    retirement: list[str] = field(default_factory=list)   # 걸린 퇴출 조건 (처음 걸린 봉과 사유). 매매는 계속한다

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Book:
        return cls(**d)


@dataclass
class Bar:
    """봉 하나의 종목별 값 (원본가 기준). 시세가 없으면 NaN, 목록에 없으면 present=False."""
    date: pd.Timestamp
    open: pd.Series
    close: pd.Series
    adj_close: pd.Series
    tradable: pd.Series
    present: pd.Series
    factor: pd.Series | None = None   # kr_adj_factor (panel["_f"], 분할·병합만). 없으면 비율 범위로 나눈다


def bars(panel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """panel → (원본 시가, 원본 종가, 거래 가능, 시세 행 있음). kr_stock 외에는 수정 = 원본."""
    c = panel.close
    ratio = (panel["raw_close"] / c) if "raw_close" in panel else pd.DataFrame(1.0, c.index, c.columns)
    ro, rc = panel.open * ratio, c * ratio
    if "tradable" in panel:
        t = panel["tradable"].reindex_like(c)
        present = t.notna()
        tradable = t.astype("boolean").fillna(False).astype(bool)
    else:
        present = c.notna()
        tradable = present.copy()
    return ro, rc, tradable, present


def bar_at(panel, raw: tuple, i: int) -> Bar:
    ro, rc, tr, pr = raw
    return Bar(panel.index[i], ro.iloc[i], rc.iloc[i], panel.close.iloc[i], tr.iloc[i], pr.iloc[i],
               panel["_f"].iloc[i] if "_f" in panel else None)


def _floor(q: float, lot: float) -> float:
    return round(math.floor(q / lot + EPS) * lot, 10)


def _ts(d: pd.Timestamp) -> datetime:
    """봉 날짜 → UTC aware (기록용. 일봉은 날짜만 의미가 있다)."""
    d = pd.Timestamp(d)
    return (d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC")).to_pydatetime()


# --- 1. 체결 -------------------------------------------------------------------------

def settle(book: Book, bar: Bar, costs: Costs, tax: float, lab: str, lot: float) -> list[dict]:
    """대기 주문을 이 봉 시가에 체결한다. 이벤트 목록을 돌려준다."""
    ev: list[dict] = []
    if not book.pending:
        return ev
    fee = costs.per_trade
    ok, gone = [], []
    for o in book.pending:
        s, px = o["symbol"], bar.open.get(o["symbol"], np.nan)
        lim = o.get("limit_price")
        why = None
        if not bool(bar.tradable.get(s, False)) or not np.isfinite(px) or px <= 0:
            why = "거래정지 또는 시가 없음"
        elif lim is not None and ((o["side"] == "buy" and px > lim) or (o["side"] == "sell" and px < lim)):
            why = f"시가 {px:,.2f}가 지정가 {lim:,.2f} 밖"
        (gone if why else ok).append((o, px, why))
    for o, _, why in gone:
        ev.append({"kind": "cancel", "order_id": o["id"], "symbol": o["symbol"], "side": o["side"],
                   "qty": o["qty"], "reason": why})

    sells = [(o, px) for o, px, _ in ok if o["side"] == "sell"]
    buys = [[o, px, o["qty"]] for o, px, _ in ok if o["side"] == "buy"]
    proceeds = sum(min(o["qty"], book.shares.get(o["symbol"], 0.0)) * px * (1 - fee - tax) for o, px in sells)

    # 현금이 모자라면 가장 큰 매수부터 lot씩 줄인다 (엔진 whole_shares와 같은 규칙)
    def short():
        return book.cash + proceeds - sum(q * px * (1 + fee) for _, px, q in buys)

    while buys and short() < -1e-9:
        j = max(range(len(buys)), key=lambda k: buys[k][2] * buys[k][1])
        if buys[j][2] <= EPS:
            break
        buys[j][2] = _floor(buys[j][2] - lot, lot)

    for o, px in sells:
        q = min(o["qty"], book.shares.get(o["symbol"], 0.0))
        if q <= EPS:
            continue
        _fill(book, ev, o, q, px, fee, tax, lab, bar.date)
    for o, px, q in buys:
        if q < o["qty"] - EPS:
            ev.append({"kind": "reduce", "order_id": o["id"], "symbol": o["symbol"],
                       "from": o["qty"], "to": q, "reason": "현금 부족 (매도세·수수료·시가 상승)"})
        if q > EPS:
            _fill(book, ev, o, q, px, fee, 0.0, lab, bar.date)
    book.pending = []
    return ev


def _fill(book, ev, o, q, px, fee, tax, lab, date) -> None:
    notional = q * px
    f = Fill(order_id=o["id"], lab=lab, symbol=o["symbol"], side=o["side"], qty=q, price=px,
             ts=_ts(date), fee=notional * fee, tax=notional * tax)
    sign = 1 if o["side"] == "buy" else -1
    book.cash -= sign * notional + f.fee + f.tax
    n = book.shares.get(o["symbol"], 0.0) + sign * q
    if abs(n) <= EPS:
        book.shares.pop(o["symbol"], None)
    else:
        book.shares[o["symbol"]] = n
    d = asdict(f)
    d["ts"] = f.ts.isoformat()
    ev.append({"kind": "fill", **d})


# --- 2. 평가 -------------------------------------------------------------------------

# 원본가·수정주가 수익 차이가 이 안이면 무시. 실데이터에서 본 크기 (2026-11-16):
#   수정주가 부동소수 오차 ±4e-6 (068270), kr_adj_factor 자체가 날마다 ±1.3e-4 흔들림 (012200·187660, 분할 아님)
#   가장 작은 실제 배당락 0.14% (000660). 1e-4로 두면 흔들림이 분할로 잡혀 하루 1주씩 단주 정산됐다.
NOISE = 5e-4
MIN_SPLIT_FACTOR = 0.7   # 수정계수 정보가 없을 때: 비율이 0.7~1이면 배당락, 그 밖은 분할·병합으로 본다


def split_dividend(f: float, split: float | None) -> tuple[float, float]:
    """원본가·수정주가 수익 비율 f = (원본 종가 수익) / (수정 종가 수익)을 (주식 수 배율, 배당 비율)로 나눈다.
    수정주가 = 원본가 × kr_adj_factor × (세후 배당 재투자) × (주식수 기반 병합 보정)이라
    f = 분할 부분 × 배당 부분 × 그 밖 보정. split = 직전 kr_adj_factor / 오늘 kr_adj_factor (없으면 None).
    반환: shares_div — 보유 수량을 이 값으로 나눈다 (2:1 분할 0.5 → 수량 2배). 1이면 수량 그대로.
          div — 배당 부분 r (0.7~1). 세후 주당 배당 = 원본 종가 × (1/r - 1). 1이면 배당 없음."""
    if abs(f - 1) <= NOISE:
        return 1.0, 1.0
    s = split if split is not None and abs(split - 1) > NOISE else 1.0
    if split is None and not (MIN_SPLIT_FACTOR < f < 1):
        s = f                                   # 수정계수 정보 없음 → 범위 밖이면 분할·병합
    r = f / s
    if abs(r - 1) <= NOISE:
        return s, 1.0
    if MIN_SPLIT_FACTOR < r < 1:
        return s, r                             # 원본가가 수정주가보다 덜 오름 → 배당락 (현금)
    return s * r, 1.0                           # 수정계수 밖의 주식 수 변화 (주식수 기반 병합·감자 보정 등)


def mark(book: Book, bar: Bar, prev_adj: pd.Series | None, costs: Costs, tax: float,
         risk: dict, lab: str, prev_factor: pd.Series | None = None, lot: float = 1.0) -> list[dict]:
    """기업행위 반영, 측정 불가 정리, 종가 평가. prev_adj = 현재 panel의 '직전 처리 봉' 수정 종가,
    prev_factor = 그 봉의 kr_adj_factor (panel["_f"]가 있을 때).
    기업행위: 배당락은 세후 현금 배당을 장부에 넣고(수량 그대로), 분할·병합은 수량을 고친다.
    정수 주식으로 내림하고 남는 단주는 그날 원본 종가로 현금 정산한다 (실제 단주 대금처럼).
    배당 현금은 배당락일에 넣는다 (실제 입금은 한두 달 뒤 — 엔진의 배당 재투자 가정과 맞추려는 것)."""
    ev: list[dict] = []
    for s in list(book.shares):
        raw, adj = bar.close.get(s, np.nan), bar.adj_close.get(s, np.nan)
        old = book.last_px.get(s)
        pa = prev_adj.get(s, np.nan) if prev_adj is not None else np.nan
        if np.isfinite(raw) and np.isfinite(adj) and old and np.isfinite(pa) and pa > 0 and adj > 0 and raw > 0:
            f = (raw / old) / (adj / pa)
            split = None
            if bar.factor is not None and prev_factor is not None:
                fn, fp = bar.factor.get(s, np.nan), prev_factor.get(s, np.nan)
                if np.isfinite(fn) and np.isfinite(fp) and fn > 0 and fp > 0:
                    split = fp / fn
            k, r = split_dividend(f, split)
            q0 = book.shares[s]
            if k != 1.0:
                q = q0 / k
                qi = round(q) if abs(q - round(q)) < 1e-6 else _floor(q, lot)
                frac = (q - qi) * raw
                book.shares[s] = qi
                book.cash += frac
                ev.append({"kind": "corporate_action", "symbol": s, "factor": k, "from": q0, "to": qi,
                           "cash_in_lieu": frac,
                           "reason": ("kr_adj_factor 분할·병합" if split is not None and abs(split - 1) > NOISE
                                      else "원본가·수정주가 수익 차이 (수정계수 밖 병합·감자 보정 등)")
                                     + " → 보유 수량 조정 (실제 계좌 내역과 대조 필요)"})
            if r != 1.0:
                per_share = raw * (1 / r - 1)
                cash = book.shares[s] * per_share
                book.cash += cash
                ev.append({"kind": "dividend", "symbol": s, "per_share_net": per_share, "qty": book.shares[s],
                           "cash": cash, "reason": "배당락: 세후 배당을 현금으로 (실제 입금은 지급일, 계좌 내역과 대조)"})
        if bool(bar.present.get(s, False)):
            book.missed.pop(s, None)
            if np.isfinite(raw) and raw > 0:
                book.last_px[s] = float(raw)
        else:
            book.missed[s] = book.missed.get(s, 0) + 1
            if book.missed[s] >= risk["data_staleness"]["unmeasurable_retries"]:
                px = book.last_px.get(s, 0.0)
                q = book.shares.pop(s)
                notional = q * px
                book.cash += notional * (1 - costs.per_trade - tax)
                book.missed.pop(s, None)
                ev.append({"kind": "unmeasurable_exit", "symbol": s, "qty": q, "price": px,
                           "reason": f"시세 {risk['data_staleness']['unmeasurable_retries']}회 연속 없음 "
                                     "(상장폐지 가능) → 마지막 가격에 가상 청산. 실전이면 수동 확인"})
    book.last_px = {s: v for s, v in book.last_px.items() if s in book.shares}   # 보유 종목만 기억
    book.prev_equity = book.equity
    book.equity = book.cash + sum(q * book.last_px.get(s, 0.0) for s, q in book.shares.items())
    book.peak = max(book.peak, book.equity)
    return ev


# --- 3. 주문 -------------------------------------------------------------------------

def buy_block(book: Book, risk: dict) -> str | None:
    """신규 매수를 막을 사유. 매도(위험 축소)는 막지 않는다."""
    if risk["global"].get("kill_switch"):
        return "kill_switch 켜짐"
    return None


def check_retirement(book: Book, rules: dict | None, bar_date) -> list[dict]:
    """lab.yaml retirement 감시. 걸리면 알림 이벤트만 낸다 — 매매는 백테스트와 같게 계속하고,
    퇴출(status: retired)은 사람이 정한다. 같은 조건은 처음 걸린 한 번만 알린다."""
    rules = {k: v for k, v in (rules or {}).items() if v not in (None, "")}
    if book.prev_equity > 0:
        book.losing_streak = book.losing_streak + 1 if book.equity < book.prev_equity - EPS else 0
    hits = []
    if rules.get("max_drawdown_pct") is not None and book.peak > 0:
        dd = (book.equity / book.peak - 1) * 100
        if dd <= -float(rules["max_drawdown_pct"]):
            hits.append(("max_drawdown_pct", f"낙폭 {dd:.2f}% ≤ -{rules['max_drawdown_pct']}%"))
    if rules.get("max_losing_days") is not None and book.losing_streak >= int(rules["max_losing_days"]):
        hits.append(("max_losing_days", f"연속 손실 {book.losing_streak}봉 ≥ {rules['max_losing_days']}"))
    ev = []
    seen = {r.split(":")[0] for r in book.retirement}
    for rule, why in hits:
        if rule in seen:
            continue
        book.retirement.append(f"{rule}: {pd.Timestamp(bar_date).date()} {why}")
        ev.append({"kind": "retirement", "rule": rule,
                   "reason": f"{why} — 퇴출 조건. 매매는 계속하며, 퇴출 여부는 사람이 정한다 (lab.yaml status)"})
    return ev


def decide(book: Book, bar: Bar, target: pd.Series, costs: Costs, risk: dict, *, lab: str, market: str,
           lot: float = 1.0, min_notional: float = 0.0, krw_per_unit: float = 1.0,
           limit_band_pct: float | None = 5.0, now: datetime | None = None) -> list[dict]:
    """목표 비중 → 대기 주문 (book.pending). 이벤트 목록을 돌려준다."""
    ev: list[dict] = []
    now = now or datetime.now(UTC)
    fee = costs.per_trade
    E = book.equity
    o_cfg = risk["orders"]
    w = target[target.abs() > EPS].astype(float)
    # 종목 수·종목 비중은 전략이 정한다 (백테스트에 들어간 그대로). 여기서는 버그로 보이는 비중만 막는다.
    gross = float(w.sum())
    bad = [] if np.isfinite(w).all() else ["비중에 NaN·무한대"]
    if (w < 0).any():
        bad.append(f"음수 비중 {int((w < 0).sum())}종목 (paper 데스크는 롱만)")
    if gross > o_cfg.get("max_gross_weight", 1.0) + 1e-6:
        bad.append(f"비중 합 {gross:.4f} > 상한 {o_cfg.get('max_gross_weight', 1.0)}")
    if bad:
        book.pending = []
        ev.append({"kind": "risk_block", "rule": "weights_invalid",
                   "reason": "; ".join(bad) + " — 전략 버그 의심, 오늘 주문 전부 보류 (보유는 유지)"})
        return ev

    blocked = buy_block(book, risk)
    max_val = o_cfg["max_order_value_krw"] / krw_per_unit
    limit = o_cfg.get("require_limit_orders", True)
    rid = f"rebalance-{lab}-{pd.Timestamp(bar.date).date()}"
    out = []
    for s in sorted(set(w.index) | set(book.shares)):
        px = bar.close.get(s, np.nan)
        cur = book.shares.get(s, 0.0)
        if not np.isfinite(px) or px <= 0:
            if abs(w.get(s, 0.0) * E) > EPS or cur:
                ev.append({"kind": "skip", "symbol": s, "reason": "오늘 종가 없음 — 수량 계산 불가"})
            continue
        want = w.get(s, 0.0) * E / px
        if want > cur + EPS:
            q = _floor((want - cur) / (1 + fee), lot)
            if q <= EPS:
                continue
            if blocked:
                ev.append({"kind": "risk_block", "rule": "buy_halt", "symbol": s, "qty": q, "reason": blocked})
                continue
            if q * px > max_val:
                q2 = _floor(max_val / px, lot)
                ev.append({"kind": "risk_clip", "rule": "max_order_value_krw", "symbol": s, "from": q, "to": q2,
                           "reason": f"1회 주문 상한 {o_cfg['max_order_value_krw']:,}원"})
                q = q2
            if o_cfg.get("min_notional_check", True) and (q <= EPS or q * px < min_notional):
                ev.append({"kind": "risk_block", "rule": "min_notional", "symbol": s, "qty": q,
                           "reason": f"주문 금액 {q * px:,.2f} < 최소 {min_notional:,.2f}"})
                continue
            side = "buy"
        elif want < cur - EPS:
            q = cur - _floor(want, lot) if want > EPS else cur
            if q <= EPS:
                continue
            side = "sell"
        else:
            continue
        band = (limit_band_pct or 0) / 100
        lp = (px * (1 + band) if side == "buy" else px * (1 - band)) if limit else None
        od = Order(plan_id=rid, lab=lab, symbol=s, market=market, side=side, qty=q,
                   order_type="limit" if limit else "market", limit_price=lp,
                   reason=f"리밸런싱: 목표 비중 {w.get(s, 0.0):.2%} (현재 {cur:g} → {cur + (q if side == 'buy' else -q):g})",
                   ts=now, mode="paper")
        d = asdict(od)
        d["ts"] = od.ts.isoformat()
        d["bar"] = str(pd.Timestamp(bar.date).date())
        out.append(d)
        ev.append({"kind": "order", **d})
    book.pending = out
    return ev


# --- 봉 하나 처리 ----------------------------------------------------------------------

def should_fire(W: pd.DataFrame, i: int, rebalance: str, first: bool = False, enter_now: bool = False) -> bool:
    """엔진 run()과 같은 판정: 봉 i 종가에 낸 목표로 다음 봉에 맞출지."""
    if rebalance == "always":
        return True
    if rebalance != "on_signal":
        raise ValueError(f"paper 데스크가 지원하지 않는 rebalance: {rebalance}")
    if first and enter_now:
        return True
    prev = W.iloc[i - 1].to_numpy() if i >= 1 else np.zeros(W.shape[1])
    if (np.abs(W.iloc[i].to_numpy() - prev) > EPS).any():
        return True
    sig = pd.DatetimeIndex(W.attrs.get("signal_dates", ()))
    dense = len(sig) >= len(W.index)      # 매 봉 신호를 내는 전략이면 신호 날 기준은 의미 없음
    return bool(W.index[i] in sig) and not dense


def step(book: Book, panel, raw: tuple, W: pd.DataFrame, i: int, costs: Costs, risk: dict, *,
         lab: str, market: str, rebalance: str, lot: float = 1.0, min_notional: float = 0.0,
         krw_per_unit: float = 1.0, limit_band_pct: float | None = 5.0, enter_now: bool = False,
         retirement: dict | None = None, now: datetime | None = None) -> list[dict]:
    """봉 i 하나: 체결 → 평가 → 퇴출 조건 감시 → (신호면) 주문. 같은 봉을 두 번 처리하지 않는다.
    retirement: lab.yaml retirement 블록 (max_drawdown_pct, max_losing_days)."""
    date = panel.index[i]
    if book.last_bar is not None and pd.Timestamp(book.last_bar) >= date:
        return [{"kind": "skip_bar", "bar": str(date.date()), "reason": "이미 처리한 봉"}]
    bar = bar_at(panel, raw, i)
    tax = float(costs.tax_rates(pd.DatetimeIndex([date]))[0])
    first = book.last_bar is None
    prev_adj = prev_factor = None
    if not first and pd.Timestamp(book.last_bar) in panel.index:
        prev_adj = panel.close.loc[pd.Timestamp(book.last_bar)]
        if "_f" in panel:
            prev_factor = panel["_f"].loc[pd.Timestamp(book.last_bar)]
    ev = settle(book, bar, costs, tax, lab, lot)
    ev += mark(book, bar, prev_adj, costs, tax, risk, lab, prev_factor=prev_factor, lot=lot)
    if first:
        book.peak = max(book.peak, book.equity)
        book.prev_equity = book.equity
    ev += check_retirement(book, retirement, date)
    if should_fire(W, i, rebalance, first, enter_now):
        ev += decide(book, bar, W.iloc[i], costs, risk, lab=lab, market=market, lot=lot,
                     min_notional=min_notional, krw_per_unit=krw_per_unit,
                     limit_band_pct=limit_band_pct, now=now)
    book.last_bar = str(date.date())
    for e in ev:
        e.setdefault("bar", book.last_bar)
    ev.append({"kind": "equity", "bar": book.last_bar, "equity": book.equity, "cash": book.cash,
               "positions": len(book.shares), "pending": len(book.pending)})
    return ev
