"""신호형(type: trade) 백테스트: 데스크 처리 코드를 과거 봉으로 재생한다 (docs/architecture.md §10).

lab의 strategy.py는 generate_trades(data, params)로 '봉 t 종가까지 보고 낸 진입 신호' 목록을 돌려준다.
재생기는 데스크와 같은 코드로 처리한다.
    신호 → risk.guards.make_plan (가드 + sizing.size, 값은 lab.yaml trade 블록) → 봉 t+1 체결
    보유 포지션 → risk.guards.check_position (손절·목표가·장 마감 전 청산)을 봉의 시가·저가·고가·종가로

신호 열 (DataFrame 또는 dict 목록)
    ts        판단 봉 (그 봉 종가까지의 데이터로 정함). panel에 없는 날짜면 다음 봉
    symbol    종목
    side      long | short
    stop      손절가 (없으면 atr 필요)
    atr       선택. stop이 없을 때 size()가 atr×2로 손절가를 정함
    target    목표가 (선택)
    entry     지정가 (선택). 비우면 다음 봉 시가에 진입 (사이징은 판단 봉 종가 기준)
    intraday  true면 당일 청산 대상 (선택, 기본 false)
    reason    사람이 읽는 사유 (선택)

체결·청산 규칙 (보수적)
- 진입: entry가 없으면 봉 t+1 시가. entry가 있으면 봉 t+1 하루짜리 지정가 — 롱은 저가 ≤ entry일 때
  min(시가, entry)에, 숏은 고가 ≥ entry일 때 max(시가, entry)에 체결. 안 닿으면 미체결로 버린다.
- 보유 봉마다 순서: ① 시가 — 시가가 이미 손절/목표 너머면 시가에 청산 (갭은 손절가보다 나쁘게 체결)
  ② 불리한 극값(롱 저가, 숏 고가)으로 손절 판정 → 손절가에 청산
  ③ 유리한 극값(롱 고가, 숏 저가)으로 목표가 판정 → 목표가에 청산
  ④ 당일 청산(intraday) 포지션이면 그날 마지막 봉 종가에 장 마감 전 정리 판정 → 종가에 청산
  **같은 봉에서 손절과 목표가가 둘 다 닿으면 손절로 친다** (봉 안의 순서를 모르므로).
- 지정가가 장중에 체결된 봉에서는 목표가를 인정하지 않는다 (체결 전에 고가가 나왔을 수 있음). 손절은 인정.
- 장 마감 전 정리는 risk.yaml position_guard.flat_before_close_min이 있는 시장만 (kr_stock·us_stock).
  일봉에서는 "마감 N분 전"을 종가로 근사한다. 24시간 시장(코인)은 데스크처럼 정리하지 않는다.
- 상장폐지(마지막 가격 이후 데이터 없음): 다음 봉에 마지막 종가로 강제 청산. 가격이 빈 봉은 판정 보류.
- 같은 종목을 이미 보유(또는 체결 대기)하면 새 신호는 버린다.

계좌·가드
- 자본: lab.yaml execution.capital (없으면 settings backtest.initial_capital_krw). 사이징은 판단 봉 종가 평가액.
- heat·보유 종목 수에는 열린 포지션과 같은 봉에 만든 체결 대기 계획을 모두 넣는다 (보수적).
- 섹터 비중은 섹터 데이터가 없어 0으로 넘긴다. Kelly 승률·손익비는 넘기지 않는다 (위험 비율 그대로).
- kill_switch는 운영 스위치라 재생에서는 끈다. 데이터 신선도는 판단 시각 = 시세 시각으로 넘겨 통과시킨다.
- 1회 주문 상한(orders.max_order_value_krw)은 데스크처럼 적용된다 → 결과가 자본 크기에 따라 달라질 수 있다.
- 계좌 통화가 원화가 아니면 lab.yaml paper.krw_per_unit (고정 환율)으로 주문 상한을 환산한다.

비용: 체결 금액 × (수수료 + 슬리피지), 롱 청산(매도) 금액 × 매도세 — 공용 엔진과 같은 Costs.
반영하지 않는 것: 공매도 대차 비용, 선물 펀딩비, 시장 충격, 손절 주문의 추가 슬리피지(갭은 시가 체결로 반영).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from ..core.schemas import Position, Signal
from ..data import store
from ..risk import guards
from . import engine
from .data import Panel

COLS = ("ts", "symbol", "side", "stop", "atr", "target", "entry", "intraday", "reason")
# 시장 기본값: 수량 단위, 최소 주문 금액 (runner/paper.py MARKETS와 같은 값)
MARKET_DEFAULTS = {"kr_stock": {"lot": 1.0, "min_notional": 0.0},
                   "us_stock": {"lot": 1.0, "min_notional": 0.0},
                   "crypto_spot": {"lot": 1e-6, "min_notional": 5.0},
                   "crypto_futures": {"lot": 1e-3, "min_notional": 5.0}}


@dataclass
class Account:
    """재생용 계좌 설정 (lab.yaml에서)."""
    capital: float
    currency: str = "KRW"
    krw_per_unit: float | None = None
    lot: float = 1.0
    min_notional: float = 0.0
    multiplier: float = 1.0

    @classmethod
    def from_lab(cls, cfg: dict) -> Account:
        m = cfg["market"]
        s = yaml.safe_load((store.ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
        p = {k: v for k, v in (cfg.get("paper") or {}).items() if v not in (None, "")}
        ex = cfg.get("execution") or {}
        d = MARKET_DEFAULTS.get(m, {"lot": 1.0, "min_notional": 0.0})
        cur = (cfg.get("currency") or "").upper() or {"kr_stock": "KRW", "us_stock": "USD"}.get(m, "USDT")
        return cls(capital=float(ex.get("capital") or p.get("capital") or s["backtest"]["initial_capital_krw"]),
                   currency=cur, krw_per_unit=1.0 if cur == "KRW" else p.get("krw_per_unit"),
                   lot=float(p.get("lot", d["lot"])), min_notional=float(p.get("min_notional", d["min_notional"])),
                   multiplier=float(p.get("multiplier", 1.0)))


def backtest_risk_cfg(cfg: dict | None = None) -> dict:
    """risk.yaml을 재생용으로: kill_switch는 끈다 (전략 검증과 무관한 운영 스위치)."""
    c = copy.deepcopy(cfg or guards.load_config())
    c["global"]["kill_switch"] = False
    return c


def normalize(trades, panel: Panel) -> pd.DataFrame:
    """generate_trades 결과 → 표준 신호표 (ts = panel 봉, 정렬, 열 채움)."""
    df = pd.DataFrame(list(trades) if not isinstance(trades, pd.DataFrame) else trades.copy())
    if df.empty:
        return pd.DataFrame(columns=list(COLS))
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.rename_axis("ts").reset_index()
        else:
            raise ValueError("generate_trades 결과에 ts 열(판단 봉)이 필요하다")
    miss = {"symbol", "side"} - set(df.columns)
    if miss:
        raise ValueError(f"generate_trades 결과에 필요한 열이 없음: {sorted(miss)}")
    for c in COLS:
        if c not in df.columns:
            df[c] = {"intraday": False, "reason": ""}.get(c, np.nan)
    df["symbol"] = df["symbol"].astype(str)
    unknown = set(df["symbol"]) - set(panel.close.columns)
    if unknown:
        raise ValueError(f"panel에 없는 종목: {sorted(unknown)[:5]}")
    bad = set(df["side"]) - {"long", "short"}
    if bad:
        raise ValueError(f"side는 long|short: {sorted(bad)}")
    ts = pd.DatetimeIndex(pd.to_datetime(df["ts"]))
    pos = panel.index.searchsorted(ts)       # 휴장일 신호는 다음 봉을 판단 봉으로
    df = df[pos < len(panel.index)].copy()
    df["ts"] = panel.index[pos[pos < len(panel.index)]]
    df["intraday"] = df["intraday"].fillna(False).astype(bool)
    df["reason"] = df["reason"].fillna("").astype(str)
    for c in ("stop", "atr", "target", "entry"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[list(COLS)].sort_values(["ts", "symbol"], kind="stable").reset_index(drop=True)


def _aware(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _bar_len(panel: Panel) -> pd.Timedelta:
    tf = panel.timeframe
    n = int(tf[:-1] or 1)
    return {"d": pd.Timedelta(days=n), "h": pd.Timedelta(hours=n), "m": pd.Timedelta(minutes=n)}[tf[-1]]


def run(panel: Panel, trades, costs: engine.Costs, account: Account, start=None, end=None,
        trade_cfg: dict | None = None, risk_cfg: dict | None = None, lab: str = "lab") -> engine.Result:
    """신호표를 [start, end) 구간에서 재생한다. start 이전 판단 봉의 신호는 버린다
    (start 봉 체결에는 start 직전 봉의 신호를 쓴다). 반환: 공용 엔진과 같은 Result
    (meta["trade_log"]: 거래 기록, meta["blocked"]: 차단 사유별 건수)."""
    for f in ("open", "high", "low", "close"):
        if f not in panel:
            raise ValueError(f"신호형 백테스트에는 {f}가 필요하다 (lab.yaml fields에 high, low)")
    rcfg = backtest_risk_cfg(risk_cfg)
    sig = normalize(trades, panel)
    idx = panel.index
    n = len(idx)
    lo = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start)))
    hi = n if end is None else int(idx.searchsorted(pd.Timestamp(end)))
    syms = sorted(set(sig["symbol"]))
    O, H, L, C = (panel[f].reindex(columns=syms).to_numpy(float) for f in ("open", "high", "low", "close"))
    col = {s: j for j, s in enumerate(syms)}
    Cf = pd.DataFrame(C).ffill().to_numpy(float)          # 평가용 종가 (빈 봉은 직전 값)
    TR = (panel["tradable"].reindex(columns=syms).fillna(False).to_numpy(bool)
          if "tradable" in panel else np.ones_like(C, dtype=bool))
    valid = ~np.isnan(C)
    last = np.where(valid.any(0), n - 1 - np.argmax(valid[::-1], axis=0), -1) if syms else np.array([], int)
    tax_t = costs.tax_rates(idx)
    fee = costs.per_trade
    blen = _bar_len(panel)
    day = idx.normalize()
    sig_pos = idx.searchsorted(pd.DatetimeIndex(sig["ts"])) if len(sig) else np.array([], int)
    by_bar: dict[int, list] = {}
    for i, row in zip(sig_pos, sig.itertuples(index=False)):
        if lo - 1 <= i < hi - 1:                       # 체결 봉(i+1)이 구간 안
            by_bar.setdefault(int(i), []).append(row)

    cash = account.capital
    pos: dict[str, dict] = {}          # symbol → {position, j, entry_bar, risk, cost_in}
    pending: list[dict] = []           # 다음 봉 체결 대기 Plan
    blocked: dict[str, int] = {}
    log: list[dict] = []
    m = account.multiplier
    out_n = hi - lo
    net, gross, cost_s, turn_s, eq_s = (np.zeros(out_n) for _ in range(5))
    held = np.zeros((out_n, len(syms)))
    trades_n = forced = 0

    def value(t):
        return cash + sum(_sgn(p["position"].side) * p["position"].qty * m * Cf[t, p["j"]] for p in pos.values())

    def block(why):
        # 사이징 차단은 어떤 한도인지까지 (숫자만 지움), 나머지는 사유 앞부분으로 묶는다
        k = re.sub(r"\d[\d,.]*", "N", why)[:60] if why.startswith("사이징") else why.split(":")[0].split(" (")[0][:40]
        blocked[k] = blocked.get(k, 0) + 1

    def close_pos(s, t, price, kind, reason):
        nonlocal cash
        p = pos.pop(s)
        P = p["position"]
        notional = P.qty * m * price
        c = notional * fee + (notional * tax_t[t] if P.side == "long" else 0.0)
        cash += _sgn(P.side) * notional - c
        pnl = _sgn(P.side) * P.qty * m * (price - P.entry) - p["cost_in"] - c
        log.append({"symbol": s, "side": P.side, "signal_ts": p["signal_ts"], "entry_ts": idx[p["entry_bar"]],
                    "exit_ts": idx[t], "entry": P.entry, "exit": price, "stop": P.stop, "target": P.target,
                    "qty": P.qty, "intraday": P.intraday, "exit_kind": kind, "exit_reason": reason,
                    "cost": p["cost_in"] + c, "pnl": pnl, "r": pnl / p["risk"] if p["risk"] > 0 else np.nan,
                    "binding": p["binding"]})
        return notional, c

    prev_v = value(max(lo, 1) - 1) if lo > 0 else cash
    for t in range(max(lo, 1) - 1, hi):
        traded = tcost = 0.0
        bar_open = _aware(idx[t])
        bar_end = bar_open + blen
        last_of_day = t == n - 1 or day[t + 1] != day[t]
        if t >= lo:
            # 1) 보유 포지션 청산 판정
            for s in list(pos):
                p = pos[s]
                j, P = p["j"], p["position"]
                if last[j] < t:                                   # 상장폐지 → 마지막 종가에 청산
                    a, c = close_pos(s, t, float(C[last[j], j]), "delisted", "상장폐지: 마지막 가격에 강제 청산")
                    traded, tcost, forced = traded + a, tcost + c, forced + 1
                    continue
                if np.isnan(C[t, j]) or not TR[t, j]:          # 가격 없음·거래 없는 날 → 판정 보류
                    continue
                ex = _exit(P, O[t, j], H[t, j], L[t, j], C[t, j], bar_open, bar_end, bar_end if last_of_day else None,
                           rcfg, allow_target=True, check_open=True)
                if ex:
                    a, c = close_pos(s, t, *ex)
                    traded, tcost = traded + a, tcost + c
            # 2) 직전 봉에 만든 계획 체결
            for pl in pending:
                s, j = pl["plan"].symbol, col[pl["plan"].symbol]
                P0 = pl["plan"]
                o, h, l, c_ = O[t, j], H[t, j], L[t, j], C[t, j]
                if np.isnan(o) or np.isnan(c_) or not TR[t, j]:
                    block("미체결: 체결 봉 가격 없음")
                    continue
                long = P0.side == "long"
                if pl["limit"] is None:
                    fill, at_open = float(o), True
                elif (long and l <= pl["limit"]) or (not long and h >= pl["limit"]):
                    at_open = (long and o <= pl["limit"]) or (not long and o >= pl["limit"])
                    fill = float(o) if at_open else float(pl["limit"])
                else:
                    block("미체결: 지정가 안 닿음")
                    continue
                notional = P0.qty * m * fill
                cin = notional * fee
                cash -= _sgn(P0.side) * notional + cin
                traded, tcost, trades_n = traded + notional, tcost + cin, trades_n + 1
                P = Position(lab=P0.lab, symbol=s, market=P0.market, side=P0.side, qty=P0.qty, entry=fill,
                             stop=P0.stop, target=P0.target, intraday=P0.intraday, opened_ts=bar_open.to_pydatetime())
                pos[s] = {"position": P, "j": j, "entry_bar": t, "cost_in": cin, "signal_ts": pl["signal_ts"],
                          "risk": P.qty * m * abs(fill - P.stop) if P.stop is not None else np.nan,
                          "binding": P0.sizing.binding if P0.sizing else None}
                # 체결 봉 안에서도 손절·(시가 체결이면) 목표가·당일 청산 판정
                ex = _exit(P, o, h, l, c_, bar_open, bar_end, bar_end if last_of_day else None, rcfg,
                           allow_target=at_open, check_open=at_open)
                if ex:
                    a, c = close_pos(s, t, *ex)
                    traded, tcost = traded + a, tcost + c
            pending = []
        # 3) 이 봉 종가 평가 → 이 봉 신호로 다음 봉 계획 (판단 시각 = 봉 끝)
        V = value(t)
        rows = by_bar.get(t, [])
        if rows:
            now = bar_end.to_pydatetime()
            heat = sum(p["risk"] for p in pos.values() if np.isfinite(p["risk"])) / V * 100 if V > 0 else 0.0
            heat += sum(pl["risk_pct"] for pl in pending)
            busy = set(pos) | {pl["plan"].symbol for pl in pending}
            for r in rows:
                if r.symbol in busy:
                    block("이미 보유 중인 종목")
                    continue
                ref = float(r.entry) if np.isfinite(r.entry) else Cf[t, col[r.symbol]]
                if not np.isfinite(ref) or ref <= 0:
                    block("판단 봉 가격 없음")
                    continue
                sgl = Signal(lab=lab, symbol=r.symbol, market=panel.market, side=r.side, price=ref,
                             reason=r.reason, ts=now, quote_ts=now,
                             stop=float(r.stop) if np.isfinite(r.stop) else None,
                             atr=float(r.atr) if np.isfinite(r.atr) else None,
                             target=float(r.target) if np.isfinite(r.target) else None, intraday=bool(r.intraday))
                acct = guards.Account(equity=V, currency=account.currency, krw_per_unit=account.krw_per_unit,
                                      open_heat_pct=heat, open_positions=len(pos) + len(pending))
                try:
                    plan = guards.make_plan(sgl, acct, now, rcfg, trade=trade_cfg, lot=account.lot,
                                            multiplier=m, min_notional=account.min_notional)
                except ValueError as e:
                    block(f"신호 오류: {e}")
                    continue
                if not plan.approved:
                    block(plan.blocked or "차단")
                    continue
                pending.append({"plan": plan, "limit": float(r.entry) if np.isfinite(r.entry) else None,
                                "signal_ts": idx[t], "risk_pct": plan.sizing.risk_pct})
                busy.add(r.symbol)
                heat += plan.sizing.risk_pct
        if t >= lo:
            i = t - lo
            base = prev_v if abs(prev_v) > 1e-12 else 1.0
            net[i] = V / base - 1
            cost_s[i], turn_s[i] = tcost / base, traded / base
            gross[i] = (V + tcost) / base - 1
            eq_s[i] = V
            for s, p in pos.items():
                held[i, p["j"]] = _sgn(p["position"].side) * p["position"].qty * m * Cf[t, p["j"]] / V
        prev_v = V

    sl = idx[lo:hi]
    first = 1 if lo == 0 else 0
    tl = pd.DataFrame(log)
    open_rows = [{"symbol": s, "side": p["position"].side, "signal_ts": p["signal_ts"], "entry_ts": idx[p["entry_bar"]],
                  "entry": p["position"].entry, "qty": p["position"].qty, "exit_kind": "open"} for s, p in pos.items()]
    if open_rows:
        tl = pd.concat([tl, pd.DataFrame(open_rows)], ignore_index=True)
    res = engine.Result(
        returns=pd.Series(net, sl).iloc[first:], gross=pd.Series(gross, sl).iloc[first:],
        cost=pd.Series(cost_s, sl).iloc[first:], turnover=pd.Series(turn_s, sl).iloc[first:],
        weights=pd.DataFrame(held, sl, syms).iloc[first:], trades=trades_n, forced_exits=forced, costs=costs,
        meta={"type": "trade", "trade_log": tl, "blocked": blocked})
    res.equity = pd.Series(eq_s, sl).iloc[first:]
    res.invested = pd.Series(account.capital, sl).iloc[first:]
    return res


def _sgn(side: str) -> int:
    return 1 if side == "long" else -1


def _exit(P: Position, o, h, l, c, bar_open, bar_end, day_close, cfg, *, allow_target: bool, check_open: bool):
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


# ── 미래참조 자동 검사 ──
def check_causality(trade_fn, panel: Panel, params: dict, n_cuts: int = 3, tol: float = 1e-9) -> dict:
    """데이터를 잘라 다시 계산했을 때, 잘린 시점까지 판단한 신호(종목·방향·가격)가 같아야 한다."""
    def key(df):
        d = df.copy()
        for c in ("stop", "atr", "target", "entry"):
            d[c] = d[c].round(9)
        return d[["ts", "symbol", "side", "stop", "atr", "target", "entry", "intraday"]].astype(str) \
            .agg("|".join, axis=1).tolist()

    full = normalize(trade_fn(panel, params), panel)
    out = {"passed": True, "cuts": [], "signals": len(full)}
    for q in np.linspace(0.4, 0.85, n_cuts):
        cut = panel.index[int(len(panel.index) * q)]
        part = panel.slice(end=cut + pd.Timedelta(microseconds=1))
        a = key(full[full["ts"] <= cut])
        b = key(normalize(trade_fn(part, params), part))
        sa, sb = set(a), set(b)
        diff = sorted(sa ^ sb)
        rec = {"cut": str(cut), "signals": len(b), "mismatch": len(diff),
               "first_mismatch": diff[0][:120] if diff else None}
        out["cuts"].append(rec)
        out["passed"] &= not diff and len(a) == len(b)
    return out


# ── 워크포워드 ──
def walk_forward(strategy, panel: Panel, params: dict, costs: engine.Costs, account: Account,
                 start=None, train: str = "24M", test: str = "6M", trade_cfg: dict | None = None,
                 risk_cfg: dict | None = None, lab: str = "lab") -> dict:
    """공용 엔진 walk_forward와 같은 창. 시험 창마다 그 창 끝까지의 데이터로 신호를 만들어(fit은 학습 창으로)
    시험 구간 판단 봉의 신호만 이어 붙인 뒤 한 번에 재생한다 (창 경계를 넘는 포지션도 이어서 관리)."""
    wins = engine.windows(panel.index, start, train, test)
    if not wins:
        raise ValueError("워크포워드 창을 만들 수 없음 — 데이터 기간이 train+test보다 짧다")
    fit = getattr(strategy, "fit", None)
    kw = {"trade_cfg": trade_cfg, "risk_cfg": risk_cfg, "lab": lab}
    stitched, rows = [], []
    first_test = pd.Timestamp(wins[0]["test"][0])
    for k, wdw in enumerate(wins):
        (a, b), (c0, c1) = wdw["train"], wdw["test"]
        p = fit(panel.slice(a, b), dict(params)) if fit else params
        part = panel.slice(end=c1)
        sig = normalize(strategy.generate_trades(part, p), part)
        lo_ts = part.index[max(part.index.searchsorted(c0) - 1, 0)] if k == 0 else c0   # 첫 창 첫 봉 체결용 직전 봉
        stitched.append(sig[(sig["ts"] >= lo_ts) & (sig["ts"] < c1)])
        pb = panel.slice(end=b)
        tr = run(pb, strategy.generate_trades(pb, p), costs, account, start=a, **kw)
        rows.append({"train": (str(a.date()), str(b.date())), "test": (str(c0.date()), str(c1.date())),
                     "params": p, "_train_result": tr})
    sig_all = pd.concat(stitched, ignore_index=True) if stitched else pd.DataFrame(columns=list(COLS))
    last_end = wins[-1]["test"][1]
    test_res = run(panel, sig_all, costs, account, start=first_test, end=last_end, **kw)
    for r, wdw in zip(rows, wins):
        c0, c1 = wdw["test"]
        r["_test_returns"] = test_res.returns[(test_res.returns.index >= c0) & (test_res.returns.index < c1)]
    return {"windows": rows, "test": test_res, "trades": sig_all}


def stats(res: engine.Result) -> dict:
    """거래 통계: 승률, 평균 R, 손익비, 청산 종류별 건수, 차단 사유."""
    tl = res.meta.get("trade_log")
    done = tl[tl["exit_kind"] != "open"] if tl is not None and len(tl) else pd.DataFrame()
    out = {"closed": len(done), "open_at_end": int(len(tl) - len(done)) if tl is not None else 0,
           "blocked": dict(res.meta.get("blocked") or {})}
    if len(done):
        win, lose = done[done["pnl"] > 0], done[done["pnl"] <= 0]
        out |= {"win_rate": float(len(win) / len(done)),
                "avg_r": float(done["r"].mean()) if done["r"].notna().any() else None,
                "payoff": float(win["pnl"].mean() / -lose["pnl"].mean()) if len(win) and len(lose) and lose["pnl"].mean() < 0 else None,
                "exit_kinds": {str(k): int(v) for k, v in done["exit_kind"].value_counts().items()},
                "avg_hold_bars": None}
        bars = res.returns.index
        h = [bars.searchsorted(x) - bars.searchsorted(e) for e, x in zip(done["entry_ts"], done["exit_ts"])]
        out["avg_hold_bars"] = float(np.mean(h)) if h else None
        b = done["binding"].dropna()
        out["binding"] = {str(k): int(v) for k, v in b.value_counts().items()}
    return out


def save_log(res: engine.Result, folder, name: str) -> None:
    tl = res.meta.get("trade_log")
    if tl is not None and len(tl):
        tl.to_csv(folder / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
