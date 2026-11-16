"""운용데스크 (paper) — 신호형(type: trade): Signal → make_plan → 주문 → 가상 체결 → 포지션 감시.

봉 처리 규칙은 execution/trade_desk.py에 있고, 백테스트 재생(backtest/trade.py)도 같은 함수를 부른다.
이 모듈은 paper 운영에만 필요한 것을 더한다.

봉 하나를 처리하는 순서 (step)
  1. 기업행위 — 보유·대기 종목의 원본가 수익과 수정주가 수익이 1% 넘게 다르면(분할·병합) 수량과
               진입가·손절가·목표가·지정가를 그 비율로 고친다 (손절이 분할로 잘못 걸리지 않게).
  2. 측정 불가 — 보유 종목 시세가 risk.yaml unmeasurable_retries번 연속 없으면 마지막 가격에 가상 청산
               (guards.check_position의 unmeasurable 판정). 백테스트의 "상장폐지 강제 청산"에 해당한다.
  3. trade_desk.manage → trade_desk.fill (손절·목표가·당일 청산, 체결)
  4. 평가 → 퇴출 조건 감시 (execution/paper.py check_retirement, 알림만)
  5. trade_desk.plan — 이 봉 신호를 make_plan으로 (kill_switch·보유 종목 수·heat·사이징·주문 상한·최소 주문)

백테스트와 다른 점 (의도)
- 가격은 원본가로 장부를 쓴다. 신호의 손절가·목표가·지정가는 판단 봉의 원본가/수정주가 비율로 원본가로 바꾼다
  (kr_stock. 다른 시장은 비율 1). 그래서 kr_stock은 사이징·주문 상한이 백테스트(수정주가)와 조금 다를 수 있다.
- kill_switch는 risk.yaml 그대로 따른다 (백테스트 재생은 끈다).
- 상장폐지는 미래를 모르므로 시세가 연속으로 없을 때 정리한다.
- 일봉 데이터로 하루 한 번 도는 모의 운용이라 판단 시각·시세 시각은 봉 끝 시각으로 본다 (데이터 신선도 통과).
- 순수 함수다. 파일 저장·알림·데이터 읽기는 runner/paper.py가 한다. 실전 주문은 하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.core.schemas import Position
from autotrader.execution import trade_desk as desk
from autotrader.execution.paper import check_retirement
from autotrader.risk import guards

EPS = 1e-9


def trade_bars(panel) -> dict[str, pd.DataFrame]:
    """panel → 원본가 시가·고가·저가·종가, 수정 종가, 거래 가능, 시세 행 있음. kr_stock 외에는 수정 = 원본."""
    c = panel.close
    ratio = (panel["raw_close"] / c) if "raw_close" in panel else pd.DataFrame(1.0, c.index, c.columns)
    out = {k: panel[k] * ratio for k in ("open", "high", "low", "close")}
    if "tradable" in panel:
        t = panel["tradable"].reindex_like(c)
        out["present"] = t.notna()
        out["tradable"] = t.astype("boolean").fillna(False).astype(bool)
    else:
        out["present"] = c.notna()
        out["tradable"] = out["present"].copy()
    out["adj_close"] = c
    out["ratio"] = ratio
    return out


def _get(df: pd.DataFrame, i: int, s: str, default=np.nan):
    return df.iat[i, df.columns.get_loc(s)] if s in df.columns else default


def _corporate_actions(book: desk.TradeBook, raw: dict, i: int, prev_i: int | None, ev: list) -> None:
    if prev_i is None:
        return
    for s in list(book.positions) + [pl["symbol"] for pl in book.pending]:
        rc, ac = _get(raw["close"], i, s), _get(raw["adj_close"], i, s)
        rp, ap = _get(raw["close"], prev_i, s), _get(raw["adj_close"], prev_i, s)
        if not all(np.isfinite(x) and x > 0 for x in (rc, ac, rp, ap)):
            continue
        f = (rc / rp) / (ac / ap)
        if abs(f - 1) <= 0.01:
            continue
        targets = [book.positions[s]] if s in book.positions else []
        targets += [pl for pl in book.pending if pl["symbol"] == s]
        for p in targets:
            q0 = p["qty"]
            p["qty"] = q0 / f
            for k in ("entry", "stop", "target", "limit"):
                if p.get(k) is not None:
                    p[k] = p[k] * f
        ev.append({"kind": "corporate_action", "symbol": s, "factor": f,
                   "reason": "원본가·수정주가 수익 차이 → 수량·진입가·손절가·목표가 조정 (실제 계좌 내역과 대조 필요)"})


def step(book: desk.TradeBook, panel, raw: dict, rows: list, i: int, fee: float, tax: float, risk: dict,
         spec: desk.DeskSpec, *, retirement: dict | None = None) -> list[dict]:
    """봉 i 하나. rows: 봉 i에 판단한 신호 행 (backtest.trade.normalize 결과의 itertuples, 수정주가 기준).
    같은 봉을 두 번 처리하지 않는다."""
    idx = panel.index
    date = idx[i]
    if book.last_bar is not None and pd.Timestamp(book.last_bar) >= date:
        return [{"kind": "skip_bar", "bar": str(date), "reason": "이미 처리한 봉"}]
    ev: list[dict] = []
    prev_i = None
    if book.last_bar is not None and pd.Timestamp(book.last_bar) in idx:
        prev_i = int(idx.get_loc(pd.Timestamp(book.last_bar)))
    first = book.last_bar is None
    day = idx.normalize()
    day_last = int(day.searchsorted(day[i], side="right") - 1)
    bar_open, bar_end, day_close = desk.bar_times(date, idx[day_last], desk.bar_length(panel.timeframe))

    # 1. 기업행위
    _corporate_actions(book, raw, i, prev_i, ev)

    # 2. 측정 불가 (시세 연속 없음)
    for s in list(book.positions):
        if bool(_get(raw["present"], i, s, False)):
            book.missed.pop(s, None)
            continue
        book.missed[s] = book.missed.get(s, 0) + 1
        p = book.positions[s]
        P = Position(lab=spec.lab, symbol=s, market=spec.market, side=p["side"], qty=p["qty"], entry=p["entry"],
                     stop=p["stop"], target=p["target"], intraday=p["intraday"])
        d = guards.check_position(P, None, bar_end.to_pydatetime(), risk, missed_quotes=book.missed[s])
        if d and d.kind == "unmeasurable":
            px = book.last_px.get(s, p["entry"])
            desk.close(book, spec, s, date, float(px), "unmeasurable", d.reason + " (상장폐지 가능, 실전이면 수동 확인)",
                       fee, tax, ev)
            book.missed.pop(s, None)

    # 3. 청산 판정 → 체결
    def quote(s):
        if s not in raw["close"].columns:
            return None
        return (_get(raw["open"], i, s), _get(raw["high"], i, s), _get(raw["low"], i, s),
                _get(raw["close"], i, s), bool(_get(raw["tradable"], i, s, False)))

    desk.manage(book, spec, quote, date, bar_open, bar_end, day_close, fee, tax, risk, ev)
    desk.fill(book, spec, quote, date, bar_open, bar_end, day_close, fee, tax, risk, ev)

    # 4. 평가 → 퇴출 조건
    for s in book.positions:
        c = _get(raw["close"], i, s)
        if np.isfinite(c) and c > 0:
            book.last_px[s] = float(c)
    book.last_px = {s: v for s, v in book.last_px.items() if s in book.positions}
    book.prev_equity = book.equity
    book.equity = desk.value(book, spec, lambda s: book.last_px.get(s, book.positions[s]["entry"]))
    if first:
        book.prev_equity = book.equity
    book.peak = max(book.peak, book.equity)
    ev += check_retirement(book, retirement, date)

    # 5. 신호 → 계획 (수정주가 기준 가격을 원본가로)
    if rows:
        conv = []
        for r in rows:
            k = _get(raw["ratio"], i, r.symbol, 1.0)
            k = k if np.isfinite(k) and k > 0 else 1.0
            conv.append(r._replace(**{f: getattr(r, f) * k for f in ("stop", "atr", "target", "entry")}))

        def ref(s):
            c = _get(raw["close"], i, s)
            return float(c) if np.isfinite(c) else np.nan

        desk.plan(book, spec, conv, book.equity, ref, date, bar_end.to_pydatetime(), risk, ev)

    book.last_bar = str(date)
    for e in ev:
        e.setdefault("bar", book.last_bar)
    ev.append({"kind": "equity", "bar": book.last_bar, "equity": book.equity, "cash": book.cash,
               "positions": len(book.positions), "pending": len(book.pending)})
    return ev
