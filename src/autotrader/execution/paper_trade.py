"""운용데스크 (paper) — 신호형(type: trade): Signal → make_plan → 주문 → 가상 체결 → 포지션 감시.

봉 처리 규칙은 execution/trade_desk.py에 있고, 백테스트 재생(backtest/trade.py)도 같은 함수를 부른다.
이 모듈은 paper 운영에만 필요한 것을 더한다.

봉 하나를 처리하는 순서 (step)
  1. 기업행위 — 보유·대기 종목의 원본가 수익과 수정주가 수익이 다르면 (수정주가에는 분할·병합과 세후 배당 재투자가 들어 있다)
               배당락(비율 0.7~1): 세후 배당을 현금으로 넣는다 (수량 그대로).
               분할·병합(그 밖): 수량을 그 비율로 고친다.
               어느 쪽이든 진입가·손절가·목표가·지정가는 같은 비율로 옮긴다 — 백테스트가 수정주가로 정한 가격 수준을
               그대로 지켜 손절이 배당락·분할로 잘못 걸리지 않게 한다. 평가액은 백테스트의 배당 재투자와 같다.
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
NOISE = 1e-4            # 원본가·수정주가 수익 차이가 이 안이면 무시 (가장 작은 실제 배당도 0.1% 안팎)
MIN_SPLIT_FACTOR = 0.7   # 원본가/수정주가 수익 비율이 이보다 작거나 1보다 크면 분할·병합, 그 사이(0.7~1)는 배당락
sgn = desk.sgn


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


def _corporate_actions(book: desk.TradeBook, spec: desk.DeskSpec, raw: dict, i: int, prev_i: int | None,
                       ev: list) -> None:
    if prev_i is None:
        return
    for s in list(book.positions) + [pl["symbol"] for pl in book.pending]:
        rc, ac = _get(raw["close"], i, s), _get(raw["adj_close"], i, s)
        rp, ap = _get(raw["close"], prev_i, s), _get(raw["adj_close"], prev_i, s)
        if not all(np.isfinite(x) and x > 0 for x in (rc, ac, rp, ap)):
            continue
        f = (rc / rp) / (ac / ap)
        if abs(f - 1) <= NOISE:            # 수정주가 계산의 부동소수 오차 (예: 068270에서 ±4e-6)
            continue
        # 원본가가 수정주가보다 덜 오른 날: 배당락(현금, 수정주가는 세후 배당 재투자) 또는 분할·병합.
        # 30% 넘게 벌어지면 분할·병합으로 보고 수량을 고치고, 그보다 작게 내려가면 배당으로 보고 현금을 넣는다.
        # 어느 쪽이든 평가액은 백테스트(수정주가)와 같고, 손절가·목표가·지정가는 같은 비율로 옮긴다
        # (백테스트에서 수정주가 기준으로 정한 가격 수준을 그대로 유지).
        dividend = MIN_SPLIT_FACTOR < f < 1
        if s in book.positions:
            p = book.positions[s]
            if dividend:
                cash = sgn(p["side"]) * p["qty"] * spec.multiplier * rc * (1 / f - 1)
                book.cash += cash
                p["income"] = p.get("income", 0.0) + cash
            else:
                p["qty"] = p["qty"] / f
        for p in [book.positions[s]] if s in book.positions else []:
            for k in ("entry", "stop", "target"):
                if p.get(k) is not None:
                    p[k] = p[k] * f
        for pl in [pl for pl in book.pending if pl["symbol"] == s]:
            if not dividend:
                pl["qty"] = pl["qty"] / f
            for k in ("stop", "target", "limit"):
                if pl.get(k) is not None:
                    pl[k] = pl[k] * f
        if dividend:
            ev.append({"kind": "dividend", "symbol": s, "factor": f,
                       "per_share_net": rc * (1 / f - 1), "cash": cash if s in book.positions else 0.0,
                       "reason": "배당락: 세후 배당을 현금으로 (수정주가 기준 손절가·목표가는 같은 비율로 이동)"})
        else:
            ev.append({"kind": "corporate_action", "symbol": s, "factor": f,
                       "reason": "원본가·수정주가 수익 차이 → 분할·병합으로 보고 수량·진입가·손절가·목표가 조정 "
                                 "(실제 계좌 내역과 대조 필요)"})


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
    _corporate_actions(book, spec, raw, i, prev_i, ev)

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
