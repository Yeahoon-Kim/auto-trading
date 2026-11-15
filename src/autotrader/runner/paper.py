"""paper 운용 실행기: lab 하나를 모의 계좌로 하루 한 번 돌린다 (수집이 끝난 뒤).

    uv run python -m autotrader.runner.paper labs/L001-이름                       # 오늘 분 처리
    uv run python -m autotrader.runner.paper labs/L001-이름 --dry-run             # 저장·알림 없이 결과만
    uv run python -m autotrader.runner.paper labs/L001-이름 --replay-days 120     # 최근 N봉을 새 계좌로 재생, 엔진과 비교

- 검증(validated) 또는 paper 상태의 lab만 돈다. 배관 점검은 --allow-unvalidated.
- 같은 strategy.py의 generate_signals()를 그대로 부른다 (백테스트·paper 같은 코드 경로).
- 잠금 구간까지 포함한 최신 데이터를 읽는다 (paper는 앞으로 가는 검증이라 판정용 잠금과 상관없다).
- 마지막 처리 봉 이후의 봉을 차례로 처리한다 (며칠 못 돌았으면 따라잡기). 첫 실행은 마지막 봉 하나만.
- 기록: results/paper/<lab폴더>/book.json(장부), events.jsonl(주문·체결·리스크), equity.csv
- 지원 시장: kr_stock, crypto_spot. 롱만. us_stock은 원본가 처리(배당 포함 수정주가) 확인 후 추가.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from autotrader.backtest import engine
from autotrader.backtest.data import load
from autotrader.backtest.lab import load_strategy
from autotrader.data import store
from autotrader.execution.paper import Book, bars, step
from autotrader.risk.guards import load_config

RUNNABLE = ("validated", "paper", "live")
MARKETS = {   # lot, 최소 주문 금액(계좌 통화), 마지막 봉 허용 경과(달력일)
    "kr_stock": {"lot": 1.0, "min_notional": 0.0, "max_bar_age_days": 4},
    "crypto_spot": {"lot": 1e-6, "min_notional": 5.0, "max_bar_age_days": 2},
}


def paper_dir(lab_dir: Path) -> Path:
    return store.ROOT / "results" / "paper" / lab_dir.name


def lab_config(lab_dir: Path) -> dict:
    return yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8"))


def options(cfg: dict) -> dict:
    """lab.yaml의 paper 블록(선택)과 시장 기본값을 합친다."""
    m, kind = cfg["market"], cfg.get("type") or "portfolio"
    if kind != "portfolio":
        raise SystemExit(f"type: {kind} — paper 데스크는 아직 비중형(portfolio)만 돈다. 신호형(trade) 경로는 2단계")
    if m not in MARKETS:
        raise SystemExit(f"paper 데스크가 아직 지원하지 않는 시장: {m} (지원: {', '.join(MARKETS)})")
    p = {**MARKETS[m], "limit_band_pct": 5.0, "krw_per_unit": 1.0 if m == "kr_stock" else None,
         **{k: v for k, v in (cfg.get("paper") or {}).items() if v is not None}}
    if p["krw_per_unit"] is None:
        raise SystemExit(f"{m}: lab.yaml paper.krw_per_unit (계좌 통화 1단위 = 몇 원)이 필요하다 — 주문 상한 환산용")
    s = yaml.safe_load((store.ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    p.setdefault("capital", (cfg.get("execution") or {}).get("capital") or s["backtest"]["initial_capital_krw"])
    return p


def load_panel(cfg: dict, today=None):
    fields = list(cfg.get("fields") or [])
    if cfg["market"] == "kr_stock" and "raw_close" not in fields:
        fields.append("raw_close")      # 원본가 장부용
    panel = load(cfg["market"], cfg.get("universe") or None, start=cfg.get("data_start") or None,
                 timeframe=cfg.get("timeframe", "1d"), fields=tuple(fields), lockbox=True, today=today)
    if cfg["market"].startswith("crypto"):     # 오늘(UTC) 봉은 아직 안 끝났다 → 버린다
        t = pd.Timestamp(today or datetime.now(UTC))
        panel = panel.slice(end=(t.tz_convert(None) if t.tzinfo else t).normalize())
    return panel


def signals(lab_dir: Path, cfg: dict, panel) -> pd.DataFrame:
    w = load_strategy(lab_dir).generate_signals(panel, cfg.get("params") or {})
    return engine.align_weights(w, panel)


def replay(panel, W, start_i: int, capital: float, costs, risk: dict, *, lab: str, market: str,
           rebalance: str, end_i: int | None = None, book: Book | None = None, **kw) -> tuple[Book, list[dict]]:
    """봉 start_i..end_i를 차례로 처리한다. book이 없으면 새 계좌."""
    book = book or Book(cash=float(capital))
    raw = bars(panel)
    ev: list[dict] = []
    for i in range(start_i, (end_i if end_i is not None else len(panel.index) - 1) + 1):
        ev += step(book, panel, raw, W, i, costs, risk, lab=lab, market=market, rebalance=rebalance, **kw)
    return book, ev


def compare_engine(panel, W, start_i: int, capital: float, costs, rebalance: str, equity: pd.Series) -> dict:
    """같은 구간을 엔진(정수 주식, 다음 봉 시가)으로 돌려 자산 곡선을 비교한다.
    데스크는 봉 start_i 종가에 주문 → start_i+1 시가 체결, 엔진 start = start_i+1."""
    res = engine.run(panel, W, costs, fill="next_open", start=panel.index[start_i + 1],
                     capital=capital, whole_shares=True, rebalance=rebalance)
    e = res.equity
    d = equity.reindex(e.index).dropna()
    gap = (d / e.reindex(d.index) - 1)
    return {"bars": len(d), "desk_final": float(d.iloc[-1]) if len(d) else None,
            "engine_final": float(e.iloc[-1]) if len(e) else None,
            "final_gap_pct": float(gap.iloc[-1] * 100) if len(gap) else None,
            "max_abs_gap_pct": float(gap.abs().max() * 100) if len(gap) else None,
            "engine_trades": res.trades}


def _equity(ev: list[dict]) -> pd.Series:
    e = [(pd.Timestamp(x["bar"]), x["equity"]) for x in ev if x["kind"] == "equity"]
    return pd.Series(dict(e)).sort_index()


def summarize(lab: str, ev: list[dict], book: Book) -> str:
    k = pd.Series([e["kind"] for e in ev]).value_counts().to_dict() if ev else {}
    lines = [f"📒 paper {lab} — {book.last_bar}",
             f"자산 {book.equity:,.0f} · 현금 {book.cash:,.0f} · 보유 {len(book.shares)}종목 · 대기 주문 {len(book.pending)}",
             f"체결 {k.get('fill', 0)} · 새 주문 {k.get('order', 0)} · 취소 {k.get('cancel', 0)}"]
    alerts = [e for e in ev if e["kind"] in ("corporate_action", "unmeasurable_exit", "reduce")
              or (e["kind"] == "risk_block")]
    for e in alerts[:8]:
        lines.append(f"⚠️ {e['kind']} {e.get('symbol', e.get('symbols', ''))}: {e.get('reason', '')}")
    if len(alerts) > 8:
        lines.append(f"… 외 {len(alerts) - 8}건 (events.jsonl)")
    if book.halted:
        lines.append(f"⛔ {book.halted}")
    return "\n".join(lines)


def _save(out: Path, book: Book, ev: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "book.json.tmp"
    tmp.write_text(json.dumps(book.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out / "book.json")
    with (out / "events.jsonl").open("a", encoding="utf-8") as f:
        for e in ev:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    eq = [e for e in ev if e["kind"] == "equity"]
    if eq:
        path = out / "equity.csv"
        new = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if new:
                f.write("bar,equity,cash,positions,pending\n")
            for e in eq:
                f.write(f"{e['bar']},{e['equity']:.4f},{e['cash']:.4f},{e['positions']},{e['pending']}\n")


def run_today(lab_dir: Path, *, allow_unvalidated=False, dry_run=False, notify=True, enter_now=False,
              today=None) -> tuple[Book, list[dict]]:
    cfg = lab_config(lab_dir)
    lab = str(cfg.get("id") or lab_dir.name)
    if cfg.get("status") not in RUNNABLE and not allow_unvalidated:
        raise SystemExit(f"{lab}: status={cfg.get('status')} — 검증된 lab만 paper로 돈다 "
                         "(배관 점검이면 --allow-unvalidated)")
    opt = options(cfg)
    panel = load_panel(cfg, today)
    last = panel.index[-1]
    now = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    age = (now.tz_localize(None) if now.tzinfo else now).normalize() - last.normalize()
    if age.days > opt["max_bar_age_days"]:
        msg = f"⛔ paper {lab}: 마지막 봉 {last.date()} ({age.days}일 전) — 데이터 수집 확인 필요, 처리 안 함"
        if notify and not dry_run:
            from autotrader.monitoring.notify import send
            send(msg)
        raise SystemExit(msg)

    out = paper_dir(lab_dir)
    path = out / "book.json"
    book = Book.from_dict(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None
    if book is None:
        start_i = len(panel.index) - 1
    else:
        start_i = int(panel.index.searchsorted(pd.Timestamp(book.last_bar), side="right"))
        if start_i >= len(panel.index):
            print(f"{lab}: 새 봉 없음 (마지막 처리 {book.last_bar})")
            return book, []
    W = signals(lab_dir, cfg, panel)
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    rebalance = (cfg.get("execution") or {}).get("rebalance") or "always"
    kw = {k: opt[k] for k in ("lot", "min_notional", "krw_per_unit", "limit_band_pct")}
    book, ev = replay(panel, W, start_i, opt["capital"], costs, load_config(), lab=lab, market=cfg["market"],
                      rebalance=rebalance, book=book, enter_now=enter_now, **kw)
    n = len(panel.index) - start_i
    if n > 1:
        ev.insert(0, {"kind": "catch_up", "bars": n, "bar": str(panel.index[start_i].date()),
                      "reason": f"{n}봉 따라잡기"})
    text = summarize(lab, ev, book)
    print(text)
    if not dry_run:
        _save(out, book, ev)
        if notify:
            from autotrader.monitoring.notify import send
            send(text)
    return book, ev


def run_replay(lab_dir: Path, days: int, today=None) -> dict:
    """최근 days봉을 새 계좌로 재생하고 엔진과 비교한다 (저장 안 함). 리스크 한도는 risk.yaml 그대로."""
    cfg = lab_config(lab_dir)
    lab = str(cfg.get("id") or lab_dir.name)
    opt = options(cfg)
    panel = load_panel(cfg, today)
    W = signals(lab_dir, cfg, panel)
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    rebalance = (cfg.get("execution") or {}).get("rebalance") or "always"
    start_i = max(1, len(panel.index) - days)
    kw = {k: opt[k] for k in ("lot", "min_notional", "krw_per_unit", "limit_band_pct")}
    out = {}
    for name, risk in (("risk.yaml", load_config()), ("한도 없음", _no_limits(load_config()))):
        book, ev = replay(panel, W, start_i, opt["capital"], costs, risk, lab=lab, market=cfg["market"],
                          rebalance=rebalance, **kw)
        k = pd.Series([e["kind"] for e in ev]).value_counts().to_dict()
        out[name] = {"events": k, "final_equity": book.equity,
                     "engine": compare_engine(panel, W, start_i, opt["capital"], costs, rebalance, _equity(ev))}
    return out


def _no_limits(risk: dict) -> dict:
    """엔진과 같은 조건 비교용: 리스크 한도를 끈 사본 (risk.yaml은 그대로)."""
    r = json.loads(json.dumps(risk))
    r["global"].update(kill_switch=False, max_daily_loss_pct=1e9, max_drawdown_pct=1e9)
    r["orders"].update(max_order_value_krw=1e18, min_notional_check=False, require_limit_orders=False)
    r["data_staleness"]["unmeasurable_retries"] = 10 ** 6
    return r


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")       # 윈도 콘솔(cp949)에서 이모지·한글 출력
    ap = argparse.ArgumentParser(description="paper 운용 실행기")
    ap.add_argument("lab_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="저장·알림 없이 결과만 출력")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--allow-unvalidated", action="store_true", help="검증 전 lab (배관 점검용)")
    ap.add_argument("--enter-now", action="store_true", help="첫 실행에 신호 날을 기다리지 않고 바로 목표로 진입")
    ap.add_argument("--replay-days", type=int, help="최근 N봉을 새 계좌로 재생하고 엔진과 비교 (저장 안 함)")
    a = ap.parse_args(argv)
    lab_dir = a.lab_dir.resolve()
    if a.replay_days:
        print(json.dumps(run_replay(lab_dir, a.replay_days), ensure_ascii=False, indent=1, default=str))
        return
    run_today(lab_dir, allow_unvalidated=a.allow_unvalidated, dry_run=a.dry_run, notify=not a.no_notify,
              enter_now=a.enter_now)


if __name__ == "__main__":
    sys.exit(main())
