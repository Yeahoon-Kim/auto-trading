"""lab 실행기: lab.yaml + strategy.py → 워크포워드 백테스트와 고정 기준 1~5·7(자동분) 점검.

type: portfolio (기본) → generate_signals(목표 비중)를 공용 엔진(engine.py)으로.
type: trade           → generate_trades(진입 신호)를 데스크 재생기(trade.py)로 — 사이징·손절·목표가·당일 청산 포함.

    uv run python -m autotrader.backtest.lab labs/L001-이름              # 개발 구간 (잠금 구간 안 봄)
    uv run python -m autotrader.backtest.lab labs/L001-이름 --quick      # ±20% 민감도 생략 (개발 중 빠르게)
    uv run python -m autotrader.backtest.lab labs/L001-이름 --lockbox    # 최종 판정 때 단 한 번 (기준 6)

결과: labs/<lab>/results/<실행시각>/report.md, summary.json, *_returns.parquet
판정은 하지 않는다 — validator 검증 뒤 사람이 정한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import benchmark, engine, trade
from .data import Panel, check_dividend_causality, check_fin_causality, check_mcap_causality, dev_end, load

REGIME_COL = {"kr_stock": "kospi_trend", "us_stock": "spx_trend",
              "crypto_spot": "btc_trend", "crypto_futures": "btc_trend"}


def load_strategy(lab_dir: Path):
    spec = importlib.util.spec_from_file_location(f"lab_{lab_dir.name.replace('-', '_')}",
                                                  lab_dir / "strategy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rf(panel: Panel, index: pd.DatetimeIndex) -> pd.Series | None:
    if not panel.timeframe.endswith("d"):
        return None
    try:
        return benchmark.rf_daily(index, krw=panel.market == "kr_stock")
    except Exception:
        return None


def judge_sharpe(r: pd.Series, panel: Panel) -> dict:
    """판정 기준 1·2·3과 보고서의 공용 Sharpe (2026-11-19부터 한 정의).

    sharpe     = (수익 − 그 구간 날짜의 무위험수익률) 초과수익의 연율 Sharpe (benchmark.metrics와 같은 식).
                 rf: KR은 CD 91일, 그 밖은 연방기금금리 (전날 관측값). 일봉이 아니거나 rf 데이터가 없으면 rf=0.
    sharpe_rf0 = rf를 빼지 않은 값 (참고·예전 결과와 비교용)."""
    r = r.dropna()
    rf = _rf(panel, r.index)
    ex = r - rf.reindex(r.index).fillna(0) if rf is not None else r
    year = panel.bars_per_year
    return {"sharpe": engine.sharpe(ex, year), "sharpe_rf0": engine.sharpe(r, year)}


def _rf_note(panel: Panel) -> str:
    """summary·보고서에 적을 Sharpe 정의 한 줄."""
    if not panel.timeframe.endswith("d"):
        return "rf=0 (일봉이 아니어서 무위험수익률을 빼지 않음)"
    src = "CD 91일(ECOS)" if panel.market == "kr_stock" else "연방기금금리(FRED DFF)"
    return (f"rf 차감 ({src}, 전날 관측값)" if _rf(panel, panel.index[-2:]) is not None
            else f"rf=0 (**{src} 데이터를 읽지 못함**)")


def _committed(lab_dir: Path) -> bool | None:
    """가설(lab.yaml·README.md)이 커밋돼 있고 수정되지 않은 상태인지."""
    try:
        files = [str(lab_dir / f) for f in ("lab.yaml", "README.md")]
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", *files], capture_output=True,
                                 cwd=lab_dir).returncode == 0
        dirty = subprocess.run(["git", "status", "--porcelain", *files], capture_output=True, text=True,
                               cwd=lab_dir).stdout.strip()
        return tracked and not dirty
    except Exception:
        return None


def _regimes(panel: Panel, r: pd.Series, year: float) -> dict:
    """국면(전날까지 알려진 추세 라벨)별 성과 — 참고용."""
    try:
        from ..data import store
        g = store.read("regime_1d", "ALL")
        lab = g.set_index(g["ts"].dt.tz_localize(None).dt.normalize())[REGIME_COL[panel.market]].shift(1)
        lab = lab[~lab.index.duplicated(keep="last")].reindex(r.index.normalize(), method="ffill")
        lab.index = r.index
        return {str(k): {"bars": len(v), **judge_sharpe(v, panel),
                         "total_return": float((1 + v).prod() - 1)}
                for k, v in r.groupby(lab) if len(v) > 1}
    except Exception as e:
        return {"error": str(e)[:200]}


def _bench(cfg: dict, panel: Panel, idx: pd.DatetimeIndex, costs: engine.Costs) -> dict | None:
    b = (cfg.get("benchmark") or "").strip()
    if not b:
        return None
    if b.startswith(("SPY_", "SPX_", "IVV_", "VOO_")):
        r = benchmark.benchmark_returns(b, idx[0], idx[-1] + pd.Timedelta(days=1)).reindex(idx).fillna(0)
        name = f"{b} (benchmark.py)"
    elif b in panel.close.columns:
        w = pd.DataFrame({b: 1.0}, index=panel.index)
        r = engine.run(panel, w, costs, start=idx[0], end=idx[-1] + pd.Timedelta(days=1)).returns
        name = f"{b} 매수 후 보유 (비용 포함)"
    else:
        return {"name": b, "error": "벤치마크를 찾을 수 없음 (SPY_* 변형 또는 panel 안의 종목)"}
    return {"name": name, **judge_sharpe(r, panel),
            "total_return": float((1 + r).prod() - 1), "mdd": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())}


def _panel(cfg: dict, lockbox: bool) -> Panel:
    fields = list(cfg.get("fields") or ())
    if _kind(cfg) == "trade":            # 손절·목표가 판정에 고가·저가가 필요
        fields += [f for f in ("high", "low") if f not in fields]
    return load(cfg["market"], cfg.get("universe") or None, start=cfg.get("data_start") or None,
                timeframe=cfg.get("timeframe", "1d"), fields=tuple(fields),
                lockbox=lockbox, currency=cfg.get("currency") or None,
                dividend_tax=cfg.get("dividend_tax"), fin_timing=cfg.get("fin_timing") or "first",
                funding=cfg.get("funding", True) is not False)


def _kind(cfg: dict) -> str:
    kind = cfg.get("type") or "portfolio"
    if kind not in ("portfolio", "trade"):
        raise SystemExit(f"type: {kind} — portfolio(비중형) 또는 trade(신호형)만 있다 (docs/architecture.md §10)")
    return kind


class _Runner:
    """비중형·신호형을 같은 모양으로 부른다: 워크포워드, 같은 신호로 비용만 바꾼 재실행, 미래참조 검사, 한 번 실행."""

    def __init__(self, cfg: dict, strat, panel: Panel, costs: engine.Costs, lab: str):
        self.kind, self.cfg, self.strat, self.panel, self.costs = _kind(cfg), cfg, strat, panel, costs
        self.wf = cfg.get("walk_forward") or {}
        self.fill = cfg.get("fill", "next_open")
        self.ex = cfg.get("execution") or None
        if self.kind == "trade":
            if not hasattr(strat, "generate_trades"):
                raise SystemExit("type: trade — strategy.py에 generate_trades(data, params)가 필요하다")
            self.account = trade.Account.from_lab(cfg)
            self.tkw = {"trade_cfg": cfg.get("trade") or None, "lab": lab}

    def walk_forward(self, params: dict, costs: engine.Costs | None = None) -> dict:
        c, wf = costs or self.costs, self.wf
        a = (wf.get("start") or None, wf.get("train", "24M"), wf.get("test", "6M"))
        if self.kind == "trade":
            return trade.walk_forward(self.strat, self.panel, params, c, self.account, *a, **self.tkw)
        return engine.walk_forward(self.strat, self.panel, params, c, self.fill, *a, execution=self.ex)

    def rerun(self, base: dict, costs: engine.Costs, t0, t1) -> engine.Result:
        """워크포워드로 만든 같은 신호를 비용만 바꿔 다시 (기준 4·참고용 비용 없음)."""
        if self.kind == "trade":
            return trade.run(self.panel, base["trades"], costs, self.account, start=t0, end=t1, **self.tkw)
        kw = engine.execution_kw(self.ex, self.panel.index, t0, t1)
        return engine.run(self.panel, base["weights"], costs, self.fill, start=t0, end=t1, **kw)

    def causality(self, params: dict) -> dict:
        if self.kind == "trade":
            return trade.check_causality(self.strat.generate_trades, self.panel, params)
        return engine.check_causality(self.strat.generate_signals, self.panel, params)

    def once(self, params: dict, start) -> engine.Result:
        if self.kind == "trade":
            return trade.run(self.panel, self.strat.generate_trades(self.panel, params), self.costs, self.account,
                             start=start, **self.tkw)
        return engine.run(self.panel, self.strat.generate_signals(self.panel, params), self.costs, self.fill,
                          start=start, **engine.execution_kw(self.ex, self.panel.index, start))

    def extra(self, res: engine.Result) -> dict:
        """신호형만: 사이징 값·계좌·거래 통계 (summary에 붙는다)."""
        if self.kind != "trade":
            return {}
        from ..risk import guards
        return {"type": "trade", "trade": guards.trade_params(self.cfg.get("trade"), guards.load_config()),
                "account": engine.asdict(self.account), "trade_stats": trade.stats(res)}

    def save(self, res: engine.Result, out: Path, name: str) -> None:
        engine.save_result(res, out, name)
        if self.kind == "trade":
            trade.save_log(res, out, name)


def _all_passed(c: dict) -> bool:
    """신호 cut 검사 통과 + 필드 재구성 검사 중 실패(False)가 없음.
    재구성하지 못한 필드(passed None)는 실패가 아니라 '미확인' — 보고서에 따로 적는다."""
    return bool(c["passed"]) and all((c.get(k) or {}).get("passed") is not False
                                     for k in ("dividend_fields", "fin_fields", "mcap_fields"))


def _weight_cap(res: engine.Result) -> dict | None:
    """보유 비중 합 (거래정지 잔량 처리 결과 확인용)."""
    w = getattr(res, "weights", None)
    if w is None or w.empty:
        return None
    tot = w.sum(axis=1)
    return {"frozen_capped_bars": res.meta.get("frozen_capped_bars", 0),
            "sum_max": float(tot.max()), "sum_mean": float(tot.mean())}


def holds_up(test_sharpe: float, train_sharpe: float) -> bool:
    """고정 기준 2: 시험 Sharpe ≥ 학습 Sharpe × 0.5. 학습 Sharpe가 0 이하면 탈락
    (그대로 두면 기준선이 음수가 돼 손실 전략도 통과한다)."""
    return train_sharpe > 0 and test_sharpe >= 0.5 * train_sharpe


def develop(lab_dir: Path, quick: bool = False) -> dict:
    cfg = yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8"))
    _kind(cfg)
    strat = load_strategy(lab_dir)
    params = cfg.get("params") or {}
    fill = cfg.get("fill", "next_open")
    panel = _panel(cfg, lockbox=False)
    year = panel.bars_per_year
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    ex = cfg.get("execution") or None
    R = _Runner(cfg, strat, panel, costs, str(cfg.get("id") or lab_dir.name))
    wfrun = R.walk_forward

    causal = R.causality(params)
    causal["dividend_fields"] = check_dividend_causality(panel)   # 배당 필드 자체의 시점 (필드가 없으면 통과)
    causal["fin_fields"] = check_fin_causality(panel)             # 재무 필드 자체의 시점 (필드가 없으면 통과)
    causal["mcap_fields"] = check_mcap_causality(panel)           # US 시가총액 필드 자체의 시점 (필드가 없으면 통과)
    causal["passed"] = _all_passed(causal)
    base = wfrun(params)
    test = base["test"]
    rf = _rf(panel, test.returns.index)
    tm = test.metrics(year, rf)
    # 기준 1·2·3은 모두 judge_sharpe 한 정의 (rf 차감). tm["sharpe"]와 같은 값이다 (같은 식·같은 rf)
    s1 = judge_sharpe(test.returns, panel)
    win_rows = []
    for w in base["windows"]:
        tr = w.pop("_train_result")
        te = w.pop("_test_returns")
        a, b = judge_sharpe(tr.returns, panel), judge_sharpe(te, panel)
        win_rows.append({**w, "train_sharpe": a["sharpe"], "test_sharpe": b["sharpe"],
                         "train_sharpe_rf0": a["sharpe_rf0"], "test_sharpe_rf0": b["sharpe_rf0"],
                         "test_return": float((1 + te).prod() - 1)})
    train_sharpe = float(np.mean([w["train_sharpe"] for w in win_rows]))
    train_sharpe_rf0 = float(np.mean([w["train_sharpe_rf0"] for w in win_rows]))
    t0, t1 = test.returns.index[0], test.returns.index[-1] + pd.Timedelta(days=1)
    no_cost = R.rerun(base, costs.scaled(0), t0, t1)
    cost2 = R.rerun(base, costs.scaled(2), t0, t1)

    sens = []
    numeric = [k for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric and not quick:
        for f in (0.8, 1.2):
            for k in numeric:   # 하나씩 바꾼다 (모두 같이 바꾸면 서로 상쇄될 수 있음)
                p = dict(params)
                p[k] = engine.perturb({k: params[k]}, f)[k]
                s = judge_sharpe(wfrun(p)["test"].returns, panel)
                sens.append({"param": k, "factor": f, "value": p[k], "test_sharpe": s["sharpe"],
                             "test_sharpe_rf0": s["sharpe_rf0"]})

    checks = [
        {"no": 1, "rule": "시험 구간 Sharpe ≥ 1.0 (비용 포함)", "value": s1["sharpe"],
         "value_rf0": s1["sharpe_rf0"], "pass": s1["sharpe"] >= 1.0},
        # 학습 Sharpe가 0 이하면 "학습의 50%"가 음수가 돼 무엇이든 통과한다 → 학습에서 수익이 없으면 탈락
        {"no": 2, "rule": "시험 Sharpe ≥ 학습 Sharpe × 0.5 (학습 Sharpe > 0일 때만)",
         "value": f"{s1['sharpe']:.2f} vs {train_sharpe:.2f}" + (" (학습 Sharpe ≤ 0)" if train_sharpe <= 0 else ""),
         "value_rf0": f"{s1['sharpe_rf0']:.2f} vs {train_sharpe_rf0:.2f}",
         "pass": holds_up(s1["sharpe"], train_sharpe)},
        {"no": 3, "rule": "파라미터 ±20%에서 Sharpe ≥ 0.5",
         "value": min((s["test_sharpe"] for s in sens), default=None),
         "value_rf0": min((s["test_sharpe_rf0"] for s in sens), default=None),
         "pass": (min(s["test_sharpe"] for s in sens) >= 0.5) if sens else None},
        {"no": 4, "rule": "비용 2배에서도 수익", "value": cost2.metrics(year)["total_return"],
         "pass": cost2.metrics(year)["total_return"] > 0},
        {"no": 5, "rule": "거래 수 ≥ 30", "value": test.trades, "pass": test.trades >= 30},
        {"no": 6, "rule": "잠금 구간 (판정 때 1회)", "value": None, "pass": None},
        {"no": 7, "rule": "미래참조 자동 검사 (생존편향·데이터 시점 규칙은 validator 확인)",
         "value": "자른 데이터와 신호 일치" if causal["passed"] else "불일치", "pass": causal["passed"] and None},
    ]
    if not causal["passed"]:
        checks[6]["pass"] = False
    for c in checks:        # numpy bool은 summary.json에 문자열 "False"로 적힌다 → 파이썬 bool로
        if c["pass"] is not None:
            c["pass"] = bool(c["pass"])
    summary = {
        "lab": lab_dir.name, "version": cfg.get("version"), "run_at": datetime.now().isoformat(timespec="seconds"),
        "market": cfg["market"], "timeframe": panel.timeframe, "fill": fill, "params": params,
        "data": panel.meta, "costs": engine.asdict(costs),
        "hypothesis_committed": _committed(lab_dir),
        "execution": ex, "money": _clean(test.money()),
        # Sharpe 키: 어디서나 sharpe = rf 차감(판정 정의), sharpe_rf0 = 참고 (2026-11-19 통일)
        "sharpe_definition": _rf_note(panel),
        "test": _clean(tm), "train_sharpe_mean": train_sharpe, "train_sharpe_mean_rf0": train_sharpe_rf0,
        "test_no_cost": _clean(no_cost.metrics(year, rf)), "test_cost_x2": _clean(cost2.metrics(year, rf)),
        "weight_cap": _weight_cap(test),
        "windows": win_rows, "sensitivity": sens, "causality": causal,
        "benchmark": _bench(cfg, panel, test.returns.index, costs),
        "regimes": _regimes(panel, test.returns, year) if panel.timeframe.endswith("d") else None,
        "checks": checks, "trials": _trials(cfg), **R.extra(test),
    }
    out = lab_dir / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    R.save(test, out, "test")
    _write(out, summary)
    return summary


def open_lockbox(lab_dir: Path) -> dict:
    cfg = yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8"))
    _kind(cfg)
    lb = cfg.get("lockbox") or {}
    if lb.get("opened"):
        raise SystemExit(f"잠금 구간은 이미 열렸다 ({lb.get('start')}~{lb.get('end')}). "
                         "전략을 바꿨다면 version을 올리고 lockbox.opened를 false로 되돌린 뒤 처음부터 다시 검증한다.")
    last = sorted((lab_dir / "results").glob("2*/summary.json"))
    if not last:
        raise SystemExit("개발 구간 결과가 없다 — 먼저 --lockbox 없이 실행한다.")
    dev = json.loads(last[-1].read_text(encoding="utf-8"))
    strat = load_strategy(lab_dir)
    params = cfg.get("params") or {}
    panel = _panel(cfg, lockbox=True)
    year = panel.bars_per_year
    cut = dev_end()
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    fit = getattr(strat, "fit", None)
    p = fit(panel.slice(end=cut), dict(params)) if fit else params
    ex = cfg.get("execution") or None
    R = _Runner(cfg, strat, panel, costs, str(cfg.get("id") or lab_dir.name))
    res = R.once(p, cut)
    m = res.metrics(year, _rf(panel, res.returns.index))
    dev_mdd = dev["test"]["mdd"]
    ok = bool(m["total_return"] >= 0 and abs(m["mdd"]) <= 1.5 * abs(dev_mdd))
    summary = {"lab": lab_dir.name, "version": cfg.get("version"), "run_at": datetime.now().isoformat(timespec="seconds"),
               "market": cfg["market"], "sharpe_definition": _rf_note(panel),
               "lockbox": {"start": str(cut.date()), "end": panel.meta["end"]}, "params": p,
               "costs": engine.asdict(costs), "data": panel.meta, "test": _clean(m),
               "execution": ex, "money": _clean(res.money()),
               "dev_result": str(last[-1].parent.name), "dev_mdd": dev_mdd,
               "trials": _trials(cfg),
               "checks": [{"no": 6, "rule": "잠금 구간 수익 ≥ 0, MDD ≤ 개발 구간 MDD × 1.5",
                           "value": f"수익 {m['total_return']:.2%}, MDD {m['mdd']:.2%} (개발 {dev_mdd:.2%})",
                           "pass": ok}], **R.extra(res)}
    _mark_lockbox(lab_dir / "lab.yaml", str(cut.date()), panel.meta["end"][:10])
    out = lab_dir / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "_lockbox")
    R.save(res, out, "lockbox")
    _write(out, summary)
    return summary


def _mark_lockbox(path: Path, start: str, end: str) -> None:
    """lab.yaml의 lockbox 블록만 고친다 (주석 유지)."""
    s = path.read_text(encoding="utf-8")
    m = re.search(r"^lockbox:.*?(?=^\S|\Z)", s, flags=re.MULTILINE | re.DOTALL)
    if not m:
        s += f"\nlockbox:\n  start: \"{start}\"\n  end: \"{end}\"\n  opened: true\n"
    else:
        blk = m.group(0)
        blk = re.sub(r"(^\s+start:\s*)\S*", rf'\g<1>"{start}"', blk, flags=re.MULTILINE)
        blk = re.sub(r"(^\s+end:\s*)\S*", rf'\g<1>"{end}"', blk, flags=re.MULTILINE)
        blk = re.sub(r"(^\s+opened:\s*)\S*", r"\g<1>true", blk, flags=re.MULTILINE)
        s = s[:m.start()] + blk + s[m.end():]
    path.write_text(s, encoding="utf-8")


def _trials(cfg: dict) -> dict:
    """lab.yaml의 trials 블록 요약 (없거나 비어 있어도 됨). 판정은 validator가 한다."""
    t = cfg.get("trials") or {}
    ex = t.get("exposed_from") or []
    return {"count": t.get("count"), "exposed_from": [ex] if isinstance(ex, str) else list(ex),
            "log_entries": len(t.get("log") or [])}


def _clean(d: dict) -> dict:
    return {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else
                (None if isinstance(v, float) and not np.isfinite(v) else v)) for k, v in d.items()}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _tax(t) -> str:
    """세율 값 또는 {시행일: %} 표를 한 줄로."""
    if isinstance(t, dict):
        return "날짜별 " + ", ".join(f"{k}~ {v}%" for k, v in sorted(t.items())[-3:])
    return f"{t}%"


def report(s: dict) -> str:
    t = s["test"]
    lines = [f"# {s['lab']} v{s.get('version')} — {'잠금 구간' if 'lockbox' in s else '개발 구간 워크포워드'}",
             f"실행: {s['run_at']}  ·  판정은 validator 검증 후",
             "",
             "## 조건",
             f"- 기간: {t.get('start')} ~ {t.get('end')} ({t.get('days')}봉)"
             + (f", 잠금 구간 {s['lockbox']['start']}~{s['lockbox']['end']}" if "lockbox" in s
                else f", 개발 구간 끝 {s['data']['dev_end']} (이후 6개월은 잠금)"),
             f"- 비용 포함: 수수료 {s['costs']['fee_pct']}% + 슬리피지 {s['costs']['slippage_pct']}% (한쪽), "
             f"매도세 {_tax(s['costs']['tax_pct_sell'])}",
             f"- 데이터: {s['data'].get('source')}  ·  생존편향: {s['data'].get('survivorship')}",
             ]
    if s.get("type") != "trade":
        sz = (s.get("execution") or {}).get("sizing") or "next_open"
        lines.append(f"- 체결·수량: {s.get('fill') or 'next_open'} 체결, 수량 sizing={sz} ("
                     + ("신호 봉 종가로 수량을 정함 — paper 데스크와 같음, 시가·종가 차이만큼 비중이 어긋남"
                        if sz == "prev_close" else "체결 가격으로 목표 비중을 정확히 맞춤") + ")")
    tr = s.get("trials") or {}
    if tr:
        lines.append(f"- 시도 횟수: {tr.get('count') if tr.get('count') is not None else '미기록'}"
                     + (f"  ·  출처 lab(이미 본 개발 구간): {', '.join(map(str, tr['exposed_from']))}"
                        if tr.get("exposed_from") else ""))
    j = s["data"].get("jumps") or {}
    if j.get("neutralized"):
        lines.append(f"- 가격제한(±35%) 초과 하루 변동 {j['neutralized']:,}건({j['symbols']}종목)을 수익 0으로 처리 "
                     f"(수정계수에 빠진 감자·병합 등). 상장폐지 직전 정리매매 {j.get('kept_delisting_tail', 0)}건은 실제 손실로 유지")
    if s["data"].get("rows_without_adj_factor"):
        lines.append(f"- 수정계수 없는 행 {s['data']['rows_without_adj_factor']:,}개 (원본가 사용)")
    lines.append(_dividend_line(s))
    if s["data"].get("dividend_fields"):
        lines.append(f"- 배당 필드 시점: {s['data']['dividend_fields']}")
        cov = s["data"].get("dividend_data") or {}
        if cov.get("years"):
            yrs = ", ".join(f"{y} {n:,}" for y, n in cov["years"].items())
            lines.append(f"- 실행 시점 배당 데이터 범위: 사업연도별 행 수 {yrs} (보고서 접수 {cov['first']}~{cov['last']}). "
                         f"최초 접수일 없어 정정일로 시점을 잡은 행 {cov['filed_fallback']:,}개, 금액 모름 {cov['dps_unknown']:,}개")
        elif cov.get("rows"):
            lines.append(f"- 실행 시점 배당 데이터 범위: {cov['rows']:,}건, 배당락 {cov['first']}~{cov['last']}")
    lines += _fin_lines(s["data"])
    lines += _mcap_lines(s["data"])
    lines += _funding_lines(s["data"])
    if "causality" in s:
        c = s["causality"]
        df = c.get("dividend_fields") or {}
        ff = c.get("fin_fields") or {}
        mf = c.get("mcap_fields") or {}
        lines.append(f"- 미래참조 자동 검사: {'통과' if c['passed'] else '**실패**'} "
                     f"(데이터를 {len(c['cuts'])}곳에서 잘라 다시 계산한 신호 비교"
                     + (f"; 배당 필드 {', '.join(df['fields'])} 재구성 비교 "
                        f"{_pass_word(df['passed'])}" if df.get("fields") else "")
                     + (f"; 재무 필드 {', '.join(ff['fields'])} 재구성 비교 "
                        f"{_pass_word(ff['passed'])}" if ff.get("fields") else "")
                     + (f"; 시가총액 필드 {', '.join(mf['fields'])} 재구성 비교 "
                        f"{_pass_word(mf['passed'])}" if mf.get("fields") else "") + ")")
        for k, d in (("배당", df), ("재무", ff), ("시가총액", mf)):
            if d.get("fields") and d.get("passed") is None:
                lines.append(f"- {k} 필드 재구성 비교 미확인: {d.get('reason', '재구성 원천 없음')}")
        lines.append(f"- 가설 커밋 상태: {s.get('hypothesis_committed')}")
    if s.get("sharpe_definition"):
        lines.append(f"- Sharpe 정의 (기준 1·2·3·보고서 공통): {s['sharpe_definition']}. 괄호 안 rf=0 값은 참고")
    wc = s.get("weight_cap") or {}
    if wc.get("frozen_capped_bars"):
        lines.append(f"- 거래정지 잔량: 팔 수 없는 보유가 남은 봉 {wc['frozen_capped_bars']:,}개에서 나머지 매수를 줄여 "
                     f"비중 합을 1로 맞춤 (빚내서 사지 않음). 보유 비중 합 평균 {wc['sum_mean']:.3f}, 최대 {wc['sum_max']:.3f}")
    lines += ["", "## 결과 (비용 포함)",
              f"- 총수익 {_fmt(t.get('total_return'))}, CAGR {_fmt(t.get('cagr'))}, Sharpe {_fmt(t.get('sharpe'))} "
              f"(rf=0: {_fmt(t.get('sharpe_rf0'))}), MDD {_fmt(t.get('mdd'))}, 거래 {t.get('trades')}회, "
              f"연 회전율 {_fmt(t.get('turnover_per_year'))}, 연 비용 {_fmt(t.get('cost_per_year'))}"]
    if "funding_per_year" in t:
        lines.append(f"- 펀딩비 (수익에 포함, +면 받음): 연 {_fmt(t['funding_per_year'])}, 기간 누적 {_fmt(t.get('funding_total'))}")
    if s.get("money"):
        mo, e = s["money"], s.get("execution") or {}
        cur = s["data"].get("currency") or (s.get("account") or {}).get("currency") or ""
        cap = e.get("capital") or (s.get("account") or {}).get("capital") or 0
        lines.append(f"- 금액 ({cur}, 초기 {cap:,.0f}"
                     + (f", 월 적립 {e['monthly_contribution']:,}" if e.get("monthly_contribution") else "")
                     + (", 정수 주식" if e.get("whole_shares") else "")
                     + (", 신호형 재생" if s.get("type") == "trade" else f", 리밸런싱 {e.get('rebalance', 'always')}") + "): "
                     f"투입 {mo['invested']:,.0f} → 평가 {mo['value']:,.0f} (투입 대비 {mo['return_on_invested']:+.2%}, "
                     f"XIRR {_fmt(mo.get('xirr'))}, 남은 현금 {mo['cash_end']:,.0f})")
    if "test_no_cost" in s:
        lines.append(f"- 참고 — 비용 없음 Sharpe {_fmt(s['test_no_cost'].get('sharpe'))} "
                     f"(rf=0: {_fmt(s['test_no_cost'].get('sharpe_rf0'))}), "
                     f"비용 2배 총수익 {_fmt(s['test_cost_x2'].get('total_return'))}")
    if s.get("benchmark"):
        b = s["benchmark"]
        lines.append(f"- 벤치마크 {b['name']}: " + (b.get("error") or
                     f"Sharpe {_fmt(b['sharpe'])} (rf=0: {_fmt(b.get('sharpe_rf0'))}), 총수익 {_fmt(b['total_return'])}, MDD {_fmt(b['mdd'])}"))
    if s.get("trade_stats"):
        ts, tp = s["trade_stats"], s.get("trade") or {}
        lines += ["", "## 신호형 거래 (데스크 재생: make_plan → 체결 → check_position)",
                  f"- 사이징: 거래당 위험 {tp.get('risk_per_trade_pct')}%, heat {tp.get('max_heat_pct')}%, "
                  f"종목 비중 {tp.get('max_position_pct')}%, 보유 종목 {tp.get('max_open_positions')}개 "
                  "(같은 봉 손절·목표가 동시 도달은 손절로 처리)",
                  f"- 청산 {ts['closed']}건 (끝에 보유 {ts['open_at_end']}건): 승률 {_fmt(ts.get('win_rate'))}, "
                  f"평균 R {_fmt(ts.get('avg_r'))}, 손익비 {_fmt(ts.get('payoff'))}, 평균 보유 {_fmt(ts.get('avg_hold_bars'))}봉"]
        if ts.get("exit_kinds"):
            lines.append("- 청산 종류: " + ", ".join(f"{k} {v}" for k, v in ts["exit_kinds"].items()))
        if ts.get("binding"):
            lines.append("- 수량을 묶은 한도: " + ", ".join(f"{k} {v}" for k, v in ts["binding"].items()))
        if ts.get("blocked"):
            lines.append("- 진입 못 한 신호: " + ", ".join(f"{k} {v}" for k, v in ts["blocked"].items()))
        if ts.get("order_cap"):
            oc = ts["order_cap"]
            lines.append(f"- 1회 주문 상한 {oc['cap_krw']:,.0f}원 (백테스트는 자르지 않음, paper·live만 적용): "
                         f"상한이 있었다면 잘렸을 주문 {oc['would_cap']}건, 그중 진입 못 했을 주문 {oc['would_block']}건")
    lines += ["", "## 고정 기준", "| # | 기준 | 값 | 통과 |", "|---|---|---|---|"]
    for c in s["checks"]:
        p = {True: "✅", False: "❌", None: "미확인"}[c["pass"]]
        rf0 = f" (rf=0: {_fmt(c['value_rf0'])})" if c.get("value_rf0") is not None else ""
        lines.append(f"| {c['no']} | {c['rule']} | {_fmt(c['value'])}{rf0} | {p} |")
    if s.get("windows"):
        lines += ["", "## 워크포워드 창", "| 학습 | 시험 | 학습 Sharpe | 시험 Sharpe | 시험 수익 |", "|---|---|---|---|---|"]
        for w in s["windows"]:
            lines.append(f"| {w['train'][0]}~{w['train'][1]} | {w['test'][0]}~{w['test'][1]} | "
                         f"{w['train_sharpe']:.2f}{_rf0(w.get('train_sharpe_rf0'))} | "
                         f"{w['test_sharpe']:.2f}{_rf0(w.get('test_sharpe_rf0'))} | {w['test_return']:.2%} |")
    if s.get("sensitivity"):
        lines += ["", "## 파라미터 민감도 (하나씩 ±20%)"]
        lines += [f"- {x['param']}={x['value']} (×{x['factor']}): Sharpe {x['test_sharpe']:.2f}{_rf0(x.get('test_sharpe_rf0'))}"
                  for x in s["sensitivity"]]
    if s.get("regimes") and "error" not in s["regimes"]:
        lines += ["", "## 국면별 (참고, 전날 추세 라벨)"]
        lines += [f"- {k}: {v['bars']}봉, Sharpe {v['sharpe']:.2f}, 수익 {v['total_return']:.2%}"
                  for k, v in s["regimes"].items()]
    no_f = "" if s["data"].get("funding") else "선물 펀딩비, "
    lines += ["", f"반영 안 한 것: 공매도 대차 비용, {no_f}시장 충격, 양도소득세, 현금 이자."]
    return "\n".join(lines) + "\n"


def _rf0(v) -> str:
    return f" (rf=0: {v:.2f})" if v is not None else ""


def _pass_word(p) -> str:
    return {True: "통과", False: "**실패**", None: "미확인"}[p]


def _funding_lines(d: dict) -> list[str]:
    """선물: 펀딩비 반영 여부와 실행 시점 원천 범위."""
    if d.get("note") == "선물 펀딩비 미반영":
        return ["- 선물 펀딩비 반영: **아니오** (lab.yaml funding: false)"]
    f = d.get("funding")
    if not f:
        return []
    miss = f.get("symbols_without_funding") or []
    return [f"- 선물 펀딩비 반영: 예 ({f.get('source', 'binance_um_funding')}, 봉 동안 든 포지션 × 비율, 롱은 양수에 냄·숏은 받음, "
            f"정산 {f.get('events', 0):,}건, 범위 {f.get('range', '없음')}"
            + (f", **펀딩 데이터 없는 종목 {len(miss)}개: {', '.join(miss[:10])}** (0으로 처리)" if miss else "") + ")"]


def _fin_lines(d: dict) -> list[str]:
    """재무 필드를 쓴 lab: 시점 규칙과 실행 시점 원천 범위."""
    if not d.get("fin_fields"):
        return []
    v = d.get("fin_data") or {}
    out = [f"- 재무 필드 시점: {d['fin_fields']}"]
    miss = ", ".join(f"{k} {x:.1%}" for k, x in (v.get("missing") or {}).items())
    us = "amendments" in v or "note" in v
    if not v.get("rows"):
        out.append(f"- 실행 시점 재무 데이터 범위: **{'edgar_facts' if us else 'dart_fin'} 없음** (재무 필드 전부 비어 있음)")
    elif us:
        lack = v.get("symbols_without_facts") or []
        out.append(f"- 실행 시점 재무 데이터 범위: edgar_facts {v['rows']:,}행·{v['symbols']:,}종목, 공시일 {v['first']}~{v['last']}, "
                   f"정정 공시(…/A) 행 {v['amendments']:,}개"
                   + (f", 사실 없는 종목 {len(lack)}개({', '.join(lack[:10])})" if lack else "")
                   + f". 결측률(가격 있는 칸 기준): {miss}")
    else:
        out.append(f"- 실행 시점 재무 데이터 범위: dart_fin {v['rows']:,}개 분기·{v['symbols']:,}종목, "
                   f"공개일 {v['first']}~{v['last']}, 최초 접수일 없어 filed로 잡은 행 {v['fallback']:,}개, "
                   f"최초 접수 뒤 정정된 행 {v['corrected']:,}개, 원화 아닌 보고서 제외 {v['non_krw_skipped']:,}개, "
                   f"지배주주 순이익 없는 분기 {v.get('nic_missing', 0):,}개"
                   + (f"(출처별 행 {v['nic_source']})" if v.get("nic_source") else "")
                   + f". 결측률(가격 있는 칸 기준): {miss}")
    if v.get("eps"):
        out.append(f"- {v['eps']}")
    if v.get("note"):
        out.append(f"- 주의: {v['note']}")
    return out


def _mcap_lines(d: dict) -> list[str]:
    """US 시가총액 필드를 쓴 lab: 시점 규칙, 원천 범위, 결측률, 다중 클래스, 단위 이상치, 알려진 한계."""
    if not d.get("mcap_fields"):
        return []
    v = d.get("mcap_data") or {}
    out = [f"- 시가총액 필드 시점: {d['mcap_fields']}"]
    if not v.get("events"):
        out.append("- 실행 시점 주식수 데이터: **없음** (edgar_facts 주식수 항목 없음 → market_cap 전부 비어 있음)")
    else:
        fl = v.get("flags") or {}
        miss = v.get("missing")
        out.append(f"- 실행 시점 주식수 데이터: 공시 {v['events']:,}건(공시일 {v['first_filed']}~{v['last_filed']}), "
                   f"원천 행 edgar_facts {v['facts_rows']:,}·클래스별 {v['class_rows']:,}·분할 이력 {v['split_rows']:,}, "
                   f"버린 공시 {fl.get('dropped', 0):,}·단위 수정 {fl.get('scale_fixed', 0):,}건"
                   + (f". 결측률(가격 있는 칸 기준) {miss:.1%}" if miss is not None else ""))
        lack = v.get("symbols_without_shares") or []
        if lack:
            out.append(f"- 주식수가 한 번도 없는 종목 {len(lack)}개: {', '.join(lack[:15])}")
        if v.get("unit_outlier_symbols"):
            out.append(f"- 단위 이상치(전 기간 중앙값의 10배↑/↓ 날 있음) 종목: {', '.join(v['unit_outlier_symbols'][:15])}")
        mc = v.get("multi_class") or {}
        if mc:
            out.append(f"- 한 회사 여러 종목(cik 같음, 둘 다 회사 전체 시총): "
                       + "; ".join("·".join(s) for s in list(mc.values())[:10]) + " → 순위는 cik로 묶어 하나만")
    if v.get("warning"):
        out.append(f"- **경고: {v['warning']}**")
    out += [f"- 시가총액 한계: {x}" for x in v.get("limits") or []]
    return out


def _dividend_line(s: dict) -> str:
    """배당 재투자 반영 여부 한 줄 (KR은 항상 적는다 — 예전 결과와 구분하려고)."""
    d, market = s["data"], s.get("market")
    if market == "kr_stock":
        v = d.get("dividends") or {}
        if not v.get("reinvested"):
            return "- KR 배당 재투자 반영: **아니오** (가격 수익만 — 고배당 종목일수록 수익이 낮게 나옴)"
        return (f"- KR 배당 재투자 반영: 예 (kr_dividend_schedule 배당락일, 배당소득세 {v['tax']:.1%} 차감 후 재투자, "
                f"{v.get('events', 0):,}건·{v.get('symbols', 0):,}종목, 가격 없어 건너뜀 {v.get('skipped_no_price', 0)}건"
                + (f", 실행 시점 배당 일정 범위 {v['ex_range']}" if v.get("ex_range") else "") + ")")
    if market == "us_stock":
        return f"- US 배당 재투자 반영: 예 (us_1d adj_close, 원천징수 {d.get('dividend_tax', 0):.1%} 차감 후 재투자)"
    return "- 배당: 해당 없음"


def _write(out: Path, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    (out / "report.md").write_text(report(summary), encoding="utf-8")
    print(report(summary))
    print(f"→ {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="lab 백테스트")
    ap.add_argument("lab", type=Path)
    ap.add_argument("--lockbox", action="store_true", help="잠금 구간을 연다 (lab당 1회)")
    ap.add_argument("--quick", action="store_true", help="파라미터 민감도 생략")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # Windows 콘솔(cp949)에서 한글·기호가 깨지지 않게
        if stream.encoding and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")
    lab_dir = a.lab.resolve()
    open_lockbox(lab_dir) if a.lockbox else develop(lab_dir, a.quick)


if __name__ == "__main__":
    main()
