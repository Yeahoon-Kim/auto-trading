"""사전에 정한 비교 기준 (고전 퀀트 프로그램 공통, 리서치 문서 4절).

    uv run python labs/L005-kr-value-bm/baseline.py labs/L005-kr-value-bm/results/<실행시각>

같은 시험 구간(summary.json의 시험 수익 시작~끝), 같은 비용·체결·리밸런싱으로:
  1. 전략 자체 (엔진 결과와 같은지 재확인)
  2. 같은 유니버스 전체 동일가중 (매월 초)
  3. 같은 유니버스에서 그달 전략 종목 수만큼 무작위로 뽑은 포트폴리오 N회 (시드 고정)
  4. KOSPI 지수 (macro_1d 가격지수, 비용·배당 없음 — 참고)
결과: <결과 폴더>/baseline.json, baseline.md. 잠금 구간은 읽지 않는다 (load lockbox=False).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from autotrader.backtest import benchmark, engine
from autotrader.backtest.data import load
from autotrader.data import store

LAB = Path(__file__).resolve().parent
N_RANDOM = 100
SEED = 20261118


def _strategy():
    spec = importlib.util.spec_from_file_location("lab_strategy", LAB / "strategy.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def kospi(index: pd.DatetimeIndex) -> pd.Series:
    d = store.read("macro_1d", "KOSPI")
    s = d.set_index(d["ts"].dt.tz_convert(None).dt.normalize())["close"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def stats(r: pd.Series, year: float, rf) -> dict:
    m = benchmark.metrics(r, rf, year=year)
    return {k: m[k] for k in ("sharpe", "sharpe_rf0", "cagr", "mdd", "total_return") if k in m}


def main(out: Path) -> None:
    cfg = yaml.safe_load((LAB / "lab.yaml").read_text(encoding="utf-8"))
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    test = pd.read_parquet(out / "test_returns.parquet")
    t0, t1 = test.index[0], test.index[-1] + pd.Timedelta(days=1)
    m = _strategy()
    params = cfg["params"]
    panel = load(cfg["market"], cfg.get("universe") or None, start=cfg.get("data_start") or None,
                 fields=tuple(cfg.get("fields") or ()), fin_timing=cfg.get("fin_timing") or "first")
    year = panel.bars_per_year
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    kw = engine.execution_kw(cfg.get("execution"), panel.index, t0, t1)
    fill = cfg.get("fill", "next_open")
    rf = benchmark.rf_daily(test.index, krw=True)

    def run(w):
        return engine.run(panel, w, costs, fill, start=t0, end=t1, **kw)

    w = m.generate_signals(panel, params)
    res = {"strategy": run(w)}
    ok = m.universe(panel, params)
    dates = w.index
    ew = ok.loc[dates].astype(float)
    ew = ew.div(ew.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    res["equal_weight_universe"] = run(ew)

    counts = (w > 0).sum(axis=1)
    rng = np.random.default_rng(SEED)
    rand, rand_w = [], []
    for i in range(N_RANDOM):
        rows = {}
        for d in dates:
            cand = np.flatnonzero(ok.loc[d].to_numpy())
            n = int(counts.loc[d])
            row = np.zeros(len(panel.close.columns))
            if n and len(cand):
                pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
                row[pick] = 1.0 / len(pick)
            rows[d] = row
        rw = pd.DataFrame.from_dict(rows, orient="index", columns=panel.close.columns)
        rr = run(rw).returns
        rand.append(stats(rr, year, rf))
        rand_w.append((1 + rr).cumprod().to_numpy())

    k = kospi(test.index)
    kr = k.pct_change().fillna(0.0)
    report = {
        "test_period": [str(test.index[0].date()), str(test.index[-1].date())],
        "engine_strategy": {"sharpe": s["test"].get("sharpe"), "total_return": s["test"].get("total_return")},
        "strategy": stats(res["strategy"].returns, year, rf),
        "strategy_cost_x2": stats(engine.run(panel, w, costs.scaled(2), fill, start=t0, end=t1, **kw).returns, year, rf),
        "equal_weight_universe": stats(res["equal_weight_universe"].returns, year, rf),
        "kospi_price_index_no_cost": stats(kr, year, rf),
        "random": {"n": N_RANDOM, "seed": SEED,
                   "sharpe_pct": {str(q): float(np.percentile([r["sharpe"] for r in rand], q)) for q in (5, 50, 95)},
                   "cagr_pct": {str(q): float(np.percentile([r["cagr"] for r in rand], q)) for q in (5, 50, 95)},
                   "mdd_median": float(np.median([r["mdd"] for r in rand])),
                   "strategy_sharpe_rank": float(np.mean([r["sharpe"] < stats(res["strategy"].returns, year, rf)["sharpe"]
                                                          for r in rand]))},
        "avg_holdings": float(counts.loc[counts.index >= t0].mean()),
        "avg_universe": float(ok.loc[dates][dates >= t0].sum(axis=1).mean()),
    }
    # 연도별 (전략 − 동일가중)
    a, b = res["strategy"].returns, res["equal_weight_universe"].returns
    yr = pd.DataFrame({"strategy": (1 + a).groupby(a.index.year).prod() - 1,
                       "equal_weight": (1 + b).groupby(b.index.year).prod() - 1,
                       "kospi": (1 + kr).groupby(kr.index.year).prod() - 1})
    report["yearly"] = {str(y): {c: float(v) for c, v in row.items()} for y, row in yr.iterrows()}
    # 그림용 일간 수익·무작위 누적가치 분위 (2026-11-20 추가, 판정과 무관)
    idx = res["strategy"].returns.index
    curves = pd.DataFrame({"strategy": res["strategy"].returns, "equal_weight": res["equal_weight_universe"].returns,
                           "kospi": kr.reindex(idx).fillna(0.0)}, index=idx)
    rw_ = np.vstack([np.asarray(x, dtype=float) for x in rand_w])
    for q in (5, 50, 95):
        curves[f"random_wealth_p{q}"] = np.percentile(rw_, q, axis=0)
    curves.to_parquet(out / "baseline_returns.parquet")
    (out / "baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(name, d):
        return f"| {name} | {d.get('sharpe', float('nan')):.2f} | {d.get('cagr', float('nan')):.1%} | {d.get('mdd', float('nan')):.1%} |"
    md = [f"# 비교 기준 ({report['test_period'][0]} ~ {report['test_period'][1]}, 비용 포함 — KOSPI만 비용 없음)", "",
          "| | Sharpe | CAGR | MDD |", "|---|---|---|---|",
          line("전략", report["strategy"]), line("전략 (비용 2배)", report["strategy_cost_x2"]),
          line("유니버스 동일가중", report["equal_weight_universe"]), line("KOSPI 가격지수", report["kospi_price_index_no_cost"]),
          f"| 무작위 {N_RANDOM}회 중앙값 | {report['random']['sharpe_pct']['50']:.2f} | {report['random']['cagr_pct']['50']:.1%} | {report['random']['mdd_median']:.1%} |",
          "", f"- 무작위 Sharpe 5/50/95%: {report['random']['sharpe_pct']}",
          f"- 전략 Sharpe가 무작위보다 높은 비율: {report['random']['strategy_sharpe_rank']:.0%}",
          f"- 평균 보유 종목 {report['avg_holdings']:.0f} / 평균 유니버스 {report['avg_universe']:.0f}",
          f"- 엔진 시험 Sharpe {report['engine_strategy']['sharpe']} (재계산과 같아야 함)", "",
          "| 연도 | 전략 | 동일가중 | KOSPI |", "|---|---|---|---|"]
    md += [f"| {y} | {v['strategy']:.1%} | {v['equal_weight']:.1%} | {v['kospi']:.1%} |" for y, v in report["yearly"].items()]
    (out / "baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
