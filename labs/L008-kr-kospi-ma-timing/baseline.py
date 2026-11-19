"""사전에 정한 비교 기준 — L008 전용.

    uv run python labs/L008-kr-kospi-ma-timing/baseline.py labs/L008-kr-kospi-ma-timing/results/<실행시각>

같은 시험 구간, 같은 비용으로:
  1. 전략 (엔진 결과 재확인)
  2. KOSPI 대리 포트폴리오 항상 보유 (같은 월초 시가총액 비중, 국면 필터만 끔) ← 핵심 비교
  3. KOSPI 지수 (macro_1d 가격지수, 비용·배당 없음 — 대리 포트폴리오가 지수를 잘 따라가는지 확인용)
결과: <결과 폴더>/baseline.json, baseline.md
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from autotrader.backtest import benchmark, engine
from autotrader.backtest.data import load

LAB = Path(__file__).resolve().parent


def main(out: Path) -> None:
    spec = importlib.util.spec_from_file_location("lab_strategy", LAB / "strategy.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = yaml.safe_load((LAB / "lab.yaml").read_text(encoding="utf-8"))
    test = pd.read_parquet(out / "test_returns.parquet")
    t0, t1 = test.index[0], test.index[-1] + pd.Timedelta(days=1)
    panel = load(cfg["market"], cfg.get("universe") or None, start=cfg.get("data_start") or None,
                 fields=tuple(cfg.get("fields") or ()))
    year = panel.bars_per_year
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    kw = engine.execution_kw(cfg.get("execution"), panel.index, t0, t1)
    rf = benchmark.rf_daily(test.index, krw=True)

    def st(r):
        x = benchmark.metrics(r, rf, year=year)
        return {k: float(x[k]) for k in ("sharpe", "sharpe_rf0", "cagr", "mdd", "total_return")}

    w = m.generate_signals(panel, cfg["params"])
    hold = m.proxy_weights(panel, w.index)
    rs = engine.run(panel, w, costs, start=t0, end=t1, **kw)
    rh = engine.run(panel, hold, costs, start=t0, end=t1, **kw)
    k = m.kospi_close(test.index).pct_change().fillna(0.0)
    on = (w.sum(axis=1) > 0)
    rep = {"test_period": [str(test.index[0].date()), str(test.index[-1].date())],
           "strategy": st(rs.returns), "strategy_cost_x2": st(engine.run(panel, w, costs.scaled(2), start=t0, end=t1, **kw).returns),
           "proxy_hold": st(rh.returns), "kospi_price_index_no_cost": st(k),
           "proxy_vs_index_daily_corr": float(rh.returns.corr(k)),
           "invested_month_share": float(on[on.index >= t0].mean()),
           "switches": int((on[on.index >= t0].astype(int).diff().abs() > 0).sum()),
           "trades": rs.trades}
    yr = pd.DataFrame({"strategy": (1 + rs.returns).groupby(rs.returns.index.year).prod() - 1,
                       "proxy_hold": (1 + rh.returns).groupby(rh.returns.index.year).prod() - 1,
                       "kospi": (1 + k).groupby(k.index.year).prod() - 1})
    rep["yearly"] = {str(y): {c: float(v) for c, v in r.items()} for y, r in yr.iterrows()}
    # 그림용 일간 수익 (2026-11-20 추가, 판정과 무관)
    pd.DataFrame({"strategy": rs.returns, "proxy_hold": rh.returns, "kospi": k.reindex(rs.returns.index).fillna(0.0)}).to_parquet(out / "baseline_returns.parquet")
    (out / "baseline.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# 비교 기준 ({rep['test_period'][0]} ~ {rep['test_period'][1]})", "", "| | Sharpe | CAGR | MDD |", "|---|---|---|---|"]
    for n, key in (("전략", "strategy"), ("전략 비용 2배", "strategy_cost_x2"), ("대리 포트폴리오 보유", "proxy_hold"),
                   ("KOSPI 가격지수 (비용 없음)", "kospi_price_index_no_cost")):
        d = rep[key]
        md.append(f"| {n} | {d['sharpe']:.2f} | {d['cagr']:.1%} | {d['mdd']:.1%} |")
    md += ["", f"- 대리 vs 지수 일간 상관 {rep['proxy_vs_index_daily_corr']:.3f}, 투자 월 비율 {rep['invested_month_share']:.0%}, 전환 {rep['switches']}회",
           "", "| 연도 | 전략 | 대리 보유 | KOSPI |", "|---|---|---|---|"]
    md += [f"| {y} | {v['strategy']:.1%} | {v['proxy_hold']:.1%} | {v['kospi']:.1%} |" for y, v in rep["yearly"].items()]
    (out / "baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
