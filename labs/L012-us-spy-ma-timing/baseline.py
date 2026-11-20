"""비교 기준 (2026-11-20 추가, 판정과 무관 — 그림용).

    uv run python labs/L012-us-spy-ma-timing/baseline.py labs/L012-us-spy-ma-timing/results/<실행시각>

같은 시험 구간·비용·체결로 전략, 전략 비용 2배, SPY 매수 후 보유를 다시 계산해
<결과 폴더>/baseline.json, baseline.md, baseline_returns.parquet에 남긴다. 잠금 구간은 읽지 않는다.
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
    fill = cfg.get("fill", "next_open")
    rf = benchmark.rf_daily(test.index, krw=False)

    def st(r):
        x = benchmark.metrics(r, rf, year=year)
        return {k: float(x[k]) for k in ("sharpe", "sharpe_rf0", "cagr", "mdd", "total_return")}

    w = m.generate_signals(panel, cfg["params"])
    hold = pd.DataFrame(0.0, index=w.index, columns=panel.close.columns)
    hold["SPY"] = 1.0
    rs = engine.run(panel, w, costs, fill, start=t0, end=t1, **kw)
    rh = engine.run(panel, hold, costs, fill, start=t0, end=t1, **kw)
    rep = {"test_period": [str(test.index[0].date()), str(test.index[-1].date())],
           "strategy": st(rs.returns),
           "strategy_cost_x2": st(engine.run(panel, w, costs.scaled(2), fill, start=t0, end=t1, **kw).returns),
           "spy_hold": st(rh.returns)}
    pd.DataFrame({"strategy": rs.returns, "spy_hold": rh.returns}).to_parquet(out / "baseline_returns.parquet")
    (out / "baseline.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# 비교 기준 ({rep['test_period'][0]} ~ {rep['test_period'][1]}, 비용 포함)", "",
          "| | Sharpe (rf 차감) | CAGR | MDD |", "|---|---|---|---|"]
    for n, key in (("전략", "strategy"), ("전략 비용 2배", "strategy_cost_x2"), ("SPY 매수 후 보유", "spy_hold")):
        d = rep[key]
        md.append(f"| {n} | {d['sharpe']:.2f} | {d['cagr']:.1%} | {d['mdd']:.1%} |")
    (out / "baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
