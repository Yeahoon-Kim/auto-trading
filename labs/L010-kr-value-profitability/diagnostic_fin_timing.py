"""진단용 (trials 미포함, 판정 기준 아님) — 재무 공개일 규칙 민감도: fin_timing first vs last.

    uv run python labs/L010-kr-value-profitability/diagnostic_fin_timing.py labs/L010-kr-value-profitability/results/<실행시각>

first = 최초 접수일부터 정정 값이 보임(경미한 미래참조), last = 마지막 접수일(정정일)부터 보임(미래참조 없음, 더 보수적).
같은 시험 구간·비용·체결로 두 설정을 한 번씩 돌려 비교한다 (방법론 점검 2026-11-19 3절 lab 정리 항목).
결과: <결과 폴더>/diagnostic_fin_timing.json, .md
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
    rf = benchmark.rf_daily(test.index, krw=True)
    rep = {"note": "진단용 실행 — trials 미포함, 판정 기준 아님",
           "test_period": [str(test.index[0].date()), str(test.index[-1].date())]}
    for timing in ("first", "last"):
        panel = load(cfg["market"], cfg.get("universe") or None, start=cfg.get("data_start") or None,
                     fields=tuple(cfg.get("fields") or ()), fin_timing=timing)
        costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
        kw = engine.execution_kw(cfg.get("execution"), panel.index, t0, t1)
        w = m.generate_signals(panel, cfg["params"])
        res = engine.run(panel, w, costs, cfg.get("fill", "next_open"), start=t0, end=t1, **kw)
        x = benchmark.metrics(res.returns, rf, year=panel.bars_per_year)
        rep[timing] = {k: float(x[k]) for k in ("sharpe", "sharpe_rf0", "cagr", "mdd", "total_return")}
        rep[timing]["trades"] = int(res.trades)
        del panel
    (out / "diagnostic_fin_timing.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# 진단용 — fin_timing first vs last ({rep['test_period'][0]} ~ {rep['test_period'][1]}, 비용 포함)", "",
          "trials 미포함. 판정 기준과 무관한 참고 실행.", "",
          "| fin_timing | Sharpe (rf 차감) | CAGR | MDD | 거래 |", "|---|---|---|---|---|"]
    md += [f"| {t} | {rep[t]['sharpe']:.3f} | {rep[t]['cagr']:.1%} | {rep[t]['mdd']:.1%} | {rep[t]['trades']} |" for t in ("first", "last")]
    (out / "diagnostic_fin_timing.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
