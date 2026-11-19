"""진단용 (trials 미포함, 사전등록 규칙 변경 아님) — 시총 분포를 맞춘 동일가중 비교.

    uv run python labs/L010-kr-value-profitability/diagnostic_size.py labs/L010-kr-value-profitability/results/<실행시각>

HQ 지시(2026-11-19): 전략이 소형주에 쏠려 동일가중 대비 초과가 커 보일 수 있으므로,
매월 초 같은 유니버스를 시가총액 10분위로 나누고, 전략 보유 종목의 분위 비중을 그대로 맞춘
동일가중 포트폴리오(분위 안에서는 유니버스 전체 동일가중)와 비교한다. 무작위성 없음.
같은 시험 구간·비용·체결. 결과: <결과 폴더>/diagnostic_size.json, diagnostic_size.md.
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

LAB = Path(__file__).resolve().parent
N_BUCKETS = 10


def _strategy():
    spec = importlib.util.spec_from_file_location("lab_strategy", LAB / "strategy.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stats(r, year, rf):
    m = benchmark.metrics(r, rf, year=year)
    return {k: m[k] for k in ("sharpe", "sharpe_rf0", "cagr", "mdd", "total_return") if k in m}


def main(out: Path) -> None:
    cfg = yaml.safe_load((LAB / "lab.yaml").read_text(encoding="utf-8"))
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

    def run(w, c=costs):
        return engine.run(panel, w, c, fill, start=t0, end=t1, **kw)

    w = m.generate_signals(panel, params)
    ok = m.universe(panel, params)
    mc = panel.market_cap
    rows, share = {}, []
    for d in w.index:
        cap = mc.loc[d].where(ok.loc[d] & (mc.loc[d] > 0)).dropna()
        row = pd.Series(0.0, index=panel.close.columns)
        picks = w.loc[d][w.loc[d] > 0].index
        if len(cap) >= N_BUCKETS and len(picks):
            b = pd.qcut(cap.rank(method="first"), N_BUCKETS, labels=False)
            pb = b.reindex(picks).dropna()
            frac = pb.value_counts(normalize=True)
            for k, f in frac.items():
                members = b.index[b == k]
                row[members] = f / len(members)
            if d >= t0:
                share.append(frac.reindex(range(N_BUCKETS), fill_value=0.0))
        rows[d] = row
    sm = pd.DataFrame(rows).T
    strat, size = run(w), run(sm)
    ew = ok.loc[w.index].astype(float)
    ew = ew.div(ew.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    eq = run(ew)
    diff = strat.returns - size.returns
    yr = lambda r: (1 + r).groupby(r.index.year).prod() - 1
    rep = {
        "note": "진단용 실행 — trials 미포함, 판정 기준 아님",
        "test_period": [str(test.index[0].date()), str(test.index[-1].date())],
        "strategy": stats(strat.returns, year, rf),
        "size_matched_ew": stats(size.returns, year, rf),
        "size_matched_ew_cost_x2": stats(run(sm, costs.scaled(2)).returns, year, rf),
        "equal_weight_universe": stats(eq.returns, year, rf),
        "excess_vs_size_matched_ann": float(diff.mean() * year),
        "excess_t": float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))),
        "pick_decile_share_avg": {str(k + 1): float(v) for k, v in pd.concat(share, axis=1).mean(axis=1).items()},
        "yearly": {str(y): {"strategy": float(a), "size_matched": float(b)}
                   for y, a, b in zip(yr(strat.returns).index, yr(strat.returns), yr(size.returns))},
    }
    (out / "diagnostic_size.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    f = lambda d: f"{d['sharpe']:.2f} | {d['cagr']:.1%} | {d['mdd']:.1%}"
    md = [f"# 진단용 — 시총 분포 맞춘 동일가중 ({rep['test_period'][0]} ~ {rep['test_period'][1]}, 비용 포함)",
          "", "trials 미포함. 사전등록 규칙·판정 기준과 무관한 참고 실행.", "",
          "| | Sharpe (rf 차감) | CAGR | MDD |", "|---|---|---|---|",
          f"| 전략 | {f(rep['strategy'])} |", f"| 시총 분포 맞춘 동일가중 | {f(rep['size_matched_ew'])} |",
          f"| 유니버스 동일가중 | {f(rep['equal_weight_universe'])} |", "",
          f"- 전략 − 시총 맞춘 동일가중: 연 {rep['excess_vs_size_matched_ann']:.1%} (일간 차이 평균의 t {rep['excess_t']:.2f})",
          f"- 전략 보유 종목의 시총 10분위 비중 평균 (1=가장 작음): " +
          ", ".join(f"{k}:{v:.0%}" for k, v in rep["pick_decile_share_avg"].items()), "",
          "| 연도 | 전략 | 시총 맞춤 |", "|---|---|---|"]
    md += [f"| {y} | {v['strategy']:.1%} | {v['size_matched']:.1%} |" for y, v in rep["yearly"].items()]
    (out / "diagnostic_size.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
