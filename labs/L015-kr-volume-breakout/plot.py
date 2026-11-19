"""비교 그림: 시험 구간 전략 vs 사전등록 대조군(무작위 종목 위약, 시드 20) — 시각화일 뿐 시도로 세지 않는다.

새 백테스트를 돌리지 않는다. 전략 곡선은 lab 실행이 남긴 `results/<최신>/test_returns.parquet`(비용 포함 net)을 그대로 쓴다.
위약 곡선은 `placebo.py`가 이미 낸 결과(시드 0~19)를 같은 코드·같은 시드로 **재생**해 일별 수익을 얻는다
(placebo json에는 시드별 Sharpe만 있어서). 재생한 Sharpe가 json과 다르면 멈춘다. 재생 결과는
`results/placebo-series.parquet`에 저장해 다시 그릴 때 쓴다.

    uv run python labs/L015-kr-volume-breakout/plot.py labs/L015-kr-volume-breakout        # KR
    uv run python labs/L015-kr-volume-breakout/plot.py labs/L015-kr-volume-breakout/us     # US 참고
결과: <lab>/chart.png
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from autotrader.backtest import engine, trade
from autotrader.backtest import lab as L
from autotrader.backtest.data import load

HERE = Path(__file__).resolve().parent


def _placebo_mod():
    spec = importlib.util.spec_from_file_location("placebo_mod", HERE / "placebo.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def replay_placebo(lab_dir: Path, pj: dict, panel) -> pd.DataFrame:
    """placebo.py와 같은 계산을 같은 시드로 다시 해서 시드별 일별 수익(열=시드)을 돌려준다."""
    P = _placebo_mod()
    cfg = yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8"))
    strat = L.load_strategy(lab_dir)
    params = cfg.get("params") or {}
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    R = L._Runner(cfg, strat, panel, costs, str(cfg.get("id")))
    base = R.walk_forward(params)
    test = base["test"]
    t0, t1 = test.returns.index[0], test.returns.index[-1] + pd.Timedelta(days=1)
    sig, el, C = base["trades"].copy(), P.eligible(panel, params), panel.close
    cols = {}
    for seed in range(len(pj["placebo"])):
        rng = np.random.default_rng(seed)
        rows = []
        for r in sig.itertuples(index=False):
            pool = el.columns[el.loc[r.ts].to_numpy(bool)]
            pool = pool[pool != r.symbol]
            if not len(pool):
                continue
            s = str(rng.choice(pool))
            c0, c1 = C.at[r.ts, r.symbol], C.at[r.ts, s]
            rows.append({"ts": r.ts, "symbol": s, "side": "long", "stop": c1 * r.stop / c0, "target": c1 * r.target / c0})
        res = trade.run(panel, pd.DataFrame(rows), costs, R.account, start=t0, end=t1, **R.tkw)
        got, want = L.judge_sharpe(res.returns, panel)["sharpe"], pj["placebo"][seed]["sharpe"]
        if abs(got - want) > 1e-6:
            raise SystemExit(f"위약 재생 불일치: seed {seed} Sharpe {got} vs json {want}")
        cols[seed] = res.returns
    return pd.DataFrame(cols)


def stats(r: pd.Series) -> tuple[float, float]:
    v = (1 + r.fillna(0)).cumprod()
    years = len(r) / 252
    return float(v.iloc[-1] ** (1 / years) - 1), float((v / v.cummax() - 1).min())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("lab")
    a = ap.parse_args(argv)
    lab_dir = Path(a.lab)
    run = sorted(glob.glob(str(lab_dir / "results" / "2*" / "summary.json")))[-1]
    summ = json.loads(Path(run).read_text(encoding="utf-8"))
    strat = pd.read_parquet(Path(run).parent / "test_returns.parquet")["net"]
    pj = json.loads(Path(sorted(glob.glob(str(lab_dir / "results" / "placebo-*.json")))[-1]).read_text(encoding="utf-8"))
    cache = lab_dir / "results" / "placebo-series.parquet"
    if cache.exists():
        pl = pd.read_parquet(cache)
    else:
        cfg = yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8"))
        # 개발 구간 끝은 실행일 기준(실행일 − 6개월)이라, 원래 실행 날짜(today)로 불러야 같은 계산이 된다
        fields = list(cfg.get("fields") or ()) + [f for f in ("high", "low") if f not in (cfg.get("fields") or ())]
        panel = load(cfg["market"], cfg.get("universe") or None, start=cfg.get("data_start") or None,
                     timeframe=cfg.get("timeframe", "1d"), fields=tuple(fields), lockbox=False,
                     today=pj["run_at"][:10], currency=cfg.get("currency") or None,
                     dividend_tax=cfg.get("dividend_tax"), fin_timing=cfg.get("fin_timing") or "first",
                     funding=cfg.get("funding", True) is not False)   # lab._panel과 같은 인자 + today
        pl = replay_placebo(lab_dir, pj, panel)
        pl.columns = [str(c) for c in pl.columns]
        pl.to_parquet(cache)
    pl = pl.reindex(strat.index).fillna(0)

    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    sv = (1 + strat.fillna(0)).cumprod()
    pv = (1 + pl).cumprod()
    lo, med, hi = pv.quantile(0.05, axis=1), pv.median(axis=1), pv.quantile(0.95, axis=1)
    s_cagr, s_mdd = stats(strat)
    med_r = med.pct_change().fillna(med.iloc[0] - 1)
    p_cagr, p_mdd = stats(med_r)
    sh = summ["test"]["sharpe"]

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 1.3, 1.6]}, sharex=False)
    ax[0].fill_between(sv.index, lo, hi, color="tab:gray", alpha=0.25, label="위약 5~95% (시드 20)")
    ax[0].plot(med.index, med, color="tab:gray", lw=1.4,
               label=f"위약 중앙값 — Sharpe 평균 {pj['placebo_sharpe_mean']:.2f}±{pj['placebo_sharpe_std']:.2f}, CAGR {p_cagr:.1%}, MDD {p_mdd:.0%}")
    ax[0].plot(sv.index, sv, color="tab:red", lw=1.8,
               label=f"전략 (비용 포함) — Sharpe {sh:.2f}, CAGR {s_cagr:.1%}, MDD {s_mdd:.0%}, 위약 대비 백분위 {pj['strategy_pct_rank']:.0%}")
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))   # 로그 눈금을 10^-1 대신 0.1처럼 (글꼴에 수학 기호 없음)
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].axhline(1, color="black", lw=0.6)
    ax[0].set_title(f"{lab_dir.parent.name + '/' if lab_dir.name == 'us' else ''}{lab_dir.name} 시험 구간 "
                    f"{strat.index[0]:%Y-%m} ~ {strat.index[-1]:%Y-%m} 누적가치 (시작=1, 로그). Sharpe는 rf 차감")
    ax[0].legend(loc="upper left", fontsize=8.5)
    ax[0].grid(alpha=0.3, which="both")

    ax[1].fill_between(sv.index, sv / sv.cummax() - 1, 0, color="tab:red", alpha=0.35, label="전략 낙폭")
    ax[1].plot(med.index, med / med.cummax() - 1, color="tab:gray", lw=1, label="위약 중앙값 낙폭")
    ax[1].set_ylabel("낙폭")
    ax[1].legend(loc="lower left", fontsize=8.5)
    ax[1].grid(alpha=0.3)

    ys = strat.groupby(strat.index.year).apply(lambda x: (1 + x).prod() - 1)
    yp = med.groupby(med.index.year).last() / med.groupby(med.index.year).last().shift(1).fillna(1) - 1
    x = np.arange(len(ys))
    ax[2].bar(x - 0.2, ys.values, 0.4, color="tab:red", label="전략")
    ax[2].bar(x + 0.2, yp.reindex(ys.index).values, 0.4, color="tab:gray", label="위약 중앙값")
    ax[2].set_xticks(x, [str(y) for y in ys.index])
    ax[2].axhline(0, color="black", lw=0.6)
    ax[2].set_ylabel("연도별 수익")
    ax[2].legend(loc="lower left", fontsize=8.5)
    ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = lab_dir / "chart.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
