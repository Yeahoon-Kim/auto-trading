"""비교 그림 — 전략 vs 대조군 (누적가치·낙폭·연도별). 판정과 무관한 시각화.

    uv run python labs/<lab>/plot.py labs/<lab>/results/<실행시각>

입력: <결과 폴더>/baseline_returns.parquet (baseline.py가 만든 일간 수익·무작위 누적가치 분위),
있으면 diagnostic_size_returns.parquet (L010 진단용). 출력: labs/<lab>/chart.png (git에 커밋).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from autotrader.backtest import benchmark

LAB = Path(__file__).resolve().parent
NAMES = {"strategy": "전략", "equal_weight": "유니버스 동일가중", "size_matched": "시총 맞춘 동일가중 (진단용)",
         "kospi": "KOSPI 가격지수 (배당·비용 없음)", "proxy_hold": "KOSPI 대리 포트폴리오 보유",
         "spy_hold": "SPY 매수 후 보유"}
COLORS = {"strategy": "#1f5fbf", "equal_weight": "#8c8c8c", "size_matched": "#b07aa1",
          "kospi": "#d9822b", "proxy_hold": "#d9822b", "spy_hold": "#d9822b"}


def main(out: Path) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cfg = yaml.safe_load((LAB / "lab.yaml").read_text(encoding="utf-8"))
    df = pd.read_parquet(out / "baseline_returns.parquet")
    if (out / "diagnostic_size_returns.parquet").exists():
        df = df.join(pd.read_parquet(out / "diagnostic_size_returns.parquet"), how="left")
    rets = [c for c in NAMES if c in df.columns]
    year = 252
    rf = benchmark.rf_daily(df.index, krw=cfg["market"].startswith("kr"))
    wealth = (1 + df[rets].fillna(0.0)).cumprod()

    fig, ax = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [3, 1.3, 1.6]})
    a = ax[0]
    if "random_wealth_p50" in df:
        a.fill_between(df.index, df["random_wealth_p5"], df["random_wealth_p95"], color="#cccccc", alpha=0.5,
                       label="무작위 100회 5~95%")
        a.plot(df.index, df["random_wealth_p50"], color="#999999", lw=1, ls=":", label="무작위 중앙값")
    for c in rets:
        m = benchmark.metrics(df[c].fillna(0.0), rf, year=year)
        a.plot(wealth.index, wealth[c], color=COLORS[c], lw=2.2 if c == "strategy" else 1.3,
               label=f"{NAMES[c]} — Sharpe {m['sharpe']:.2f}, CAGR {m['cagr']:.1%}, MDD {m['mdd']:.0%}")
    a.set_yscale("log")
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.5, 0.7, 1.5, 3, 5, 7) else ""))
    a.set_ylabel("누적가치 (시작 = 1, 로그)")
    a.legend(loc="upper left", fontsize=8.5)
    a.grid(alpha=0.3)
    a.set_title(f"{cfg['id']} {cfg['name']} — 시험 구간 {df.index[0].date()} ~ {df.index[-1].date()}, "
                f"비용 포함 (Sharpe는 rf 차감)", fontsize=11)

    b = ax[1]
    for c in rets:
        dd = wealth[c] / wealth[c].cummax() - 1
        b.plot(dd.index, dd, color=COLORS[c], lw=1.6 if c == "strategy" else 1.0)
    b.set_ylabel("낙폭")
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    b.grid(alpha=0.3)

    c_ = ax[2]
    yr = (1 + df[rets].fillna(0.0)).groupby(df.index.year).prod() - 1
    x = np.arange(len(yr))
    wbar = 0.8 / len(rets)
    for i, c in enumerate(rets):
        c_.bar(x + (i - (len(rets) - 1) / 2) * wbar, yr[c], wbar, color=COLORS[c], label=NAMES[c])
    c_.set_xticks(x, [str(y) for y in yr.index])
    c_.axhline(0, color="black", lw=0.6)
    c_.set_ylabel("연도별 수익")
    c_.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    c_.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(LAB / "chart.png", dpi=110)
    print(LAB / "chart.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
