"""사전등록 비교 기준: 같은 날·같은 손절/목표 거리의 무작위 종목 진입 (위약).

lab 엔진과 같은 워크포워드로 전략 신호를 만든 뒤, 시험 구간 신호마다 종목만 그날 유니버스 조건(거래대금·주가·이력)을
만족하는 다른 종목으로 바꾸고, 손절·목표를 그 종목 종가 기준 같은 비율 거리로 옮긴다. 같은 엔진(trade.run)으로 재생.
시드 n개 → 위약 Sharpe 분포 대비 전략 Sharpe의 위치(백분위)를 낸다. 이 실행은 해당 lab 시도 1회에 포함된다
(lab 엔진 실행과 같은 시도로 사전등록, 결과를 보고 따로 돌리지 않는다).

    uv run python labs/L0xx-이름/placebo.py labs/L0xx-이름 [--seeds 20]
결과: labs/<lab>/results/placebo-<시각>.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from autotrader.backtest import engine, trade
from autotrader.backtest import lab as L


def eligible(panel, p):
    val = (panel.value if "value" in panel else panel.raw_close * panel.volume).rolling(20, min_periods=20).mean()
    px = panel.raw_close if "raw_close" in panel else panel.close
    need = max([int(k) for k in p.get("ma_periods", [250])] + [250]) + 5
    hist = panel.close.notna().rolling(need, min_periods=1).sum() >= need
    return (val >= p["min_value"]) & (px >= p["min_price"]) & hist


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("lab")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args(argv)
    lab_dir = Path(a.lab)
    cfg = yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8"))
    strat = L.load_strategy(lab_dir)
    params = cfg.get("params") or {}
    panel = L._panel(cfg, lockbox=False)
    costs = engine.Costs.from_config(cfg["market"], cfg.get("costs"))
    R = L._Runner(cfg, strat, panel, costs, str(cfg.get("id")))
    base = R.walk_forward(params)
    test = base["test"]
    t0, t1 = test.returns.index[0], test.returns.index[-1] + pd.Timedelta(days=1)
    sig = base["trades"].copy()
    el = eligible(panel, params)
    C = panel.close
    s_main = L.judge_sharpe(test.returns, panel)["sharpe"]
    out = {"lab": lab_dir.name, "run_at": datetime.now().isoformat(timespec="seconds"),
           "test": [str(t0.date()), str(t1.date())], "signals": len(sig), "strategy_sharpe": s_main,
           "strategy_total_return": float((1 + test.returns).prod() - 1), "placebo": []}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        rows = []
        for r in sig.itertuples(index=False):
            pool = el.columns[el.loc[r.ts].to_numpy(bool)]
            pool = pool[pool != r.symbol]
            if not len(pool):
                continue
            s = str(rng.choice(pool))
            c0, c1 = C.at[r.ts, r.symbol], C.at[r.ts, s]
            rows.append({"ts": r.ts, "symbol": s, "side": "long",
                         "stop": c1 * r.stop / c0, "target": c1 * r.target / c0})
        res = trade.run(panel, pd.DataFrame(rows), costs, R.account, start=t0, end=t1, **R.tkw)
        out["placebo"].append({"seed": seed, "sharpe": L.judge_sharpe(res.returns, panel)["sharpe"],
                               "total_return": float((1 + res.returns).prod() - 1), "trades": res.trades})
    ps = np.array([x["sharpe"] for x in out["placebo"]])
    out["placebo_sharpe_mean"], out["placebo_sharpe_std"] = float(ps.mean()), float(ps.std(ddof=1))
    out["strategy_pct_rank"] = float((ps < s_main).mean())
    path = lab_dir / "results" / f"placebo-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "placebo"}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
