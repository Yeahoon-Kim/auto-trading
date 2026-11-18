"""L003 라운드 2 (학습형 예측 모델) 실행.

    $env:AUTOTRADER_DATA_DIR = "C:\\Users\\apple\\Claude\\auto-trading\\data"
    # verifier 점검 (subagent도 실행 가능 — 실제 목표값 성과를 보지 않는다)
    uv run python labs/L003-chart-patterns/model/run.py --null                    # (a) 헛발견: 섞은 가짜 시장
    uv run python labs/L003-chart-patterns/model/run.py --null --leak             # (b) 검출력: 가짜 시장 + 미래 정보 특징
    # 실제 탐색 구간 워크포워드 (시도 횟수에 들어감 → lead agent만)
    uv run python labs/L003-chart-patterns/model/run.py --allow-real
    # 확인 구간 1회 (사전등록 뒤 lead agent만. 탐색 구간 전체로 최종 모델 1개 학습 → 확인 구간 예측)
    uv run python labs/L003-chart-patterns/model/run.py --period confirm --model hgb --explore-ic 0.0xx --allow-confirm

출력: labs/L003-chart-patterns/results/R2/<YYYYMMDD-HHMM>-<period>-<tag>/ report.md, summary.csv, folds.csv, predictions.parquet, run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset as D  # noqa: E402
import walkforward as W  # noqa: E402

OUT_ROOT = D.LAB / "results" / "R2"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=("explore", "confirm"), default="explore")
    ap.add_argument("--markets", nargs="+", default=list(D.MARKETS))
    ap.add_argument("--models", nargs="+", default=["zero", "ridge", "hgb"])
    ap.add_argument("--seeds", type=int, default=5, help="noise floor용 모델 시드 수 (hgb만 시드 영향)")
    ap.add_argument("--null", action="store_true", help="헛발견 검사: study shuffle_stock(boot) 가짜 시장")
    ap.add_argument("--null-seed", type=int, default=12345)
    ap.add_argument("--drift", choices=("remove", "keep"), default="remove",
                    help="가짜 시장에서 종목별 평균 수익 지우기 (remove 기본, 이유는 dataset.remove_drift)")
    ap.add_argument("--leak", action="store_true", help="검출력 시험: 목표값 + 잡음 특징 'leak'을 넣는다")
    ap.add_argument("--leak-corr", type=float, default=0.3, help="leak 특징과 y의 목표 상관 (잡음 크기로 정함)")
    ap.add_argument("--model", choices=("ridge", "hgb"), help="confirm: 사전등록한 최종 모델")
    ap.add_argument("--explore-ic", type=float, help="confirm: 탐색 워크포워드 IC (통과 기준 '50% 이상' 비교용)")
    ap.add_argument("--allow-real", action="store_true", help="실제 시장·실제 목표값으로 탐색 성과 계산 (lead agent만)")
    ap.add_argument("--allow-confirm", action="store_true", help="확인 구간 1회 실행 (lead agent만)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rebuild", action="store_true", help="데이터셋 캐시 무시")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    return ap.parse_args(argv)


def add_leak(df: pd.DataFrame, corr: float, seed: int = 99) -> pd.DataFrame:
    """y + 잡음. 잡음 표준편차 = sd(y) × sqrt(1/corr² − 1) → 피어슨 상관 ≈ corr. 하네스 검출력 시험 전용."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy(float)
    sd = np.nanstd(y) * np.sqrt(1 / corr ** 2 - 1)
    df = df.copy()
    df["leak"] = y + rng.normal(0, sd, len(df))
    return df


def run(a) -> Path:
    t0 = time.time()
    real = not a.null and not a.leak
    if a.period == "explore" and real and not a.allow_real:
        raise SystemExit("실제 시장 탐색 성과 계산은 시도 횟수에 들어간다 → lead agent만 --allow-real로 실행")
    if a.period == "confirm":
        if not a.allow_confirm:
            raise SystemExit("확인 구간은 라운드당 1회, lead agent만 --allow-confirm으로 실행")
        if not a.model:
            raise SystemExit("confirm은 사전등록한 --model 1개가 필요")
    ds_period = "explore" if a.period == "explore" else "both"
    timing = {}
    df, info = D.build_dataset(ds_period, a.markets, a.null_seed if a.null else None, a.drift, a.workers,
                               rebuild=a.rebuild)
    timing["dataset"] = round(time.time() - t0, 1)
    features = list(D.NUM_FEATURES)
    if a.leak:
        df = add_leak(df, a.leak_corr)
        features.append("leak")
    tag = "-".join(x for x in [a.tag, "null" if a.null else "", f"drift{a.drift}" if a.null and a.drift != "remove" else "",
                               "leak" if a.leak else "", a.model or ""] if x)
    stamp = time.strftime("%Y%m%d-%H%M")
    out = Path(a.out) if a.out else OUT_ROOT / f"{stamp}-{a.period}{'-' + tag if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    tw = time.time()
    if a.period == "explore":
        preds, summ, folds, hp = W.walk_forward(df, a.models, tuple(range(a.seeds)), features, "explore")
    else:
        preds, summ, folds, hp = W.walk_forward(df, ("zero", a.model), (0,), features, "confirm")
        if a.explore_ic is not None:
            summ["c_ic_half"] = summ["ic"] >= 0.5 * a.explore_ic
            summ["passed_confirm"] = summ["passed"] & summ["c_ic_half"]
    timing["walkforward"] = round(time.time() - tw, 1)
    nf = W.noise_floor(summ)
    main_col = next((f"pred_{m}_s0" for m in ("hgb", "ridge") if f"pred_{m}_s0" in preds), None)
    bd = W.breakdown(preds, main_col) if main_col else pd.DataFrame()
    timing["total"] = round(time.time() - t0, 1)
    preds.to_parquet(out / "predictions.parquet", index=False)
    summ.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out / "folds.csv", index=False, encoding="utf-8-sig")
    run_info = {"argv": sys.argv, "args": vars(a), "features": features, "dataset": info, "timing": timing,
                "rows": len(df), "hyperparams": hp}
    (out / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(a, df, summ, nf, folds, bd, features, timing, main_col), encoding="utf-8")
    print(f"출력: {out}  ({timing['total']}초)")
    print(summ[["model", "seed", "ic", "top_n", "top_exc", "top_net", "top_p_date", "top_p_blk", "passed"]].to_string(index=False))
    return out


def _pct(x, d=2):
    return "" if pd.isna(x) else f"{x * 100:+.{d}f}"


def _p(x):
    return "" if pd.isna(x) else (f"{x:.2g}" if x < 0.001 else f"{x:.3f}")


def report(a, df, summ, nf, folds, bd, features, timing, main_col) -> str:
    kind = "헛발견 검사(섞은 가짜 시장)" if a.null else "실제 시장"
    if a.leak:
        kind += f" + 검출력 시험(leak 특징, 목표 상관 {a.leak_corr})"
    L = [f"# L003 R2 학습형 예측 — {a.period} ({kind})", "",
         f"- 실행: `{' '.join(sys.argv)}`",
         f"- 구간: 탐색 = 신호일·20봉 청산일 모두 < {D.H.BOUNDARY.date()}. "
         + ("확인 구간 데이터는 읽지 않았다." if a.period == "explore" else "확인 = 신호일 ≥ 경계 (1회)."),
         "- 비용: 포함 (study 하네스 Costs, 청산일 기준 KR 매도세). 생존편향: KR 없음, US 있음(현재 구성종목).",
         "- 미래참조: 특징은 신호 봉 t 종가까지 (model/dataset.py 머리말, 잘라내기 검사 results/R2/harness-check.md).",
         f"- 가짜 시장: {'study shuffle_stock(boot), seed ' + str(a.null_seed) + ', drift ' + a.drift if a.null else '없음'}",
         f"- 데이터셋 {len(df)}행. 특징 {len(features)}개 + 범주 {D.CAT_FEATURE}: {', '.join(features)}",
         "- 목표값 y = 20봉 수익 − 같은 날·같은 시장·같은 봉 유니버스 평균 20봉 수익 (방향 부호 없음).",
         "- 상위 20% = 예측 연도마다 예측 백분위 > 0.8. 비용 후 = 평균(y − 왕복 비용). p = 날짜 묶음 / 20봉 블록 부트스트랩.",
         f"- 통과(탐색 verifier) = 상위 20% 표본 ≥ {W.MIN_TOP_N} ∧ 비용 후 > 0 ∧ 두 p < {W.ALPHA} ∧ IC > 0.", "",
         "## 데이터셋 크기", "", df.groupby(["market", "timeframe"]).size().rename("신호 수").reset_index().to_markdown(index=False), "",
         "## 모델·시드별 (예측 구간 전체)", "",
         "| 모델 | 시드 | 예측 신호 | IC | IC 날짜 | IC p(날짜/블록) | 상위 20% 수 | 상위 초과 % | 비용 % | 상위 비용 후 % | p(날짜/블록) | 전체 초과 % | 통과 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        L.append(f"| {r['model']} | {r['seed']} | {r['n_test']} | {r['ic']:+.4f} | {r['ic_dates']} | {_p(r['ic_p_date'])} / {_p(r['ic_p_blk'])} | "
                 f"{r['top_n']} | {_pct(r['top_exc'])} | {_pct(r['top_cost'])} | {_pct(r['top_net'])} | "
                 f"{_p(r['top_p_date'])} / {_p(r['top_p_blk'])} | {_pct(r['all_exc'])} | {'예' if r['passed'] else '아니오'} |")
    L += ["", "## noise floor (모델 시드)", "", "ridge·zero는 결정적이라 시드 1개 (흔들림 0).", "",
          nf.to_markdown(index=False, floatfmt=".4f"), "", "## 예측 연도별 (fold)", "",
          folds[["model", "seed", "fold", "n_test", "ic", "top_n", "top_net"]].to_markdown(index=False, floatfmt=".4f")]
    if len(bd):
        L += ["", f"## 시장·봉별 참고 ({main_col})", "", bd.to_markdown(index=False, floatfmt=".4f")]
    L += ["", "실행 시간(초): " + ", ".join(f"{k} {v}" for k, v in timing.items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    run(parse())
