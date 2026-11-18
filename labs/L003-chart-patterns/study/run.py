"""L003 A 단계 event study 실행.

    $env:AUTOTRADER_DATA_DIR = "C:\\Users\\apple\\Claude\\auto-trading\\data"
    uv run python labs/L003-chart-patterns/study/run.py --period explore --markets kr_stock us_stock --timeframes 1d 1w
    uv run python labs/L003-chart-patterns/study/run.py --period explore --detector fake          # 가짜 탐지기 A·B·C
    uv run python labs/L003-chart-patterns/study/run.py --period explore --detector fake --null   # 헛발견 검사

출력: labs/L003-chart-patterns/results/A/<YYYYMMDD-HHMM>-<period>[-tag]/ events.parquet, summary.csv, report.md, run.json
확인 구간(--period confirm|both)은 lead agent만 실행한다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

PERIODS = {"explore": (0,), "confirm": (1,), "both": (0, 1)}
PERIOD_NAME = {0: "탐색", 1: "확인"}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=PERIODS, default="explore")
    ap.add_argument("--markets", nargs="+", default=["kr_stock", "us_stock"])
    ap.add_argument("--timeframes", nargs="+", default=["1d", "1w"])
    ap.add_argument("--detector", default="patterns", help="patterns (실제 탐지기) | fake (가짜 탐지기 A·B·C)")
    ap.add_argument("--tools", nargs="*", help="도구 일부만 (기본 = 탐지기 모듈의 TOOLS 전체)")
    ap.add_argument("--null", action="store_true", help="헛발견 검사: 종목별 일봉 묶음을 섞은 가짜 시장으로 실행")
    ap.add_argument("--null-seed", type=int, default=12345)
    ap.add_argument("--seeds", type=int, default=20, help="noise floor용 위약 추출 시드 수")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-symbols", type=int, help="개발용: 시장마다 종목 수 제한 (무작위, 시드 고정)")
    ap.add_argument("--causality-cuts", type=int, default=2, help="유니버스·분위 잘라내기 검사 지점 수 (0 = 생략)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", help="출력 폴더 (기본 results/A/<시각>-<period>[-tag])")
    return ap.parse_args(argv)


def detector_module(name: str):
    if name == "fake":
        return str(HERE), "fakes"
    if name == "patterns":
        return str(H.LAB), "patterns"
    raise SystemExit(f"--detector는 patterns|fake: {name}")


def week_end_cuts(idx: pd.DatetimeIndex, n: int, lo, hi) -> list[pd.Timestamp]:
    """[lo, hi) 안에서 주의 마지막 거래일 n개 (고르게)."""
    wk = H.week_key(idx.values)
    last = idx[np.r_[wk[1:] != wk[:-1], False]]
    last = last[(last >= pd.Timestamp(lo)) & (last < pd.Timestamp(hi))]
    if n <= 0 or len(last) == 0:
        return []
    return [last[int(i)] for i in np.linspace(len(last) * 0.3, len(last) - 1, n)]


def run(args) -> Path:
    t0 = time.time()
    mod_path, mod_name = detector_module(args.detector)
    try:
        sys.path.insert(0, mod_path)
        mod = importlib.import_module(mod_name)
    except Exception as e:  # 실제 탐지기가 아직 없거나 깨졌으면 여기서 멈춘다
        raise SystemExit(f"탐지기 모듈 {mod_name} import 실패 ({type(e).__name__}: {e}). "
                         f"가짜 탐지기로 돌리려면 --detector fake") from None
    tools = tuple(args.tools) if args.tools else tuple(mod.TOOLS)
    periods = PERIODS[args.period]
    stamp = time.strftime("%Y%m%d-%H%M")
    tag = "-".join(x for x in [args.tag, "null" if args.null else "", args.detector if args.detector != "patterns" else ""] if x)
    out = Path(args.out) if args.out else H.LAB / "results" / "A" / f"{stamp}-{args.period}{'-' + tag if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    info = {"args": vars(args), "tools": tools, "timing": {}, "markets": {}, "causality": {}}
    all_ev, noise_rows = [], []
    for market in args.markets:
        tm = time.time()
        wide = H.load_wide(market, use_cache=not args.no_cache)
        info["timing"][f"{market}_load"] = round(time.time() - tm, 1)
        if args.period == "explore":
            wide = H.slice_wide(wide, H.BOUNDARY)        # 확인 구간 데이터는 아예 쓰지 않는다
        costs = H.Costs(market)
        univ = H.universe_wide(market, wide)
        stocks = H.stocks_from_wide(wide)
        if args.max_symbols and len(stocks) > args.max_symbols:
            pick = np.random.default_rng(7).choice(len(stocks), args.max_symbols, replace=False)
            stocks = [stocks[i] for i in sorted(pick)]
            keep = [s.sym for s in stocks]
            wide = {k: v[keep] for k, v in wide.items()}
            univ = univ[keep] if univ is not None else None
        if args.null:
            stocks = [H.shuffle_stock(s, args.null_seed) for s in stocks]
        panel_last = wide["close"].index[-1]
        minfo = {"symbols": len(stocks), "data_start": str(wide["close"].index[0].date()), "data_end": str(panel_last.date()),
                 "costs": costs.describe(), "roundtrip_base_pct": round(costs.base * 100, 4),
                 "survivorship": H.SURVIVORSHIP[market]}
        td = time.time()
        det = H.run_detection(stocks, tools, args.timeframes, mod_path, mod_name, args.workers)
        info["timing"][f"{market}_detect"] = round(time.time() - td, 1)
        for tf in args.timeframes:
            tb = time.time()
            bt = H.build_bars(market, tf, stocks, univ, panel_last, costs)
            ev, blocked, stat = H.attach_events(det[det["timeframe"] == tf], bt, periods)
            ev["market"] = market
            ev = ev.drop(columns=["sym"]).rename(columns={"sid": "sid"})
            ev["symbol"] = np.array(bt.syms)[ev["sid"].values]
            sign = ev["direction"].values
            for kind, matched in (("m", True), ("s", False)):
                idx = H.sample_placebos(ev, bt, blocked, seed=0, matched=matched)
                ev[f"n_plc_{kind}"] = (idx >= 0).sum(1)
                for h in H.HORIZONS:
                    pm = H.placebo_means(idx, bt, f"ret{h}")
                    pc = H.placebo_means(idx, bt, f"cost{h}")
                    ev[f"plc{h}_{kind}"] = pm
                    ev[f"plc{h}_{kind}_net"] = sign * pm - pc       # 위약도 같은 방향 부호·같은 비용
                    ev[f"d{h}_{kind}"] = sign * (ev[f"ret{h}"].values - pm)
            for h in H.HORIZONS:
                ev[f"r{h}"] = sign * ev[f"ret{h}"].values
                ev[f"r{h}_net"] = ev[f"r{h}"] - ev[f"cost{h}"]
            # noise floor: 위약 추출 시드만 바꿔 다시 (seed 0 포함 args.seeds개)
            for seed in range(1, args.seeds):
                for kind, matched in (("m", True), ("s", False)):
                    idx = H.sample_placebos(ev, bt, blocked, seed=seed, matched=matched)
                    d = sign * (ev[f"ret{H.MAIN_H}"].values - H.placebo_means(idx, bt, f"ret{H.MAIN_H}"))
                    g = pd.DataFrame({k: ev[k] for k in H.UNIT + ["period"]}).assign(d=d)
                    g = g[np.isfinite(ev[f"d{H.MAIN_H}_m"].values)]   # 본 결과와 같은 이벤트 집합
                    nr = g.groupby(H.UNIT + ["period"])["d"].mean().rename(f"exc_{kind}").reset_index()
                    nr["seed"], nr["kind"] = seed, kind
                    noise_rows.append(nr)
            info["timing"][f"{market}_{tf}_bars_placebo"] = round(time.time() - tb, 1)
            if args.causality_cuts:
                tc = time.time()
                hi = H.BOUNDARY if args.period == "explore" else panel_last
                cuts = week_end_cuts(wide["close"].index, args.causality_cuts, "2012-01-01", hi)
                info["causality"][f"{market}_{tf}"] = H.check_universe_causality(market, wide, tf, cuts, costs)
                info["timing"][f"{market}_{tf}_causality"] = round(time.time() - tc, 1)
            minfo[f"{tf}_events"] = stat
            all_ev.append(ev)
        info["markets"][market] = minfo
    ev = pd.concat(all_ev, ignore_index=True) if all_ev else pd.DataFrame()
    if ev.empty:
        raise SystemExit("이벤트가 없다")
    summ = H.unit_stats(ev)
    # noise floor 합치기 (seed 0 = 본 결과)
    base = summ[H.UNIT + ["period", f"exc{H.MAIN_H}_m", f"exc{H.MAIN_H}_s"]].rename(
        columns={f"exc{H.MAIN_H}_m": "exc_m", f"exc{H.MAIN_H}_s": "exc_s"})
    nz = [base.assign(seed=0)]
    if noise_rows:
        nd = pd.concat(noise_rows)
        nz.append(nd[nd["kind"] == "m"].drop(columns=["kind", "exc_s"], errors="ignore").merge(
            nd[nd["kind"] == "s"][H.UNIT + ["period", "seed", "exc_s"]], on=H.UNIT + ["period", "seed"], how="outer"))
    nz = pd.concat(nz, ignore_index=True)
    nf = nz.groupby(H.UNIT + ["period"]).agg(noise_m=("exc_m", "std"), noise_s=("exc_s", "std"),
                                              noise_seeds=("seed", "nunique")).reset_index()
    summ = summ.merge(nf, on=H.UNIT + ["period"], how="left")
    verdict = H.judge(summ)
    summ = summ.merge(verdict[H.UNIT + ["p_holm", "p_holm_blk", "verdict_blk", "c1_holm", "c2_cost", "c4_n", "c3_confirm", "verdict"]].assign(period=0),
                      on=H.UNIT + ["period"], how="left")
    info["timing"]["total"] = round(time.time() - t0, 1)
    keep_cols = [c for c in ev.columns if not c.startswith("_")]
    ev[keep_cols].to_parquet(out / "events.parquet", index=False)
    summ.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(args, info, summ, verdict), encoding="utf-8")
    print(f"출력: {out}  ({info['timing']['total']}초)")
    return out


def _pct(x, d=2):
    return "" if pd.isna(x) else f"{x * 100:+.{d}f}"


def _p(x):
    return "" if pd.isna(x) else (f"{x:.2g}" if x < 0.001 else f"{x:.3f}")


def report(args, info, summ, verdict) -> str:
    L = [f"# L003 A 유효성 — {args.period}{' (헛발견 검사: 섞은 가짜 시장)' if args.null else ''}", ""]
    L += [f"- 실행: `{' '.join(sys.argv)}`", f"- 탐지기: `{args.detector}` 도구 {', '.join(info['tools'])}",
          f"- 구간: 탐색 = 신호일 < {H.BOUNDARY.date()} 이고 20봉 청산일도 < 경계 (5·60봉도 청산일이 경계를 넘으면 뺌), "
          f"확인 = 신호일 ≥ {H.BOUNDARY.date()}. 이번 실행: **{args.period}**"
          + (" — 탐색 실행은 경계 이후 데이터를 읽지 않는다" if args.period == "explore" else ""),
          "- 잠금 구간: `load()` 기본(lockbox=False)으로 제외"]
    for m, mi in info["markets"].items():
        L.append(f"- **{m}**: 종목 {mi['symbols']}, 데이터 {mi['data_start']} ~ {mi['data_end']}, "
                 f"비용 포함 = 예 ({mi['costs']}; 기본 왕복 {mi['roundtrip_base_pct']:.3f}%), 생존편향 {mi['survivorship']}")
        for tf in args.timeframes:
            if f"{tf}_events" in mi:
                L.append(f"  - {tf} 신호 집계: {mi[f'{tf}_events']}")
    L.append("- 미래참조 점검:")
    for k, c in info["causality"].items():
        L.append(f"  - 유니버스·60봉 수익률·5분위 잘라내기 ({k}): {'통과' if c['ok'] else '**실패**'} — 비교 {c['compared']}행, "
                 f"불일치 {c['mismatch']}, 잘린 데이터에서 빠진 유니버스 행 {c['missing_in_cut']}, 지점 {[x['cut'] for x in c['cuts']]}")
    if not info["causality"]:
        L.append("  - 유니버스·분위 잘라내기: 이번 실행에서 생략 (--causality-cuts 0)")
    L.append("  - 탐지기 자체의 미래참조는 A0 잘라내기 검사(탐지기 담당)가 본다. 진입은 신호 봉 다음 일봉 시가.")
    L += ["", "## 정의", "",
          "- 수익률: 진입 = 신호 봉 다음 일봉 시가, 청산 = 진입 봉을 1봉째로 h봉째 종가 (주봉은 그 주 마지막 거래일 종가). 방향 부호를 곱함.",
          "- 초과수익 = 평균[방향 × (이벤트 수익 − 그 이벤트 위약 10개 평균)]. 위약도 같은 비용을 물리면 비용이 상쇄되므로 비용 전 값.",
          "  **비용 후 초과** = 초과수익 − 이벤트 평균 왕복 비용 (통과 기준 2 '초과수익 > 왕복 비용' ⇔ 비용 후 초과 > 0).",
          "- 추세 맞춘 위약 = 같은 종목·같은 연도·같은 구간·유니버스 안, 직전 60봉 수익률 횡단면 5분위가 같은 날 (주 지표). 단순 위약 = 분위 무시.",
          f"  같은 종목·같은 도구 신호 ±{H.BLOCK_BARS}봉은 위약 후보에서 뺀다.",
          f"- p = 날짜 묶음(일봉 = 같은 날, 주봉 = 같은 주) 부트스트랩 {H.N_BOOT}회 표준오차로 만든 양측 정규 p. "
          "Holm = 이번 실행의 모든 탐색 구간 검정 단위, 20봉, 추세 맞춘 위약.",
          f"- noise floor = 위약 추출 시드만 바꾼 {args.seeds}회의 20봉 초과수익 표준편차.",
          f"- 통과: Holm p < {H.ALPHA} ∧ 비용 후 초과 > 0 ∧ 표본 ≥ {H.MIN_N} ∧ (확인 구간이 있으면) 같은 부호·크기 ≥ 탐색의 {H.CONFIRM_RATIO:.0%}.",
          "", "## 검정 단위별 결과 (탐색 구간, 수익률 단위 %)", "",
          "- 참고: '블록 Holm p'는 묶음을 연속 20봉 블록으로 넓힌 보수적 p (가짜 탐지기 검사에서 주봉 날짜 묶음 p가 약간 낙관적이었음 — "
          "results/A/harness-check.md). '판정'은 사양대로, '블록 판정'은 참고.",
          "",
          "| 도구 | 변형 | 방향 | 시장 | 봉 | 표본 | 상폐청산 | 20봉 수익(비용 후) | 20봉 초과(추세 위약) | 비용 후 초과 | 단순 위약 대비 | 5봉 초과 | 60봉 초과 | noise floor | 원 p | Holm p | 블록 Holm p | 확인 초과 | 판정 | 블록 판정 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in verdict.sort_values(["market", "timeframe", "tool", "variant", "direction"]).iterrows():
        s = summ[(summ[H.UNIT].values == r[H.UNIT].values).all(1) & (summ["period"] == 0)].iloc[0]
        L.append(f"| {r['tool']} | {r['variant']} | {int(r['direction']):+d} | {r['market']} | {r['timeframe']} | {int(r['n'])} | "
                 f"{int(r['n_delisted_exit20'])} | {_pct(r['ret20_net'])} | {_pct(r['exc20_m'])} | {_pct(r['exc20_m_after_cost'])} | "
                 f"{_pct(r['exc20_s'])} | {_pct(r['exc5_m'])} | {_pct(r['exc60_m'])} | {_pct(s['noise_m'], 3)} | {_p(r['p_m'])} | "
                 f"{_p(r['p_holm'])} | {_p(r['p_holm_blk'])} | {_pct(r['confirm_exc20'])} | {r['verdict']} | {r['verdict_blk']} |")
    n_pass = int(verdict["verdict"].isin(["통과", "탐색 통과(확인 대기)"]).sum())
    L += ["", f"- 검정 단위 {len(verdict)}개, 탐색 기준(Holm·비용·표본) 통과 {n_pass}개, "
          f"Holm 전 p<0.05 비율 {np.mean(verdict['p_m'] < 0.05):.1%}", ""]
    L.append("실행 시간(초): " + ", ".join(f"{k} {v}" for k, v in info["timing"].items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    run(parse())
