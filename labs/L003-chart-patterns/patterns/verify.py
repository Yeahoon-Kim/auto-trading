"""A0 탐지기 verifier.

실행 (PowerShell, 저장소 루트에서):
    $env:AUTOTRADER_DATA_DIR = "C:\\Users\\apple\\Claude\\auto-trading\\data"
    uv run python labs/L003-chart-patterns/patterns/verify.py            # 전체
    uv run python labs/L003-chart-patterns/patterns/verify.py --quick    # 표본 줄임 (개발용)

1. 합성 재현율: 무작위 보행에 패턴(변형별)을 심고 완성 봉 ±3봉 안에서 같은 변형을 찾는 비율. 기준 ≥ 90%.
2. 기저 발생률: 순수 무작위 보행(일봉 변동성 2%)에서 1000봉당 이벤트 수. 보고만.
3. 잘라내기 검사: detect(bars[:t]) 의 date ≤ t 이벤트 == detect(bars) 의 date ≤ t 이벤트 (값까지). 기준 불일치 0건.
4. 육안 표본: 실제 데이터(2018-03-09까지, 탐색 구간)에서 도구·변형별 차트 PNG. 신호 봉까지만 그린다.
결과 → labs/L003-chart-patterns/results/A0/verify.md
실제 데이터는 2018-03-09 이전만 읽고, 패턴 뒤 수익률은 계산하지 않는다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from patterns import TOOLS, _detect, detect, pivots, params, to_weekly  # noqa: E402
from patterns import synth  # noqa: E402

OUT = LAB / "results" / "A0"
REAL_END = "2018-03-09"     # 탐색 구간 경계(2018-03-12) 직전 금요일
REAL_START = "2010-01-01"
KR_SYMS = ["005930", "000660", "005380", "035420", "051910", "006400", "012330", "105560",
           "055550", "017670", "015760", "034220", "009150", "028260", "066570", "032830",
           "003550", "096770", "010950", "000270", "011170", "010130", "030200", "033780",
           "086790", "004020", "000810", "090430", "024110", "021240"]
US_SYMS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "INTC", "CSCO", "WMT", "PFE",
           "CVX", "HD", "MRK", "DIS", "BA", "CAT", "MMM", "IBM", "MCD"]

# 합성 재현율 대상: (도구, 변형, 생성 종류, 뒤집기)
PLANTS = [
    ("head_shoulders", "ihs", "ihs", False), ("head_shoulders", "hs", "ihs", True),
    ("double_triple", "double_bottom", "double_bottom", False),
    ("double_triple", "double_top", "double_bottom", True),
    ("double_triple", "triple_bottom", "triple_bottom", False),
    ("double_triple", "triple_top", "triple_bottom", True),
    ("fib_retrace", "bull_retrace", "bull_retrace", False),
    ("fib_retrace", "bear_retrace", "bull_retrace", True),
    ("trendline", "up_break", "up_break", False), ("trendline", "down_break", "up_break", True),
]
MA_PLANTS = [("ma200_cross", "up_break", "down_break", 1), ("ma50_200_cross", "golden", "dead", 50)]


# ------------------------------------------------------------------ 1. 재현율
def recall(n_per: int, noise: float, tf: str, seed0: int = 0):
    rows = []
    for tool, variant, kind, mir in PLANTS:
        hit = tot = skipped = 0
        for s in range(n_per):
            r = synth.plant(kind, tf, noise=noise, seed=seed0 + 7919 * s + (1 if mir else 0), for_mirror=mir)
            if r is None:
                skipped += 1
                continue
            df, comp, _ = r
            if mir:
                df = synth.mirror(df)
            ev = detect(df, tool, tf)
            pos = np.searchsorted(df.index.to_numpy(), ev.loc[ev.variant == variant, "date"].to_numpy())
            tot += 1
            hit += bool(np.any(np.abs(pos - comp) <= 3))
        rows.append(dict(timeframe=tf, noise=noise, tool=tool, variant=variant, planted=tot,
                         found=hit, recall=hit / max(tot, 1), skipped=skipped))
    # 이동평균: 하락→상승 추세 시계열의 모든 교차를 독립 구현(순수 파이썬)으로 구해 비교
    pr = params(tf)
    for tool, up_name, dn_name, fast in MA_PLANTS:
        fast_n = 1 if fast == 1 else pr["ma_short"]
        for mir, variant, d in ((False, up_name, 1), (True, dn_name, -1)):
            hit = tot = 0
            for s in range(max(n_per // 4, 20)):
                df = synth.plant_ma("up", tf, seed=seed0 + 104729 * s)
                if mir:
                    df = synth.mirror(df)
                exp_ = [t for t, dd in synth.naive_ma_events(df["close"].tolist(), fast_n, pr["ma_long"])
                        if dd == d]
                ev = detect(df, tool, tf)
                pos = np.searchsorted(df.index.to_numpy(),
                                      ev.loc[ev.variant == variant, "date"].to_numpy())
                for t in exp_:
                    tot += 1
                    hit += bool(np.any(np.abs(pos - t) <= 3))
            rows.append(dict(timeframe=tf, noise=0.02 if tf == "1d" else 0.045, tool=tool, variant=variant, planted=tot,
                             found=hit, recall=hit / max(tot, 1), skipped=0))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. 기저 발생률
def base_rate(n_series: int, n_bars: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    cnt: dict = {}
    bars_tot = {"1d": 0, "1w": 0}
    for _ in range(n_series):
        d = synth.random_walk(n_bars, 0.02, rng=rng)
        for tf, bars in (("1d", d), ("1w", to_weekly(d))):
            bars_tot[tf] += len(bars)
            for tool in TOOLS:
                ev = detect(bars, tool, tf)
                for v, k in ev["variant"].value_counts().items():
                    cnt[(tf, tool, v)] = cnt.get((tf, tool, v), 0) + int(k)
    rows = []
    for (tf, tool, v), k in sorted(cnt.items()):
        rows.append(dict(timeframe=tf, tool=tool, variant=v, events=k,
                         per_1000_bars=1000 * k / bars_tot[tf]))
    return pd.DataFrame(rows), bars_tot


# ------------------------------------------------------------------ 실제 데이터
def load_real():
    from autotrader.backtest.data import load
    out = {}
    for market, syms in (("kr_stock", KR_SYMS), ("us_stock", US_SYMS)):
        try:
            p = load(market, symbols=syms, start=REAL_START, end=REAL_END, fields=("high", "low"))
        except Exception as e:  # 일부 종목 없음 등
            print(f"[경고] {market} 로드 실패: {e}")
            continue
        for s in p.close.columns:
            df = pd.DataFrame({k: getattr(p, k)[s] for k in ("open", "high", "low", "close")}).dropna()
            assert df.index.max() <= pd.Timestamp(REAL_END), "탐색 구간 밖 데이터"
            if len(df) > 300:
                out[(market, s)] = df
    return out


# ------------------------------------------------------------------ 3. 잘라내기 검사
def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a = a.reset_index(drop=True); b = b.reset_index(drop=True)
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=0, atol=0)
        return True
    except AssertionError:
        return False


def truncation(series: dict, n_rand: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    bad_examples = []
    for name, (bars, tf) in series.items():
        n = len(bars)
        for tool in TOOLS:
            full = detect(bars, tool, tf)
            pos = np.searchsorted(bars.index.to_numpy(), full["date"].to_numpy())
            ts = set(rng.integers(30, n - 1, n_rand).tolist())
            near = np.r_[pos, pos - 1, pos + 1]
            near = near[(near >= 30) & (near < n - 1)]
            if len(near):
                ts |= set(rng.choice(near, min(len(near), n_rand), replace=False).tolist())
            bad = 0
            for t in sorted(ts):
                cut = bars.index[t]
                tr = detect(bars.iloc[:t + 1], tool, tf)
                if not _same(tr[tr.date <= cut], full[full.date <= cut]):
                    bad += 1
                    if len(bad_examples) < 5:
                        bad_examples.append((name, tool, str(cut.date())))
            rows.append(dict(series=name, timeframe=tf, tool=tool, checks=len(ts), mismatches=bad,
                             events=len(full)))
    return pd.DataFrame(rows), bad_examples


# ------------------------------------------------------------------ 4. 차트
def charts(real: dict, per_variant_d: int = 3, per_variant_w: int = 1, seed: int = 3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = OUT / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    for f in cdir.glob("*.png"):
        f.unlink()
    rng = np.random.default_rng(seed)
    made = []
    for tf, per in (("1d", per_variant_d), ("1w", per_variant_w)):
        pool: dict = {}
        for (mk, sym), d in real.items():
            bars = d if tf == "1d" else to_weekly(d)
            for tool in TOOLS:
                ev = _detect(bars, tool, tf)
                for _, e in ev.iterrows():
                    pool.setdefault((tool, e["variant"]), []).append((mk, sym, bars, e))
        for (tool, variant), lst in sorted(pool.items(), key=lambda x: x[0]):
            pick = rng.choice(len(lst), min(per, len(lst)), replace=False)
            for k, i in enumerate(sorted(pick)):
                mk, sym, bars, e = lst[i]
                fn = f"{tf}_{tool}_{variant}_{k + 1}_{sym}_{e['date'].date()}.png"
                _plot(plt, bars, e, tool, tf, f"{mk}:{sym}", cdir / fn)
                made.append(dict(timeframe=tf, tool=tool, variant=variant, file=fn,
                                 n_events_pool=len(lst)))
    return pd.DataFrame(made)


def _plot(plt, bars, e, tool, tf, label, path):
    ix = bars.index
    t = int(ix.get_loc(e["date"]))
    s0 = int(ix.get_loc(e["pattern_start"]))
    pr = params(tf)
    if tool in ("ma200_cross", "ma50_200_cross"):
        lo = max(0, t - 150)
    else:
        lo = max(0, s0 - 25)
    seg = bars.iloc[:t + 1]                # 신호 봉까지만 (그 뒤는 보지 않는다)
    x = np.arange(lo, t + 1)
    sub = seg.iloc[lo:]
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=90)
    up = (sub["close"] >= sub["open"]).to_numpy()
    ax.vlines(x, sub["low"], sub["high"], color="#555", lw=0.7)
    ax.vlines(x[up], sub["open"][up], sub["close"][up], color="#d62728", lw=2.6)
    ax.vlines(x[~up], sub["close"][~up], sub["open"][~up], color="#1f77b4", lw=2.6)
    if tool in ("ma200_cross", "ma50_200_cross"):
        c = seg["close"]
        ax.plot(x, c.rolling(pr["ma_long"]).mean().iloc[lo:], color="purple", lw=1.4,
                label=f"MA{pr['ma_long']}")
        if tool == "ma50_200_cross":
            ax.plot(x, c.rolling(pr["ma_short"]).mean().iloc[lo:], color="orange", lw=1.4,
                    label=f"MA{pr['ma_short']}")
    else:
        pv = pivots(seg)
        pv = pv[(pv["conf_date"] <= e["date"]) & (pv["date"] >= ix[lo])]
        pi = ix.get_indexer(pv["date"])
        ax.scatter(pi[pv.kind == 1], pv.price[pv.kind == 1], marker="v", color="k", s=25,
                   zorder=3, label="고점(확정)")
        ax.scatter(pi[pv.kind == -1], pv.price[pv.kind == -1], marker="^", color="k", s=25,
                   zorder=3, label="저점(확정)")
        info = e["info"]
        pts = info.get("points", [])
        if pts:
            ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], "o-", color="green", lw=1.2,
                    ms=6, alpha=0.8, label="패턴 꼭짓점")
        if "line" in info:
            i0, v0, i1, v1 = info["line"]
            xs = np.arange(i0, i1 + 1)
            if tool == "trendline":   # 로그 직선
                ys = np.exp(np.log(v0) + (np.log(v1) - np.log(v0)) * (xs - i0) / max(i1 - i0, 1))
            else:
                ys = v0 + (v1 - v0) * (xs - i0) / max(i1 - i0, 1)
            ax.plot(xs, ys, "--", color="magenta", lw=1.5,
                    label="추세선" if tool == "trendline" else "넥라인")
        if "hline" in info:
            i0, v, i1 = info["hline"]
            ax.hlines(v, i0, i1, colors="magenta", linestyles="--", lw=1.5, label="돌파 기준(넥)")
        if "zone" in info:
            i0, zl, zh, i1 = info["zone"]
            ax.fill_between([i0, i1], zl, zh, color="gold", alpha=0.3, label="38.2~61.8% 구간")
    ax.axvline(t, color="red", lw=0.8, alpha=0.6)
    ax.set_xlim(lo - 1, t + 3)
    ttl = (f"{label} [{tf}] {tool} / {e['variant']}  신호 {e['date'].date()} "
           f"(dir {e['direction']:+d}, stop {e['stop_level']:.4g})")
    ax.set_title(ttl, fontsize=10)
    step = max(1, (t - lo) // 8)
    ticks = list(range(lo, t + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(ix[i].date()) for i in ticks], rotation=30, fontsize=7)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ 속도
def speed(n_series: int = 20, n_bars: int = 4000):
    rng = np.random.default_rng(99)
    data = [synth.random_walk(n_bars, 0.02, rng=rng) for _ in range(n_series)]
    t0 = time.perf_counter()
    for d in data:
        for tool in TOOLS:
            detect(d, tool, "1d")
    el = time.perf_counter() - t0
    return el / n_series   # 4000봉 1종목 × 도구 6개 초


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args()
    q = a.quick
    import matplotlib
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    T0 = time.perf_counter()

    n_per = 40 if q else 200
    rec = pd.concat([recall(n_per, 0.004, "1d"), recall(n_per, 0.004, "1w")])
    rec_noisy = pd.concat([recall(n_per // 2, 0.010, "1d", seed0=555),
                           recall(n_per // 2, 0.010, "1w", seed0=555)])
    rec_noisy = rec_noisy[~rec_noisy.tool.isin(["ma200_cross", "ma50_200_cross"])]
    print(rec.to_string()); print(rec_noisy.to_string())

    br, bt = base_rate(5 if q else 40, 5000)
    print(br.to_string())

    real = {} if a.no_real else load_real()
    print("실제 데이터 종목:", len(real))

    series = {}
    rng = np.random.default_rng(1)
    for i, (tool, variant, kind, mir) in enumerate(PLANTS):
        r = synth.plant(kind, "1d", seed=900 + i, for_mirror=mir)
        if r is not None:
            series[f"synth_{variant}_1d"] = ((synth.mirror(r[0]) if mir else r[0]), "1d")
        r = synth.plant(kind, "1w", seed=950 + i, for_mirror=mir)
        if r is not None:
            series[f"synth_{variant}_1w"] = ((synth.mirror(r[0]) if mir else r[0]), "1w")
    for i in range(2 if q else 4):
        series[f"rw_{i}_1d"] = (synth.random_walk(1500, 0.02, rng=rng), "1d")
        series[f"rw_{i}_1w"] = (to_weekly(synth.random_walk(2500, 0.02, rng=rng)), "1w")
    real_keys = list(real)[:2] if q else [k for k in real if k[1] in
                                          ("005930", "000660", "035420", "005380", "AAPL", "MSFT", "JPM")]
    for k in real_keys:
        series[f"{k[0]}:{k[1]}_1d"] = (real[k], "1d")
        series[f"{k[0]}:{k[1]}_1w"] = (to_weekly(real[k]), "1w")
    tr, bad_ex = truncation(series, 8 if q else 25)
    print(tr.groupby("timeframe")[["checks", "mismatches"]].sum())

    ch = charts(real) if real else pd.DataFrame()
    sp = speed(5 if q else 20)
    total = time.perf_counter() - T0
    write_report(rec, rec_noisy, br, bt, tr, bad_ex, ch, sp, total, q, len(real))


def write_report(rec, rec_noisy, br, bt, tr, bad_ex, ch, sp, total, q, n_real):
    L = []
    w = L.append
    w("# A0 탐지기 verifier 결과")
    w("")
    w(f"- 실행: `uv run python labs/L003-chart-patterns/patterns/verify.py{' --quick' if q else ''}`"
      f" ({pd.Timestamp.now():%Y-%m-%d %H:%M}), 총 {total:.0f}초")
    w("- 정의·세부 수치: [patterns/DEFINITIONS.md](../../patterns/DEFINITIONS.md)")
    w(f"- 실제 데이터: KR·US {n_real}종목, {REAL_START} ~ {REAL_END} (탐색 구간만). 패턴 뒤 수익률은 계산하지 않았다.")
    w("")
    ok1 = bool((rec.recall >= 0.9).all())
    w(f"## 1. 합성 재현율 — 기준 ≥ 90% → **{'통과' if ok1 else '실패'}** (최저 {rec.recall.min():.1%})")
    w("")
    w("심은 패턴의 완성 봉(정의상 신호가 나야 할 봉) ±3봉 안에서 같은 도구·같은 변형 이벤트를 찾은 비율.")
    w("배경 = 무작위 보행(일봉 2%, 주봉 4.5%), 패턴 구간 잡음 = 봉당 0.4%(주봉 0.9%). "
      "패턴 높이 12~30%(주봉 20~50%), 다리 길이 8~22봉(주봉 5~12봉), 같은 높이 꼭짓점 어긋남 ±0.4%.")
    w("이동평균은 하락→상승 추세 시계열(변동성 2%)의 모든 교차를 순수 파이썬 독립 구현과 대조.")
    w("")
    w("| 시간 축 | 도구 | 변형 | 심은 수 | 찾은 수 | 재현율 | 판정 |")
    w("|---|---|---|---|---|---|---|")
    for _, r in rec.iterrows():
        w(f"| {r.timeframe} | {r.tool} | {r.variant} | {r.planted} | {r.found} | {r.recall:.1%} | "
          f"{'통과' if r.recall >= 0.9 else '실패'} |")
    w("")
    w("참고 (판정 아님): 패턴 구간 잡음을 봉당 1.0%(주봉 2.2%)로 키웠을 때")
    w("")
    w("| 시간 축 | 도구 | 변형 | 심은 수 | 재현율 |")
    w("|---|---|---|---|---|")
    for _, r in rec_noisy.iterrows():
        w(f"| {r.timeframe} | {r.tool} | {r.variant} | {r.planted} | {r.recall:.1%} |")
    w("")
    w(f"## 2. 기저 발생률 (보고만) — 순수 무작위 보행, 일봉 변동성 2%")
    w("")
    w(f"일봉 {bt['1d']:,}봉, 주봉(같은 경로를 W-FRI로 묶음) {bt['1w']:,}봉. 1000봉당 이벤트 수.")
    w("무작위 보행에도 '패턴'은 이만큼 나온다 → A 단계에서 위약 비교가 필요한 이유.")
    w("")
    w("| 시간 축 | 도구 | 변형 | 이벤트 | 1000봉당 |")
    w("|---|---|---|---|---|")
    for _, r in br.iterrows():
        w(f"| {r.timeframe} | {r.tool} | {r.variant} | {r.events} | {r.per_1000_bars:.2f} |")
    w("")
    nb = int(tr.mismatches.sum())
    w(f"## 3. 잘라내기 검사 — 기준 불일치 0건 → **{'통과' if nb == 0 else '실패'}** "
      f"(검사 {int(tr.checks.sum()):,}회, 불일치 {nb}건)")
    w("")
    w("각 시계열·도구마다 무작위 t와 이벤트 날짜 근처(t, t±1)에서 `detect(bars[:t])`의 date ≤ t 이벤트가 "
      "`detect(bars)`의 date ≤ t 이벤트와 모든 열·값까지 같은지 확인.")
    w("")
    g = tr.assign(kind=np.where(tr.series.str.startswith("synth"), "합성(패턴 심음)",
                                np.where(tr.series.str.startswith("rw"), "무작위 보행", "실제")))
    s = g.groupby(["kind", "timeframe", "tool"])[["checks", "mismatches", "events"]].sum().reset_index()
    w("| 데이터 | 시간 축 | 도구 | 검사 수 | 불일치 | 전체 이벤트 수 |")
    w("|---|---|---|---|---|---|")
    for _, r in s.iterrows():
        w(f"| {r.kind} | {r.timeframe} | {r.tool} | {r.checks} | {r.mismatches} | {r.events} |")
    if bad_ex:
        w("")
        w("불일치 예: " + ", ".join(f"{a}/{b}/{c}" for a, b, c in bad_ex))
    w("")
    w("실제 종목: " + ", ".join(sorted({x.rsplit('_', 1)[0] for x in tr.series
                                    if not x.startswith(('synth', 'rw'))})))
    w("")
    w("## 4. 육안 표본 (사용자 확인 대기)")
    w("")
    w("`charts/` — 일봉은 도구·변형별 3개, 주봉은 1개. 패턴 시작 25봉 전 ~ 신호 봉까지만 그렸다 (이후 가격은 안 그림).")
    w("빨간 세로선 = 신호 봉, ▼▲ = 그 시점까지 확정된 고점·저점, 초록 = 패턴 꼭짓점, 자홍 점선 = 넥라인·추세선·돌파 기준, 노랑 = 피보나치 구간.")
    w("")
    if len(ch):
        w("| 시간 축 | 도구 | 변형 | 파일 | 표본 풀(이벤트 수) |")
        w("|---|---|---|---|---|")
        for _, r in ch.iterrows():
            w(f"| {r.timeframe} | {r.tool} | {r.variant} | [{r.file}](charts/{r.file}) | {r.n_events_pool} |")
    w("")
    w("## 속도")
    w("")
    w(f"무작위 보행 4000봉 1종목에 도구 6개 = {sp:.2f}초 → KR 1500종목 × 4000일 일봉 추정 {sp * 1500 / 60:.1f}분 (단일 프로세스).")
    (OUT / "verify.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("->", OUT / "verify.md")


if __name__ == "__main__":
    main()
