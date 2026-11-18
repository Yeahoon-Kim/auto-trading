"""한국·미국 시장 국면(상승장/횡보장/하락장) 구분.

두 가지 라벨을 만든다.
1. 사후 라벨(정답지): 종가 zigzag로 큰 전환점을 찾고, 전환점 사이 구간을 기울기로 상승/횡보/하락으로 나눈다.
   전환점은 그 뒤 가격을 봐야 확정되므로 "과거를 설명"하는 용도로만 쓴다 (미래참조).
2. 실시간 라벨: 그날까지의 종가만으로 계산 (200일선, 200일선 기울기, 고점 대비 낙폭, 6개월 수익률).
   현재 국면 판단은 이 라벨로 한다. 사후 라벨과의 일치율·전환 지연이 verifier.

실행: AUTOTRADER_DATA_DIR=<메인>/data uv run python notebooks/market_cycle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.data import store

OUT = Path(__file__).resolve().parents[1] / "docs" / "research" / "market-cycle"

# 전환 임계: 반대 방향으로 이만큼 움직여야 전환점 확정. 한국은 변동성이 커서 더 크게.
ZZ = {"KOSPI": 0.15, "KOSDAQ": 0.20, "SPX": 0.10, "NASDAQ": 0.15}
SIDEWAYS_ANN = 0.10   # 구간 연율 수익이 ±10% 안이면 횡보
START = "2015-01-01"


def close(sym: str) -> pd.Series:
    d = store.read("macro_1d", sym)
    s = d.set_index(d["ts"].dt.tz_convert(None).dt.normalize())["close"].astype(float)
    return s[~s.index.duplicated(keep="last")].dropna().rename(sym)


def zigzag(px: pd.Series, th: float) -> list[tuple[pd.Timestamp, float, str]]:
    """전환점 목록 [(날짜, 가격, 'peak'|'trough')]. 마지막 원소는 미확정 극점."""
    piv = []
    trend = 0  # 1 상승 추적, -1 하락 추적
    hi_t, hi = px.index[0], px.iloc[0]
    lo_t, lo = hi_t, hi
    for t, p in px.items():
        if trend >= 0:
            if p > hi:
                hi_t, hi = t, p
            if p <= hi * (1 - th):
                if trend == 1 or not piv:
                    piv.append((hi_t, hi, "peak"))
                trend, lo_t, lo = -1, t, p
                continue
        if trend <= 0:
            if p < lo:
                lo_t, lo = t, p
            if p >= lo * (1 + th):
                if trend == -1 or not piv:
                    piv.append((lo_t, lo, "trough"))
                trend, hi_t, hi = 1, t, p
    # 미확정 마지막 극점
    piv.append((hi_t, hi, "peak*") if trend == 1 else (lo_t, lo, "trough*"))
    return piv


def segments(px: pd.Series, piv) -> pd.DataFrame:
    rows = []
    pts = [(px.index[0], px.iloc[0], "start")] + piv
    for (t0, p0, k0), (t1, p1, k1) in zip(pts[:-1], pts[1:]):
        days = (t1 - t0).days or 1
        ret = p1 / p0 - 1
        ann = (1 + ret) ** (365 / days) - 1
        seg = px[t0:t1]
        rows.append(dict(start=t0.date(), end=t1.date(), days=days, start_px=p0, end_px=p1, ret=ret, ann=ann,
                         max_dd=(seg / seg.cummax() - 1).min(), confirmed=not k1.endswith("*"),
                         dir="상승" if ret > 0 else "하락"))
    return pd.DataFrame(rows)


def label_segments(seg: pd.DataFrame) -> pd.DataFrame:
    """긴 상승·하락 구간 안의 횡보: 구간 연율 수익이 작으면 횡보."""
    seg = seg.copy()
    seg["regime"] = np.where(seg["ann"].abs() < SIDEWAYS_ANN, "횡보", seg["dir"].map({"상승": "상승장", "하락": "하락장"}))
    return seg


def expost_daily(px: pd.Series, piv, win=126) -> pd.Series:
    """사후 일별 라벨: zigzag 구간 방향 + 구간 안 126일 창이 ±10% 연율 안이면 횡보로 덮어씀."""
    lab = pd.Series(index=px.index, dtype=object)
    pts = [(px.index[0], px.iloc[0], "start")] + piv
    for (t0, p0, _), (t1, p1, _) in zip(pts[:-1], pts[1:]):
        lab[t0:t1] = "상승장" if p1 > p0 else "하락장"
    # 중심 창 기울기(사후): 앞뒤 63일씩 본 연율 수익
    fwd = px.shift(-win // 2) / px.shift(win // 2) - 1
    ann = (1 + fwd) ** (252 / win) - 1
    lab[(ann.abs() < SIDEWAYS_ANN)] = "횡보"
    # 20거래일 미만 짧은 조각은 앞 구간에 흡수 (사후 라벨의 깜빡임 제거)
    runs = (lab != lab.shift()).cumsum()
    size = lab.groupby(runs).transform("size")
    lab[size < 20] = None
    return lab.ffill().bfill()


def realtime(px: pd.Series) -> pd.DataFrame:
    """그날까지 정보만 사용."""
    ma200 = px.rolling(200).mean()
    slope = ma200 / ma200.shift(20) - 1          # 200일선 20일 기울기
    r126 = px / px.shift(126) - 1
    dd = px / px.cummax().rolling(1).max() - 1    # 전고점(전 기간) 대비
    dd252 = px / px.rolling(252).max() - 1        # 1년 고점 대비
    above = px > ma200
    bull = above & (slope > 0) & (r126 > 0.05)
    bear = (~above & (slope < 0)) | (dd252 < -0.20)
    lab = pd.Series(np.select([bear, bull], ["하락장", "상승장"], "횡보"), index=px.index)
    lab[ma200.isna()] = None
    # 3일 연속 같은 판정이어야 바뀜 (잦은 깜빡임 방지)
    stable = lab.copy()
    cur = None
    for i in range(len(lab)):
        v = lab.iloc[i]
        if v is not None and i >= 2 and lab.iloc[i - 1] == v and lab.iloc[i - 2] == v:
            cur = v
        stable.iloc[i] = cur
    return pd.DataFrame(dict(close=px, ma200=ma200, ma200_slope20=slope, ret_126d=r126, dd_1y=dd252,
                             dd_ath=dd, raw=lab, regime=stable))


def regime_runs(lab: pd.Series) -> pd.DataFrame:
    lab = lab.dropna()
    grp = (lab != lab.shift()).cumsum()
    out = lab.groupby(grp).agg(["first", "size"])
    idx = lab.groupby(grp).apply(lambda s: (s.index[0].date(), s.index[-1].date()))
    out["start"] = [a for a, _ in idx]
    out["end"] = [b for _, b in idx]
    return out.rename(columns={"first": "regime", "size": "days"})[["start", "end", "regime", "days"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for sym, th in ZZ.items():
        px = close(sym)
        px = px[px.index >= "2010-01-01"]
        piv = zigzag(px[px.index >= START], th)
        seg = label_segments(segments(px[px.index >= START], piv))
        seg.to_csv(OUT / f"{sym}_zigzag.csv", index=False, encoding="utf-8-sig")
        rt = realtime(px)
        ex = expost_daily(px[px.index >= START], piv)
        rt["expost"] = ex
        rt = rt[rt.index >= START]
        rt.to_csv(OUT / f"{sym}_daily.csv", encoding="utf-8-sig")
        runs = regime_runs(rt["regime"])
        runs.to_csv(OUT / f"{sym}_realtime_runs.csv", index=False, encoding="utf-8-sig")

        both = rt.dropna(subset=["regime", "expost"])
        both = both[both.index <= both.index[-1] - pd.Timedelta(days=95)]  # 사후 라벨이 확정된 구간만
        agree = (both["regime"] == both["expost"]).mean()
        conf = pd.crosstab(both["expost"], both["regime"], normalize="index").round(2)
        # 전환 지연: 확정된 zigzag 전환점 → 실시간 라벨이 같은 방향(상승장/하락장)으로 바뀐 날까지 거래일
        lags = []
        for t, p, k in piv[:-1]:
            target = "하락장" if k == "peak" else "상승장"
            after = rt.loc[t:, "regime"]
            hit = after[after == target]
            if len(hit):
                lags.append((t.date(), k, target, after.index.get_loc(hit.index[0])))
        report.append(f"\n## {sym} (zigzag {th:.0%})\n")
        report.append(seg.assign(ret=seg.ret.map("{:+.1%}".format), ann=seg.ann.map("{:+.1%}".format),
                                 max_dd=seg.max_dd.map("{:.1%}".format)).to_string(index=False))
        report.append(f"\n실시간 vs 사후 일치율: {agree:.1%}\n{conf}\n")
        report.append("전환 지연(거래일): " + "; ".join(f"{d} {k}->{tg} {n}일" for d, k, tg, n in lags))
        report.append("\n실시간 국면 연속 구간(최근 12개):\n" + runs.tail(12).to_string(index=False))
        last = rt.iloc[-1]
        report.append(
            f"\n최근 {rt.index[-1].date()}: 종가 {last.close:,.2f}, 200일선 {last.ma200:,.2f} ({last.close / last.ma200 - 1:+.1%}),"
            f" 200일선 20일 기울기 {last.ma200_slope20:+.2%}, 6개월 {last.ret_126d:+.1%}, 1년고점대비 {last.dd_1y:+.1%},"
            f" 실시간 판정 {last.regime} (당일 원판정 {last.raw})")
        r = rt.tail(60)
        report.append("최근 60일 추이(5일 간격): " + ", ".join(f"{i:%m-%d} {c:,.0f}({g})" for i, c, g in
                                                     zip(r.index[::5], r.close[::5], r.raw[::5])))
        if lags:
            report.append(f"전환 지연 중앙값: {np.median([n for *_, n in lags]):.0f}거래일")
        plot(sym, px[px.index >= START], seg, rt)
    (OUT / "summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


# 주요 사건 (KR/US). 2025-06 이후 일부는 저장소 news 데이터(Benzinga 헤드라인)로만 확인.
EVENTS = {
    "KR": [("2018-01-29", "반도체 정점·미중 무역분쟁"), ("2020-03-19", "코로나 바닥→동학개미"),
           ("2021-01-07", "KOSPI 3000"), ("2021-07-06", "KOSPI 고점(유동성 정점)"),
           ("2022-09-30", "금리 인상·강달러 바닥"), ("2023-07-25", "이차전지(에코프로) 정점"),
           ("2024-08-05", "엔캐리 청산"), ("2024-12-03", "비상계엄"), ("2025-04-09", "관세 유예 바닥"),
           ("2025-06-03", "대선·상법개정 기대"), ("2026-03-04", "이란 전쟁 -12%"),
           ("2026-06-22", "KOSPI 9,115 고점"), ("2026-07-28", "中 메모리 공포"),
           ("2026-10-15", "뜨거운 CPI·연준 인상"), ("2026-11-12", "젠슨 황 사임")],
    "US": [("2018-12-24", "연준 긴축 바닥"), ("2020-03-23", "코로나 바닥·무제한 QE"),
           ("2020-11-09", "백신·대선"), ("2022-01-03", "고점→금리 인상"), ("2022-10-12", "인플레 정점 바닥"),
           ("2023-05-24", "엔비디아 실적(AI 1차)"), ("2023-10-27", "10년물 5% 바닥→AI 2차"),
           ("2024-08-05", "엔캐리 청산"), ("2024-11-05", "트럼프 당선"), ("2025-04-08", "관세 충격 바닥"),
           ("2026-03-02", "이란 전쟁·유가↑"), ("2026-09-16", "연준 인상 재개"),
           ("2026-10-06", "S&P 고점"), ("2026-11-12", "젠슨 황 사임")],
}
COLOR = {"상승장": "#2e9d5b", "하락장": "#d64545", "횡보": "#b8b8b8"}


def plot(sym, px, seg, rt) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True, gridspec_kw={"height_ratios": [6, 1]})
    ex = rt["expost"].dropna()
    for _, s in ex.groupby((ex != ex.shift()).cumsum()):
        ax.axvspan(s.index[0], s.index[-1] + pd.Timedelta(days=1), color=COLOR[s.iloc[0]], alpha=0.22, lw=0)
    ax.plot(px.index, px.values, color="#222", lw=1)
    ax.plot(rt.index, rt.ma200, color="#3366cc", lw=0.8, alpha=0.7, label="200일선")
    ax.set_yscale("log")
    ev = EVENTS["KR" if sym.startswith("KOS") else "US"]
    ymax = px.max()
    for i, (d, lab) in enumerate(ev):
        t = pd.Timestamp(d)
        if t < px.index[0]:
            continue
        ax.axvline(t, color="#555", lw=0.5, ls=":")
        ax.text(t, ymax * (0.97 ** (i % 6 * 3 + 1)), f" {lab}", fontsize=7.5, rotation=0, va="top", color="#333")
    ax.set_title(f"{sym} 국면 — 배경: 사후 zigzag 라벨 (초록 상승장 · 빨강 하락장 · 회색 횡보), 아래 띠: 실시간 판정")
    ax.legend(loc="lower right", fontsize=8)
    lab = rt["regime"].dropna()
    for g, s in lab.groupby((lab != lab.shift()).cumsum()):
        ax2.axvspan(s.index[0], s.index[-1] + pd.Timedelta(days=1), color=COLOR[s.iloc[0]], alpha=0.8, lw=0)
    ax2.set_yticks([])
    ax2.set_ylabel("실시간", rotation=0, labelpad=25)
    fig.tight_layout()
    fig.savefig(OUT / f"{sym}_regime.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
