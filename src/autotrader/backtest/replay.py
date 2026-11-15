"""진입 직전 점검: 타임머신 재현 (day-trader-skill의 trader-timemachine에서 가져온 아이디어).

진입가·손절가·목표가가 정해진 하루짜리 계획에 대해 두 가지를 계산한다.
1) 모의 재현(simulate): 그 종목의 최근 N일 장중 봉을 "같은 시각대"끼리 무작위로 뽑아
   남은 장을 10,000번 다시 만든다. 목표 먼저 / 손절 먼저 / 둘 다 아님 확률과 평균 R.
2) 실제 기록(history): 같은 규칙을 과거 각 세션에 다시 돌려 실제로 어땠는지.

규칙 (원본과 같음, 보수적으로):
- 봉 하나가 손절과 목표를 둘 다 건드리면 손절로 친다.
- 손절을 갭으로 뚫어도 -1R로 계산한다 (실제 체결은 더 나쁠 수 있음 → 결과에 명시).
- 판단일과 그 이후 봉은 표본에 넣지 않는다. 완결된 세션만 쓴다.
- 시드를 항상 결과에 남긴다 (같은 계획·데이터·시드 = 같은 숫자).

판정은 R을 반영한다: 목표 R에서 손익분기 승률은 1/(1+R) (1R 50%, 2R 33%, 3R 25%).
이 모듈은 주문·사이징을 하지 않는다. 결과는 예측이 아니라 최근 분포의 재현이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Plan:
    entry: float
    stop: float
    target: float
    entry_ts: pd.Timestamp   # 진입 봉의 시작 시각 (이 봉의 종가에 진입했다고 본다)
    side: str = "long"       # long | short

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def target_r(self) -> float:
        return abs(self.target - self.entry) / self.risk

    def check(self) -> None:
        ok = (self.stop < self.entry < self.target) if self.side == "long" else (self.target < self.entry < self.stop)
        if not ok:
            raise ValueError(f"계획이 맞지 않음 ({self.side}): entry={self.entry} stop={self.stop} target={self.target}")


@dataclass
class Result:
    p_target: float
    p_stop: float
    p_neither: float
    p_green: float
    avg_r: float
    close_pctl: dict
    n: int
    seed: int
    sessions: int
    bars_left: int
    verdict: str
    notes: list[str] = field(default_factory=list)


def sessions(bars: pd.DataFrame, tz: str) -> dict:
    """장중 봉을 현지 날짜별 세션으로 나누고, 가장 흔한 봉 개수를 가진 세션만 '완결 세션'으로 남긴다
    (반일장·결측 세션 제외). 반환: {date: DataFrame(ts, open, high, low, close)}"""
    local = bars["ts"].dt.tz_convert(tz)
    groups = {d: g.reset_index(drop=True) for d, g in bars.groupby(local.dt.date)}
    full = pd.Series({d: len(g) for d, g in groups.items()}).mode().iloc[0]
    return {d: g for d, g in groups.items() if len(g) == full}


def _ratios(sess: dict) -> np.ndarray:
    """세션 × 봉 × (open, high, low, close)/직전 종가. 첫 봉은 전날 종가 기준."""
    days = sorted(sess)
    arr = np.stack([sess[d][["open", "high", "low", "close"]].to_numpy(float) for d in days])
    prev = np.concatenate([[arr[0, 0, 0]], arr[:-1, -1, 3]])     # 전 세션 종가
    prev_close = np.concatenate([prev[:, None], arr[:, :-1, 3]], axis=1)
    return arr / prev_close[:, :, None]


def verdict(p_target: float, p_stop: float, target_r: float) -> str:
    hit = p_target + p_stop
    if hit == 0:
        return "판정 불가: 어느 쪽에도 닿은 재현이 없음"
    share, be = p_target / hit, 1 / (1 + target_r)
    gap = (share - be) * 100
    if abs(gap) < 5:
        base = "동전 던지기 수준" if abs(target_r - 1) < 1e-9 else "손익분기 근처"
    else:
        side = "목표" if gap > 0 else "손절"
        base = f"{'강하게 ' if abs(gap) >= 15 else ''}{side} 쪽으로 기울어짐"
    return f"{base}: 한쪽에 닿은 경우 중 목표 먼저 {share:.1%} (손익분기 {be:.1%})"


def simulate(bars: pd.DataFrame, plan: Plan, tz: str, n: int = 10_000, seed: int = 19,
             window: int = 2, lookback_sessions: int = 60, until_bars: int | None = None,
             min_sessions: int = 20) -> Result:
    """bars: 한 종목 장중 봉 (ts UTC). 판단일 세션에 진입 봉이 있어야 한다.
    until_bars: 진입 후 몇 봉까지만 볼지 (예: 장 마감 N분 전 청산). 기본은 세션 끝."""
    plan.check()
    sess = sessions(bars, tz)
    day = plan.entry_ts.tz_convert(tz).date()
    if day not in sess:
        raise ValueError(f"판단일 {day} 세션이 없거나 완결되지 않음")
    hit = np.flatnonzero((sess[day]["ts"] == plan.entry_ts).to_numpy())
    if len(hit) == 0:
        raise ValueError(f"진입 봉 {plan.entry_ts}이 데이터에 없음 (봉 간격·시각 확인)")
    k = int(hit[0])
    past = {d: g for d, g in sess.items() if d < day}
    past = dict(sorted(past.items())[-lookback_sessions:])
    if len(past) < min_sessions:
        raise ValueError(f"완결 세션 {len(past)}개 < {min_sessions}개: 표본 부족")

    R = _ratios(past)                                  # (S, B, 4)
    S, B, _ = R.shape
    end = B if until_bars is None else min(B, k + 1 + until_bars)
    m = end - (k + 1)
    if m <= 0:
        raise ValueError("진입 후 남은 봉이 없음")
    rng = np.random.default_rng(seed)
    idx = np.arange(k + 1, end)
    bar_idx = np.clip(idx[None, :] + rng.integers(-window, window + 1, (n, m)), 1, B - 1)
    pick = R[rng.integers(0, S, (n, m)), bar_idx]      # (n, m, 4)

    closes = plan.entry * np.cumprod(pick[:, :, 3], axis=1)
    prev = np.concatenate([np.full((n, 1), plan.entry), closes[:, :-1]], axis=1)
    hi, lo = prev * pick[:, :, 1], prev * pick[:, :, 2]
    if plan.side == "long":
        stop_hit, tgt_hit = lo <= plan.stop, hi >= plan.target
    else:
        stop_hit, tgt_hit = hi >= plan.stop, lo <= plan.target
    first = lambda a: np.where(a.any(1), a.argmax(1), m)
    s_i, t_i = first(stop_hit), first(tgt_hit)
    stop_first = (s_i <= t_i) & (s_i < m)              # 같은 봉이면 손절
    tgt_first = (t_i < s_i)
    neither = ~(stop_first | tgt_first)
    sign = 1 if plan.side == "long" else -1
    last = closes[:, -1]
    r = np.where(tgt_first, plan.target_r, np.where(stop_first, -1.0, sign * (last - plan.entry) / plan.risk))
    green = tgt_first | (neither & (sign * (last - plan.entry) > 0))

    pt, ps, pn = tgt_first.mean(), stop_first.mean(), neither.mean()
    notes = ["손절 갭 돌파도 -1R로 계산 (실제 체결은 더 나쁠 수 있음)",
             f"최근 {S}개 완결 세션의 같은 시각대(±{window}봉) 봉으로 만든 재현이며 예측이 아님"]
    if pn > 0.4:
        notes.append(f"재현의 {pn:.1%}가 끝까지 어느 쪽에도 닿지 않음 → 마감 가격이 결과를 좌우")
    v = verdict(pt, ps, plan.target_r)
    if (r.mean() > 0) != (pt / max(pt + ps, 1e-12) > 1 / (1 + plan.target_r)):
        notes.append(f"평균 R({r.mean():+.3f})의 방향이 판정과 반대 (미도달 재현 때문)")
    return Result(pt, ps, pn, green.mean(), float(r.mean()),
                  {q: float(np.percentile(last, q)) for q in (5, 50, 95)},
                  n, seed, S, m, v, notes)


def history(sess: dict, plan_fn, before=None) -> pd.DataFrame:
    """같은 규칙을 과거 각 세션에 다시 적용한 실제 결과.
    plan_fn(session_df) -> (entry_idx, entry, stop, target, side) 또는 None(그날 신호 없음).
    반환: date, outcome(target|stop|neither), r"""
    rows = []
    for d, g in sorted(sess.items()):
        if before is not None and d >= before:
            continue
        p = plan_fn(g)
        if p is None:
            continue
        k, entry, stop, target, side = p
        risk, sign = abs(entry - stop), (1 if side == "long" else -1)
        outcome, r = "neither", sign * (g["close"].iloc[-1] - entry) / risk
        for _, b in g.iloc[k + 1:].iterrows():
            s_hit = b["low"] <= stop if side == "long" else b["high"] >= stop
            t_hit = b["high"] >= target if side == "long" else b["low"] <= target
            if s_hit:
                outcome, r = "stop", -1.0
                break
            if t_hit:
                outcome, r = "target", abs(target - entry) / risk
                break
        rows.append({"date": d, "outcome": outcome, "r": r})
    return pd.DataFrame(rows)


def report(res: Result, hist: pd.DataFrame | None = None) -> str:
    lines = [
        f"{res.n:,}회 재현, 진입 후 {res.bars_left}봉, 표본 {res.sessions}세션, seed {res.seed}",
        f"  목표 먼저 {res.p_target:.1%} · 손절 먼저 {res.p_stop:.1%} · 미도달 {res.p_neither:.1%} · 수익 마감 {res.p_green:.1%}",
        f"  마감 가격 중앙값 {res.close_pctl[50]:,.2f} (90% 구간 {res.close_pctl[5]:,.2f} ~ {res.close_pctl[95]:,.2f})",
        f"  평균 R {res.avg_r:+.3f}",
    ]
    if hist is not None and not hist.empty:
        c = hist["outcome"].value_counts()
        lines.append(f"실제 기록: {len(hist)}회 발생 — 목표 {c.get('target', 0)} · 손절 {c.get('stop', 0)} · "
                     f"미도달 {c.get('neither', 0)}, 평균 R {hist['r'].mean():+.3f}")
    lines.append(res.verdict)
    lines += [f"  - {x}" for x in res.notes]
    return "\n".join(lines)
