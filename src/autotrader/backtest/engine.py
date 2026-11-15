"""공용 백테스트 엔진: 목표 비중 → 비용 포함 수익률.

lab의 strategy.py는 generate_signals(data, params)로 '봉 t 종가 시점에 정한 목표 비중'(날짜 × 종목)을 돌려준다.
엔진은 그 비중을 다음 봉에 체결한다고 보고 수익률을 계산한다. 같은 함수가 paper/live에서도 쓰인다.

체결 규칙 (fill)
- next_open (기본): 봉 t에 정한 비중을 봉 t+1 시가에 맞춘다. 밤사이(종가→시가)는 이전 비중으로 보유.
- next_close      : 봉 t+1 종가에 맞춘다 (시가가 없거나 장 마감 동시호가 체결을 가정할 때).
비중 규칙
- 1.0 = 자산 100%. 합이 1 미만이면 나머지는 현금(이자 0). 음수는 공매도(선물). 합이 1 초과면 레버리지.
- 신호 표에 없는 날짜는 직전 비중 유지(ffill), 처음 신호 전은 0. NaN 비중은 0.
- 체결 가격이 없는 종목(거래정지 등)은 그 봉에 매매하지 않고 기존 보유 유지.
- 상장폐지(마지막 가격 이후 데이터 없음): 마지막 가격 다음 봉에 그 가격으로 강제 청산 (비용·세금 부과).
  실제 정리매매는 더 낮은 값에 팔리는 경우가 많아 이 가정은 낙관적이다 — krx_1d에 정리매매 시세가 있으면 그만큼은 반영된다.
비용 (Costs, 모두 %)
- 매매한 금액 × (수수료 + 슬리피지), 매도한 금액(롱 감소분) × 매도세. 비중이 가격 변동으로 흘러간 것(drift)까지 맞추는 회전율 기준.
- 반영하지 않는 것: 공매도 대차 비용, 선물 펀딩비, 시장 충격(주문 크기에 따른), 양도소득세·배당세.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data import store
from . import benchmark
from .data import Panel

EPS = 1e-9


@dataclass
class Costs:
    fee_pct: float = 0.0          # 한쪽 수수료 %
    slippage_pct: float = 0.0     # 한쪽 슬리피지 %
    tax_pct_sell: float | dict = 0.0   # 매도 시 세금 %. {"시행일": %} 표면 날짜별로 적용
    mult: float = 1.0             # 비용 배수 (검증 기준 4: 2배)

    @property
    def per_trade(self) -> float:
        return (self.fee_pct + self.slippage_pct) / 100 * self.mult

    @property
    def sell_tax(self) -> float:
        """세율 표면 가장 최근 값 (요약 표시용). 계산은 tax_rates()."""
        if isinstance(self.tax_pct_sell, dict):
            return float(self.tax_pct_sell[max(self.tax_pct_sell)]) / 100 * self.mult
        return self.tax_pct_sell / 100 * self.mult

    def tax_rates(self, index: pd.DatetimeIndex) -> np.ndarray:
        """봉마다 적용할 매도세 (비율). 표면 시행일 이후 첫 값, 표 시작 전은 첫 값."""
        if not isinstance(self.tax_pct_sell, dict):
            return np.full(len(index), self.tax_pct_sell / 100 * self.mult)
        tab = pd.Series({pd.Timestamp(k): float(v) for k, v in self.tax_pct_sell.items()}).sort_index()
        pos = np.clip(tab.index.searchsorted(index, side="right") - 1, 0, None)
        return tab.to_numpy()[pos] / 100 * self.mult

    def scaled(self, mult: float) -> Costs:
        return Costs(self.fee_pct, self.slippage_pct, self.tax_pct_sell, mult)

    @classmethod
    def from_config(cls, market: str, overrides: dict | None = None, mult: float = 1.0) -> Costs:
        s = yaml.safe_load((store.ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
        m = s["markets"][market]
        o = {k: v for k, v in (overrides or {}).items() if v is not None}
        return cls(fee_pct=o.get("fee_pct", m.get("fee_pct", 0.0)),
                   slippage_pct=o.get("slippage_pct", s["backtest"]["slippage_pct"]),
                   tax_pct_sell=o.get("tax_pct_sell", m.get("tax_pct_sell", 0.0)), mult=mult)


@dataclass
class Result:
    returns: pd.Series            # 봉별 순수익률 (비용 후)
    gross: pd.Series              # 비용 전
    cost: pd.Series               # 봉별 비용 (자산 대비)
    turnover: pd.Series           # 봉별 회전율 (매매 금액 / 자산)
    weights: pd.DataFrame         # 각 봉 종가 시점 보유 비중
    trades: int                   # 신규 진입 횟수 (0 → 보유로 바뀐 횟수, 종목별)
    forced_exits: int             # 상장폐지 강제 청산 횟수
    costs: Costs
    meta: dict = field(default_factory=dict)
    equity: pd.Series | None = None     # capital을 줬을 때: 봉 종가 평가액 (통화 단위)
    invested: pd.Series | None = None   # 누적 투입액 (초기 자본 + 적립금)
    shares: pd.DataFrame | None = None  # whole_shares일 때 보유 주식 수

    def money(self) -> dict:
        """금액 기준 결과 (capital을 줬을 때). 적립식이면 수익률은 투입 대비, xirr은 돈 들어온 시점을 반영한 연수익률.
        MDD는 metrics()의 시간가중 수익률 기준을 쓴다 (평가액 기준은 적립금 때문에 낙폭이 작게 보임)."""
        if self.equity is None:
            return {}
        eq, inv = self.equity, self.invested
        pnl = (eq / inv.where(inv > 0) - 1)
        flows = inv.diff().fillna(inv.iloc[0])
        cf = [(d, -float(v)) for d, v in flows.items() if v] + [(eq.index[-1], float(eq.iloc[-1]))]
        return {"value": float(eq.iloc[-1]), "invested": float(inv.iloc[-1]),
                "profit": float(eq.iloc[-1] - inv.iloc[-1]), "return_on_invested": float(pnl.iloc[-1]),
                "xirr": xirr(cf), "worst_return_on_invested": float(pnl.min()),
                "worst_date": str(pnl.idxmin().date()) if pnl.notna().any() else None,
                "cash_end": float(eq.iloc[-1] * (1 - self.weights.iloc[-1].sum()))}

    def metrics(self, year: float, rf: pd.Series | None = None) -> dict:
        r = self.returns
        if len(r) < 2 or r.std() == 0:
            return {"days": len(r), "trades": self.trades, "sharpe": 0.0, "sharpe_rf0": 0.0,
                    "total_return": float((1 + r).prod() - 1), "mdd": 0.0}
        m = benchmark.metrics(r, rf, year=year)
        m |= {"trades": self.trades, "forced_exits": self.forced_exits,
              "turnover_per_year": float(self.turnover.mean() * year),
              "cost_per_year": float(self.cost.mean() * year),
              "exposure": float(self.weights.abs().sum(axis=1).mean()),
              "gross_total_return": float((1 + self.gross).prod() - 1)}
        return m


def align_weights(w: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """신호 표를 panel의 날짜·종목에 맞춘다 (없는 날짜는 직전 비중 유지)."""
    if not isinstance(w, pd.DataFrame):
        raise TypeError("generate_signals는 날짜 × 종목 DataFrame(목표 비중)을 돌려줘야 한다")
    unknown = set(w.columns) - set(panel.close.columns)
    if unknown:
        raise ValueError(f"panel에 없는 종목: {sorted(unknown)[:5]}")
    w = w.sort_index()
    w = w[~w.index.duplicated(keep="last")]
    w = w.reindex(panel.index.union(w.index)).ffill().reindex(panel.index)
    return w.reindex(columns=panel.close.columns).fillna(0.0).astype(float)


def monthly(index: pd.DatetimeIndex, amount: float, start=None, end=None) -> pd.Series:
    """매월 첫 봉에 amount씩 넣는 적립금 표 (run의 flows로)."""
    d = index[(index >= pd.Timestamp(start)) if start is not None else slice(None)]
    if end is not None:
        d = d[d < pd.Timestamp(end)]
    first = pd.Series(d, index=d).groupby([d.year, d.month]).first()
    return pd.Series(float(amount), index=pd.DatetimeIndex(first.values))


def xirr(flows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """현금흐름(투입 음수, 마지막 평가액 양수)의 연환산 내부수익률. 이분법."""
    if len(flows) < 2:
        return None
    t0 = flows[0][0]
    ts = np.array([(d - t0).total_seconds() / 86400 / 365.25 for d, _ in flows])
    cf = np.array([c for _, c in flows])
    lo, hi = -0.9999, 100.0
    if (cf / (1 + lo) ** ts).sum() * (cf / (1 + hi) ** ts).sum() > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if (cf / (1 + mid) ** ts).sum() > 0:
            lo = mid
        else:
            hi = mid
    return float(mid)


def run(panel: Panel, weights: pd.DataFrame, costs: Costs, fill: str = "next_open",
        start=None, end=None, capital: float | None = None, flows: pd.Series | None = None,
        whole_shares: bool = False, rebalance: str = "always") -> Result:
    """weights(봉 t 종가에 정한 목표)를 봉 t+1에 체결. [start, end) 구간의 수익만 남긴다.
    start 이전 신호는 start 봉의 체결에 쓰인다 (워크포워드 이음매에서 끊기지 않게).

    capital      초기 자본 (통화 단위). 주면 평가액(equity)·투입액(invested)을 같이 계산한다.
    flows        적립금 표 {날짜: 금액}. 그 날짜(없으면 다음 봉)의 체결 시점에 현금으로 들어오고
                 목표 비중대로 투자된다. capital 없이 flows만 주면 0원에서 시작.
    whole_shares 정수 주식만 산다 (capital 또는 flows 필요). 살 때 수수료까지 감안해 내림, 1주 미만은 현금.
    rebalance    "always": 매 봉 목표 비중으로 되돌림 (가격 변동으로 흘러간 비중도 맞춤)
                 "on_change": 목표가 바뀐 종목, 적립금이 들어온 봉, 상장폐지 청산 때만 매매.
    returns(시간가중)는 적립금 영향을 뺀 수익률이라 MDD·Sharpe는 그대로 의미가 있다."""
    if fill not in ("next_open", "next_close"):
        raise ValueError(f"fill: {fill}")
    if rebalance not in ("always", "on_change"):
        raise ValueError(f"rebalance: {rebalance}")
    money = capital is not None or flows is not None
    if whole_shares and not money:
        raise ValueError("whole_shares에는 capital이나 flows가 필요하다")
    w = align_weights(weights, panel)
    idx = panel.index
    o = panel.open.reindex(columns=w.columns).to_numpy(float)
    c = panel.close.reindex(columns=w.columns).to_numpy(float)
    W = w.to_numpy()
    n, k = c.shape

    # 종목별 마지막 유효 종가 위치 → 그 다음 봉에서 강제 청산
    valid = ~np.isnan(c)
    last = np.where(valid.any(0), n - 1 - np.argmax(valid[::-1], axis=0), -1)
    last_px = np.full(k, np.nan)

    lo = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start)))
    hi = n if end is None else int(idx.searchsorted(pd.Timestamp(end)))
    flow = np.zeros(n)
    if flows is not None and len(flows):
        pos = idx.searchsorted(pd.DatetimeIndex(flows.index))
        for p_, v in zip(pos, flows.to_numpy(float)):
            if lo <= p_ < hi:
                flow[max(p_, lo, 1)] += v      # 0번째 봉은 체결이 없으므로 다음 봉으로
    tax_t = costs.tax_rates(idx)
    V = float(capital or 0.0) if money else 1.0
    inv = V if money else 0.0
    h = np.zeros(k)                 # 현재 보유 비중 (직전 봉 종가 기준, drift 반영)
    net, gross, cost_s, turn_s, eq_s, inv_s = (np.zeros(hi - lo) for _ in range(6))
    held = np.zeros((hi - lo, k))
    sh_s = np.zeros((hi - lo, k)) if whole_shares else None
    sh = np.zeros(k)                # whole_shares: 보유 주식 수
    trades = forced = 0
    fee = costs.per_trade

    def drift(h, r):
        p = float(h @ r)
        return (h * (1 + r) / (1 + p) if abs(1 + p) > EPS else h * 0), p

    for t in range(max(lo, 1), hi):
        prev_c = np.where(np.isnan(c[t - 1]), last_px, c[t - 1])
        last_px = np.where(np.isnan(c[t - 1]), last_px, c[t - 1])
        target = W[t - 1].copy()

        # 상장폐지: 마지막 가격 다음 봉 → 그 가격에 청산
        dead = (last < t) & (np.abs(h) > EPS)
        target[last < t] = 0.0

        if fill == "next_open":
            px = o[t]
            g = np.nan_to_num(o[t] / prev_c - 1)            # 밤사이
            trade_ok = ~np.isnan(o[t]) | dead
            seg2 = np.nan_to_num(c[t] / o[t] - 1)            # 장중
        else:
            px = c[t]
            g = np.nan_to_num(c[t] / prev_c - 1)
            trade_ok = ~np.isnan(c[t]) | dead
            seg2 = np.zeros(k)
        g[dead] = 0.0
        px = np.where(np.isnan(px), last_px, px)            # 상장폐지 청산 가격
        h, p1 = drift(h, g)
        V *= 1 + p1

        # 적립금: 현금으로 들어온 뒤 비중이 그만큼 희석된다
        f = flow[t]
        if f:
            if V + f > EPS:
                h = h * V / (V + f)
            V += f
            inv += f

        if rebalance == "on_change":
            prev_target = W[t - 2] if t >= 2 else np.zeros(k)
            changed = (np.abs(W[t - 1] - prev_target) > EPS) | dead | (last < t)
            trade_ok = trade_ok & (changed | bool(f))
        new = np.where(trade_ok, target, h)

        if whole_shares and V > EPS:
            cur = np.round(h * V / np.where(px > 0, px, np.inf))       # 지금 보유 주식 수 (정수)
            want = new * V / np.where(px > 0, px, np.inf)
            buy = want > cur
            n_new = np.where(buy, cur + np.floor((want - cur) * np.where(want > 0, 1 / (1 + fee), 1) + 1e-9),
                             np.where(want >= 0, np.floor(want + 1e-9), -np.floor(-want + 1e-9)))
            n_new = np.where(trade_ok, n_new, cur)
            # 같은 봉에 팔고 사면 매도세·수수료만큼 현금이 모자랄 수 있다 → 현금이 음수가 안 될 때까지
            # 가장 큰 매수부터 1주씩 줄인다
            px0 = np.nan_to_num(px)

            def cash_after(nn):
                pv, hv = nn * px0, cur * px0
                sold = np.clip(np.clip(hv, 0, None) - np.clip(pv, 0, None), 0, None).sum()
                return V - pv.sum() - np.abs(pv - hv).sum() * fee - sold * tax_t[t]

            while cash_after(n_new) < -1e-9:
                buying = np.where(n_new > cur, n_new * px0, -np.inf)
                j = int(np.argmax(buying))
                if not np.isfinite(buying[j]):
                    break
                n_new[j] -= 1
            sh = n_new
            new = np.nan_to_num(n_new * px / V)

        dw = new - h
        entries = (np.abs(h) <= EPS) & (np.abs(new) > EPS)
        trades += int(entries.sum())
        forced += int(dead.sum())
        sold_long = np.clip(np.clip(h, 0, None) - np.clip(new, 0, None), 0, None)
        turnover = float(np.abs(dw).sum())
        cost = turnover * fee + float(sold_long.sum()) * tax_t[t]
        # 정수 주식: 비용은 남는 현금에서 나간다 (보유 금액 그대로, 자산만 줄어 비중이 조금 커짐).
        # 소수점(비중) 모드: 비용만큼 산 금액이 줄어든다고 본다 (비중 1 = 비용 떼고 전액 투자).
        h = (new / (1 - cost) if cost < 1 else new * 0) if whole_shares else new
        V *= 1 - cost
        h, p2 = drift(h, seg2)     # next_close면 seg2 = 0 (종가에 맞추고 봉이 끝남)
        V *= 1 + p2
        i = t - lo
        gross[i] = (1 + p1) * (1 + p2) - 1
        net[i] = (1 + p1) * (1 - cost) * (1 + p2) - 1   # 적립금이 들어온 뒤 구간까지 이어 붙인 시간가중
        cost_s[i], turn_s[i] = cost, turnover
        held[i] = h
        eq_s[i], inv_s[i] = V, inv
        if whole_shares:
            sh_s[i] = sh

    sl = idx[lo:hi]
    first = 1 if lo == 0 else 0     # 0번째 봉은 전일 대비 수익이 없음
    res = Result(
        returns=pd.Series(net, sl).iloc[first:], gross=pd.Series(gross, sl).iloc[first:],
        cost=pd.Series(cost_s, sl).iloc[first:], turnover=pd.Series(turn_s, sl).iloc[first:],
        weights=pd.DataFrame(held, sl, w.columns).iloc[first:],
        trades=trades, forced_exits=forced, costs=costs,
        meta={"fill": fill, "rebalance": rebalance, "whole_shares": whole_shares})
    if money:
        res.equity = pd.Series(eq_s, sl).iloc[first:]
        res.invested = pd.Series(inv_s, sl).iloc[first:]
        if whole_shares:
            res.shares = pd.DataFrame(sh_s, sl, w.columns).iloc[first:]
    return res


# ── 미래참조 자동 검사 ──
def check_causality(signal_fn, panel: Panel, params: dict, n_cuts: int = 3, tol: float = 1e-9) -> dict:
    """데이터를 중간에서 잘라 다시 계산했을 때, 잘린 시점까지의 신호가 같아야 한다.
    다르면 generate_signals가 미래 데이터를 쓰고 있다 (center=True rolling, 전체 기간 정규화, bfill 등)."""
    full = align_weights(signal_fn(panel, params), panel)
    idx = panel.index
    out = {"passed": True, "cuts": []}
    for q in np.linspace(0.4, 0.85, n_cuts):
        cut = idx[int(len(idx) * q)]
        part = panel.slice(end=cut + pd.Timedelta(microseconds=1))
        w = align_weights(signal_fn(part, params), part)
        a = full.loc[w.index, w.columns].to_numpy()
        diff = np.abs(a - w.to_numpy())
        bad = np.where(diff.max(axis=1) > tol)[0]
        rec = {"cut": str(cut), "max_diff": float(diff.max()) if diff.size else 0.0,
               "mismatch_bars": len(bad),
               "first_mismatch": str(w.index[bad[0]]) if len(bad) else None}
        out["cuts"].append(rec)
        out["passed"] &= rec["mismatch_bars"] == 0
    return out


# ── 워크포워드 ──
def windows(index: pd.DatetimeIndex, start=None, train: str = "24M", test: str = "6M") -> list[dict]:
    """학습 창 train, 시험 창 test를 test만큼씩 앞으로. 마지막 시험 창은 데이터 끝에서 잘린다."""
    def off(s):
        n, u = int(s[:-1]), s[-1].upper()
        return {"M": pd.DateOffset(months=n), "Y": pd.DateOffset(years=n), "D": pd.DateOffset(days=n)}[u]

    t0 = pd.Timestamp(start) if start else index[0]
    end = index[-1] + pd.Timedelta(days=1)
    out, tr_start = [], t0
    while True:
        te_start = tr_start + off(train)
        te_end = min(te_start + off(test), end)
        if te_start >= end or (index[(index >= te_start) & (index < te_end)].size < 2):
            break
        out.append({"train": (tr_start, te_start), "test": (te_start, te_end)})
        tr_start = tr_start + off(test)
    return out


def execution_kw(execution: dict | None, index: pd.DatetimeIndex, start=None, end=None) -> dict:
    """lab.yaml execution 블록 → run() 인자. monthly_contribution은 그 구간의 월초 적립금 표로 바꾼다."""
    e = dict(execution or {})
    kw = {k: e[k] for k in ("capital", "whole_shares", "rebalance") if e.get(k) is not None}
    if e.get("monthly_contribution"):
        kw["flows"] = monthly(index, e["monthly_contribution"], start, end)
    return kw


def walk_forward(strategy, panel: Panel, params: dict, costs: Costs, fill: str = "next_open",
                 start=None, train: str = "24M", test: str = "6M", execution: dict | None = None) -> dict:
    """strategy: generate_signals(data, params)와 선택적으로 fit(train_data, params) -> params를 가진 모듈.
    시험 창마다 그 창 끝까지의 데이터로 신호를 계산해(과거만 보임) 시험 구간 비중만 이어 붙인 뒤 한 번에 돌린다.
    execution: {capital, monthly_contribution, whole_shares, rebalance} (lab.yaml) — 학습·시험 실행 각각에 적용."""
    wins = windows(panel.index, start, train, test)
    if not wins:
        raise ValueError("워크포워드 창을 만들 수 없음 — 데이터 기간이 train+test보다 짧다")
    fit = getattr(strategy, "fit", None)
    stitched, rows = [], []
    for wdw in wins:
        (a, b), (c0, c1) = wdw["train"], wdw["test"]
        p = fit(panel.slice(a, b), dict(params)) if fit else params
        sig = align_weights(strategy.generate_signals(panel.slice(end=c1), p), panel.slice(end=c1))
        stitched.append(sig.loc[(sig.index >= c0) & (sig.index < c1)])
        pb = panel.slice(end=b)
        tr = run(pb, strategy.generate_signals(pb, p), costs, fill, start=a,
                 **execution_kw(execution, pb.index, a, b))
        rows.append({"train": (str(a.date()), str(b.date())), "test": (str(c0.date()), str(c1.date())),
                     "params": p, "_train_result": tr})
    # 시험 구간 첫 봉 체결에는 직전 학습 끝 신호가 필요 → 첫 창 직전 봉 신호도 붙인다
    first_test = pd.Timestamp(wins[0]["test"][0])
    p0 = rows[0]["params"]
    pre = align_weights(strategy.generate_signals(panel.slice(end=first_test), p0), panel.slice(end=first_test))
    w_all = pd.concat([pre.iloc[-1:]] + stitched)
    w_all = w_all[~w_all.index.duplicated(keep="last")]
    last_end = wins[-1]["test"][1]
    test_res = run(panel, w_all, costs, fill, start=first_test, end=last_end,
                   **execution_kw(execution, panel.index, first_test, last_end))
    for r, wdw in zip(rows, wins):
        c0, c1 = wdw["test"]
        r["_test_returns"] = test_res.returns[(test_res.returns.index >= c0) & (test_res.returns.index < c1)]
    return {"windows": rows, "test": test_res, "weights": w_all}


def sharpe(r: pd.Series, year: float) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(year)) if len(r) > 1 and r.std() > 0 else 0.0


def perturb(params: dict, factor: float) -> dict:
    """숫자 파라미터를 factor배 (정수는 반올림, 원래 1 이상이면 1 이상 유지). bool·문자열은 그대로."""
    out = {}
    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, int):
            out[k] = max(1 if v >= 1 else 0, int(round(v * factor)))
        else:
            out[k] = v * factor
    return out


def save_result(res: Result, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"net": res.returns, "gross": res.gross, "cost": res.cost, "turnover": res.turnover}) \
        .to_parquet(folder / f"{name}_returns.parquet")
    (folder / f"{name}_costs.yaml").write_text(yaml.safe_dump(asdict(res.costs), allow_unicode=True),
                                               encoding="utf-8")
