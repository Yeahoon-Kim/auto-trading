"""L010 — KR 가치+수익성: B/M 순위와 영업이익/총자산 순위의 평균 상위 20% 동일가중, 월 1회 리밸런싱.

매월 첫 거래일 종가로 계산 → 엔진이 다음 거래일 시가에 체결한다.
    uv run python -m autotrader.backtest.lab labs/L010-kr-value-profitability [--quick]
공통 유니버스(universe)·월초(month_starts)는 고전 퀀트 프로그램 공통 규칙
(docs/research/2026-11-18-classic-quant-factors.md 4절)을 그대로 옮긴 것이다. lab끼리 import하지 않는다.
"""

import numpy as np
import pandas as pd


def month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """각 달의 첫 거래일 (그날 장이 끝나면 바로 알 수 있다)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).min().to_numpy())


def universe(data, params: dict) -> pd.DataFrame:
    """그날 투자할 수 있는 종목 (날짜 × 종목 bool). 그날까지의 값만 쓴다.
    보통주(코드 끝 0), 원본 종가 ≥ min_price, liq_days일 평균 거래대금 ≥ min_value, 가격 이력 ≥ min_hist봉, 그날 거래."""
    close, raw, value = data.close, data["raw_close"], data["value"]
    common = pd.Series([str(c).endswith("0") for c in close.columns], index=close.columns)
    liq_days, min_hist = int(params["liq_days"]), int(params["min_hist"])
    liq = value.where(close.notna()).rolling(liq_days, min_periods=liq_days).mean()
    hist = close.notna().cumsum()
    traded = data["tradable"].astype("boolean").fillna(False).astype(bool) if "tradable" in data else close.notna()
    ok = close.notna() & traded & (raw >= params["min_price"]) & (liq >= params["min_value"]) & (hist >= min_hist)
    return ok & common.to_numpy()[None, :]


def score(data, ok: pd.DataFrame) -> pd.DataFrame:
    """Novy-Marx(2013) 방식: 그날 후보(ok) 안에서 B/M 백분위 순위와 영업수익성 백분위 순위의 평균.
    둘 중 하나라도 없으면 NaN. B/M = 자본총계/시가총액(자본 ≤ 0 제외), 영업수익성 = 영업이익 TTM/총자산."""
    eq = data.equity.where(data.equity > 0)
    bm = eq / data.market_cap.where(data.market_cap > 0)
    opa = data.operating_income_ttm / data.total_assets.where(data.total_assets > 0)
    both = ok & bm.notna() & opa.notna()
    r1 = bm.where(both).rank(axis=1, pct=True)
    r2 = opa.where(both).rank(axis=1, pct=True)
    return (r1 + r2) / 2


def top_quantile(sc: pd.DataFrame, ok: pd.DataFrame, dates, q: float, cols) -> pd.DataFrame:
    """날짜마다 후보 중 점수 상위 q 비율을 동일가중. 후보가 없으면 전액 현금."""
    rows = {}
    for d in dates:
        s = sc.loc[d].where(ok.loc[d]).dropna()
        s = s[np.isfinite(s)]
        w = pd.Series(0.0, index=cols)
        n = int(np.floor(len(s) * q))
        if n >= 1:
            pick = s.sort_values(ascending=False, kind="mergesort").index[:n]
            w[pick] = 1.0 / n
        rows[d] = w
    return pd.DataFrame(rows).T


def generate_signals(data, params: dict) -> pd.DataFrame:
    close = data.close
    dates = month_starts(close.index)
    if len(dates) == 0:
        return pd.DataFrame(0.0, index=close.index[:1], columns=close.columns)
    ok = universe(data, params)
    return top_quantile(score(data, ok), ok, dates, float(params["quantile"]), close.columns)
