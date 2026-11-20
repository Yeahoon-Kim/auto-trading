"""L012 — US 국면 필터 단독: SPY를 200일선 위에서만 보유 (Faber 2007).

SPY 하나만 쓰므로 us_1d의 현재 구성종목 생존편향이 없다. 매월 첫 거래일 종가로 판단 → 다음 거래일 시가 체결.
    uv run python -m autotrader.backtest.lab labs/L012-us-spy-ma-timing [--quick]
비교: lab.yaml benchmark: SPY (같은 비용으로 매수 후 보유) — 엔진 보고서에 붙는다.
"""

import pandas as pd


def month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).min().to_numpy())


def generate_signals(data, params: dict) -> pd.DataFrame:
    c = data.close["SPY"]
    n = int(params["ma_days"])
    on = c > c.rolling(n, min_periods=n).mean()     # 봉 t 종가까지 (체결은 t+1 시가)
    dates = month_starts(c.index)
    w = pd.DataFrame(0.0, index=dates, columns=data.close.columns)
    if len(dates):
        w["SPY"] = on.reindex(dates).fillna(False).astype(float).to_numpy()
    return w
