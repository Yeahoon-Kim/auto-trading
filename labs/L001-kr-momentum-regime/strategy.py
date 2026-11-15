"""L001 — KR 12-1 모멘텀 + KOSPI 200일선 국면 필터.

매월 첫 거래일 종가로 계산 → 엔진이 다음 거래일 시가에 체결한다.
    uv run python -m autotrader.backtest.lab labs/L001-kr-momentum-regime [--quick | --lockbox]
"""

import numpy as np
import pandas as pd

from autotrader.data import store


def kospi_close(index: pd.DatetimeIndex) -> pd.Series:
    """KOSPI 지수 종가를 panel 날짜에 맞춘다 (빈 날은 직전 값 — 과거 방향으로만 채움)."""
    d = store.read("macro_1d", "KOSPI")
    s = d.set_index(d["ts"].dt.tz_convert(None).dt.normalize())["close"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """각 달의 첫 거래일. 그날 장이 끝나면 바로 알 수 있다 (월말은 다음 거래일을 봐야 알 수 있어
    데이터 끝·워크포워드 이음매에서 신호가 빠지므로 쓰지 않는다)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).min().to_numpy())


def generate_signals(data, params: dict) -> pd.DataFrame:
    lb, skip, top_n = int(params["lookback"]), int(params["skip"]), int(params["top_n"])
    ma_days, liq_days = int(params["ma_days"]), int(params["liq_days"])
    close, raw, value = data.close, data["raw_close"], data["value"]
    idx = close.index

    # 국면: KOSPI > 200일선
    k = kospi_close(idx)
    risk_on = k > k.rolling(ma_days, min_periods=ma_days).mean()

    # 12-1 모멘텀 (거래정지일 결측은 직전 가격으로 — 과거 방향만)
    px = close.ffill()
    score = px.shift(skip) / px.shift(lb) - 1
    hist = close.notna().cumsum()
    liq = value.where(close.notna()).rolling(liq_days, min_periods=liq_days).mean()
    ok = close.notna() & (raw >= params["min_price"]) & (liq >= params["min_value"]) & (hist >= lb)

    rows = {}
    for d in month_starts(idx):
        w = pd.Series(0.0, index=close.columns)
        if bool(risk_on.get(d, False)):
            s = score.loc[d].where(ok.loc[d]).dropna()
            s = s[np.isfinite(s)]
            pick = s.nlargest(top_n).index
            if len(pick):
                w[pick] = 1.0 / top_n        # 후보가 top_n보다 적으면 남는 비중은 현금
        rows[d] = w
    if not rows:
        return pd.DataFrame(0.0, index=idx[:1], columns=close.columns)
    return pd.DataFrame(rows).T
