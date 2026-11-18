"""L008 — KR 국면 필터 단독: KOSPI 대리 포트폴리오를 지수 200일선 위에서만 보유.

KR 데이터에 ETF가 없어 KOSPI 지수를 직접 살 수 없다. 대신 그날 KOSPI 소속 보통주 전체를 시가총액 비중으로 담은
포트폴리오(KOSPI 대리)를 쓴다. 매월 첫 거래일 종가로 판단 → 다음 거래일 시가 체결.
    uv run python -m autotrader.backtest.lab labs/L008-kr-kospi-ma-timing [--quick]
"""

import pandas as pd

from autotrader.data import store


def month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).min().to_numpy())


def kospi_close(index: pd.DatetimeIndex) -> pd.Series:
    """KOSPI 지수 종가(macro_1d, 관측일)를 panel 날짜에 맞춘다 (빈 날은 직전 값 — 과거 방향만)."""
    d = store.read("macro_1d", "KOSPI")
    s = d.set_index(d["ts"].dt.tz_convert(None).dt.normalize())["close"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def risk_on(index: pd.DatetimeIndex, ma_days: int) -> pd.Series:
    """전날까지 알려진 KOSPI 종가 > 전날까지의 ma_days 이동평균 (거시 지표 shift(1) 규칙)."""
    k = kospi_close(index).shift(1)
    return k > k.rolling(ma_days, min_periods=ma_days).mean()


def proxy_weights(data, dates) -> pd.DataFrame:
    """그날 KOSPI 소속 보통주(코드 끝 0) 시가총액 비중. 종가·시가총액이 있는 종목만."""
    mc = data.market_cap.where(data.close.notna() & (data.market_cap > 0))
    common = [c for c in mc.columns if str(c).endswith("0")]
    mc = mc.loc[dates].copy()
    mc.loc[:, [c for c in mc.columns if c not in set(common)]] = float("nan")
    return mc.div(mc.sum(axis=1), axis=0).fillna(0.0)


def generate_signals(data, params: dict) -> pd.DataFrame:
    idx = data.close.index
    dates = month_starts(idx)
    if len(dates) == 0:
        return pd.DataFrame(0.0, index=idx[:1], columns=data.close.columns)
    on = risk_on(idx, int(params["ma_days"])).reindex(dates).fillna(False).astype(bool)
    w = proxy_weights(data, dates)
    return w.mul(on.astype(float), axis=0)
