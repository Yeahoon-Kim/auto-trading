"""벤치마크: 시장 평균(매수 후 보유) 수익.

lab 결과를 "그냥 지수를 샀으면"과 비교하기 위한 기준선을 만든다.

    uv run python -m autotrader.backtest.benchmark            # S&P500 기준선 생성 → results/benchmarks/sp500/

계산 규칙
- 첫 거래일 종가에 전액 매수, 마지막 거래일 종가에 전액 매도 (거래 2회).
- 비용: 매수·매도 각각 수수료(config/settings.yaml us_stock.fee_pct) + 슬리피지(backtest.slippage_pct).
- 변형:
    SPX_price      ^GSPC 가격지수. 배당 없음 (실제로 살 수 없는 지수 — 참고용)
    SPY_price      SPY 종가만. 배당 없음
    SPY_tr         SPY adj_close. 배당 전액 재투자 (ETF 운용보수는 NAV에 이미 반영)
    SPY_tr_wht     배당에 미국 원천징수 15%를 떼고 재투자 (한국 거주자 현실값)
    SPY_tr_wht_krw 위 값을 원화로 환산 (한국은행 ECOS 매매기준율, 같은 날짜 값)
    IVV_tr, VOO_tr 다른 S&P500 ETF로 같은 계산 (데이터 교차 확인용)
- 배당 = adj_close 일간 수익률 - close 일간 수익률 (배당락일에 생긴 차이).
- Sharpe: 연 252일 기준, 무위험금리를 하루 늦춰(shift 1) 뺀 초과수익 기준. rf=0 값도 같이 남긴다.
  무위험금리: 달러 변형은 FRED DFF(연방기금금리), 원화 변형은 ECOS CD 91일.
- 반영 안 한 것: 해외주식 양도소득세(22%, 연 250만원 공제), 배당소득 종합과세, 환전 수수료.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data import store

ROOT = store.ROOT
OUT = ROOT / "results" / "benchmarks"
YEAR = 252


def _settings() -> dict:
    return yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def _daily(dataset: str, symbol: str, col: str = "close") -> pd.Series:
    d = store.read(dataset, symbol)
    s = pd.Series(d[col].values, index=d["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
    return s[~s.index.duplicated(keep="last")].sort_index()


def _macro(symbol: str, dataset: str = "macro_1d", col: str = "close") -> pd.Series:
    d = store.read(dataset, symbol)
    s = pd.Series(d[col].values, index=d["ts"].dt.tz_localize(None).dt.normalize())
    return s[~s.index.duplicated(keep="last")].sort_index()


def returns_variants() -> pd.DataFrame:
    """변형별 일간 수익률 표 (비용 전). 인덱스: 미국 거래일."""
    spy_c, spy_a = _daily("us_1d", "SPY"), _daily("us_1d", "SPY", "adj_close")
    r_price = spy_c.pct_change()
    r_tr = spy_a.pct_change()
    div = (r_tr - r_price).clip(lower=0)          # 배당락일 외에는 0 (부동소수 오차는 잘라냄)
    r_wht = r_price + 0.85 * div

    # 한국은행 매매기준율 (Yahoo KRW=X는 하루 만에 되돌아오는 이상 틱이 있어 쓰지 않음).
    # 같은 날짜 서울 고시값 = 그날 뉴욕 종가보다 먼저 알려진 값 → 미래참조 없음. 원화 휴장일은 직전 값.
    krw = _macro("usdkrw_ecos", "ecos", "value").reindex(spy_c.index, method="ffill")
    r_krw = (1 + r_wht) * (krw / krw.shift(1)) - 1

    out = pd.DataFrame({
        "SPX_price": _macro("SPX").reindex(spy_c.index).pct_change(),
        "SPY_price": r_price,
        "SPY_tr": r_tr,
        "SPY_tr_wht": r_wht,
        "SPY_tr_wht_krw": r_krw,
    })
    for t in ("IVV", "VOO"):
        out[f"{t}_tr"] = _daily("us_1d", t, "adj_close").pct_change().reindex(out.index)
    return out


def rf_daily(index: pd.DatetimeIndex, krw: bool = False) -> pd.Series:
    """일간 무위험수익률. USD: 연방기금금리(FRED DFF), KRW: CD 91일(ECOS).
    관측일 다음 날부터 알 수 있으므로 shift(1)."""
    s = _macro("cd_91d", "ecos", "value") if krw else _macro("DFF", "fred", "value")
    return (s.shift(1).reindex(index, method="ffill") / 100 / YEAR).fillna(0)


def apply_costs(r: pd.Series, cost: float) -> pd.Series:
    """첫날 매수 비용, 마지막 날 매도 비용을 수익률에 반영."""
    r = r.dropna().copy()
    r.iloc[0] = (1 + r.iloc[0]) * (1 - cost) - 1     # 첫 수익률 = 매수 직후 하루. 매수 비용을 여기서 뗌
    r.iloc[-1] = (1 + r.iloc[-1]) * (1 - cost) - 1
    return r


def metrics(r: pd.Series, rf: pd.Series | None = None, year: float = YEAR) -> dict:
    """year: 연간 봉 수 (주식 일봉 252, 코인 일봉 365)."""
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / year
    peak = eq.cummax()
    dd = eq / peak - 1
    # 최장 낙폭 기간: 고점 갱신 사이 거래일 수
    under = dd < 0
    runs = under.groupby((~under).cumsum()).sum()
    vol = r.std() * np.sqrt(year)
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    m = {
        "start": r.index[0].date(), "end": r.index[-1].date(), "days": len(r),
        "total_return": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "vol": vol,
        "sharpe_rf0": r.mean() / r.std() * np.sqrt(year),
        "sharpe": ex.mean() / ex.std() * np.sqrt(year),
        "sortino_rf0": r.mean() / r[r < 0].std() * np.sqrt(year),
        "mdd": dd.min(),
        "mdd_date": dd.idxmin().date(),
        "max_underwater_days": int(runs.max()) if len(runs) else 0,
        "best_day": r.max(), "worst_day": r.min(),
    }
    m["calmar"] = m["cagr"] / abs(m["mdd"]) if m["mdd"] else np.nan
    return m


def build(start: str | None = None, cost_mult: float = 1.0) -> dict[str, pd.DataFrame]:
    s = _settings()
    cost = (s["markets"]["us_stock"]["fee_pct"] + s["backtest"]["slippage_pct"]) / 100 * cost_mult
    raw = returns_variants()
    if start:
        raw = raw[raw.index >= start]
    raw = raw.iloc[1:]                                   # 첫 행은 전일 대비 값이 없음
    rf, rf_krw = rf_daily(raw.index), rf_daily(raw.index, krw=True)

    today = pd.Timestamp.now().normalize()
    lock_start = today - pd.DateOffset(months=6)
    first, after = pd.Timestamp(raw.index[0]), today + pd.DateOffset(days=1)
    periods = {"full": (first, after), "dev": (first, lock_start), "lockbox": (lock_start, after)}

    rows, net = [], {}
    for name, col in raw.items():
        for pname, (a, b) in periods.items():
            seg = col[(col.index >= a) & (col.index < b)].dropna()
            if len(seg) < 20:
                continue
            for costed in (True, False):
                r = apply_costs(seg, cost) if costed else seg
                m = metrics(r, rf_krw if name.endswith("_krw") else rf)
                rows.append({"variant": name, "period": pname, "costs": costed, **m})
                if pname == "full" and costed:
                    net[name] = r
    summary = pd.DataFrame(rows)

    rets = pd.DataFrame(net)
    equity = (1 + rets).cumprod()
    yearly = rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly.index.name = "year"
    meta = pd.DataFrame([{
        "cost_one_way_pct": cost * 100, "cost_mult": cost_mult,
        "lockbox_start": lock_start.date(), "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
    }])
    return {"summary": summary, "yearly": yearly, "equity": equity, "meta": meta}


def save(res: dict[str, pd.DataFrame], name: str = "sp500") -> Path:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    res["summary"].to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    res["yearly"].to_csv(out / "yearly.csv", encoding="utf-8-sig")
    res["meta"].to_csv(out / "meta.csv", index=False, encoding="utf-8-sig")
    res["equity"].to_parquet(out / "equity.parquet")
    return out


def benchmark_returns(variant: str = "SPY_tr_wht", start=None, end=None) -> pd.Series:
    """lab에서 벤치마크 비교용으로 쓰는 일간 수익률 (비용 전)."""
    r = returns_variants()[variant].dropna()
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index < pd.Timestamp(end)]
    return r


def main() -> None:
    res = build()
    out = save(res)
    stress = build(cost_mult=2.0)["summary"]
    stress.to_csv(out / "summary_cost2x.csv", index=False, encoding="utf-8-sig")
    s = res["summary"]
    view = s[(s["period"] == "full") & s["costs"]][["variant", "start", "end", "cagr", "vol", "sharpe", "sharpe_rf0",
                                                     "mdd", "mdd_date", "max_underwater_days", "calmar"]]
    pd.set_option("display.width", 220)
    print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
