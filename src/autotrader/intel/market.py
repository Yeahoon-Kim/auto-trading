"""시장정보팀: 국면 지표 계산과 장 전 브리핑.

국면 지표는 전부 코드로 계산하는 결정론적 값이고, 과거 전 기간에 대해 다시 계산할 수 있다.
→ 전략이 "국면"을 조건으로 쓸 때 이 표(regime_1d)를 그대로 쓴다.

시점 규칙: ts는 관측일(UTC 00:00). 거시 지표는 관측일 다음 날에야 알 수 있으므로,
전략에서는 반드시 shift(1) 해서 쓴다. 브리핑은 "어제까지 확정된 값"을 보여 준다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import store

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "market"
ANN = np.sqrt(252)


def _close(dataset: str, symbol: str, col: str = "close") -> pd.Series:
    d = store.read(dataset, symbol)
    if d.empty:
        return pd.Series(dtype=float, name=symbol)
    return d.set_index(d["ts"].dt.normalize())[col].rename(symbol)


def _daily_from_1m(dataset: str, symbol: str) -> pd.Series:
    d = store.read(dataset, symbol)
    if d.empty:
        return pd.Series(dtype=float, name=symbol)
    s = d.set_index("ts")["close"].resample("1D").last().dropna().rename(symbol)
    return s[s.index < pd.Timestamp.now(tz="UTC").normalize()]  # 아직 안 끝난 오늘(UTC) 제외


def _trend_block(px: pd.Series, prefix: str) -> pd.DataFrame:
    ret = px.pct_change()
    rv = ret.rolling(20).std() * ANN
    out = pd.DataFrame({
        f"{prefix}_close": px,
        f"{prefix}_above_200d": px > px.rolling(200).mean(),
        f"{prefix}_ret_20d": px.pct_change(20),
        f"{prefix}_rv_20d": rv,
        # 변동성 백분위: 최근 3년(756일) 중 현재 위치
        f"{prefix}_rv_pctile": rv.rolling(756, min_periods=250).rank(pct=True),
    })
    return out


def compute() -> pd.DataFrame:
    """관측일별 국면 지표 표."""
    spx, kospi = _close("macro_1d", "SPX"), _close("macro_1d", "KOSPI")
    btc = _daily_from_1m("binance_spot_1m", "BTCUSDT")
    parts = [_trend_block(spx, "spx"), _trend_block(kospi, "kospi"), _trend_block(btc, "btc")]

    vix, vix3m = _close("macro_1d", "VIX"), _close("macro_1d", "VIX3M")
    usdkrw = _close("macro_1d", "USDKRW")
    hy = _close("fred", "BAMLH0A0HYM2", "value")
    macro = pd.DataFrame({
        "vix": vix, "vix_term": vix / vix3m,  # 1 초과 = 단기 공포가 장기보다 큼(스트레스)
        "hy_spread": hy, "hy_spread_chg_20d": hy - hy.shift(20),
        "t10y2y": _close("fred", "T10Y2Y", "value"), "nfci": _close("fred", "NFCI", "value"),
        "usdkrw": usdkrw, "usdkrw_chg_20d": usdkrw.pct_change(20),
        "dxy": _close("macro_1d", "DXY"), "fng": _close("crypto_fear_greed", "FNG", "value"),
    })

    # 펀딩비: 유니버스 전체 평균, 7일 평균을 연율로 (8시간마다 3회/일)
    fund = []
    for sym in store.load_universe("binance", max_age_days=10**6).get("symbol", []):
        f = store.read("binance_um_funding", sym)
        if not f.empty:
            fund.append(f.set_index("ts")["funding_rate"].resample("1D").sum().rename(sym))
    crypto = pd.DataFrame(index=btc.index)
    if fund:
        daily_fund = pd.concat(fund, axis=1).mean(axis=1)
        crypto["funding_avg_7d_ann"] = daily_fund.rolling(7).mean() * 365
    # 김치 프리미엄: 업비트 BTC 원화가 / (바이낸스 BTC 달러가 × 환율) - 1  (USDT ≈ USD 가정)
    up = _daily_from_1m("upbit_1m", "KRW-BTC")
    if not up.empty:
        crypto["kimchi_premium_btc"] = up / (btc * usdkrw.reindex(btc.index).ffill()) - 1

    df = pd.concat(parts + [macro, crypto], axis=1).sort_index()
    df = df[df.index >= "2015-01-01"]
    for p in ("spx", "kospi", "btc"):  # 실제 관측일 표시 (아래 ffill로 채운 날과 구분)
        df[f"{p}_observed"] = df[f"{p}_close"].notna()
    df = df.ffill(limit=5)  # 시장별 휴일 차이 메우기 (최대 5일)
    return _labels(df)


def _labels(df: pd.DataFrame) -> pd.DataFrame:
    for p in ("spx", "kospi", "btc"):
        up = df[f"{p}_above_200d"].astype(bool) & (df[f"{p}_ret_20d"] > 0)
        down = ~df[f"{p}_above_200d"].astype(bool) & (df[f"{p}_ret_20d"] < 0)
        df[f"{p}_trend"] = np.select([up, down], ["상승", "하락"], "혼조")
        df[f"{p}_vol"] = np.select([df[f"{p}_rv_pctile"] < 0.3, df[f"{p}_rv_pctile"] > 0.8], ["낮음", "높음"], "보통")
    # 위험선호 점수: 위험회피 신호 개수 (0~4)
    off = (
        (df["vix_term"] > 1).astype(int)
        + (df["vix"] > 25).astype(int)
        + (df["hy_spread_chg_20d"] > 0.5).astype(int)
        + (~df["spx_above_200d"].astype(bool)).astype(int)
    )
    df["risk_off_signals"] = off
    df["risk_regime"] = np.select([off >= 3, off >= 1], ["위험회피", "중립"], "위험선호")
    return df


def save(df: pd.DataFrame) -> None:
    out = df.reset_index().rename(columns={"index": "ts"})
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    import shutil
    shutil.rmtree(store.DATA_DIR / "regime_1d", ignore_errors=True)  # 매번 전체 재계산
    store.write("regime_1d", "ALL", out)


def _fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return {"pct": f"{v:+.1%}", "pct0": f"{v:.1%}", "num": f"{v:,.2f}", "int": f"{v:,.0f}"}[kind]


def brief(df: pd.DataFrame, health_line: str = "") -> str:
    last = df[df["spx_observed"]].iloc[-1]
    btc = df[df["btc_observed"]].iloc[-1]
    kos = df[df["kospi_observed"]].iloc[-1]
    lines = [
        f"📊 장 전 브리핑 — {pd.Timestamp.now(tz='Asia/Seoul'):%Y-%m-%d (%a)}",
        f"(확정 종가 기준: 미국 {last.name:%m-%d} · 한국 {kos.name:%m-%d} · 코인 {btc.name:%m-%d} UTC)",
        "",
        f"■ 위험 국면: {last['risk_regime']} (위험회피 신호 {int(last['risk_off_signals'])}/4)",
        f"  VIX {_fmt(last['vix'])} · 기간구조 {_fmt(last['vix_term'])} {'(역전·스트레스)' if last['vix_term'] > 1 else ''}",
        f"  하이일드 스프레드 {_fmt(last['hy_spread'])}%p (20일 {_fmt(last['hy_spread_chg_20d'])}) · 10Y-2Y {_fmt(last['t10y2y'])}",
        "",
        f"■ 미국 S&P500 {_fmt(last['spx_close'], 'int')} · 추세 {last['spx_trend']} · 변동성 {last['spx_vol']}",
        f"  20일 {_fmt(last['spx_ret_20d'], 'pct')} · 실현변동성 {_fmt(last['spx_rv_20d'], 'pct0')}",
        f"■ 한국 KOSPI {_fmt(kos['kospi_close'], 'num')} · 추세 {kos['kospi_trend']} · 변동성 {kos['kospi_vol']}",
        f"  20일 {_fmt(kos['kospi_ret_20d'], 'pct')} · 원/달러 {_fmt(kos['usdkrw'], 'num')} (20일 {_fmt(kos['usdkrw_chg_20d'], 'pct')})",
        f"■ 코인 BTC {_fmt(btc['btc_close'], 'int')} · 추세 {btc['btc_trend']} · 변동성 {btc['btc_vol']}",
        f"  20일 {_fmt(btc['btc_ret_20d'], 'pct')} · 펀딩비(7일, 연율) {_fmt(btc.get('funding_avg_7d_ann'), 'pct')}"
        f" · 김프 {_fmt(btc.get('kimchi_premium_btc'), 'pct')} · 공포탐욕 {_fmt(btc.get('fng'), 'int')}",
    ]
    if health_line:
        lines += ["", health_line]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog="autotrader.intel.market")
    p.add_argument("--notify", action="store_true", help="텔레그램으로 보내기")
    a = p.parse_args()
    from ..monitoring import health, notify

    df = compute()
    save(df)
    issues = health.check()
    text = brief(df, health.report(issues))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{pd.Timestamp.now(tz='Asia/Seoul'):%Y-%m-%d}.md").write_text(text, encoding="utf-8")
    print(text)
    if a.notify:
        notify.send(text)


if __name__ == "__main__":
    main()
