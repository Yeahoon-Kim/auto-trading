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
START = "2010-01-01"  # macro_1d가 2010-01부터 있음 (2026-11-18 이전에는 2015-01-01)
RISK_OFF_N = 4  # 위험회피 신호 정의 개수


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
    ma200 = px.rolling(200).mean()
    out = pd.DataFrame({
        f"{prefix}_close": px,
        # 200일선이 아직 없는 워밍업 구간은 NaN (False로 두면 '200일선 아래'로 잘못 읽힌다)
        f"{prefix}_above_200d": (px > ma200).astype(object).where(ma200.notna()),
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
    df = df[df.index >= START]  # 지표는 자르기 전 전체 원천으로 계산 → 원천이 앞선 만큼 워밍업이 짧아진다
    for p in ("spx", "kospi", "btc"):  # 실제 관측일 표시 (아래 ffill로 채운 날과 구분)
        df[f"{p}_observed"] = df[f"{p}_close"].notna()
    df = df.ffill(limit=5)  # 시장별 휴일 차이 메우기 (최대 5일)
    return _labels(df)


def _labels(df: pd.DataFrame) -> pd.DataFrame:
    """추세·변동성·위험 국면 라벨. 판단에 필요한 값이 없는 날(워밍업·원천 공백)은 라벨을 비운다(None)."""
    for p in ("spx", "kospi", "btc"):
        above, ret = df[f"{p}_above_200d"], df[f"{p}_ret_20d"]
        ok = above.notna() & ret.notna()
        up = ok & above.eq(True) & (ret > 0)
        down = ok & above.eq(False) & (ret < 0)
        df[f"{p}_trend"] = np.select([up, down, ok], ["상승", "하락", "혼조"], None)
        pct = df[f"{p}_rv_pctile"]
        df[f"{p}_vol"] = np.select([pct < 0.3, pct > 0.8, pct.notna()], ["낮음", "높음", "보통"], None)
    # 위험회피 신호 (정의 4개). 원천이 없는 날 그 신호는 '없음'이 아니라 '셀 수 없음'으로 본다:
    #   risk_off_available = 그날 셀 수 있는 신호 개수, risk_off_signals = 그중 켜진 개수.
    #   하나도 셀 수 없으면 둘 다 NaN·국면 None. (예: HY 스프레드는 FRED에 2023-11부터만 있어
    #   그 전에는 available이 최대 3.) 국면 기준(3개 이상 위험회피, 1개 이상 중립)은 개수 그대로 쓴다.
    sig = pd.DataFrame({
        "vix_term": df["vix_term"] > 1,
        "vix": df["vix"] > 25,
        "hy": df["hy_spread_chg_20d"] > 0.5,
        "spx_below_200d": df["spx_above_200d"].eq(False),
    })
    avail = pd.DataFrame({
        "vix_term": df["vix_term"].notna(), "vix": df["vix"].notna(),
        "hy": df["hy_spread_chg_20d"].notna(), "spx_below_200d": df["spx_above_200d"].notna(),
    })
    n = avail.sum(axis=1)
    off = (sig & avail).sum(axis=1).astype(float).where(n > 0)
    df["risk_off_available"] = n.astype(float).where(n > 0)
    df["risk_off_signals"] = off
    df["risk_regime"] = np.select([off >= 3, off >= 1, off.notna()], ["위험회피", "중립", "위험선호"], None)
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
        f"■ 위험 국면: {last['risk_regime'] or '-'} (위험회피 신호 {_fmt(last['risk_off_signals'], 'int')}/{RISK_OFF_N}"
        + ("" if last.get("risk_off_available") == RISK_OFF_N
           else f", 셀 수 있는 신호 {_fmt(last.get('risk_off_available'), 'int')}개") + ")",
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
    p.add_argument("--no-llm", action="store_true", help="해석 문단(LLM) 없이")
    p.add_argument("--no-scan", action="store_true", help="스캐너 없이")
    a = p.parse_args()
    from ..monitoring import health, notify
    from . import analyst, scanner

    df = compute()
    save(df)
    issues = health.check()
    text = brief(df)
    scan_text = ""
    if not a.no_scan:
        try:
            scan_text = scanner.summary(scanner.run())
        except Exception as e:  # noqa: BLE001 — 스캐너 실패가 브리핑을 막지 않게
            scan_text = f"■ 스캐너 실패: {type(e).__name__}: {e}"
        text += "\n\n" + scan_text
    if not a.no_llm:
        note = analyst.commentary(df, text, scan_text)
        text += "\n\n■ 해석 (market-analyst · LLM 참고용)\n" + (note or "  (생성 실패 — logs/llm_calls.jsonl 확인)")
    hl = health.report(issues)
    if hl:
        text += "\n\n" + hl
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{pd.Timestamp.now(tz='Asia/Seoul'):%Y-%m-%d}.md").write_text(text, encoding="utf-8")
    print(text)
    if a.notify:
        notify.send(text)


if __name__ == "__main__":
    main()
