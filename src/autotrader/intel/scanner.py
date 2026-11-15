"""시장정보팀: 장 전 차트 스캐너 (day-trader-skill의 trader-scanner에서 가져온 아이디어).

전 종목 일봉에 정해진 규칙(셋업)을 돌려 "규칙에 맞은 종목" 목록을 만든다.
- 스캔 결과는 매수 추천이 아니다. "규칙에 맞음"일 뿐이며 수익을 약속하지 않는다.
- 셋업은 순수 함수(setups)라 과거 어느 날짜에도 그대로 다시 돌릴 수 있다
  → lab에서 "이 셋업이 실제로 통하는가"를 검증할 때 같은 코드를 쓴다.
- 매일의 결과는 scan_hits 데이터셋에 쌓아 둔다 (그날 무엇이 보였는지에 대한 기록).

시장별 입력: KR = krx_1d(전 종목, 생존편향 없음), US = us_1d, 코인 = binance_spot_1m → 일봉.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import store

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "scan"
LOOKBACK_DAYS = 120  # 셋업 계산에 필요한 과거 (RSI·20일선·플래그 30일 + 여유)

# 시장별 유동성 하한: 최근 20일 평균 거래대금, 최소 가격
LIQUIDITY = {
    "kr": {"min_value": 5e8, "min_price": 1000},     # 5억 원
    "us": {"min_value": 2e7, "min_price": 5},        # 2천만 달러
    "crypto": {"min_value": 1e7, "min_price": 0},    # 1천만 달러
}

SETUPS = ("breakout_20d", "bull_flag", "reclaim_ma20", "oversold", "oversold_bounce")


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def setups(bars: pd.DataFrame) -> dict[str, bool]:
    """한 종목의 일봉(날짜 오름차순, 마지막 행 = 판단 기준일)에 대해 셋업별 해당 여부.
    마지막 행까지 확정된 봉만 넣어야 한다 (미래 정보 없음)."""
    c, h = bars["close"], bars["high"]
    if len(c) < 31:
        return {s: False for s in SETUPS}
    ma20 = c.rolling(20).mean()
    rsi = _rsi(c)
    last5, prior10 = c.iloc[-5:], c.iloc[-15:-5]
    return {
        # 마지막 종가가 직전 20거래일 고가를 모두 넘음
        "breakout_20d": bool(c.iloc[-1] > h.iloc[-21:-1].max()),
        # 직전 10일 +10% 넘게 오른 뒤 최근 5일 종가가 5% 범위 안에서 횡보
        "bull_flag": bool(prior10.iloc[-1] / prior10.iloc[0] - 1 > 0.10
                          and last5.max() / last5.min() - 1 < 0.05),
        # 어제는 20일선 아래, 오늘은 위
        "reclaim_ma20": bool(c.iloc[-2] < ma20.iloc[-2] and c.iloc[-1] > ma20.iloc[-1]),
        "oversold": bool(rsi.iloc[-1] < 30),
        # 어제 RSI 30 미만이었고 오늘 상승 마감
        "oversold_bounce": bool(rsi.iloc[-2] < 30 and c.iloc[-1] > c.iloc[-2]),
    }


def _load_kr(start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    d = store.read("krx_1d", "ALL", start=start)
    d = d[d["close"] > 0]  # 거래정지일(시가 0 등) 제외
    return {s: g.drop(columns="symbol") for s, g in d.groupby("symbol")}


def _load_per_symbol(dataset: str, start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    out = {}
    folder = store.DATA_DIR / dataset
    for sym_dir in folder.iterdir() if folder.exists() else []:
        d = store.read(dataset, sym_dir.name, start=start)
        if d.empty:
            continue
        d["value"] = d["close"] * d["volume"]
        out[sym_dir.name] = d
    return out


def _load_crypto(start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    out = {}
    today = pd.Timestamp.now(tz="UTC").normalize()
    for sym, d in _load_per_symbol("binance_spot_1m", start).items():
        g = d.set_index("ts").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "value": "sum"}).dropna()
        out[sym] = g[g.index < today].reset_index()  # 끝나지 않은 오늘 봉 제외
    return out


LOADERS = {"kr": _load_kr, "us": lambda s: _load_per_symbol("us_1d", s), "crypto": _load_crypto}


def scan(market: str, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """asof(포함) 이전 확정 일봉으로 스캔. 반환: symbol, setup, close, value_20d, asof (거래대금 큰 순)."""
    asof = pd.Timestamp.now(tz="UTC") if asof is None else store._utc(asof)
    data = LOADERS[market](asof - pd.Timedelta(days=int(LOOKBACK_DAYS * 1.6)))
    liq = LIQUIDITY[market]
    rows = []
    for sym, d in data.items():
        d = d[d["ts"] <= asof].sort_values("ts")
        if len(d) < 31:
            continue
        v20 = d["value"].iloc[-20:].mean()
        if v20 < liq["min_value"] or d["close"].iloc[-1] < liq["min_price"]:
            continue
        for s, hit in setups(d).items():
            if hit:
                rows.append({"ts": d["ts"].iloc[-1], "market": market, "symbol": sym,
                             "name": d["name"].iloc[-1] if "name" in d else sym,
                             "setup": s, "close": float(d["close"].iloc[-1]), "value_20d": float(v20)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.attrs["scanned"] = len(data)
    return out.sort_values("value_20d", ascending=False).reset_index(drop=True)


def run(markets=("kr", "us", "crypto")) -> dict[str, pd.DataFrame]:
    res = {}
    for m in markets:
        hits = scan(m)
        res[m] = hits
        if not hits.empty:
            store.write("scan_hits", "ALL", hits, key=["ts", "market", "symbol", "setup"])
    return res


def summary(res: dict[str, pd.DataFrame], top: int = 3) -> str:
    """브리핑용 요약. 개수 + 셋업별 거래대금 상위 몇 개."""
    names = {"kr": "한국", "us": "미국", "crypto": "코인"}
    lines = ["■ 스캐너 (규칙에 맞은 종목일 뿐, 매수 신호 아님)"]
    for m, hits in res.items():
        if hits.empty:
            lines.append(f"  {names[m]}: 해당 없음")
            continue
        asof = hits["ts"].max().tz_convert("Asia/Seoul" if m == "kr" else "UTC")
        counts = hits["setup"].value_counts()
        lines.append(f"  {names[m]} ({asof:%m-%d} 종가 기준, {len(hits)}건)")
        for s in SETUPS:
            if s in counts:
                ex = ", ".join(hits.loc[hits["setup"] == s, "name"].head(top).astype(str))
                lines.append(f"    {s} {counts[s]}: {ex}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog="autotrader.intel.scanner")
    p.add_argument("--markets", nargs="*", default=["kr", "us", "crypto"])
    a = p.parse_args()
    res = run(a.markets)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    day = f"{pd.Timestamp.now(tz='Asia/Seoul'):%Y-%m-%d}"
    payload = {m: json.loads(h.to_json(orient="records", date_format="iso")) for m, h in res.items()}
    (REPORT_DIR / f"{day}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summary(res))


if __name__ == "__main__":
    main()
