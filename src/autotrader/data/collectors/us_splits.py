"""미국 주식 분할 이력 (us_splits). us_1d 가격(분할 소급 반영)과 EDGAR 주식수(공시 당시 값)를 같은 기준으로 맞출 때 쓴다.

- us_splits (symbol=ALL): 한 행 = (종목, 분할 효력일). us_1d 유니버스 + 벤치마크 ETF, 2010~.
      열: ts(효력일 = 분할 반영 첫 거래일, NY 자정의 UTC), symbol, ratio(새 주식수/옛 주식수 — 4:1 분할 4.0,
          1:10 병합 0.1), source(yahoo), alpaca_date(Alpaca가 준 발표일 추정, 2020~ 일부만, 틀린 경우 있음 — 참고용), alpaca_ratio(교차 확인용).
- 소스 비교 (2026-11-17 확인):
      Yahoo(yfinance Ticker.splits) — 키 없음, 2010 이전부터 전 기간, 발표일 없음. us_1d와 같은 소스라 가격 기준이 일치 → 기본.
      Alpaca v1 corporate-actions(forward/reverse_split) — 2016 이전은 비어 있음(2014 AAPL 7:1 없음), 발표일 없음.
      Alpaca v2 announcements(ca_types=split) — 2020~. declaration_date는 없고 effective_date가 발표일과 맞아
          (NVDA 2024-05-22) 경우도 있지만 틀린 경우도 있어(GOOGL 2022 분할 2021-12-27, 실제 발표 2022-02-01)
          참고용 alpaca_date로만 남긴다. 무료 소스에서 믿을 만한 분할 발표일은 찾지 못했다.
      us_1d의 adj_close/close 비율 — Yahoo close가 이미 분할 반영이라 비율에 분할이 안 나타난다(배당만) → 못 씀.
      EDGAR — 분할을 구조화된 값으로 주지 않는다(StockholdersEquityNoteStockSplitConversionRatio1 드묾).
- 갱신: 실행마다 전 종목 Ticker.splits를 다시 받는다(종목당 1회, 약 520회, 병렬). Yahoo가 과거 분할을 고치면 그대로 덮는다.
한계: 현재 구성종목만(생존편향). Yahoo 비율이 소수(예: GOOGL 2014 1.998 — C주 배당을 분할로 표시)인 경우 그대로 둔다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from .. import store
from .common import config, get_json, logger

NY = "America/New_York"
DS = "us_splits"
KEY = ["symbol", "ts"]


def parse_yahoo(symbol: str, s: pd.Series, start: str) -> pd.DataFrame:
    """yfinance Ticker.splits(날짜 → 비율) → us_splits 행."""
    s = s[s > 0] if s is not None else pd.Series(dtype=float)
    if s.empty:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(s.index)
    idx = (idx.tz_convert(NY) if idx.tz is not None else idx.tz_localize(NY)).normalize()
    d = pd.DataFrame({"ts": idx.tz_convert("UTC"), "symbol": symbol, "ratio": s.values.astype(float),
                      "source": "yahoo"})
    return d[d["ts"] >= pd.Timestamp(start, tz=NY)].reset_index(drop=True)


def parse_alpaca_ann(rows: list[dict]) -> pd.DataFrame:
    """Alpaca v2 announcements(ca_type=split) → (symbol, ex_date, alpaca_date, alpaca_ratio).
    분할 행은 initiating_symbol이 비고 target_symbol에 종목이 있으며 declaration_date가 없다(2026-11-17 확인).
    effective_date가 발표일과 맞는 경우가 있어(NVDA 10:1 → 2024-05-22) 효력일보다 앞설 때만 alpaca_date로 남긴다.
    틀린 경우도 있다(GOOGL 20:1 → 2021-12-27, 실제 발표 2022-02-01) → 참고용."""
    if not rows:
        return pd.DataFrame(columns=["symbol", "ex_date", "alpaca_date", "alpaca_ratio"])
    d = pd.DataFrame(rows)
    for c in ("ex_date", "declaration_date", "effective_date", "old_rate", "new_rate", "initiating_symbol",
              "target_symbol"):
        if c not in d:
            d[c] = None
    sym = d["initiating_symbol"].replace("", None).fillna(d["target_symbol"])
    ex = pd.to_datetime(d["ex_date"], errors="coerce")
    eff = pd.to_datetime(d["effective_date"], errors="coerce")
    decl = pd.to_datetime(d["declaration_date"], errors="coerce").fillna(eff.where(eff < ex))
    out = pd.DataFrame({
        "symbol": sym.astype(str).str.replace(".", "-", regex=False),
        "ex_date": ex,
        "alpaca_date": decl,
        "alpaca_ratio": pd.to_numeric(d["new_rate"], errors="coerce") / pd.to_numeric(d["old_rate"], errors="coerce"),
    })
    return out.dropna(subset=["ex_date"])


def attach_declared(sp: pd.DataFrame, ann: pd.DataFrame, tol_days: int = 5) -> pd.DataFrame:
    """Yahoo 분할 행에 Alpaca 날짜(참고용 발표일 추정)를 붙인다 (같은 종목, 효력일 ±tol_days)."""
    sp = sp.copy()
    sp["alpaca_date"], sp["alpaca_ratio"] = pd.NaT, float("nan")
    if sp.empty or ann.empty:
        return sp
    day = sp["ts"].dt.tz_convert(NY).dt.tz_localize(None)
    for i, r in sp.iterrows():
        a = ann[(ann["symbol"] == r["symbol"]) & ((ann["ex_date"] - day[i]).abs() <= pd.Timedelta(days=tol_days))]
        if not a.empty:
            sp.at[i, "alpaca_date"] = a["alpaca_date"].iloc[0]
            sp.at[i, "alpaca_ratio"] = a["alpaca_ratio"].iloc[0]
    return sp


def _alpaca_announcements(headers: dict, start: str) -> pd.DataFrame:
    import httpx

    from .dividends import windows
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    parts, client = [], httpx.Client(timeout=60)
    for s, e in windows(pd.Timestamp(start), today + pd.Timedelta(days=60), 85):  # API 창 한도 90일
        rows = get_json(client, "https://paper-api.alpaca.markets/v2/corporate_actions/announcements",
                        {"ca_types": "split", "since": f"{s:%Y-%m-%d}", "until": f"{e:%Y-%m-%d}",
                         "date_type": "ex_date"}, headers=headers)
        parts.append(parse_alpaca_ann(rows))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_yahoo(tickers: list[str], start: str, workers: int = 8) -> tuple[pd.DataFrame, list[str]]:
    def one(t: str):
        try:
            return t, parse_yahoo(t, yf.Ticker(t).splits, start), None
        except Exception as e:  # 한 종목 실패가 전체를 막지 않게
            return t, pd.DataFrame(), str(e)[:200]

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, tickers))
    fails = [t for t, _, err in res if err]
    parts = [d for _, d, _ in res if not d.empty]
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), fails


def run(symbols: list[str] | None = None) -> None:
    from .us_alpaca import keys
    from .us_stock import universe

    cfg = config()["us"]
    start = cfg.get("start_1d", "2010-01-01")
    tickers = symbols or universe()["yahoo"].tolist() + [t for t in cfg.get("benchmarks_1d", [])]
    try:
        sp, fails = fetch_yahoo(sorted(set(tickers)), start)
        headers = keys()
        if headers and not sp.empty:
            try:
                sp = attach_declared(sp, _alpaca_announcements(headers, "2020-01-01"))
            except Exception as e:  # Alpaca 날짜는 보조 정보 — 실패해도 분할 이력은 저장
                logger.warning(f"{DS}: Alpaca 날짜 실패 ({e})")
                sp = attach_declared(sp, pd.DataFrame())
        else:
            sp = attach_declared(sp, pd.DataFrame())
        n = store.write(DS, "ALL", sp, key=KEY)
        store.set_state(DS, "ALL", backfill_done=not fails, status="ok" if not fails else "error",
                        error=f"Yahoo 실패 {len(fails)}개 {fails[:5]}" if fails else None)
        logger.info(f"{DS}: {len(tickers)}종목 중 분할 있는 {sp['symbol'].nunique() if n else 0}종목, {n}행, "
                    f"Alpaca 날짜 {int(sp['alpaca_date'].notna().sum()) if n else 0}건, 실패 {len(fails)}")
    except Exception as e:
        store.set_state(DS, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{DS} 실패: {e}")
