"""백테스트용 데이터: 시장별 시세를 '날짜 × 종목' 표(Panel)로 읽는다.

    panel = load("kr_stock", start="2015-01-01")          # 개발 구간만 (잠금 구간은 잘라냄)
    panel.close, panel.open, panel["volume"]              # index: 시장 현지 날짜(tz 없음), columns: 종목

시장별 원천과 처리
- kr_stock  : krx_1d (KRX 전 종목, 상장폐지 포함 → 생존편향 없음). 가격은 kr_adj_factor로 수정주가 환산.
              수정계수가 없는 종목·날짜는 원본가 그대로 (분할·증자 날 가짜 수익이 생길 수 있음 → meta에 개수 기록).
              raw_close(원본 종가), market_cap, value(거래대금)도 필요하면 fields로 받는다.
- us_stock  : us_1d (현재 수집 종목 = 지금 살아 있는 종목 → 생존편향 있음). adj_close/close 비율로 OHLC 수정.
- crypto_spot / crypto_futures : binance_spot_1m / binance_um_1m을 timeframe(1d, 1h …)으로 묶음. UTC 기준.
              선물 펀딩비는 엔진에 반영하지 않는다 (필요하면 binance_um_funding을 lab에서 직접 읽어 신호·비용에 넣는다).

잠금 구간: 실행일 기준 최근 6개월. lockbox=False(기본)면 그 이전까지만 읽는다.
"""

from __future__ import annotations

import pandas as pd
import yaml

from ..data import store

LOCKBOX_MONTHS = 6
TZ = {"kr_stock": "Asia/Seoul", "us_stock": "America/New_York", "crypto_spot": "UTC", "crypto_futures": "UTC"}
YEAR_BARS = {"kr_stock": 252, "us_stock": 252, "crypto_spot": 365, "crypto_futures": 365}
SURVIVORSHIP = {
    "kr_stock": "없음 — krx_1d는 상장폐지 종목 포함",
    "us_stock": "있음 — us_1d는 현재 수집 종목(살아남은 종목)만 있음. 결과가 과대평가될 수 있음",
    "crypto_spot": "있음 — 현재 수집 중인 코인만 있음 (상장폐지·급락 코인 빠짐)",
    "crypto_futures": "있음 — 현재 수집 중인 코인만 있음 (상장폐지·급락 코인 빠짐)",
}


class Panel(dict):
    """필드 이름 → 날짜×종목 DataFrame. panel.close처럼 속성으로도 읽는다."""

    def __init__(self, frames: dict[str, pd.DataFrame], market: str, timeframe: str = "1d",
                 meta: dict | None = None):
        super().__init__(frames)
        self.market, self.timeframe = market, timeframe
        self.meta = meta or {}

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @property
    def index(self) -> pd.DatetimeIndex:
        return self["close"].index

    def slice(self, start=None, end=None) -> Panel:
        """[start, end) 구간. 원본은 그대로 둔다."""
        idx = self.index
        m = pd.Series(True, index=idx)
        if start is not None:
            m &= idx >= pd.Timestamp(start)
        if end is not None:
            m &= idx < pd.Timestamp(end)
        return Panel({k: v.loc[m.values] for k, v in self.items()}, self.market, self.timeframe, self.meta)

    @property
    def bars_per_year(self) -> float:
        base = YEAR_BARS[self.market]
        tf = self.timeframe
        if tf.endswith("d"):
            return base / int(tf[:-1] or 1)
        hours = {"crypto_spot": 24, "crypto_futures": 24}.get(self.market, 6.5)
        if tf.endswith("h"):
            return base * hours / int(tf[:-1])
        if tf.endswith("m"):
            return base * hours * 60 / int(tf[:-1])
        raise ValueError(f"timeframe {tf}")


def dev_end(today=None) -> pd.Timestamp:
    """개발 구간 끝 = 잠금 구간 시작 (실행일 기준 6개월 전, 날짜 단위)."""
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    return (t.normalize() - pd.DateOffset(months=LOCKBOX_MONTHS)).tz_localize(None)


def load(market: str, symbols: list[str] | str | None = None, start=None, end=None,
         timeframe: str = "1d", fields: tuple[str, ...] = ("open", "close"),
         lockbox: bool = False, today=None, currency: str | None = None,
         dividend_tax: float | None = None) -> Panel:
    """symbols: None/"ALL" = 전체. kr_stock은 "KOSPI"/"KOSDAQ"도 받는다 (그날 소속 시장 기준).
    lockbox=False면 end를 dev_end()로 자른다.
    us_stock 전용:
      currency="KRW"  가격을 그날 원/달러(ECOS)로 환산 → 수익에 환율 변동 포함. close_usd·fx 필드가 같이 온다.
      dividend_tax    배당 재투자 때 떼는 비율 (기본: settings us_stock.dividend_withholding_pct, 0이면 전액 재투자)."""
    cut = dev_end(today)
    if not lockbox:
        end = min(pd.Timestamp(end), cut) if end is not None else cut
    fields = tuple(dict.fromkeys(("open", "close") + tuple(fields)))
    if market == "kr_stock":
        long, meta = _kr(symbols, start, end, fields)
    elif market == "us_stock":
        long, meta = _us(symbols, start, end, fields, currency, dividend_tax)
        if currency == "KRW":
            fields = tuple(dict.fromkeys(fields + ("close_usd", "fx")))
    elif market in ("crypto_spot", "crypto_futures"):
        long, meta = _crypto(market, symbols, start, end, fields, timeframe)
    else:
        raise ValueError(f"지원하지 않는 시장: {market}")
    if long.empty:
        raise ValueError(f"{market} 데이터 없음 (symbols={symbols}, {start}~{end})")
    frames = {f: long.pivot(index="date", columns="symbol", values=f).sort_index() for f in fields if f in long}
    meta |= {"dev_end": str(cut.date()), "lockbox": lockbox, "survivorship": SURVIVORSHIP[market],
             "start": str(frames["close"].index[0]), "end": str(frames["close"].index[-1])}
    return Panel(frames, market, timeframe, meta)


def _bounds(start, end, tz: str) -> str:
    """현지 날짜 경계를 UTC 조건으로."""
    w = []
    if start is not None:
        w.append(f"CAST(ts AS TIMESTAMPTZ) >= TIMESTAMPTZ '{pd.Timestamp(start).tz_localize(tz).tz_convert('UTC')}'")
    if end is not None:
        w.append(f"CAST(ts AS TIMESTAMPTZ) < TIMESTAMPTZ '{pd.Timestamp(end).tz_localize(tz).tz_convert('UTC')}'")
    return " AND ".join(w) or "TRUE"


def _sym_filter(symbols, col: str = "symbol") -> str:
    if symbols is None or symbols == "ALL":
        return "TRUE"
    if isinstance(symbols, str):
        symbols = [symbols]
    return f"{col} IN ({', '.join(repr(s) for s in symbols)})"


def _files(dataset: str, symbols=None, start=None, end=None) -> str:
    """필요한 종목 폴더·기간 파일만 골라 duckdb 목록으로 (전체 glob보다 훨씬 빠르다).
    파일 이름은 YYYY 또는 YYYY-MM. 경계 하루 오차는 _bounds가 다시 거른다."""
    root = store.DATA_DIR / dataset
    if symbols is None or symbols == "ALL" or symbols in ("KOSPI", "KOSDAQ"):
        dirs = [p for p in root.iterdir() if p.is_dir()]
    else:
        dirs = [root / store._safe(s) for s in ([symbols] if isinstance(symbols, str) else symbols)]
    lo = str(pd.Timestamp(start).year - 1) if start is not None else ""
    hi = str(pd.Timestamp(end).year + 1) if end is not None else "9999"
    files = [f.as_posix() for d in dirs if d.exists() for f in d.glob("*.parquet") if lo <= f.stem[:4] <= hi]
    return "[" + ", ".join(repr(f) for f in files) + "]" if files else ""


def _local_date(ts: pd.Series, tz: str) -> pd.Series:
    if ts.dt.tz is None:              # duckdb가 TIMESTAMPTZ_NS를 tz 없이 돌려주는 경우 (값은 UTC)
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()


def _kr(symbols, start, end, fields):
    tz = TZ["kr_stock"]
    if symbols in ("KOSPI", "KOSDAQ"):
        sym_where = f"market = '{symbols}'"
    else:
        sym_where = _sym_filter(symbols)
    extra = [f for f in ("high", "low", "volume", "value", "market_cap", "shares") if f in fields]
    cols = ", ".join(["ts", "symbol", "open", "close"] + extra)
    src = _files("krx_1d", "ALL", start, end)
    if not src:
        return pd.DataFrame(), {}
    px = store.query(f"SELECT {cols} FROM read_parquet({src}) WHERE {_bounds(start, end, tz)} AND {sym_where}")
    if px.empty:
        return px, {}
    px["date"] = _local_date(px["ts"], tz)
    fsrc = _files("kr_adj_factor", sorted(px["symbol"].unique()) if symbols not in (None, "ALL") else "ALL",
                  start, end)
    fac = store.query(
        f"SELECT ts, regexp_extract(filename, '[\\\\/]([^\\\\/]+)[\\\\/][^\\\\/]+$', 1) AS symbol, factor "
        f"FROM read_parquet({fsrc}, filename=true) WHERE {_bounds(start, end, tz)}") if fsrc else pd.DataFrame()
    if not fac.empty:
        fac["date"] = _local_date(fac["ts"], tz)
        px = px.merge(fac[["date", "symbol", "factor"]].drop_duplicates(["date", "symbol"]),
                      on=["date", "symbol"], how="left")
    else:
        px["factor"] = float("nan")
    missing = int(px["factor"].isna().sum())
    f = px["factor"].fillna(1.0)
    px["raw_close"] = px["close"]
    for c in ("open", "close", "high", "low"):
        if c in px:
            px[c] = px[c].where(px[c] > 0) * f          # 거래정지일 가격 0 → 결측
    px = px.drop_duplicates(["date", "symbol"], keep="last")
    return px, {"source": "krx_1d + kr_adj_factor", "rows": len(px), "rows_without_adj_factor": missing}


def _settings() -> dict:
    return yaml.safe_load((store.ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def _net_dividend_ratio(g: pd.DataFrame, tax: float) -> pd.Series:
    """한 종목의 '배당을 (1-tax)만 재투자한' 수정 비율 (수정가 / 원본가).
    배당 = adj_close 수익률 - close 수익률 (배당락일에 생기는 차이). 마지막 날 비율 = adj_close/close."""
    c, a = g["close"].where(g["close"] > 0), g["adj_close"]
    r_p, r_t = c.pct_change(), a.pct_change()
    div = (r_t - r_p).clip(lower=0).fillna(0)
    net = (1 + r_p.fillna(0) + (1 - tax) * div).cumprod()
    scale = (a.iloc[-1] / c.iloc[-1]) / (net.iloc[-1] / c.iloc[-1]) if c.iloc[-1] else 1.0
    return net * scale / c


def usdkrw(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """날짜별 원/달러 (한국은행 ECOS 매매기준율). 같은 날 서울 고시값 = 뉴욕 장 시작 전에 알려진 값.
    원화 휴장일은 직전 값."""
    e = store.read("ecos", "usdkrw_ecos")
    if e.empty:
        raise ValueError("ecos/usdkrw_ecos 데이터 없음 — 원화 환산 불가")
    s = pd.Series(e["value"].values, index=_local_date(e["ts"], "UTC"))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    d = pd.DatetimeIndex(pd.unique(pd.Series(dates)))
    return s.reindex(s.index.union(d)).ffill().reindex(d)


def _us(symbols, start, end, fields, currency=None, dividend_tax=None):
    tz = TZ["us_stock"]
    src = _files("us_1d", symbols, start, end)
    if not src:
        return pd.DataFrame(), {}
    px = store.query(
        f"SELECT *, regexp_extract(filename, '[\\\\/]([^\\\\/]+)[\\\\/][^\\\\/]+$', 1) AS symbol "
        f"FROM read_parquet({src}, filename=true, union_by_name=true) WHERE {_bounds(start, end, tz)}")
    if px.empty:
        return px, {}
    px["date"] = _local_date(px["ts"], tz)
    px = px.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])
    if dividend_tax is None:
        dividend_tax = _settings()["markets"]["us_stock"].get("dividend_withholding_pct", 0) / 100
    if dividend_tax:
        ratio = pd.concat([_net_dividend_ratio(g, dividend_tax) for _, g in px.groupby("symbol")]).reindex(px.index)
    else:
        ratio = (px["adj_close"] / px["close"]).where(px["close"] > 0)
    px["raw_close"] = px["close"]
    for c in ("open", "close", "high", "low"):
        px[c] = px[c] * ratio
    meta = {"source": "us_1d (adj_close로 배당 재투자 수정)", "rows": len(px),
            "dividend_tax": dividend_tax, "currency": "USD"}
    if currency == "KRW":
        fx = usdkrw(px["date"])
        px["close_usd"] = px["close"]
        px["fx"] = px["date"].map(fx)
        for c in ("open", "close", "high", "low", "raw_close"):
            px[c] = px[c] * px["fx"]
        meta |= {"currency": "KRW", "fx": "ECOS 매매기준율 (같은 날, 휴장일은 직전 값). 환전 스프레드 미반영"}
    elif currency not in (None, "USD"):
        raise ValueError(f"지원하지 않는 통화: {currency}")
    return px, meta


def _crypto(market, symbols, start, end, fields, timeframe):
    ds = "binance_spot_1m" if market == "crypto_spot" else "binance_um_1m"
    unit = {"m": "minutes", "h": "hours", "d": "days"}[timeframe[-1]]
    n = int(timeframe[:-1] or 1)
    sym = "regexp_extract(filename, '[\\\\/]([^\\\\/]+)[\\\\/][^\\\\/]+$', 1)"
    src = _files(ds, symbols, start, end)
    if not src:
        return pd.DataFrame(), {}
    t = "CAST(ts AS TIMESTAMPTZ)"
    px = store.query(
        f"SELECT time_bucket(INTERVAL {n} {unit}, {t}) AS bucket, {sym} AS symbol, "
        f"arg_min(open, {t}) AS open, max(high) AS high, min(low) AS low, arg_max(close, {t}) AS close, "
        f"sum(volume) AS volume, count(*) AS n_bars "
        f"FROM read_parquet({src}, filename=true) WHERE {_bounds(start, end, 'UTC')} GROUP BY 1, 2")
    if px.empty:
        return px, {}
    px["date"] = px["bucket"].dt.tz_convert("UTC").dt.tz_localize(None)
    return px, {"source": f"{ds} → {timeframe}", "rows": len(px),
                "note": "선물 펀딩비 미반영" if market == "crypto_futures" else ""}
