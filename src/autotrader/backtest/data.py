"""백테스트용 데이터: 시장별 시세를 '날짜 × 종목' 표(Panel)로 읽는다.

    panel = load("kr_stock", start="2015-01-01")          # 개발 구간만 (잠금 구간은 잘라냄)
    panel.close, panel.open, panel["volume"]              # index: 시장 현지 날짜(tz 없음), columns: 종목

시장별 원천과 처리
- kr_stock  : krx_1d (KRX 전 종목, 상장폐지 포함 → 생존편향 없음). 가격은 kr_adj_factor로 수정주가 환산.
              수정계수가 없는 종목·날짜는 원본가 그대로 (분할·증자 날 가짜 수익이 생길 수 있음 → meta에 개수 기록).
              kr_adj_factor에는 배당이 없어서, kr_dividend_schedule 배당락일에 세후 배당금을 재투자한 것으로
              가격을 보정한다 (총수익, meta["dividends"]).
              raw_close(원본 종가), market_cap, value(거래대금)도 필요하면 fields로 받는다.
- us_stock  : us_1d (현재 수집 종목 = 지금 살아 있는 종목 → 생존편향 있음). adj_close/close 비율로 OHLC 수정.
- 배당 신호 필드 (kr_stock·us_stock, fields에 적으면): dps(그날 알 수 있는 주당배당금), div_yield(dps / raw_close).
              시점 규칙은 dividends() 참고. check_dividend_causality()가 데이터를 잘라 다시 만든 값과 비교한다.
- crypto_spot / crypto_futures : binance_spot_1m / binance_um_1m을 timeframe(1d, 1h …)으로 묶음. UTC 기준.
              선물 펀딩비는 엔진에 반영하지 않는다 (필요하면 binance_um_funding을 lab에서 직접 읽어 신호·비용에 넣는다).

잠금 구간: 실행일 기준 최근 6개월. lockbox=False(기본)면 그 이전까지만 읽는다.
"""

from __future__ import annotations

import numpy as np
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
        self.div_basis: pd.DataFrame | None = None   # 배당 필드를 다시 만드는 재료 (check_dividend_causality용)

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
         dividend_tax: float | None = None, neutralize_jumps: bool = True,
         reinvest_dividends: bool = True) -> Panel:
    """symbols: None/"ALL" = 전체. kr_stock은 "KOSPI"/"KOSDAQ"도 받는다 (그날 소속 시장 기준).
    lockbox=False면 end를 dev_end()로 자른다.
    dividend_tax: 배당 재투자 때 떼는 비율 (0이면 전액 재투자).
      us_stock 기본 settings us_stock.dividend_withholding_pct, kr_stock 기본 settings kr_stock.dividend_tax_pct
      (없으면 15.4% = 소득세 14% + 지방세 1.4%).
    us_stock 전용:
      currency="KRW"  가격을 그날 원/달러(ECOS)로 환산 → 수익에 환율 변동 포함. close_usd·fx 필드가 같이 온다.
    kr_stock 전용:
      reinvest_dividends 배당락일에 세후 배당 재투자 (총수익). False면 예전처럼 가격 수익만.
      neutralize_jumps 가격제한을 넘는 하루 변동을 0으로 (수정계수에 빠진 감자·병합 등). 정리매매 구간은 제외.
      항상 하는 처리: config/kr_exclude.yaml 종목 제외, 코드 재사용 분리, 주식수 기반 병합·감자 보정.
      panel["tradable"] (거래량 > 0)을 같이 준다 → 엔진이 거래정지일에는 매매하지 않는다."""
    cut = dev_end(today)
    if not lockbox:
        end = min(pd.Timestamp(end), cut) if end is not None else cut
    fields = tuple(dict.fromkeys(("open", "close") + tuple(fields)))
    want_div = [f for f in DIV_FIELDS if f in fields]
    if want_div and market not in ("kr_stock", "us_stock"):
        raise ValueError(f"배당 필드 {want_div}는 kr_stock·us_stock만 있다")
    if market == "kr_stock":
        long, meta = _kr(symbols, start, end, fields, neutralize_jumps, dividend_tax, reinvest_dividends)
        fields = fields + ("tradable",)
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
    basis = None
    if want_div:
        basis = _div_basis(long, market)
        long = long.merge(dividend_fields(market, basis), on=["date", "symbol"], how="left")
        meta["dividend_fields"] = DIV_TIMING[market]
    frames = {f: long.pivot(index="date", columns="symbol", values=f).sort_index() for f in fields if f in long}
    meta |= {"dev_end": str(cut.date()), "lockbox": lockbox, "survivorship": SURVIVORSHIP[market],
             "start": str(frames["close"].index[0]), "end": str(frames["close"].index[-1])}
    panel = Panel(frames, market, timeframe, meta)
    panel.div_basis = basis
    return panel


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


def _kr(symbols, start, end, fields, neutralize_jumps=True, dividend_tax=None, reinvest_dividends=True):
    tz = TZ["kr_stock"]
    if symbols in ("KOSPI", "KOSDAQ"):
        sym_where = f"market = '{symbols}'"
    else:
        sym_where = _sym_filter(symbols)
    extra = [f for f in ("high", "low", "value", "market_cap") if f in fields]
    # shares: 병합·감자 보정, volume: 거래정지 판정에 항상 필요
    cols = ", ".join(["ts", "symbol", "open", "close", "shares", "volume"] + extra)
    excl = _kr_exclude()
    if excl:
        sym_where += f" AND symbol NOT IN ({', '.join(repr(s) for s in sorted(excl))})"
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
    px = px.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
    reused = _split_reused_codes(px)
    missing = int(px["factor"].isna().sum())
    g = px["symbol"]
    fac = px["factor"].where(px["factor"] > 0)                 # 0 이하 계수는 오류
    nonpos = int((px["factor"] <= 0).sum())
    f = fac.groupby(g).ffill().groupby(g).bfill().fillna(1.0)  # 없는 날은 직전(없으면 다음) 계수, 그것도 없으면 1
    px["raw_close"] = px["close"]
    px["_f"] = f                                     # 배당 필드의 주식 기준 환산용
    for c in ("open", "close", "high", "low"):
        if c in px:
            px[c] = px[c].where(px[c] > 0) * f          # 거래정지일 가격 0 → 결측
    if dividend_tax is None:
        dividend_tax = _settings()["markets"]["kr_stock"].get("dividend_tax_pct", KR_DIVIDEND_TAX_PCT) / 100
    divs = _kr_reinvest(px, dividend_tax) if reinvest_dividends else {"reinvested": False}
    share_events = _fix_share_events(px)
    jumps = _neutralize_jumps(px) if neutralize_jumps else {"neutralized": 0}
    px["tradable"] = px["volume"] > 0          # 거래량 0 = 거래정지·무거래 (KRX는 종가를 직전 값으로 채운다)
    px = px.drop(columns=[c for c in ("shares", "volume") if c not in fields])
    src = "krx_1d + kr_adj_factor" + (" + kr_dividend_schedule (배당 재투자)" if divs["reinvested"] else "")
    return px, {"source": src, "rows": len(px), "excluded": len(excl),
                "untradable_rows": int((~px["tradable"]).sum()), "rows_without_adj_factor": missing,
                "adj_factor_nonpositive": nonpos, "reused_codes": reused, "share_events": share_events,
                "jumps": jumps, "dividends": divs}


KR_DIVIDEND_TAX_PCT = 15.4   # 배당소득세 14% + 지방소득세 1.4% (settings kr_stock.dividend_tax_pct가 없을 때)


def _kr_dividend_events(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """kr_dividend_schedule → (symbol, ex_date, dps). dps = 배당락일 기준 원본 주당 현금배당(원).
    ex_date가 비어 있으면 panel 거래일로 다시 계산. 같은 종목·배당락일 중복(0원 중복 행 등)은 큰 값 하나만."""
    s = store.read("kr_dividend_schedule", "ALL")
    if s.empty:
        return pd.DataFrame(columns=["symbol", "ex_date", "dps"])
    s = s[s["dps"] > 0].copy()
    na = s["ex_date"].isna()
    if na.any():
        from ..data.collectors.dividends import ex_dates
        rec = _local_date(s.loc[na, "ts"], TZ["kr_stock"])
        s.loc[na, "ex_date"] = ex_dates(rec, trading_days).to_numpy()
    s = s.dropna(subset=["ex_date"])
    return s.groupby(["symbol", "ex_date"], as_index=False)["dps"].max()


def _kr_reinvest(px: pd.DataFrame, tax: float) -> dict:
    """배당락일 수익에 세후 배당을 더한다 (px를 고친다, 종목·날짜순 전제).
    그날 총수익 = (종가 + 세후 배당) / 전일 종가 → 그날 이전 가격 전체에 종가 / (종가 + 세후 배당)을 곱한다.
    배당금과 종가는 둘 다 그날 원본 주식 기준이라 수정계수와 무관하다. 최근 가격은 그대로 (US adj_close와 같은 방식).
    배당락일에 행이 없으면(휴장 등) 7일 안의 다음 행에 붙인다. 코드 재사용으로 나뉜 앞 상장(코드@날짜)도 원래 코드로 맞춘다."""
    ev = _kr_dividend_events(pd.DatetimeIndex(px["date"].unique()))
    meta = {"reinvested": True, "tax": tax, "source": "kr_dividend_schedule ex_date·dps"}
    if ev.empty:
        return meta | {"events": 0, "skipped_no_price": 0}
    rows = px[["date", "symbol"]].assign(base=px["symbol"].str.split("@").str[0], row=np.arange(len(px)))
    ev = ev.rename(columns={"symbol": "base"})
    ev = ev[ev["base"].isin(set(rows["base"]))].sort_values("ex_date")
    m = pd.merge_asof(ev, rows.sort_values("date"), left_on="ex_date", right_on="date", by="base",
                      direction="forward", tolerance=pd.Timedelta(days=7)).dropna(subset=["row"])
    pos = m["row"].astype(int).to_numpy()
    first = (px.groupby("symbol").cumcount() == 0).to_numpy()
    m, pos = m[~first[pos]], pos[~first[pos]]            # 첫 행은 이전 가격이 없어 의미 없음
    raw = px["raw_close"].to_numpy(dtype=float)[pos]
    ok = raw > 0
    k = raw[ok] / (raw[ok] + (1 - tax) * m["dps"].to_numpy(dtype=float)[ok])
    step = np.ones(len(px))
    np.multiply.at(step, pos[ok], k)
    _scale_history(px, pd.Series(step, index=px.index))
    y = 1 / k - 1
    return meta | {"events": int(ok.sum()), "skipped_no_price": int((~ok).sum()),
                   "symbols": int(m.loc[ok, "symbol"].nunique()),
                   "mean_net_yield": float(y.mean()) if len(y) else 0.0}


KR_JUMP = 0.35        # 하루 가격제한(2015-06 이후 ±30%, 이전 ±15%)을 넘는 변동
DELIST_TAIL = 15      # 상장폐지 종목의 마지막 N거래일(정리매매)은 가격제한이 없어 실제 폭락으로 둔다
TAIL_UP_MAX = 1.0     # 단, 정리매매 구간이라도 하루 +100% 넘는 급등은 오류로 본다 (보수적: 이익을 지움)
REUSE_GAP_DAYS = 30   # 같은 코드의 행 사이가 이만큼(달력일) 비면 다른 상장으로 본다 (KRX는 거래정지일에도 행을 준다)
SHARE_EVENT = 1.5     # 상장주식수가 하루에 이 배수 이상 바뀌면 병합·감자·분할 후보
LIMIT_CHANGE = pd.Timestamp("2015-06-15")   # 가격제한폭 ±15% → ±30%
SHARE_LAG = 30        # 권리락 뒤 신주 상장까지 기다려 보는 거래일 수
SHARE_LAG_TOL = 0.02  # 그 경우 시가총액 기준 변동이 이 안이어야 보정


def _events(px: pd.DataFrame, flag: pd.Series) -> list[str]:
    """보정한 (종목, 날짜) 목록 — 감사·교차검증용. "036220:2016-05-04" 형식."""
    return (px.loc[flag, "symbol"] + ":" + px.loc[flag, "date"].dt.strftime("%Y-%m-%d")).tolist()


def _kr_exclude() -> set[str]:
    """config/kr_exclude.yaml: 투자회사(펀드)·데이터 오류 종목."""
    p = store.ROOT / "config" / "kr_exclude.yaml"
    return set((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("exclude") or {}) if p.exists() else set()


def _split_reused_codes(px: pd.DataFrame) -> dict:
    """한 코드가 상장폐지 뒤 다른 회사에 다시 쓰인 경우(예: 036220 인포피아 2016 폐지 → 오상헬스케어 2024 상장)
    행 사이 공백(REUSE_GAP_DAYS 초과)을 경계로 나눈다. 마지막 구간은 원래 코드를 그대로 쓰고,
    앞 구간은 "코드@마지막날짜"(예: 036220@20160504)로 바꾼다 → 엔진이 그 구간 끝에서 상장폐지 청산한다.
    px를 고친다 (종목·날짜순 전제)."""
    g = px["symbol"]
    new_seg = px["date"].groupby(g).diff() > pd.Timedelta(days=REUSE_GAP_DAYS)
    if not new_seg.any():
        return {"split": 0}
    seg = new_seg.astype(int).groupby(g).cumsum()
    last_seg = seg.groupby(g).transform("max")
    old = seg < last_seg
    seg_end = px["date"].groupby([g, seg]).transform("max")
    codes = sorted(set(px.loc[new_seg, "symbol"]))
    px.loc[old, "symbol"] = px.loc[old, "symbol"] + "@" + seg_end[old].dt.strftime("%Y%m%d")
    px.sort_values(["symbol", "date"], inplace=True, ignore_index=True)
    return {"split": len(codes), "codes": codes[:15]}


def _limit(dates: pd.Series) -> pd.Series:
    return pd.Series(np.where(dates < LIMIT_CHANGE, 0.15, 0.30), index=dates.index)


def _scale_history(px: pd.DataFrame, step: pd.Series) -> None:
    """step(그날의 배율, 기본 1)이 붙은 날 *이전* 가격 전체에 그 배율을 곱한다. 이후 가격(최근값)은 그대로.
    종목·날짜순 전제. 여러 날이면 뒤에서부터 누적."""
    g = px["symbol"]
    scale = step[::-1].groupby(g[::-1]).cumprod()[::-1]     # 그날 포함 이후 배율의 곱
    scale = scale.groupby(g).shift(-1).fillna(1.0)          # 다음 날부터의 곱 → 그날 이전 가격에 적용
    for col in ("open", "close", "high", "low"):
        if col in px:
            px[col] = px[col] * scale


def _delist_tail(px: pd.DataFrame) -> pd.Series:
    """상장폐지 종목의 마지막 DELIST_TAIL거래일(정리매매)."""
    last = px.groupby("symbol")["date"].transform("max")
    from_end = px.groupby("symbol").cumcount(ascending=False)
    return (last < px["date"].max() - pd.Timedelta(days=7)) & (from_end < DELIST_TAIL)


def _fix_share_events(px: pd.DataFrame) -> dict:
    """수정계수에 빠진 병합·감자·분할을 상장주식수 변화로 보정한다 (px를 고친다, 종목·날짜순 전제).
    조건: 수정주가 하루 변동이 가격제한을 넘고, 같은 날 상장주식수가 SHARE_EVENT배 이상 바뀌었으며,
    주식수 비율로 되돌린 변동(= 시가총액 변동)은 가격제한 안에 들어온다 (정리매매 구간은 -100%~+100%).
    예: 196490 정지 중 30:1 병합, 203원 → 6,090원 (수정계수 1.0) → 시가총액 기준 0%.
    보정: 그날 이전 가격 전체에 (이전 주식수 / 이후 주식수)를 곱한다 → 그날 수익 = 시가총액 변동."""
    g = px["symbol"]
    c = px["close"]
    prev = c.groupby(g).ffill().groupby(g).shift()
    sh = px["shares"].where(px["shares"] > 0)
    s = sh / sh.groupby(g).ffill().groupby(g).shift()       # 이후/이전 주식수
    r = c / prev
    base = _limit(px["date"])
    lim = base + 0.05
    tail = _delist_tail(px)
    cap_r = r * s - 1                                        # 시가총액 기준 하루 변동
    big = ((r - 1).abs() > lim) & ((s >= SHARE_EVENT) | (s <= 1 / SHARE_EVENT))
    ok = np.where(tail, cap_r.abs() <= 1.0, cap_r.abs() <= lim)
    same_day = big & ok
    # 권리락처럼 가격이 먼저 바뀌고 새 주식은 몇 주 뒤 상장되는 경우: SHARE_LAG거래일 뒤 주식수로 본다.
    # 우연한 일치를 피하려고 시가총액 기준 변동이 거의 0(±SHARE_LAG_TOL)일 때만.
    # 예: 900010 2010-04-30 무상증자 권리락 2,035원 → 1,700원(-16.5%), 주식수는 05-31에 48,000,000 → 57,600,000(×1.2).
    s_fwd = sh.groupby(g).shift(-SHARE_LAG).fillna(sh.groupby(g).transform("last")) /         sh.groupby(g).ffill().groupby(g).shift()
    lagged = ((r - 1).abs() > base + 0.005) & ((s_fwd - 1).abs() > 0.05) &         ((r * s_fwd - 1).abs() <= SHARE_LAG_TOL) & ~same_day & (s - 1).abs().lt(0.05).fillna(True)
    flag = same_day | lagged
    if not flag.any():
        return {"fixed": 0}
    k = (1 / s).where(same_day, (1 / s_fwd).where(lagged, 1.0)).fillna(1.0)
    _scale_history(px, k)
    syms = px.loc[flag, "symbol"]
    return {"fixed": int(flag.sum()), "same_day": int(same_day.sum()), "lagged": int(lagged.sum()),
            "symbols": int(syms.nunique()), "events": _events(px, flag)}


def _neutralize_jumps(px: pd.DataFrame) -> dict:
    """가격제한을 넘는 하루 변동(|수익| > 35%)을 0으로 만든다 (px를 고친다, 종목·날짜순 전제).
    정상 거래일에는 나올 수 없는 변동이라, 수정계수에 빠진 감자·병합·분할(거래정지 뒤 재상장)이나
    잘못된 계수로 본다. 예: 정지 중 35원 → 재상장 960원. 직전 유효 종가와 비교하므로 정지 기간을 건너뛴 점프도 잡는다.
    예외: 상장폐지 종목의 마지막 15거래일(정리매매는 가격제한이 없고 폭락이 실제 손실).
    그 구간에서도 하루 +100% 넘는 급등은 0으로 (예: 008080 원본가 1원 → 67,000원).
    수익 0으로 둔 날 이전 가격 전체를 그 비율만큼 옮겨 이후 가격(최근값)은 그대로 둔다."""
    g = px["symbol"]
    c = px["close"]
    prev = c.groupby(g).ffill().groupby(g).shift()
    r = c / prev - 1
    tail = _delist_tail(px)
    # 거래가 없는 날(거래량 0)에 가격제한을 넘게 바뀐 값은 체결가가 아니라 정지 중 기준가 변경이다 (35% 미만도)
    # 예: 182400 2025-08-06 감자 중 정지, 2,745 → 3,685원(+34%), 네이버 수정주가는 0%.
    idle = (px["volume"] == 0) if "volume" in px else pd.Series(False, index=px.index)
    flag = ((r.abs() > KR_JUMP) & ~tail) | (tail & (r > TAIL_UP_MAX)) |         (idle & (r.abs() > _limit(px["date"]) + 0.005) & ~tail)
    if not flag.any():
        return {"neutralized": 0, "kept_delisting_tail": int(((r.abs() > KR_JUMP) & tail).sum())}
    # 플래그 날의 비율(c/prev)만큼 그 이전 가격을 곱해 이어 붙인다 → 그날 수익 0
    _scale_history(px, (c / prev).where(flag, 1.0).fillna(1.0))
    syms = px.loc[flag, "symbol"]
    return {"neutralized": int(flag.sum()), "up": int((r[flag] > 0).sum()), "down": int((r[flag] < 0).sum()),
            "symbols": int(syms.nunique()), "events": _events(px, flag),
            "kept_delisting_tail": int(((r.abs() > KR_JUMP) & tail & ~flag).sum())}


def _settings() -> dict:
    return yaml.safe_load((store.ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


US_ADJ_STEP = 0.25   # 하루 수정비율(adj_close/close) 변화가 이보다 크면 배당일 수 없다 → 수정 오류


def _fix_adj_steps(px: pd.DataFrame) -> dict:
    """us_1d의 close는 이미 분할 반영, adj_close는 거기에 배당만 더한 값이다. 그래서 adj/close 비율은
    배당락일에 조금씩만 바뀐다. 하루에 25% 넘게 바뀌었는데 원본가는 거의 그대로(±10%)이고 수정 수익만 ±25%를
    넘으면 오류로 보고 그 변화를 없앤다 (px의 adj_close를 고친다).
    예: DHR 2016-07-05 포티브 분사가 close와 adj_close에 이중 반영 → 수정 수익 +61% (원본 +3.6%).
    실제 특별배당(KDP 2018, BKR 2017, LDOS 2016 등)은 원본가가 배당만큼 떨어지고 수정 수익은 평범하므로 건드리지 않는다."""
    g = px["symbol"]
    ratio = (px["adj_close"] / px["close"]).where(px["close"] > 0)
    step = ratio / ratio.groupby(g).shift()
    r_raw = px["close"] / px["close"].groupby(g).shift() - 1
    r_adj = px["adj_close"] / px["adj_close"].groupby(g).shift() - 1
    flag = ((step - 1).abs() > US_ADJ_STEP) & (r_adj.abs() > US_ADJ_STEP) & (r_raw.abs() < 0.10)
    if not flag.any():
        return {"fixed": 0}
    k = step.where(flag, 1.0).fillna(1.0)
    scale = k[::-1].groupby(g[::-1]).cumprod()[::-1].groupby(g).shift(-1).fillna(1.0)
    px["adj_close"] = px["adj_close"] * scale                # 그날 이전 비율에 곱해 이어 붙임 (최근값 유지)
    return {"fixed": int(flag.sum()), "examples": sorted(set(px.loc[flag, "symbol"]))[:15]}


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
    bad_adj = _fix_adj_steps(px)
    if dividend_tax is None:
        dividend_tax = _settings()["markets"]["us_stock"].get("dividend_withholding_pct", 0) / 100
    if dividend_tax:
        ratio = pd.concat([_net_dividend_ratio(g, dividend_tax) for _, g in px.groupby("symbol")]).reindex(px.index)
    else:
        ratio = (px["adj_close"] / px["close"]).where(px["close"] > 0)
    px["raw_close"] = px["close"]
    # 배당 필드용: 그날 배당액(close 기준 = 분할 반영) = adj 수익과 close 수익의 차 × 전일 close. 통화 환산 전 USD.
    g = px["symbol"]
    c0, a0 = px["close"].where(px["close"] > 0), px["adj_close"]
    px["_div_amt"] = (a0 / a0.groupby(g).shift() * c0.groupby(g).shift() - c0).clip(lower=0)
    px["_rc"] = c0
    for c in ("open", "close", "high", "low"):
        px[c] = px[c] * ratio
    meta = {"source": "us_1d (adj_close로 배당 재투자 수정)", "rows": len(px), "bad_adj_steps": bad_adj,
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


# ── 배당: 신호용 필드 (dps, div_yield) ──
DIV_FIELDS = ("dps", "div_yield")
DIV_STALE_DAYS = 550          # KR: 마지막 사업보고서 값이 이보다 오래되면 낡은 값으로 보고 비운다 (코드 재사용 등)
US_DIV_START = pd.Timestamp("2016-01-01")   # us_dividends 시작 → 그 뒤 1년이 지나야 12개월 합계가 온전하다
US_DIV_PLACE_DAYS = 5         # US 배당락일에 봉이 없으면 이 안의 다음 봉에 붙인다
DIV_TIMING = {
    "kr_stock": "dps = 최신 사업보고서 주당 현금배당(dart_dividend dps_common, 원, 연간 합계), "
                "최초 접수일(first_filed) 다음 날부터. 그날 주식 기준으로 분할 환산. div_yield = dps / raw_close",
    "us_stock": "dps = 직전 365일 정기 현금배당 합계(us_dividends 배당락일, 특별배당 제외), 배당락일 당일부터. "
                "금액은 us_1d close 기준(분할 반영, USD). div_yield = dps / 그날 close(USD)",
}


def dividends(market: str, symbols=None, asof=None) -> pd.DataFrame:
    """전략·검사용 배당 이벤트 표. 열: symbol, src_date(원천에 나타난 날), known(이 날짜 봉부터 쓸 수 있음), dps.
    kr_stock: dart_dividend 사업보고서 연간 dps_common(원본 주당 원). src_date = 최초 접수일(first_filed()).
              filed 열은 정정일일 수 있어 시점으로 쓰지 않는다. 최초 접수일이 없는 행만 filed로 대신한다
              (더 늦은 날이라 보수적, filed_fallback=True). known = src_date 다음 날. year, ref(결산일)도 준다.
    us_stock: us_dividends 정기 현금배당(특별배당·주식배당 제외). src_date = known = 배당락일. dps = Alpaca rate
              (원본 금액, 분할 미반영). us_dividend_announcements의 발표일은 실제 결의일이 아니라 쓰지 않는다.
    asof: 그날까지 원천에 있던 행만 (KR 최초 접수일 < asof, US 배당락일 ≤ asof)."""
    cols = ["symbol", "src_date", "known", "dps"]
    if market == "kr_stock":
        dv = store.read("dart_dividend", "ALL")
        if dv.empty:
            return pd.DataFrame(columns=cols)
        dv = dv[dv["has_data"].astype(bool)]
        if "reprt_code" in dv:
            dv = dv[dv["reprt_code"] == "11011"]
        if symbols not in (None, "ALL", "KOSPI", "KOSDAQ"):
            dv = dv[dv["symbol"].isin([symbols] if isinstance(symbols, str) else symbols)]
        dv = dv.assign(period=dv["ts"].dt.tz_convert(TZ["kr_stock"]).dt.strftime("%Y.%m"))
        from ..data.collectors.dividends import first_filed
        try:
            ff = first_filed()
        except ValueError:            # 공시 목록이 둘 다 비어 있으면 first_filed()의 concat이 실패한다
            ff = pd.DataFrame()
        if not ff.empty:
            ff = ff.loc[ff["kind"] == "사업보고서", ["corp_code", "period", "first_filed"]]
            dv = dv.merge(ff, on=["corp_code", "period"], how="left")
        else:
            dv = dv.assign(first_filed=pd.NaT)
        fallback = dv["first_filed"].isna()
        src = _local_date(pd.to_datetime(dv["first_filed"], utc=True).where(~fallback, dv["filed"]), TZ["kr_stock"])
        out = pd.DataFrame({"symbol": dv["symbol"].str.strip(), "src_date": src, "known": src + pd.Timedelta(days=1),
                            "dps": dv["dps_common"].fillna(0.0).astype(float),   # 보고서는 있는데 값 없음 = 무배당
                            "year": dv["bsns_year"], "ref": _local_date(dv["ts"], TZ["kr_stock"]),
                            "filed_fallback": fallback})
        if asof is not None:
            out = out[out["src_date"] < pd.Timestamp(asof)]
    elif market == "us_stock":
        u = store.read("us_dividends", "ALL")
        if u.empty:
            return pd.DataFrame(columns=cols)
        u = u[(u["kind"] == "cash_dividend") & ~u["special"].astype(bool)]
        if symbols not in (None, "ALL"):
            u = u[u["symbol"].isin([symbols] if isinstance(symbols, str) else symbols)]
        ex = _local_date(u["ts"], TZ["us_stock"])
        out = pd.DataFrame({"symbol": u["symbol"], "src_date": ex, "known": ex, "dps": u["rate"].astype(float)})
        if asof is not None:
            out = out[out["src_date"] <= pd.Timestamp(asof)]
    else:
        raise ValueError(f"배당 데이터는 kr_stock·us_stock만 있다: {market}")
    return out.sort_values(["symbol", "known"]).reset_index(drop=True)


def _div_basis(long: pd.DataFrame, market: str) -> pd.DataFrame:
    cols = ["raw_close", "_f"] if market == "kr_stock" else ["_rc", "_div_amt"]
    return long[["date", "symbol"] + cols].sort_values(["symbol", "date"]).reset_index(drop=True)


def dividend_fields(market: str, basis: pd.DataFrame, asof=None) -> pd.DataFrame:
    """basis(_div_basis) 날짜·종목마다 그날 알 수 있는 dps, div_yield. 반환: date, symbol, dps, div_yield."""
    b = basis.sort_values(["symbol", "date"]).reset_index(drop=True)
    syms = sorted(b["symbol"].unique())
    ev = dividends(market, syms, asof)
    if market == "kr_stock":
        # 늦게 나온 옛 사업연도 보고서는 이미 알려진 최신 연도 값을 덮지 않는다 (cummax는 과거만 본다)
        ev = ev.sort_values(["symbol", "known", "year"])
        ev = ev[ev["year"] >= ev.groupby("symbol")["year"].cummax()]
        ev = ev.drop_duplicates(["symbol", "known"], keep="last").sort_values("known")
        bd = b.sort_values("date")
        # 알려진 날(그 뒤 첫 봉)의 수정계수 → 그날 계수로 나눠 그 사이 분할·병합만 반영 (미래 정보 없음)
        ev = pd.merge_asof(ev, bd[["date", "symbol", "_f"]], left_on="known", right_on="date", by="symbol",
                           direction="forward").drop(columns="date").rename(columns={"_f": "_fk"})
        m = pd.merge_asof(bd, ev[["symbol", "known", "dps", "_fk"]], left_on="date", right_on="known", by="symbol",
                          direction="backward", tolerance=pd.Timedelta(days=DIV_STALE_DAYS))
        m["dps"] = m["dps"] * m["_fk"] / m["_f"]
        m["div_yield"] = m["dps"] / m["raw_close"].where(m["raw_close"] > 0)
    else:
        ev = ev.drop_duplicates(["symbol", "known"]).sort_values("known")
        rows = b[["date", "symbol"]].assign(row=np.arange(len(b))).sort_values("date")
        hit = pd.merge_asof(ev[["symbol", "known"]], rows, left_on="known", right_on="date", by="symbol",
                            direction="forward", tolerance=pd.Timedelta(days=US_DIV_PLACE_DAYS))["row"].dropna()
        amt = np.zeros(len(b))
        pos = np.unique(hit.astype(int).to_numpy())
        amt[pos] = b["_div_amt"].to_numpy(dtype=float)[pos]
        m = b.assign(_amt=np.nan_to_num(amt))
        m["dps"] = m.set_index("date").groupby("symbol", sort=True)["_amt"].rolling("365D").sum().to_numpy()
        m.loc[m["date"] < US_DIV_START + pd.Timedelta(days=365), "dps"] = np.nan
        m["div_yield"] = m["dps"] / m["_rc"]
    return m[["date", "symbol", "dps", "div_yield"]]


def check_dividend_causality(panel: Panel, n_cuts: int = 3, tol: float = 1e-9) -> dict:
    """배당 필드(dps, div_yield) 미래참조 검사: 중간 날짜 cut에서 '그때까지 원천에 있던 배당 기록'과
    cut까지의 가격만으로 다시 만든 값이, 전체로 만든 panel 값과 cut까지 같아야 한다.
    (시점 규칙을 어기면 — 예: 최초 접수일 당일이나 결산일부터 보이게 하면 — cut 근처에서 어긋난다.)"""
    fields = [f for f in DIV_FIELDS if f in panel]
    out = {"passed": True, "fields": fields, "cuts": []}
    if not fields or panel.div_basis is None:
        return out
    b, idx = panel.div_basis, panel.index
    for q in np.linspace(0.4, 0.85, n_cuts):
        cut = idx[int(len(idx) * q)]
        part = dividend_fields(panel.market, b[b["date"] <= cut], asof=cut)
        bad = 0
        for f in fields:
            p = part.pivot(index="date", columns="symbol", values=f)
            full = panel[f].reindex(index=p.index, columns=p.columns).to_numpy(dtype=float)
            same = np.isclose(full, p.to_numpy(dtype=float), rtol=tol, atol=tol, equal_nan=True)
            bad += int((~same).any(axis=1).sum())
        out["cuts"].append({"cut": str(cut), "mismatch_bars": bad})
        out["passed"] &= bad == 0
    return out


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
        f"sum(volume) AS volume, count(*) AS n_bars, max({t}) AS last_ts "
        f"FROM read_parquet({src}, filename=true) WHERE {_bounds(start, end, 'UTC')} GROUP BY 1, 2")
    if px.empty:
        return px, {}
    for col in ("bucket", "last_ts"):
        if px[col].dt.tz is None:
            px[col] = px[col].dt.tz_localize("UTC")
    # 아직 끝나지 않은 마지막 봉(오늘 일봉 등)은 뺀다: 봉 끝이 그 종목의 마지막 1분봉보다 뒤면 미완성
    step = pd.Timedelta(**{unit: n})
    sym_last = px.groupby("symbol")["last_ts"].transform("max")
    partial = px["bucket"] + step > sym_last + pd.Timedelta(minutes=1)
    px = px[~partial]
    px["date"] = px["bucket"].dt.tz_convert("UTC").dt.tz_localize(None)
    return px, {"source": f"{ds} → {timeframe}", "rows": len(px), "partial_bars_dropped": int(partial.sum()),
                "note": "선물 펀딩비 미반영" if market == "crypto_futures" else ""}
