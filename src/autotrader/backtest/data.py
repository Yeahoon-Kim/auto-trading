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
- 재무 필드 (kr_stock, fields에 적으면): dart_fin 분기 재무. 재무상태 FIN_BS(equity, total_liabilities …)는 최근 분기말 값,
              손익 FIN_TTM(net_income_ttm …)은 최근 4개 분기 3개월 값 합. 공개일 다음 거래일부터 보인다 (fin_reports()).
              check_fin_causality()가 cut 전에 공개된 보고서만으로 다시 만든 값과 비교한다.
- 시가총액 필드 (us_stock, fields에 market_cap·shares·cik): us_1d 원본 종가 × 분할 보정 주식수 (data.shares).
              공시일 다음 거래일부터. 회사 단위라 GOOG·GOOGL은 둘 다 회사 전체 시총 → cik로 묶어 하나만 고른다.
              check_mcap_causality()가 cut 전 공시만으로 다시 만든 값과 비교한다.
- crypto_spot / crypto_futures : binance_spot_1m / binance_um_1m을 timeframe(1d, 1h …)으로 묶음. UTC 기준.
              선물은 panel["funding"](봉 안에 정산된 펀딩비율 합, binance_um_funding)을 같이 주고 엔진이 손익에 반영한다.

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
        self.fin_reports: pd.DataFrame | None = None  # 재무 필드 원천 보고서 표 (check_fin_causality용)
        self.fin_basis: pd.DataFrame | None = None    # 재무 필드를 붙인 (date, symbol) 행
        self.mcap_source: dict | None = None          # US 시가총액 원천 (check_mcap_causality용)
        self.mcap_basis: pd.DataFrame | None = None   # US 시가총액을 붙인 (date, symbol, ts, _rc) 행

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
         reinvest_dividends: bool = True, fin_timing: str = "first", funding: bool = True) -> Panel:
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
      panel["tradable"] (거래량 > 0)을 같이 준다 → 엔진이 거래정지일에는 매매하지 않는다.
      fin_timing  재무 필드(FIN_FIELDS)의 공개일: "first"(최초 접수일, 배당과 같은 규칙) | "last"(마지막 접수일 = filed).
    crypto_futures 전용:
      funding  True(기본)면 panel["funding"] = 그 봉 [시작, 끝) 안에 정산된 펀딩비율 합 (binance_um_funding).
               엔진이 그 봉 동안 든 포지션에 비중 × 비율만큼 물린다 (롱은 양수 비율에 냄, 숏은 받음).
               봉 끝에 이미 알려진 값이라 신호로 써도 된다. False면 예전처럼 미반영."""
    cut = dev_end(today)
    if not lockbox:
        end = min(pd.Timestamp(end), cut) if end is not None else cut
    fields = tuple(dict.fromkeys(("open", "close") + tuple(fields)))
    want_div = [f for f in DIV_FIELDS if f in fields]
    if want_div and market not in ("kr_stock", "us_stock"):
        raise ValueError(f"배당 필드 {want_div}는 kr_stock·us_stock만 있다")
    want_fin = [f for f in ALL_FIN_FIELDS if f in fields]
    if want_fin:
        ok = FIN_FIELDS if market == "kr_stock" else tuple(US_FIN_FIELDS) if market == "us_stock" else ()
        if bad := [f for f in want_fin if f not in ok]:
            raise ValueError(f"재무 필드 {bad}는 {market}에 없다. kr_stock: {list(FIN_FIELDS)}, us_stock: {list(US_FIN_FIELDS)}")
    if market == "kr_stock":
        long, meta = _kr(symbols, start, end, fields, neutralize_jumps, dividend_tax, reinvest_dividends)
        fields = fields + ("tradable",)
    elif market == "us_stock":
        long, meta = _us(symbols, start, end, fields, currency, dividend_tax)
        if currency == "KRW":
            fields = tuple(dict.fromkeys(fields + ("close_usd", "fx")))
    elif market in ("crypto_spot", "crypto_futures"):
        long, meta = _crypto(market, symbols, start, end, fields, timeframe, funding)
        if market == "crypto_futures" and funding:
            fields = fields + ("funding",)
    else:
        raise ValueError(f"지원하지 않는 시장: {market}")
    if long.empty:
        raise ValueError(f"{market} 데이터 없음 (symbols={symbols}, {start}~{end})")
    basis = None
    if want_div:
        basis = _div_basis(long, market)
        long = long.merge(dividend_fields(market, basis), on=["date", "symbol"], how="left")
        meta["dividend_fields"] = DIV_TIMING[market]
        meta["dividend_data"] = dividend_coverage(market)
    rep_fin = fin_basis = None
    if want_fin:
        if market == "kr_stock":
            rep_fin = fin_reports(timing=fin_timing)
            meta["fin_fields"] = FIN_TIMING[fin_timing]
            meta["fin_data"] = fin_coverage(rep_fin, fin_timing)
            if "eps_ttm" in want_fin:
                long = long.assign(_base=long["symbol"].str.split("@").str[0]).merge(
                    _kr_class_shares(start, end).rename(columns={"base": "_base"}), on=["date", "_base"], how="left")
                meta["fin_data"]["eps"] = ("eps_ttm = 지배주주 순이익 TTM ÷ (보통주 + 우선주 상장주식수, 자사주 미차감), "
                                           "그날 주식 기준 원, 보통주 코드만 (우선주는 NaN). PER = raw_close / eps_ttm")
        else:
            if fin_timing not in US_FIN_TIMING:
                raise ValueError(f"fin_timing은 {list(US_FIN_TIMING)} 중 하나: {fin_timing}")
            rep_fin = us_fin_facts()
            meta["fin_fields"] = US_FIN_TIMING[fin_timing]
            meta["fin_data"] = us_fin_coverage(rep_fin, fin_timing, long["symbol"].unique())
        fin_basis = long[["date", "symbol", *[c for c in ("_sh", "_f") if c in long and "eps_ttm" in want_fin]]]
        fin_basis = fin_basis.reset_index(drop=True)
        long = long.merge(fin_build(market, fin_basis, rep_fin, want_fin, fin_timing), on=["date", "symbol"], how="left")
    want_mcap = [f for f in US_MCAP_FIELDS if f in fields] if market == "us_stock" else []
    mcap_src = mcap_basis = None
    if want_mcap:
        mcap_src = us_mcap_source(long["symbol"].unique())
        ev = _mcap_events(mcap_src)
        mcap_basis = long[["date", "symbol", "ts", "_rc"]].reset_index(drop=True)
        vals = us_mcap_fields(mcap_basis, ev)
        long = long.drop(columns=[c for c in US_MCAP_FIELDS if c in long]).merge(vals, on=["date", "symbol"], how="left")
        meta["mcap_fields"] = US_MCAP_TIMING
        meta["mcap_data"] = us_mcap_coverage(mcap_src, ev, vals)
    frames = {f: long.pivot(index="date", columns="symbol", values=f).sort_index() for f in fields if f in long}
    if want_fin:   # 결측률: 가격이 있는 칸 중 재무 값이 비어 있는 비율
        has_px = frames["close"].notna()
        meta["fin_data"]["missing"] = {f: round(float(frames[f].isna()[has_px].sum().sum() / max(has_px.sum().sum(), 1)), 4)
                                       for f in want_fin}
    if want_mcap:
        has_px = frames["close"].notna()
        meta["mcap_data"]["missing"] = round(float(frames["market_cap"].isna()[has_px].sum().sum()
                                                   / max(has_px.sum().sum(), 1)), 4) if "market_cap" in frames else None
    meta |= {"dev_end": str(cut.date()), "lockbox": lockbox, "survivorship": SURVIVORSHIP[market],
             "start": str(frames["close"].index[0]), "end": str(frames["close"].index[-1])}
    panel = Panel(frames, market, timeframe, meta)
    panel.div_basis = basis
    panel.fin_reports = rep_fin
    panel.fin_basis = fin_basis
    panel.mcap_source, panel.mcap_basis = mcap_src, mcap_basis
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
    meta["ex_range"] = f"{ev['ex_date'].min().date()}~{ev['ex_date'].max().date()}"   # 실행 시점 원천 범위
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
    "kr_stock": "dps = 최신 사업보고서 주당 현금배당(dart_dividend, 보통주 dps_common·첫 우선주 dps_pref, 원, 연간 합계), "
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
        dps = _dps_or_zero(dv, dv["dps_common"])
        if "dps_pref" in dv:   # 우선주·종류주식에만 배당하고 보통주 칸이 "-"인 회사 → 보통주는 무배당
            dps = dps.where(dps.notna() | dv["dps_pref"].isna(), 0.0)
        dv = dv.assign(symbol=dv["symbol"].str.strip(), dps=dps)
        dv = pd.concat([dv, _kr_pref_rows(dv)], ignore_index=True)
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
        out = pd.DataFrame({"symbol": dv["symbol"], "src_date": src, "known": src + pd.Timedelta(days=1),
                            "dps": dv["dps"].astype(float),
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


def _dps_or_zero(dv: pd.DataFrame, dps: pd.Series) -> pd.Series:
    """보고서는 있는데 주당배당금이 비면 무배당(0). 단 배당성향·배당총액이 있으면 파싱 누락이라 모름(NaN)으로 둔다."""
    col = lambda c: dv[c] if c in dv else pd.Series(np.nan, index=dv.index)
    paid = (col("payout_ratio").fillna(0) > 0) | (col("cash_div_total").fillna(0) > 0)
    return dps.where(dps.notna(), np.where(paid, np.nan, 0.0)).astype(float)


def _kr_pref_rows(dv: pd.DataFrame) -> pd.DataFrame:
    """우선주(종목코드 따로)에 같은 회사 사업보고서의 dps_pref를 붙인 행.
    우선주 목록·종류는 kr_dividend_schedule의 stock_kind, 보통주는 우선주 코드 앞 5자리 + '0' (예: 005935 → 005930).
    사업보고서에는 우선주 칸이 하나뿐이라 첫 번째 우선주('우선')에만 붙인다. 2우선 이하는 금액이 다를 수 있어 비운다."""
    s = store.read("kr_dividend_schedule", "ALL")
    if s.empty or "dps_pref" not in dv:
        return dv.iloc[:0]
    kinds = s.drop_duplicates("symbol", keep="last").set_index("symbol")["stock_kind"].str.strip()
    pref = kinds[kinds == "우선"].index
    m = pd.DataFrame({"_pref": pref, "_common": pref.str[:5] + "0"})
    out = dv.merge(m, left_on="symbol", right_on="_common")
    out = out.assign(symbol=out["_pref"], dps=_dps_or_zero(out, out["dps_pref"])).drop(columns=["_pref", "_common"])
    return out.drop_duplicates(["symbol", "corp_code", "bsns_year"])


def dividend_coverage(market: str) -> dict:
    """실행 시점 배당 원천 범위 (백필 중이면 결과가 데이터와 함께 바뀌므로 report에 남긴다)."""
    ev = dividends(market)
    if ev.empty:
        return {"rows": 0}
    out = {"rows": len(ev), "first": str(ev["src_date"].min().date()), "last": str(ev["src_date"].max().date())}
    if market == "kr_stock":
        n = ev.groupby("year").size()
        out |= {"years": {int(k): int(v) for k, v in n.items()},
                "filed_fallback": int(ev["filed_fallback"].sum()),   # 최초 접수일이 없어 정정일로 잡은 행
                "dps_unknown": int(ev["dps"].isna().sum())}         # 파싱 누락 등으로 금액을 모르는 행
    return out


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


# ── 재무: dart_fin (kr_stock) ──
FIN_BS = {  # 필드 이름 → dart_fin 열 (분기말 잔액). equity = 자본총계 (연결이면 비지배지분 포함)
    "equity": "total_equity", "total_equity": "total_equity", "total_assets": "total_assets",
    "total_liabilities": "total_liabilities", "current_assets": "current_assets",
    "current_liabilities": "current_liabilities", "noncurrent_assets": "noncurrent_assets",
    "noncurrent_liabilities": "noncurrent_liabilities", "capital_stock": "capital_stock",
    "retained_earnings": "retained_earnings",
}
FIN_FLOW = ("revenue", "operating_income", "pretax_income", "net_income", "net_income_controlling")
FIN_TTM = {f"{c}_ttm": c for c in FIN_FLOW}    # 최근 4개 분기 3개월 값 합 (연결·별도가 섞이면 NaN)
FIN_PER_SHARE = ("eps_ttm",)   # 지배주주 순이익 TTM ÷ 보통주+우선주 상장주식수 (자사주 미차감), 보통주 코드만
FIN_FIELDS = tuple(FIN_BS) + tuple(FIN_TTM) + FIN_PER_SHARE
ALL_FIN_FIELDS = FIN_FIELDS + ("shares_outstanding",)       # + us_stock 전용 (US_FIN_FIELDS)
FIN_STALE_DAYS = 400      # 마지막 공개 뒤 이만큼 새 보고서가 없으면 낡은 값으로 보고 비운다
FIN_KIND = {"Q1": "분기보고서", "H1": "반기보고서", "Q3": "분기보고서", "FY": "사업보고서"}
FIN_TIMING = {
    "first": "dart_fin 최신(정정 반영) 값을 최초 접수일(first_filed) 다음 거래일부터. 최초 접수일이 없으면 filed(정정일일 수 "
             "있어 더 늦음) 다음 거래일부터. 정정 전 원래 값은 저장돼 있지 않아, 정정된 보고서는 정정 값이 최초 접수일부터 "
             "보인다(경미한 미래참조). 4분기 = FY − Q3 누적 (사업보고서·3분기보고서 중 늦은 공개일부터)",
    "last": "dart_fin 값을 filed(마지막 접수일, 정정이면 정정일) 다음 거래일부터 (보수적: 정정된 보고서는 정정 전까지 안 보임). "
            "4분기 = FY − Q3 누적",
}


def fin_reports(symbols=None, asof=None, timing: str = "first") -> pd.DataFrame:
    """dart_fin 보고서 1행 = (종목, 분기). 열: symbol, qidx(연*4+분기, 분기말 기준), period, fs_div,
    src_date(공개일, 현지 날짜), known(이 날짜 봉부터 쓸 수 있음 = src_date 다음 날 → 다음 거래일 봉),
    flow_known(손익을 쓸 수 있는 날), FIN_BS 원천 열, FIN_FLOW 3개월 값,
    fallback(최초 접수일이 없어 filed 사용), corrected(최초 접수 뒤 정정됨).
    FY 행의 손익은 4분기 3개월 값(FY − Q3 누적)으로 바꾼다. 그 flow_known은 사업보고서·3분기보고서 중 늦은 공개일
    다음 날, 재무상태(known)는 사업보고서 공개일 다음 날. 원화 아닌 보고서는 뺀다.
    asof: 그날 전에 공개된 행만 (src_date < asof)."""
    if timing not in FIN_TIMING:
        raise ValueError(f"fin_timing은 {list(FIN_TIMING)} 중 하나: {timing}")
    bs = sorted(set(FIN_BS.values()))
    cols = ["symbol", "qidx", "period", "fs_div", "src_date", "known", "flow_known", *bs, *FIN_FLOW,
            "fallback", "corrected"]
    d = store.read("dart_fin", "ALL")
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d[d["currency"].fillna("KRW") == "KRW"].copy()
    d["symbol"] = d["symbol"].astype(str).str.strip()
    if symbols not in (None, "ALL", "KOSPI", "KOSDAQ"):
        d = d[d["symbol"].isin([symbols] if isinstance(symbols, str) else symbols)]
    for c in [*bs, *FIN_FLOW, *(f"{c}_cum" for c in FIN_FLOW)]:
        if c not in d:
            d[c] = np.nan
    tz = TZ["kr_stock"]
    qend = _local_date(d["ts"], tz)
    filed = _local_date(d["filed"], tz)
    d["qidx"] = qend.dt.year * 4 + (qend.dt.month - 1) // 3
    d["fallback"], d["corrected"] = True, False
    src = filed
    if timing == "first":
        from ..data.collectors.dividends import first_filed
        try:
            ff = first_filed()
        except ValueError:
            ff = pd.DataFrame()
        if not ff.empty and "corp_code" in d:
            key = d.assign(kind=d["period"].map(FIN_KIND), _p=qend.dt.strftime("%Y.%m"))
            m = key[["corp_code", "kind", "_p"]].merge(
                ff.rename(columns={"period": "_p"})[["corp_code", "kind", "_p", "first_filed"]],
                on=["corp_code", "kind", "_p"], how="left")
            first = _local_date(pd.to_datetime(m["first_filed"], utc=True), tz)
            first.index = d.index
            use = first.notna() & (first <= filed)          # 최초 접수일이 filed보다 늦으면 잘못 붙은 것 → filed
            src = first.where(use, filed)
            d["fallback"], d["corrected"] = ~use, use & (first < filed)
    d["src_date"] = src
    rows = d[["symbol", "bsns_year", "period", "qidx", "fs_div", "src_date", "fallback", "corrected", *bs,
              *FIN_FLOW, *(f"{c}_cum" for c in FIN_FLOW)]]
    rows = rows.drop_duplicates(["symbol", "qidx"], keep="last")
    # 4분기: 사업보고서 값(연간) − 같은 사업연도 3분기 누적. 3분기보고서가 없으면 3개월 값을 모른다(NaN)
    # 손익(flow)은 사업보고서·3분기보고서 중 늦은 공개일부터 (flow_known), 재무상태는 사업보고서 공개일부터 (known)
    fy = rows[rows["period"] == "FY"]
    q3 = rows[rows["period"] == "Q3"].drop_duplicates(["symbol", "bsns_year"], keep="last")
    m = fy[["symbol", "bsns_year"]].merge(q3, on=["symbol", "bsns_year"], how="left", suffixes=("", "_q3"))
    m.index = fy.index
    rows = rows.copy()
    rows["flow_src"] = rows["src_date"]
    for c in FIN_FLOW:
        rows.loc[fy.index, c] = fy[c] - m[f"{c}_cum"]
    later = m["src_date"].notna() & (m["src_date"] > fy["src_date"])
    rows.loc[fy.index[later], "flow_src"] = m.loc[later, "src_date"]
    rows.loc[fy.index[m["src_date"].isna()], "flow_src"] = pd.NaT    # 3분기보고서가 없으면 4분기 손익은 알 수 없음
    rows["known"] = rows["src_date"] + pd.Timedelta(days=1)
    rows["flow_known"] = rows["flow_src"] + pd.Timedelta(days=1)
    if asof is not None:
        rows = rows[rows["src_date"] < pd.Timestamp(asof)]
    return rows[cols].sort_values(["symbol", "qidx"]).reset_index(drop=True)


def _fin_ttm(rep: pd.DataFrame) -> pd.DataFrame:
    """분기 행마다 그 분기까지 4개 분기 합(TTM). known = 네 분기 중 가장 늦게 공개된 날.
    네 분기 보고서가 다 있어야 행이 생기고, 연결·별도 구분이 섞이면 값은 NaN (값 누락이면 합도 NaN)."""
    rep = rep.drop(columns="known").rename(columns={"flow_known": "known"})
    acc = rep[["symbol", "qidx", "fs_div", "known"]].copy()
    ok = pd.Series(True, index=acc.index)
    have = acc["known"].notna()
    for c in FIN_FLOW:
        acc[c] = rep[c].to_numpy(dtype=float)
    for lag in (1, 2, 3):
        p = rep[["symbol", "qidx", "fs_div", "known", *FIN_FLOW]].assign(qidx=rep["qidx"] + lag)
        m = acc[["symbol", "qidx"]].merge(p, on=["symbol", "qidx"], how="left")
        m.index = acc.index
        for c in FIN_FLOW:
            acc[c] = acc[c] + m[c]
        have &= m["known"].notna()
        ok &= m["fs_div"] == acc["fs_div"]
        acc["known"] = acc["known"].where(m["known"].isna() | (m["known"] <= acc["known"]), m["known"])
    for c in FIN_FLOW:
        acc[c] = acc[c].where(ok)
    # 분기가 빠진 행은 언제 다 모일지 모르므로 이벤트로 두지 않는다 (직전 TTM이 FIN_STALE_DAYS 안에서 유지)
    acc = acc[have]
    return acc.rename(columns={c: f"{c}_ttm" for c in FIN_FLOW}).drop(columns="fs_div")


def _fin_asof(bars: pd.DataFrame, ev: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """bars(date, base)마다 그날까지 공개된 가장 최근 분기(qidx 최대)의 값. 늦게 공개된 옛 분기는 새 분기를 덮지 않는다."""
    ev = ev.sort_values(["symbol", "known", "qidx"])
    ev = ev[ev["qidx"] >= ev.groupby("symbol")["qidx"].cummax()]
    ev = ev.drop_duplicates(["symbol", "known"], keep="last").sort_values("known")
    ev = ev[["symbol", "known", *cols]].rename(columns={"symbol": "base"})
    return pd.merge_asof(bars, ev, left_on="date", right_on="known", by="base", direction="backward",
                         tolerance=pd.Timedelta(days=FIN_STALE_DAYS))


def fin_fields(basis: pd.DataFrame, rep: pd.DataFrame, fields) -> pd.DataFrame:
    """basis(date, symbol) 날짜·종목마다 그날 알 수 있는 재무 값. 반환: date, symbol, *fields.
    코드 재사용으로 나뉜 앞 상장(코드@날짜)도 원래 코드의 보고서를 쓴다 (FIN_STALE_DAYS로 낡은 값은 비움)."""
    fields = list(fields)
    extra = [c for c in ("_sh", "_f") if c in basis]
    bars = basis[["date", "symbol", *extra]].assign(base=basis["symbol"].str.split("@").str[0]).sort_values("date")
    out = bars.copy()
    bs = [f for f in fields if f in FIN_BS]
    if bs:
        src = sorted({FIN_BS[f] for f in bs})
        m = _fin_asof(bars, rep[["symbol", "qidx", "known", *src]], src)
        for f in bs:
            out[f] = m[FIN_BS[f]].to_numpy()
    ttm = [f for f in fields if f in FIN_TTM]
    if "eps_ttm" in fields and "net_income_controlling_ttm" not in ttm:
        ttm.append("net_income_controlling_ttm")
    if ttm:
        m = _fin_asof(bars, _fin_ttm(rep), ttm)
        for f in ttm:
            out[f] = m[f].to_numpy()
        if "eps_ttm" in fields:
            out["eps_ttm"] = _kr_eps(bars, m)
    return out[["date", "symbol", *fields]]


def _kr_eps(bars: pd.DataFrame, m: pd.DataFrame) -> np.ndarray:
    """eps_ttm (그날 주식 기준 원): 지배주주 순이익 TTM ÷ 그 TTM이 알려진 날(그 뒤 첫 봉)의 보통주+우선주 상장주식수,
    그 사이 분할·병합은 수정계수로 환산 (× 알려진 날 계수 / 그날 계수 — 배당 dps와 같은 방식, 미래 정보 없음).
    krx_1d shares는 자사주를 빼지 않은 값이다. 우선주 코드(끝자리 0 아님)는 NaN — 보통주 EPS만 둔다.
    bars: date, symbol, base, _sh(같은 회사 전 종류 상장주식수 합), _f(수정계수). m: _fin_asof 결과(known 포함)."""
    if "_sh" not in bars or "_f" not in bars:
        raise ValueError("eps_ttm에는 basis의 _sh(상장주식수 합)·_f(수정계수)가 필요하다")
    k = m[["symbol", "known"]].dropna().drop_duplicates().sort_values("known")
    ref = bars[["date", "symbol", "_sh", "_f"]].rename(columns={"date": "_kd", "_sh": "_shk", "_f": "_fk"})
    k = pd.merge_asof(k, ref.sort_values("_kd"), left_on="known", right_on="_kd", by="symbol", direction="forward")
    j = m[["symbol", "known"]].merge(k[["symbol", "known", "_shk", "_fk"]], on=["symbol", "known"], how="left")
    sh = j["_shk"].where(j["_shk"] > 0).to_numpy(float)
    eps = m["net_income_controlling_ttm"].to_numpy(float) / sh * j["_fk"].to_numpy(float) / bars["_f"].to_numpy(float)
    return np.where(bars["base"].str.endswith("0").to_numpy(), eps, np.nan)


def fin_coverage(rep: pd.DataFrame, timing: str) -> dict:
    """실행 시점 재무 원천 범위 (백필·정정으로 결과가 바뀌므로 report에 남긴다)."""
    if rep.empty:
        return {"rows": 0, "timing": timing}
    raw = store.read("dart_fin", "ALL")
    out = {"rows": len(rep), "symbols": int(rep["symbol"].nunique()), "timing": timing,
           "first": str(rep["src_date"].min().date()), "last": str(rep["src_date"].max().date()),
           "fallback": int(rep["fallback"].sum()), "corrected": int(rep["corrected"].sum()),
           "non_krw_skipped": int((raw["currency"].fillna("KRW") != "KRW").sum()) if "currency" in raw else 0,
           "nic_missing": int(rep["net_income_controlling"].isna().sum())}   # 지배주주 순이익 없음 (백필 전 포함)
    if "nic_source" in raw:
        out["nic_source"] = {str(k): int(v) for k, v in raw["nic_source"].fillna("없음").value_counts().items()}
    return out


def _kr_class_shares(start, end) -> pd.DataFrame:
    """(date, base, _sh): 같은 회사 보통주 + 우선주 상장주식수 합. 우선주 코드(끝자리 0 아님) → 앞 5자리 + '0'.
    한국 우선주는 이익 배분에 보통주와 같이 참여하므로 EPS 분모에 넣는다 (우선 배당분은 빼지 않음)."""
    tz = TZ["kr_stock"]
    src = _files("krx_1d", "ALL", start, end)
    if not src:
        return pd.DataFrame(columns=["date", "base", "_sh"])
    s = store.query(f"SELECT ts, symbol, shares FROM read_parquet({src}) WHERE {_bounds(start, end, tz)}")
    s["date"] = _local_date(s["ts"], tz)
    s["base"] = s["symbol"].where(s["symbol"].str.endswith("0"), s["symbol"].str[:5] + "0")
    s = s.drop_duplicates(["date", "symbol"], keep="last")
    return s.groupby(["date", "base"], as_index=False)["shares"].sum().rename(columns={"shares": "_sh"})


# ── 재무: edgar_facts (us_stock) ──
US_FIN_FIELDS = {  # 필드 → (edgar field, 종류). instant = 시점 값(최근 기간 끝), flow = 최근 4개 분기 합
    "equity": ("equity", "instant"), "shares_outstanding": ("shares_outstanding", "instant"),
    "revenue_ttm": ("revenue", "flow"), "operating_income_ttm": ("operating_income", "flow"),
    "net_income_ttm": ("net_income", "flow"),
    "net_income_controlling_ttm": ("net_income", "flow"),   # 1순위 태그 NetIncomeLoss = 지배주주 귀속
}
US_FIN_TIMING = {
    "first": "edgar_facts 기간마다 최초 공시값(정정 전 원래 숫자)을 공시일(filed) 다음 거래일부터. 같은 공시에 태그가 "
             "여럿이면 tag_rank가 작은 것. 4분기 = 연간 − 9개월 누계. 미래참조 없음",
    "last": "edgar_facts 기간마다 그날까지 공시된 가장 최신 값(10-K/A·10-Q/A 정정, 다음 해 비교 재무제표의 재작성 값)을 "
            "그 공시일 다음 거래일부터 바꿔 쓴다. 같은 공시에 태그가 여럿이면 tag_rank가 작은 것. 4분기 = 연간 − 9개월 누계",
}
US_FIN_NOTE = ("재무 값은 USD. 주당 값(EPS)은 아직 만들지 않는다 (시가총액은 market_cap 필드: 분할 보정 주식수 × 종가). "
               "edgar_facts는 현재 구성종목만 → 생존편향 있음")
QDAYS = 365.25 / 4


def us_fin_facts() -> pd.DataFrame:
    """edgar_facts → 엔진용 표: symbol, field, tag_rank, start_d·end_d·filed_d(뉴욕 날짜), val, form.
    금액은 USD, 주식수는 shares 단위만 쓴다."""
    cols = ["symbol", "field", "tag_rank", "start_d", "end_d", "filed_d", "val", "form"]
    f = store.read("edgar_facts", "ALL")
    if f.empty:
        return pd.DataFrame(columns=cols)
    want = {v[0] for v in US_FIN_FIELDS.values()}
    f = f[f["field"].isin(want)]
    f = f[np.where(f["field"] == "shares_outstanding", f["unit"] == "shares", f["unit"] == "USD")]
    tz = TZ["us_stock"]
    f = f.assign(start_d=_local_date(pd.to_datetime(f["start"], utc=True), tz),
                 end_d=_local_date(pd.to_datetime(f["end"], utc=True), tz),
                 filed_d=_local_date(pd.to_datetime(f["filed"], utc=True), tz))
    return f[cols].reset_index(drop=True)


def _us_versions(f: pd.DataFrame, timing: str) -> pd.DataFrame:
    """같은 공시 안 여러 태그 → tag_rank 최소. first면 기간마다 가장 이른 공시 하나, last면 공시마다 하나(버전).
    t = 쓸 수 있는 날 (공시일 다음 날 → 다음 거래일 봉)."""
    f = f.sort_values("tag_rank").drop_duplicates(["symbol", "field", "start_d", "end_d", "filed_d"])
    if timing == "first":
        f = f.sort_values(["filed_d", "tag_rank"]).drop_duplicates(["symbol", "field", "start_d", "end_d"])
    return f.assign(t=f["filed_d"] + pd.Timedelta(days=1), v=f["val"].astype(float))


def _tl_combine(parts: list[pd.DataFrame], signs: list[float]) -> pd.DataFrame:
    """시간에 따라 바뀌는 값(symbol, k, t, v) 여러 개를 더한다. 시각 = 모든 부분의 변경 시각 합집합,
    그 시각의 각 부분 값 = 그때까지 알려진 최신 값. 한 부분이라도 아직 모르면 행을 만들지 않는다."""
    key = ["symbol", "k"]
    out = pd.concat([p[key + ["t"]] for p in parts]).drop_duplicates().sort_values("t").reset_index(drop=True)
    total = np.zeros(len(out))
    for p, sg in zip(parts, signs):
        m = pd.merge_asof(out, p[key + ["t", "v"]].sort_values("t"), on="t", by=key, direction="backward")
        total = total + sg * m["v"].to_numpy(float)
    return out.assign(v=total).dropna(subset=["v"])


def _us_timeline(f: pd.DataFrame, field: str, kind: str, timing: str) -> pd.DataFrame:
    """한 항목의 (symbol, k, t, v). instant: k = 기간 끝 날짜(일 번호). flow: k = 분기 번호, v = 그 분기까지 4개 분기 합."""
    v = _us_versions(f[f["field"] == field], timing)
    empty = pd.DataFrame(columns=["symbol", "k", "t", "v"])
    if v.empty:
        return empty
    if kind == "instant":
        return v.assign(k=(v["end_d"] - pd.Timestamp("1970-01-01")).dt.days)[["symbol", "k", "t", "v"]]
    days = (v["end_d"] - v["start_d"]).dt.days
    v = v.assign(k=((v["end_d"] - pd.Timestamp("1970-01-01")).dt.days / QDAYS).round().astype("Int64"))
    q = v[days.between(80, 100)]
    fy, m9 = v[days.between(350, 380)], v[days.between(260, 290)]
    # 4분기 3개월 = 연간 − 같은 시작일의 9개월 누계 (분기 보고서가 따로 낸 3개월 값이 있으면 그것을 쓴다)
    m9 = m9.drop(columns="k").merge(fy[["symbol", "start_d", "k"]].drop_duplicates(), on=["symbol", "start_d"])
    parts = [p[["symbol", "k", "t", "v"]] for p in (fy, m9)]
    q4 = _tl_combine(parts, [1.0, -1.0]) if len(fy) and len(m9) else empty
    have = set(zip(q["symbol"], q["k"]))
    q4 = q4[[(s, k) not in have for s, k in zip(q4["symbol"], q4["k"])]]
    quarters = pd.concat([q[["symbol", "k", "t", "v"]], q4], ignore_index=True)
    if quarters.empty:
        return empty
    quarters["k"] = quarters["k"].astype("int64")
    return _tl_combine([quarters.assign(k=quarters["k"] + j) for j in range(4)], [1.0] * 4)


def _us_attach(bars: pd.DataFrame, tl: pd.DataFrame) -> np.ndarray:
    """bars(date, symbol, 날짜순)마다: 그날까지 처음 알려진 가장 최근 k(늦게 알려진 옛 기간은 새 기간을 덮지 않음)의
    그날까지 최신 값. FIN_STALE_DAYS 넘게 새 기간이 없으면 NaN."""
    if tl.empty:
        return np.full(len(bars), np.nan)
    ev = tl.groupby(["symbol", "k"], as_index=False)["t"].min().rename(columns={"t": "known"})
    ev = ev.sort_values(["symbol", "known", "k"])
    ev = ev[ev["k"] >= ev.groupby("symbol")["k"].cummax()]
    ev = ev.drop_duplicates(["symbol", "known"], keep="last").sort_values("known")
    b = bars[["date", "symbol"]].reset_index(drop=True).assign(_row=lambda x: np.arange(len(x)))
    m = pd.merge_asof(b, ev, left_on="date", right_on="known", by="symbol", direction="backward",
                      tolerance=pd.Timedelta(days=FIN_STALE_DAYS)).dropna(subset=["k"])
    m["k"] = m["k"].astype("int64")
    tl = tl.assign(k=tl["k"].astype("int64")).sort_values("t")
    m = pd.merge_asof(m.sort_values("date"), tl, left_on="date", right_on="t", by=["symbol", "k"],
                      direction="backward")
    out = np.full(len(bars), np.nan)
    out[m["_row"].to_numpy()] = m["v"].to_numpy(float)
    return out


def us_fin_fields(basis: pd.DataFrame, facts: pd.DataFrame, fields, timing: str = "first") -> pd.DataFrame:
    """basis(date, symbol)마다 그날 알 수 있는 US 재무 값. 반환: date, symbol, *fields."""
    bars = basis[["date", "symbol"]].sort_values("date").reset_index(drop=True)
    out, cache = bars.copy(), {}
    for f in fields:
        src, kind = US_FIN_FIELDS[f]
        if (src, kind) not in cache:
            cache[(src, kind)] = _us_attach(bars, _us_timeline(facts, src, kind, timing))
        out[f] = cache[(src, kind)]
    return out


def us_fin_coverage(facts: pd.DataFrame, timing: str, symbols) -> dict:
    if facts.empty:
        return {"rows": 0, "timing": timing, "note": US_FIN_NOTE}
    syms = set(symbols)
    return {"rows": len(facts), "symbols": int(facts["symbol"].nunique()), "timing": timing,
            "first": str(facts["filed_d"].min().date()), "last": str(facts["filed_d"].max().date()),
            "amendments": int(facts["form"].astype(str).str.endswith("/A").sum()),
            "symbols_without_facts": sorted(syms - set(facts["symbol"]))[:30], "note": US_FIN_NOTE}


# ── 시가총액: 분할 보정 주식수 (us_stock) ──
US_MCAP_FIELDS = ("market_cap", "shares", "cik")
US_MCAP_TIMING = (
    "market_cap = us_1d 원본 종가(분할 반영·배당 미반영, USD) × 분할 보정 주식수(data.shares.daily_shares, 같은 분할 기준). "
    "주식수는 EDGAR 공시일 다음 거래일부터, 기준일(asof)이 us_shares.max_age_days(400일)보다 오래되면 NaN. "
    "회사 단위: 한 회사 종목이 여럿(GOOG·GOOGL 등)이면 둘 다 회사 전체 시총 → 순위는 cik로 묶어 하나만 (엔진은 빼지 않음). "
    "shares = 회사 전체 주식수를 그 종목 주식으로 환산한 값(close와 같은 분할 기준). currency=KRW여도 USD")
US_MCAP_LIMITS = (
    "V: B·C주를 빼서 시총이 약 5~20% 작다",
    "교환비가 1:1 아닌 합병 종목(LIN·DD·BKR·VTRS·AMCR·SW·SKYD·TKO 등)은 합병 전 기간이 비어 있다",
    "us_1d·edgar_facts는 현재 구성종목만 → 생존편향 (과거에 컸다가 빠진 회사가 순위에 없다)",
    "2011-06 이전 XBRL 없는 소형사(첫 1~1.5년)·외국 회사(20-F)는 비어 있다",
    "데이터 점검(shares_check)에서 단위 이상치가 남은 종목: BE(상장 직후 주식수), EXE(파산 재편 전 옛 주식)",
)


def us_mcap_source(symbols) -> dict:
    """시가총액 원천: edgar_facts 주식수 항목, edgar_class_shares, us_splits, us_1d 전체 기간 종가(공시일 시총 점검용).
    미래참조 검사가 cut 전 공시만으로 다시 만들 수 있게 원천 그대로 둔다."""
    from ..data import shares as sh
    syms = sorted(set(symbols))
    folder = store.DATA_DIR / "edgar_facts" / "ALL"
    facts = pd.DataFrame()
    if folder.exists() and any(folder.glob("*.parquet")):
        facts = store.query(f"SELECT symbol, cik, tag, unit, start, \"end\", val, accn, form, filed FROM "
                            f"'{folder.as_posix()}/*.parquet' WHERE field LIKE 'shares%' AND {_sym_filter(syms)}")
    return {"facts": facts, "class_shares": store.read("edgar_class_shares", "ALL"),
            "splits": store.read("us_splits", "ALL"), "bars": sh._bars(syms)}


def _mcap_events(src: dict, asof=None) -> pd.DataFrame:
    """공시별 채택 주식수 (shares.share_events). asof: 그날 전에 공시된 것만, 공시일 시총 점검 가격도 그날까지만.
    분할 이력은 자르지 않는다 — us_1d 가격이 오늘 기준으로 모든 분할을 반영해 저장돼 있어 배수가 가격과 상쇄된다."""
    from ..data import shares as sh
    facts, cls, bars = src["facts"], src["class_shares"], src["bars"]
    if asof is not None:
        a, tz = pd.Timestamp(asof), TZ["us_stock"]
        before = lambda d, c: d[_local_date(pd.to_datetime(d[c], utc=True), tz) < a] if len(d) else d
        facts, cls = before(facts, "filed"), before(cls, "filed")
        bars = bars[_local_date(pd.to_datetime(bars["ts"], utc=True), tz) <= a] if len(bars) else bars
    if facts is None or facts.empty:
        return pd.DataFrame()
    return sh.share_events(facts=facts, class_shares=cls, splits=src["splits"], bars=bars)


def us_mcap_fields(basis: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """basis(date, symbol, ts, _rc = us_1d 원본 종가)마다 그날 알 수 있는 shares, market_cap, cik."""
    from ..data import shares as sh
    b = basis[["date", "symbol", "ts"]].assign(close=basis["_rc"].to_numpy())
    if events is None or events.empty:
        return b[["date", "symbol"]].assign(shares=np.nan, market_cap=np.nan, cik=np.nan)
    # 같은 날 공시가 둘 이상이면(10-Q와 옛 기준일 8-K 등) 기준일(asof)이 가장 최근인 것 하나만.
    # 그대로 두면 merge_asof가 정렬 순서에 따라 아무거나 골라 결과가 입력 크기에 따라 달라진다 (KO 2016-10-27)
    ev = events[events["flag"] != "dropped"]
    fday = _local_date(pd.to_datetime(ev["filed"], utc=True), TZ["us_stock"])
    ev = ev.assign(_fd=fday).sort_values(["symbol", "_fd", "asof", "accn"])
    ev = ev.drop_duplicates(["symbol", "_fd"], keep="last").drop(columns="_fd")
    ev.attrs = dict(events.attrs)
    d = sh.daily_shares(events=ev, bars=b[["ts", "symbol", "close"]])
    out = b[["date", "symbol", "ts"]].merge(d[["ts", "symbol", "shares", "market_cap", "cik"]],
                                            on=["ts", "symbol"], how="left")
    out["cik"] = out["cik"].astype(float)
    return out[["date", "symbol", "shares", "market_cap", "cik"]]


def us_mcap_coverage(src: dict, events: pd.DataFrame, vals: pd.DataFrame) -> dict:
    """실행 시점 시가총액 원천 범위와 품질 (데이터 보강·수집으로 결과가 바뀌므로 report에 남긴다)."""
    from ..data import shares as sh
    out = {"timing": US_MCAP_TIMING, "limits": list(US_MCAP_LIMITS),
           "facts_rows": len(src["facts"]), "class_rows": len(src["class_shares"]), "split_rows": len(src["splits"])}
    if not len(src["splits"]):
        out["warning"] = "us_splits 없음 → 분할한 종목은 분할 전 시총이 분할 비율만큼 작게 나온다 (데이터 수집 전)"
    if events is None or events.empty:
        return out | {"events": 0}
    out |= {"events": len(events), "first_filed": str(pd.Timestamp(events["filed"].min()).date()),
            "last_filed": str(pd.Timestamp(events["filed"].max()).date()),
            "flags": {str(k): int(v) for k, v in events["flag"].value_counts().items()}}
    v = vals.dropna(subset=["market_cap"])
    if len(v):
        q = sh.quality(events, vals.rename(columns={"date": "ts"}))
        out["unit_outlier_symbols"] = sorted(q.loc[q["unit_outlier"] > 0, "symbol"])
    g = vals.dropna(subset=["cik"]).groupby("cik")["symbol"].unique()
    out["multi_class"] = {str(int(k)): sorted(s) for k, s in g.items() if len(s) > 1}
    out["symbols_without_shares"] = sorted(set(vals["symbol"]) - set(v["symbol"]))[:30]
    return out


def check_mcap_causality(panel: Panel, n_cuts: int = 3, tol: float = 1e-9) -> dict:
    """시가총액 필드 미래참조 검사: cut 전에 공시된 주식수(와 그때까지 가격)만으로 공시별 채택값을 다시 고르고
    cut까지 봉에 붙인 값이 전체로 만든 panel 값과 같아야 한다 (공시 당일부터 보이게 하거나, 뒤 공시가
    앞 채택값을 바꾸면 어긋난다)."""
    fields = [f for f in US_MCAP_FIELDS if f in panel]
    out = {"passed": True, "fields": fields, "cuts": []}
    if not fields or panel.mcap_source is None or panel.mcap_basis is None:
        return out
    idx, basis = panel.index, panel.mcap_basis
    for q in np.linspace(0.4, 0.85, n_cuts):
        cut = idx[int(len(idx) * q)]
        part = us_mcap_fields(basis[basis["date"] <= cut], _mcap_events(panel.mcap_source, asof=cut))
        bad = 0
        for f in fields:
            p = part.pivot(index="date", columns="symbol", values=f)
            full = panel[f].reindex(index=p.index, columns=p.columns).to_numpy(dtype=float)
            same = np.isclose(full, p.to_numpy(dtype=float), rtol=tol, atol=tol, equal_nan=True)
            bad += int((~same).any(axis=1).sum())
        out["cuts"].append({"cut": str(cut), "mismatch_bars": bad})
        out["passed"] &= bad == 0
    return out


def fin_build(market: str, basis: pd.DataFrame, src: pd.DataFrame, fields, timing: str, asof=None) -> pd.DataFrame:
    """시장별 재무 필드. asof: 그날 전에 공개된 원천만 (미래참조 검사용)."""
    if market == "kr_stock":
        return fin_fields(basis, src if asof is None else src[src["src_date"] < pd.Timestamp(asof)], fields)
    return us_fin_fields(basis, src if asof is None else src[src["filed_d"] < pd.Timestamp(asof)], fields, timing)


def check_fin_causality(panel: Panel, n_cuts: int = 3, tol: float = 1e-9) -> dict:
    """재무 필드 미래참조 검사: 중간 날짜 cut에서 '그날 전에 공개된 보고서'만으로 다시 만든 값이
    전체로 만든 panel 값과 cut까지 같아야 한다 (공개 당일이나 분기말부터 보이게 하면 어긋난다)."""
    fields = [f for f in ALL_FIN_FIELDS if f in panel]
    out = {"passed": True, "fields": fields, "cuts": []}
    if not fields or panel.fin_reports is None or panel.fin_basis is None:
        return out
    idx, basis = panel.index, panel.fin_basis
    timing = (panel.meta.get("fin_data") or {}).get("timing", "first")
    for q in np.linspace(0.4, 0.85, n_cuts):
        cut = idx[int(len(idx) * q)]
        part = fin_build(panel.market, basis[basis["date"] <= cut], panel.fin_reports, fields, timing, asof=cut)
        bad = 0
        for f in fields:
            p = part.pivot(index="date", columns="symbol", values=f)
            full = panel[f].reindex(index=p.index, columns=p.columns).to_numpy(dtype=float)
            same = np.isclose(full, p.to_numpy(dtype=float), rtol=tol, atol=tol, equal_nan=True)
            bad += int((~same).any(axis=1).sum())
        out["cuts"].append({"cut": str(cut), "mismatch_bars": bad})
        out["passed"] &= bad == 0
    return out


def _funding(px: pd.DataFrame, symbols, start, end, n: int, unit: str) -> tuple[pd.DataFrame, dict]:
    """binance_um_funding → (bucket, symbol, funding). 가격 봉과 같은 time_bucket으로 묶는다.
    정산 시각은 밀리초 단위로 흔들려(08:00:00.001) 분 단위로 내린 뒤 묶는다 → 08:00 정산은 08:00에 시작하는 봉."""
    src = _files("binance_um_funding", symbols, start, end)
    syms = set(px["symbol"])
    if not src:
        return pd.DataFrame(columns=["bucket", "symbol", "funding"]), {"symbols_without_funding": sorted(syms)}
    sym = "regexp_extract(filename, '[\\\\/]([^\\\\/]+)[\\\\/][^\\\\/]+$', 1)"
    t = "date_trunc('minute', CAST(ts AS TIMESTAMPTZ))"
    f = store.query(
        f"SELECT time_bucket(INTERVAL {n} {unit}, {t}) AS bucket, {sym} AS symbol, sum(funding_rate) AS funding, "
        f"count(*) AS n_events, min({t}) AS first_ts, max({t}) AS last_ts "
        f"FROM read_parquet({src}, filename=true) WHERE {_bounds(start, end, 'UTC')} GROUP BY 1, 2")
    f = f[f["symbol"].isin(syms)]
    for col in ("bucket", "first_ts", "last_ts"):
        if len(f) and f[col].dt.tz is None:
            f[col] = f[col].dt.tz_localize("UTC")
    meta = {"source": "binance_um_funding (봉 안 정산 합)", "events": int(f["n_events"].sum()),
            "symbols_without_funding": sorted(syms - set(f["symbol"]))}
    if len(f):
        meta["range"] = f"{f['first_ts'].min():%Y-%m-%d %H:%M}~{f['last_ts'].max():%Y-%m-%d %H:%M} UTC"
    return f[["bucket", "symbol", "funding"]], meta


def _crypto(market, symbols, start, end, fields, timeframe, funding: bool = True):
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
    meta = {"source": f"{ds} → {timeframe}", "rows": len(px), "partial_bars_dropped": int(partial.sum()), "note": ""}
    if market == "crypto_futures":
        if funding:
            f, meta["funding"] = _funding(px, symbols, start, end, n, unit)
            px = px.merge(f, on=["bucket", "symbol"], how="left")
            px["funding"] = px["funding"].fillna(0.0)       # 정산이 없는 봉은 0
        else:
            meta["note"] = "선물 펀딩비 미반영"
    px["date"] = px["bucket"].dt.tz_convert("UTC").dt.tz_localize(None)
    return px, meta
