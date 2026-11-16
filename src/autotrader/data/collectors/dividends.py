"""배당 데이터 수집 (KR: OpenDART, US: Alpaca). 소스 검토는 docs/research/2026-11-16-dividend-data-sources.md.

- dart_dividend (symbol=ALL): 사업보고서 "배당에 관한 사항"(OpenDART alotMatter), 사업연도 2015~.
      회사·사업연도당 1행: 주당 현금배당금(보통·우선), 현금배당수익률, 현금배당성향, 배당총액, 주식배당 등.
      ts = 결산일, filed = 접수번호 앞 8자리. DART는 최신(정정) 보고서를 주므로 filed가 정정일일 수 있다
      (22%가 결산일 +120일 뒤). 최초 접수일은 first_filed()로 붙인다 → "최초 접수일 다음 날부터 알 수 있음".
      (값은 정정 후 값이라, 정정이 배당 수치를 바꿨다면 최초 접수일 기준으로는 아주 약한 미래참조가 남는다.)
- dart_periodic_filings (symbol=ALL): 2015~2023 정기보고서(사업·반기·분기) 목록, 정정 포함. 회사당 1회 백필.
      2024~는 dart_disclosures에 있다. first_filed()가 둘을 합쳐 (회사, 보고서 종류, 기간)별 최초 접수일을 준다.
      값은 사업연도 연간 합계(중간·분기배당 포함). 상장폐지 종목도 DART 고유번호가 있으면 들어간다.
      회사당 연 1회 호출이라 하루 호출 한도(약 2만, dart_disclosures·dart_fin과 공유) 때문에
      실행마다 최대 dart_max_calls 만큼 진행하고 다음 실행에 이어받는다.
      배당 기록이 없는 회사·연도도 has_data=False 행으로 남겨 다시 묻지 않는다(사업보고서 마감 전이면 다시 확인).
- kr_dividend_schedule (symbol=ALL): KIS 예탁원 배당 일정, 기준일 2010~, 종목별(상장폐지 포함).
      ts = 기준일, ex_date = 배당락일(거래일로 계산), dps = 주당 현금배당금. 공시일이 없어 배당락 가격 보정용.
- us_dividends (symbol=ALL): Alpaca corporate actions v1, 현금·주식배당, 배당락일 2016~.
      날짜 창으로 시장 전체를 받으므로 상장폐지 종목도 들어 있다(생존편향 없음). rate는 원본(분할 미반영) 주당 금액.
      공시일이 없다 → ts = 배당락일(ex_date). 배당은 배당락 전에 발표되므로 "ex_date에 알 수 있음"은 보수적이다.
- us_dividend_announcements (symbol=ALL): Alpaca corporate action announcements v2, 2020~. ts = declaration_date.
      주의: 이 declaration_date는 실제 이사회 결의일이 아니다. 2022~는 대개 배당락 1~4일 전(AAPL 2026-02 배당:
      실제 결의 1/29, 여기 2/5), 2020~21은 배당락 뒤 날짜도 많다(나중에 채운 기록). 실제보다 늦으므로
      "다음 날부터 알 수 있음"으로 써도 미래참조는 없지만, 발표 이벤트 전략의 공시일로 쓰면 안 된다.
      2020~21에는 뮤추얼펀드(5글자 티커)가 많이 섞여 있다.
키: DART_API_KEY, ALPACA_API_KEY/ALPACA_SECRET_KEY (없으면 해당 소스는 건너뛴다).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd

from .. import store
from .common import RateLimiter, config, get_json, logger
from .kr_public import _corp_codes, _kst_ts, _num
from .us_alpaca import keys

KST, NY = "Asia/Seoul", "America/New_York"
_http = httpx.Client(timeout=60)
_dart_limit, _alpaca_limit = RateLimiter(8), RateLimiter(3)


# ── KR: OpenDART alotMatter ──
_FIELDS = [  # (se 라벨에 들어 있는 말, 컬럼). 공백 제거 후 비교. 앞에 있는 것부터 맞춘다.
    ("주당현금배당금", "dps"), ("주당주식배당", "stock_dps"), ("현금배당수익률", "cash_yield"),
    ("주식배당수익률", "stock_yield"), ("현금배당성향", "payout_ratio"), ("현금배당금총액", "cash_div_total"),
    ("주식배당금총액", "stock_div_total"), ("주당순이익", "eps"), ("당기순이익", "net_income"),
    ("주당액면가액", "face_value"),
]
PARSER = 2  # 파싱 규칙을 고치면 올린다 → 영향받는 행만 다시 받는다 (todo_pairs)
PER_CLASS = {"dps", "stock_dps", "cash_yield", "stock_yield"}  # 보통주·우선주 따로 나오는 항목
COLUMNS = ["face_value", "net_income", "eps", "cash_div_total", "stock_div_total", "payout_ratio",
           "dps_common", "dps_pref", "cash_yield_common", "cash_yield_pref",
           "stock_dps_common", "stock_dps_pref", "stock_yield_common", "stock_yield_pref"]


def parse_alot(rows: list[dict]) -> dict:
    """alotMatter 한 회사·한 보고서의 list → {컬럼: 값}. 금액 단위는 원본 그대로
    (배당총액·순이익 백만원, 주당 값 원, 비율 %). 순이익·EPS는 연결 우선.
    주식 종류(stock_knd)가 "-"로 오는 회사도 있다 → 그때는 순서대로 보통주, 우선주."""
    out: dict = {}
    for r in sorted(rows, key=lambda r: "연결" not in r.get("se", "")):  # 연결 행을 먼저
        se = r.get("se", "").replace(" ", "")
        col = next((c for k, c in _FIELDS if k in se), None)
        if col is None:
            continue
        if col in PER_CLASS:
            knd = r.get("stock_knd", "")
            if "우선" in knd:
                col += "_pref"
            elif "보통" in knd or f"{col}_common" not in out:
                col += "_common"
            elif "-" in knd or not knd.strip():
                col += "_pref"
            else:  # 종류주 등
                continue
        if col in out:
            continue
        v = _num(pd.Series([r.get("thstrm")])).iloc[0]
        out[col] = None if pd.isna(v) else float(v)
    return out


def _alot(corp_code: str, year: int) -> tuple[str, list[dict]]:
    _dart_limit.wait()
    j = get_json(_http, "https://opendart.fss.or.kr/api/alotMatter.json",
                 {"crtfc_key": os.environ["DART_API_KEY"], "corp_code": corp_code,
                  "bsns_year": str(year), "reprt_code": "11011"})
    return j.get("status", ""), j.get("list", []) or []


def _alot_row(corp_code: str, symbol: str, year: int, rows: list[dict], fetched: pd.Timestamp) -> dict:
    vals = parse_alot(rows)
    stlm = next((r.get("stlm_dt") for r in rows if r.get("stlm_dt")), None)
    rcept = next((r.get("rcept_no") for r in rows if r.get("rcept_no")), None)
    return {
        "ts": pd.Timestamp(stlm or f"{year}-12-31", tz=KST).tz_convert("UTC"),
        "corp_code": corp_code, "symbol": symbol, "bsns_year": year, "reprt_code": "11011",
        "rcept_no": rcept,
        "filed": _kst_ts(pd.Series([rcept[:8]])).iloc[0] if rcept else pd.NaT,
        "has_data": bool(rows), "fetched": fetched, "parser": PARSER, **{c: vals.get(c) for c in COLUMNS},
    }


def todo_pairs(codes: pd.DataFrame, years: range, done: pd.DataFrame, now: pd.Timestamp) -> list[tuple]:
    """아직 물어볼 (corp_code, symbol, year). 받은 값이 있으면 끝. 값이 없다는 답(has_data=False)은
    사업보고서 마감(다음 해 3월 말)과 지연 제출을 넉넉히 본 다음 해 7월 이후에 받은 것이면 끝,
    그 전에 받은 것이면 14일 지나서 다시 묻는다."""
    final: set = set()
    if not done.empty:
        d = done.copy()
        d["fetched"] = pd.to_datetime(d["fetched"], utc=True)
        deadline = pd.to_datetime((d["bsns_year"] + 1).astype(str) + "-07-01", utc=True)
        ver = d["parser"].fillna(1) if "parser" in d else pd.Series(1, index=d.index)
        # 1판 파서는 stock_knd가 "-"인 회사의 주당배당금을 놓쳤다 → 배당총액은 있는데 DPS가 빈 행은 다시
        col = lambda c: d[c] if c in d else pd.Series(float("nan"), index=d.index)
        stale = (ver < PARSER) & col("dps_common").isna() & (col("cash_div_total").fillna(0) > 0)
        ok = (d["has_data"] & ~stale) | (~d["has_data"] & ((d["fetched"] >= deadline)
                                                            | (d["fetched"] > now - pd.Timedelta(days=14))))
        final = set(zip(d.loc[ok, "corp_code"], d.loc[ok, "bsns_year"].astype(int)))
    out = []
    for y in years:  # 오래된 연도부터 (재시작해도 같은 순서)
        if now < pd.Timestamp(f"{y + 1}-03-01", tz="UTC"):  # 사업보고서가 나오기 전
            continue
        out += [(c, s, y) for c, s in zip(codes["corp_code"], codes["symbol"]) if (c, y) not in final]
    return out


def collect_dart_dividend(budget: list[int]) -> None:
    ds = "dart_dividend"
    cfg = config().get("dividends", {})
    now = pd.Timestamp.now(tz="UTC")
    buf: list[dict] = []
    calls = 0
    try:
        codes = _corp_codes()
        prev = store.read(ds, "ALL")
        years = range(int(cfg.get("dart_start_year", 2015)), now.tz_convert(KST).year)
        todo = todo_pairs(codes, years, prev, now)
        for corp, sym, y in todo[:max(0, budget[0])]:
            status, rows = _alot(corp, y)
            calls += 1
            budget[0] -= 1
            if status == "020":  # 하루 한도 초과 → 다음 실행에 이어서
                logger.warning(f"{ds}: DART 호출 한도 도달")
                break
            if status not in ("000", "013"):
                raise RuntimeError(f"DART alotMatter {status}")
            buf.append(_alot_row(corp, sym, y, rows if status == "000" else [], now))
            if len(buf) >= 500:
                store.write(ds, "ALL", pd.DataFrame(buf), key=["corp_code", "bsns_year", "reprt_code"])
                buf = []
        if buf:
            store.write(ds, "ALL", pd.DataFrame(buf), key=["corp_code", "bsns_year", "reprt_code"])
        store.set_state(ds, "ALL", backfill_done=calls >= len(todo))
        logger.info(f"{ds} {calls:,}회 호출, 남은 회사·연도 {len(todo) - calls:,}")
    except Exception as e:
        if buf:
            store.write(ds, "ALL", pd.DataFrame(buf), key=["corp_code", "bsns_year", "reprt_code"])
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


# ── KR: 정기보고서 최초 접수일 ──
# alotMatter·fnlttMultiAcnt는 최신(정정) 보고서의 접수번호를 준다 → 정정이 있으면 filed가 몇 년 늦어진다
# (예: 현대차 2015 사업보고서 → 2022-02 정정일). 최초 접수일은 공시 목록에서 얻는다.
# 2024~는 dart_disclosures에 이미 있으므로, 여기서는 그 전(2015~2023)만 회사당 1회 받는다.
FILINGS_DS, FILINGS_END = "dart_periodic_filings", "20231231"
_PERIOD = r"(사업보고서|반기보고서|분기보고서)\s*\((\d{4}\.\d{2})\)"


def parse_filings(d: pd.DataFrame) -> pd.DataFrame:
    """공시 목록(report_nm, rcept_no, rcept_dt|ts, corp_code, stock_code|symbol) → 정기보고서 행.
    period = 보고서 기간 끝 'YYYY.MM', correction = 정정·첨부 보고서 여부."""
    if d.empty:
        return pd.DataFrame()
    m = d["report_nm"].str.extract(_PERIOD)
    d = d.assign(kind=m[0], period=m[1]).dropna(subset=["kind"])
    ts = d["ts"] if "ts" in d else _kst_ts(d["rcept_dt"])
    return pd.DataFrame({
        "ts": ts, "rcept_no": d["rcept_no"], "corp_code": d["corp_code"],
        "symbol": (d["symbol"] if "symbol" in d else d["stock_code"]).str.strip(),
        "kind": d["kind"], "period": d["period"], "report_nm": d["report_nm"].str.strip(),
        "correction": d["report_nm"].str.strip().str.startswith("["),
    })


def first_filed() -> pd.DataFrame:
    """(corp_code, kind, period)별 최초 접수일. dart_periodic_filings(2015~2023) + dart_disclosures(2024~).
    dart_dividend는 kind='사업보고서', period=결산일 'YYYY.MM'으로 붙인다:
        dv['period'] = dv['ts'].dt.tz_convert('Asia/Seoul').dt.strftime('%Y.%m')"""
    parts = [store.read(FILINGS_DS, "ALL"), parse_filings(store.read("dart_disclosures", "ALL"))]
    f = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if f.empty:
        return f
    return (f.groupby(["corp_code", "kind", "period"], as_index=False)
            .agg(first_filed=("ts", "min"), last_filed=("ts", "max"), n_filings=("rcept_no", "nunique")))


def _filings(corp_code: str, start_year: int) -> tuple[str, pd.DataFrame, int]:
    rows, page, calls = [], 1, 0
    while True:
        _dart_limit.wait()
        calls += 1
        j = get_json(_http, "https://opendart.fss.or.kr/api/list.json",
                     {"crtfc_key": os.environ["DART_API_KEY"], "corp_code": corp_code, "bgn_de": f"{start_year}0101",
                      "end_de": FILINGS_END, "pblntf_ty": "A", "page_no": page, "page_count": 100})
        status = j.get("status", "")
        if status != "000":
            return status, pd.DataFrame(rows), calls
        rows += j["list"]
        if page >= int(j["total_page"]):
            return status, pd.DataFrame(rows), calls
        page += 1


def collect_dart_filings(budget: list[int]) -> None:
    """회사당 한 번(2015~2023 정기보고서 목록). 진행은 (dataset, corp_code) 수집 상태로 남긴다."""
    ds = FILINGS_DS
    try:
        codes = _corp_codes()
        start = int(config().get("dividends", {}).get("dart_start_year", 2015))
        todo = [c for c in codes["corp_code"] if not store.get_state(ds, c)["backfill_done"]]
        buf: list[pd.DataFrame] = []
        got: list[str] = []

        def flush() -> None:
            df = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame()
            store.write(ds, "ALL", df, key=["rcept_no"])
            for c in got:  # 저장한 뒤에 완료 표시 (중간에 죽어도 빠지는 회사가 없게)
                store.set_state(ds, c, backfill_done=True)
            buf.clear()
            got.clear()

        done = 0
        for corp in todo:
            if budget[0] <= 0:
                break
            status, d, calls = _filings(corp, start)
            budget[0] -= calls
            if status == "020":
                logger.warning(f"{ds}: DART 호출 한도 도달")
                budget[0] = 0
                break
            if status not in ("000", "013"):
                raise RuntimeError(f"DART list {status}")
            buf.append(parse_filings(d))
            got.append(corp)
            done += 1
            if len(got) >= 200:
                flush()
        flush()
        logger.info(f"{ds} 이번 실행 {done:,}개사, 남은 {len(todo) - done:,}개사")
    except Exception as e:
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


# ── KR: KIS 예탁원 배당 일정 (기준일·배당락일) ──
# 종목별로 조회하면 2010~ 전체가 나온다 (시장 전체를 날짜로 조회하면 100건에서 잘린다).
# 공시일은 없다 → 신호가 아니라 배당락일 가격 보정(총수익)용. 신호 시점은 dart_dividend + first_filed().
SCHED_DS = "kr_dividend_schedule"


def ex_dates(record: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """기준일 → 배당락일. 결제가 T+2라 '기준일 이하 마지막 거래일'의 바로 전 거래일이 배당락일이다.
    예: 2024-12-31(휴장) 기준일 → 마지막 거래일 12-30 → 배당락 12-27. 거래일 범위 밖이면 NaT."""
    out = pd.Series(pd.NaT, index=record.index, dtype="datetime64[ns]")
    td = trading_days.sort_values()
    if not len(td):
        return out
    r = record.to_numpy(dtype="datetime64[ns]")
    pos = td.searchsorted(r, side="right") - 2  # 기준일 이하 마지막 거래일의 위치 − 1
    ok = (pos >= 0) & ~pd.isna(r) & (r <= td[-1].to_datetime64())  # 마지막 거래일 뒤 기준일은 아직 모름
    out.iloc[ok] = td[pos[ok]]
    return out


def parse_schedule(rows: list[dict], code: str, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """KIS HHKDB669102C0 output1 → 행. dps = 주당 현금배당금(원), divi_rate = 액면 대비 배당률(%)."""
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d = d[d["sht_cd"].str.strip() == code]  # 조회 종목만
    if d.empty:
        return pd.DataFrame()
    rec = pd.to_datetime(d["record_date"], format="%Y%m%d", errors="coerce")
    date = lambda c: pd.to_datetime(d[c].str.replace("/", ""), format="%Y%m%d", errors="coerce")
    out = pd.DataFrame({
        "ts": rec.dt.tz_localize(KST).dt.tz_convert("UTC"), "symbol": code, "name": d["isin_name"].str.strip(),
        "kind": d["divi_kind"].str.strip(), "stock_kind": d["stk_kind"].str.strip(),
        "dps": _num(d["per_sto_divi_amt"]), "divi_rate": _num(d["divi_rate"]),
        "stock_divi_rate": _num(d["stk_divi_rate"]), "face_value": _num(d["face_val"]),
        "pay_date": date("divi_pay_dt"), "stock_pay_date": date("stk_div_pay_dt"),
        "high_dividend": d["high_divi_gb"].str.strip(),
        "ex_date": ex_dates(rec, trading_days).to_numpy(),
    })
    return out.dropna(subset=["ts"]).drop_duplicates(["ts", "symbol", "kind", "stock_kind"], keep="last")


def schedule_rows(fetch, code: str, start: str, end: str) -> list[dict]:
    """fetch(code, start, end) → rows. 한 번에 100건까지만 오므로 100건이면 기간을 반으로 나눠 다시 받는다 (월배당 종목)."""
    rows = fetch(code, start, end)
    if len(rows) < 100:
        return rows
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if (b - a).days < 2:
        logger.warning(f"{SCHED_DS} {code} {start}~{end}: 100건 — 더 나눌 수 없음")
        return rows
    mid = a + (b - a) / 2
    return (schedule_rows(fetch, code, start, mid.strftime("%Y%m%d"))
            + schedule_rows(fetch, code, (mid + pd.Timedelta(days=1)).strftime("%Y%m%d"), end))


def _updated_at(ds: str) -> dict[str, pd.Timestamp]:
    with store._db() as con:
        rows = con.execute("SELECT symbol, updated_at FROM state WHERE dataset=? AND backfill_done=1", (ds,)).fetchall()
    return {s: pd.Timestamp(t) for s, t in rows}


def collect_kr_schedule() -> None:
    """종목별 1회 조회(2010~). 상장 중인 종목은 refresh_days마다 다시 묻고(새 배당 반영), 상장폐지 종목은 한 번이면 끝."""
    from .kr_stock import _get  # KIS 클라이언트는 쓸 때만 만든다

    ds = SCHED_DS
    cfg = config().get("dividends", {})
    start = cfg.get("kis_start", "2010-01-01").replace("-", "")
    refresh = pd.Timedelta(days=int(cfg.get("kis_refresh_days", 7)))
    now = pd.Timestamp.now(tz="UTC")
    try:
        src = f"'{store.DATA_DIR.as_posix()}/krx_1d/ALL/*.parquet'"
        kday = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert(KST).dt.tz_localize(None).dt.normalize()
        days = pd.DatetimeIndex(sorted(kday(store.query(f"SELECT DISTINCT ts FROM {src}")["ts"])))
        syms = store.query(f"SELECT symbol, max(ts) last FROM {src} WHERE ts >= '{cfg.get('kis_start', '2010-01-01')}' "
                           "GROUP BY symbol")
        listed = set(syms.loc[kday(syms["last"]) == days[-1], "symbol"])
        seen = _updated_at(ds)
        todo = [s for s in sorted(syms["symbol"])
                if s not in seen or (s in listed and seen[s] < now - refresh)]
        end = (pd.Timestamp.now(tz=KST) + pd.Timedelta(days=365)).strftime("%Y%m%d")  # 발표된 미래 기준일도
        buf, got = [], []

        def flush() -> None:
            store.write(ds, "ALL", pd.concat(buf, ignore_index=True) if buf else pd.DataFrame(),
                        key=["ts", "symbol", "kind", "stock_kind"])
            for s in got:
                store.set_state(ds, s, backfill_done=True)
            buf.clear()
            got.clear()

        def fetch(code: str, f: str, t: str) -> list[dict]:
            j = _get("/uapi/domestic-stock/v1/ksdinfo/dividend", "HHKDB669102C0",
                     {"CTS": "", "GB1": "0", "F_DT": f, "T_DT": t, "SHT_CD": code, "HIGH_GB": ""})
            return j.get("output1") or []

        for i, code in enumerate(todo, 1):
            rows = schedule_rows(fetch, code, start, end)
            buf.append(parse_schedule(rows, code, days))
            got.append(code)
            if len(got) >= 300:
                flush()
                logger.info(f"{ds} {i:,}/{len(todo):,}")
        flush()
        store.set_state(ds, "ALL", backfill_done=True)
        logger.info(f"{ds} {len(todo):,}종목 조회")
    except Exception as e:
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


# ── US: Alpaca ──
def _ny_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(NY).dt.tz_convert("UTC")


def parse_ca(j: dict) -> pd.DataFrame:
    """v1 corporate-actions 응답 → 배당 행 (현금·주식배당)."""
    parts = []
    for kind in ("cash_dividends", "stock_dividends"):
        rows = (j.get("corporate_actions") or {}).get(kind) or []
        if rows:
            parts.append(pd.DataFrame(rows).assign(kind=kind.removesuffix("s")))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    for c in ("cusip", "record_date", "payable_date", "process_date", "special", "foreign"):
        if c not in d:
            d[c] = None
    out = pd.DataFrame({
        "ts": _ny_ts(d["ex_date"]), "id": d["id"], "symbol": d["symbol"], "cusip": d["cusip"].replace("", None),
        "kind": d["kind"], "rate": pd.to_numeric(d["rate"], errors="coerce"),
        "record_date": pd.to_datetime(d["record_date"], errors="coerce"),
        "payable_date": pd.to_datetime(d["payable_date"], errors="coerce"),
        "process_date": pd.to_datetime(d["process_date"], errors="coerce"),
        "special": d["special"].eq(True), "foreign": d["foreign"].eq(True),
    })
    return out.dropna(subset=["ts"])


def parse_ann(rows: list[dict]) -> pd.DataFrame:
    """v2 announcements 응답 → 배당 발표 행."""
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    for c in ("declaration_date", "ex_date", "record_date", "payable_date", "cash", "old_rate", "new_rate",
              "ca_sub_type", "target_symbol", "initiating_original_cusip", "corporate_action_id"):
        if c not in d:
            d[c] = None
    out = pd.DataFrame({
        "ts": _ny_ts(d["declaration_date"]), "id": d["id"], "corporate_action_id": d["corporate_action_id"],
        "symbol": d["initiating_symbol"], "target_symbol": d["target_symbol"],
        "cusip": d["initiating_original_cusip"], "sub_type": d["ca_sub_type"],
        "cash": pd.to_numeric(d["cash"], errors="coerce"),
        "old_rate": pd.to_numeric(d["old_rate"], errors="coerce"),
        "new_rate": pd.to_numeric(d["new_rate"], errors="coerce"),
        "ex_date": pd.to_datetime(d["ex_date"], errors="coerce"),
        "record_date": pd.to_datetime(d["record_date"], errors="coerce"),
        "payable_date": pd.to_datetime(d["payable_date"], errors="coerce"),
    })
    return out.dropna(subset=["ts"])


def windows(start: pd.Timestamp, end: pd.Timestamp, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end]를 days일짜리 창(양 끝 포함)으로 나눈다."""
    out, s = [], start.normalize()
    while s <= end:
        e = min(s + pd.Timedelta(days=days - 1), end)
        out.append((s, e))
        s = e + pd.Timedelta(days=1)
    return out


def _resume_start(ds: str, first: str, lookback_days: int) -> pd.Timestamp:
    """처음이면 first부터, 아니면 마지막으로 끝낸 날짜 − lookback (늦게 들어온 정정·추가분 반영)."""
    st = store.get_state(ds, "ALL")
    if st["last_ts"] is None:
        return pd.Timestamp(first)
    return max(pd.Timestamp(first), st["last_ts"].tz_localize(None).normalize() - pd.Timedelta(days=lookback_days))


def collect_us_dividends(headers: dict) -> None:
    ds = "us_dividends"
    cfg = config().get("dividends", {})
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    # store.write가 last_ts를 미래 배당락일까지 올리므로 미래 창(90일) + 늦은 정정분(60일)만큼 되돌아간다
    start = _resume_start(ds, cfg.get("alpaca_start", "2016-01-01"), 150)
    n = 0
    try:
        for s, e in windows(start, today + pd.Timedelta(days=90), 31):  # 발표된 미래 배당락도 받는다
            token, parts = None, []
            while True:
                _alpaca_limit.wait()
                p = {"types": "cash_dividend,stock_dividend", "start": f"{s:%Y-%m-%d}", "end": f"{e:%Y-%m-%d}",
                     "limit": 1000, "sort": "asc"}
                if token:
                    p["page_token"] = token
                j = get_json(_http, "https://data.alpaca.markets/v1/corporate-actions", p, headers=headers)
                parts.append(parse_ca(j))
                token = j.get("next_page_token")
                if not token:
                    break
            df = pd.concat(parts, ignore_index=True) if any(not x.empty for x in parts) else pd.DataFrame()
            n += store.write(ds, "ALL", df, key=["id"])
            store.set_state(ds, "ALL", last_ts=pd.Timestamp(min(e, today), tz="UTC"))  # 빈 창도 진행 위치 기록
        store.set_state(ds, "ALL", backfill_done=True)
        logger.info(f"{ds} {n:,}행")
    except Exception as e:
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


def collect_us_announcements(headers: dict) -> None:
    ds = "us_dividend_announcements"
    cfg = config().get("dividends", {})
    today = pd.Timestamp.now(tz=NY).tz_localize(None).normalize()
    start = _resume_start(ds, cfg.get("alpaca_ann_start", "2020-01-01"), 30)
    n = 0
    try:
        for s, e in windows(start, today, 30):  # API 창 한도 90일
            _alpaca_limit.wait()
            rows = get_json(_http, "https://paper-api.alpaca.markets/v2/corporate_actions/announcements",
                            {"ca_types": "dividend", "since": f"{s:%Y-%m-%d}", "until": f"{e:%Y-%m-%d}",
                             "date_type": "declaration_date"}, headers=headers)
            n += store.write(ds, "ALL", parse_ann(rows), key=["id"])
            store.set_state(ds, "ALL", last_ts=pd.Timestamp(e, tz="UTC"))
        store.set_state(ds, "ALL", backfill_done=True)
        logger.info(f"{ds} {n:,}행")
    except Exception as e:
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


def run(symbols: list[str] | None = None) -> None:
    headers = keys()
    if headers is None:
        logger.info("us_dividends: ALPACA 키 없음 → 건너뜀")
    else:
        collect_us_dividends(headers)
        collect_us_announcements(headers)
    if os.environ.get("KIS_LIVE_APP_KEY"):
        collect_kr_schedule()
    else:
        logger.info(f"{SCHED_DS}: KIS 키 없음 → 건너뜀")
    if not os.environ.get("DART_API_KEY"):
        logger.info("dart_dividend: DART_API_KEY 없음 → 건너뜀")
        return
    # 오래 걸리고 한도 소진형이라 마지막에. 두 수집기가 실행당 호출 상한을 나눠 쓴다
    budget = [int(config().get("dividends", {}).get("dart_max_calls", 8000))]
    collect_dart_filings(budget)  # 회사당 1회, 약 3천 호출로 끝남
    collect_dart_dividend(budget)
