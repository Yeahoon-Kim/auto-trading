"""배당 데이터 수집 (KR: OpenDART, US: Alpaca). 소스 검토는 docs/research/2026-11-16-dividend-data-sources.md.

- dart_dividend (symbol=ALL): 사업보고서 "배당에 관한 사항"(OpenDART alotMatter), 사업연도 2015~.
      회사·사업연도당 1행: 주당 현금배당금(보통·우선), 현금배당수익률, 현금배당성향, 배당총액, 주식배당 등.
      ts = 결산일, filed = 접수번호 앞 8자리(접수일; 정정보고서면 정정일) → "filed 다음 날부터 알 수 있음".
      값은 사업연도 연간 합계(중간·분기배당 포함). 상장폐지 종목도 DART 고유번호가 있으면 들어간다.
      회사당 연 1회 호출이라 하루 호출 한도(약 2만, dart_disclosures·dart_fin과 공유) 때문에
      실행마다 최대 dart_max_calls 만큼 진행하고 다음 실행에 이어받는다.
      배당 기록이 없는 회사·연도도 has_data=False 행으로 남겨 다시 묻지 않는다(사업보고서 마감 전이면 다시 확인).
- us_dividends (symbol=ALL): Alpaca corporate actions v1, 현금·주식배당, 배당락일 2016~.
      날짜 창으로 시장 전체를 받으므로 상장폐지 종목도 들어 있다(생존편향 없음). rate는 원본(분할 미반영) 주당 금액.
      공시일이 없다 → ts = 배당락일(ex_date). 배당은 배당락 전에 발표되므로 "ex_date에 알 수 있음"은 보수적이다.
- us_dividend_announcements (symbol=ALL): Alpaca corporate action announcements v2, 2020~.
      **declaration_date(이사회 결의·발표일)** 가 있다 → ts = declaration_date, "다음 날부터 알 수 있음".
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
PER_CLASS = {"dps", "stock_dps", "cash_yield", "stock_yield"}  # 보통주·우선주 따로 나오는 항목
COLUMNS = ["face_value", "net_income", "eps", "cash_div_total", "stock_div_total", "payout_ratio",
           "dps_common", "dps_pref", "cash_yield_common", "cash_yield_pref",
           "stock_dps_common", "stock_dps_pref", "stock_yield_common", "stock_yield_pref"]


def parse_alot(rows: list[dict]) -> dict:
    """alotMatter 한 회사·한 보고서의 list → {컬럼: 값}. 금액 단위는 원본 그대로
    (배당총액·순이익 백만원, 주당 값 원, 비율 %). 순이익·EPS는 연결 우선."""
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
            elif "보통" in knd:
                col += "_common"
            else:
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
        "has_data": bool(rows), "fetched": fetched, **{c: vals.get(c) for c in COLUMNS},
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
        ok = d["has_data"] | (d["fetched"] >= deadline) | (d["fetched"] > now - pd.Timedelta(days=14))
        final = set(zip(d.loc[ok, "corp_code"], d.loc[ok, "bsns_year"].astype(int)))
    out = []
    for y in years:  # 오래된 연도부터 (재시작해도 같은 순서)
        if now < pd.Timestamp(f"{y + 1}-03-01", tz="UTC"):  # 사업보고서가 나오기 전
            continue
        out += [(c, s, y) for c, s in zip(codes["corp_code"], codes["symbol"]) if (c, y) not in final]
    return out


def collect_dart_dividend() -> None:
    ds = "dart_dividend"
    if not os.environ.get("DART_API_KEY"):
        logger.info(f"{ds}: DART_API_KEY 없음 → 건너뜀")
        return
    cfg = config().get("dividends", {})
    budget = int(cfg.get("dart_max_calls", 8000))
    now = pd.Timestamp.now(tz="UTC")
    buf: list[dict] = []
    calls = 0
    try:
        codes = _corp_codes()
        prev = store.read(ds, "ALL")
        years = range(int(cfg.get("dart_start_year", 2015)), now.tz_convert(KST).year)
        todo = todo_pairs(codes, years, prev, now)
        for corp, sym, y in todo[:budget]:
            status, rows = _alot(corp, y)
            calls += 1
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
    collect_dart_dividend()  # 오래 걸리고 한도 소진형이라 마지막에
