"""DART 기업 연고표 (L004 국내 테마주 lab 요청, 사양: labs/L004-kr-theme-events/data-request.md 1-6).

사업보고서 주요정보 3종 + 기업개황을 **원문 그대로**(응답 문자열 그대로, 파싱은 lab에서) 저장한다.
- dart_executives (symbol=ALL)     : 임원 현황 `exctvSttus` — 이름·성별·출생연월·직위·등기 여부·상근 여부·담당 업무·
      주요 경력(`main_career`, 학력 포함 텍스트)·최대주주와의 관계·재직 기간·임기 만료일. 1행 = 보고서 1개의 임원 1명.
- dart_investments (symbol=ALL)    : 타법인 출자 현황 `otrCprInvstmntSttus` — 피출자 법인명(`inv_prm`)·출자 목적·
      기초/증감/기말 수량·지분율·장부가·피출자사 최근 재무. 1행 = 출자 1건.
- dart_major_holders (symbol=ALL)  : 최대주주 현황 `hyslrSttus` — 주주명·관계·주식 종류·기초/기말 주식 수·지분율. 1행 = 주주 1명.
- dart_company (symbol=ALL)        : 기업개황 `company` — 본사 주소(`adres`)·대표자·업종코드·설립일·결산월 등.
      **현재 값만 준다(과거 이력 없음)**. ts = 받은 날. company_refresh_days마다 다시 받아 그때부터의 변경만 쌓인다.
- dart_profile_calls (symbol=ALL)  : 진행 기록. (api, corp_code, bsns_year)당 1행 — status('000' 있음, '013' 없음),
      n_rows, rcept_no, fetched. 이어받기(받은 것 건너뜀)와 "없음" 재확인에 쓴다.

공통 열 (보고서 3종): ts = 결산일(stlm_dt, 없으면 사업연도 말), corp_code, symbol, bsns_year, reprt_code='11011',
  seq(응답 안 순서, 0부터), rcept_no, filed(접수번호 앞 8자리 = 받은 보고서의 접수일, 정정이면 정정일),
  first_filed(받을 때 dividends.first_filed()로 찾은 최초 접수일, 못 찾으면 NaT), fetched, 그리고 응답 필드 전부(문자열).
  키 = (corp_code, bsns_year, reprt_code, seq).

시점 규칙: 사업보고서 **최초 접수일 다음 날**부터 쓴다 (dart_dividend와 같은 규칙). first_filed 열이 비었으면
  dividends.first_filed()(kind='사업보고서', period=결산 'YYYY.MM')로 다시 붙인다. 값은 최신(정정) 보고서 값이다.

대상: dart_dividend·dart_periodic_filings·dart_disclosures로 **사업보고서가 있다고 확인된** (회사, 사업연도)만 묻는다
  (상장 전·상장폐지 뒤 연도에 헛호출하지 않게). 범위 = 사업연도 profile_start_year(2015)~, 2015년 이후 KRX 상장 종목
  (상장폐지 포함, dart_dividend와 같은 _corp_codes()).

호출 예산: 새 상한을 만들지 않는다. dividends.run()이 dividends.dart_max_calls(실행당 8천)를 아래 순서로 나눠 쓴다.
  1) dart_periodic_filings 백필  2) dart_dividend 백필·새 사업보고서  3) 이 연고표(남은 예산 전부).
  연고표 안에서는 기업개황(회사당 1회) → 보고서 3종(최근 사업연도부터, 회사마다 3종을 이어서).
  DART 020(하루 한도 초과)을 받으면 그 자리에서 멈추고 다음 실행이 이어받는다.
"""

from __future__ import annotations

import os

import pandas as pd

from .. import store
from .common import config, get_json, logger
from .dividends import KST, _dart_limit, _http, first_filed
from .kr_public import _corp_codes

APIS = {  # dataset → OpenDART 엔드포인트
    "dart_executives": "exctvSttus",
    "dart_investments": "otrCprInvstmntSttus",
    "dart_major_holders": "hyslrSttus",
}
CALLS_DS, COMPANY_DS = "dart_profile_calls", "dart_company"
REPORT_KEY = ["corp_code", "bsns_year", "reprt_code", "seq"]
CALLS_KEY = ["api", "corp_code", "bsns_year"]
_URL = "https://opendart.fss.or.kr/api/{}.json"


class DartLimit(RuntimeError):
    """OpenDART 020: 하루 요청 한도 초과."""


def _call(endpoint: str, params: dict) -> dict:
    _dart_limit.wait()
    j = get_json(_http, _URL.format(endpoint), {"crtfc_key": os.environ["DART_API_KEY"], **params})
    s = j.get("status", "")
    if s == "020":
        raise DartLimit(endpoint)
    if s not in ("000", "013"):
        raise RuntimeError(f"DART {endpoint} {s}")
    return j


# ── 무엇을 물을지 ──
def report_pairs(codes: pd.DataFrame, dividend: pd.DataFrame, filings: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """사업보고서가 있다고 확인된 (corp_code, symbol, bsns_year, period, first_filed).
    dividend: dart_dividend (has_data, ts=결산일). filings: dividends.first_filed() 결과.
    bsns_year = 결산일의 연도 (DART 규칙: 2016.03 결산 → 2016)."""
    parts = []
    if not dividend.empty:
        d = dividend[dividend["has_data"].astype(bool)]
        parts.append(pd.DataFrame({"corp_code": d["corp_code"], "bsns_year": d["bsns_year"].astype(int),
                                   "period": pd.to_datetime(d["ts"], utc=True).dt.tz_convert(KST).dt.strftime("%Y.%m")}))
    ff = filings[filings["kind"] == "사업보고서"] if not filings.empty else filings
    if not ff.empty:
        parts.append(pd.DataFrame({"corp_code": ff["corp_code"], "bsns_year": ff["period"].str[:4].astype(int),
                                   "period": ff["period"]}))
    if not parts:
        return pd.DataFrame(columns=["corp_code", "symbol", "bsns_year", "period", "first_filed"])
    p = pd.concat(parts, ignore_index=True).drop_duplicates(["corp_code", "bsns_year"])
    p = p[p["bsns_year"] >= start_year].merge(codes[["corp_code", "symbol"]], on="corp_code")
    if not ff.empty:
        p = p.merge(ff[["corp_code", "period", "first_filed"]], on=["corp_code", "period"], how="left")
    else:
        p["first_filed"] = pd.NaT
    return p[["corp_code", "symbol", "bsns_year", "period", "first_filed"]].reset_index(drop=True)


def finished(calls: pd.DataFrame, now: pd.Timestamp) -> set[tuple]:
    """더 묻지 않을 (api, corp_code, bsns_year). 받은 값이 있으면 끝. '없음'(013)은 사업보고서 마감·지연 제출을
    넉넉히 본 다음 해 7월 이후에 받은 것이면 끝, 그 전이면 14일 뒤 다시 묻는다 (dividends.todo_pairs와 같은 규칙)."""
    if calls.empty:
        return set()
    c = calls.copy()
    c["fetched"] = pd.to_datetime(c["fetched"], utc=True)
    deadline = pd.to_datetime((c["bsns_year"].astype(int) + 1).astype(str) + "-07-01", utc=True)
    ok = (c["status"] == "000") | (c["fetched"] >= deadline) | (c["fetched"] > now - pd.Timedelta(days=14))
    return set(zip(c.loc[ok, "api"], c.loc[ok, "corp_code"], c.loc[ok, "bsns_year"].astype(int)))


def todo_calls(pairs: pd.DataFrame, calls: pd.DataFrame, now: pd.Timestamp) -> list[tuple]:
    """물어볼 (dataset, corp_code, symbol, bsns_year, period, first_filed). 최근 사업연도부터, 회사마다 3종을 이어서
    (재시작해도 같은 순서)."""
    done = finished(calls, now)
    p = pairs.sort_values(["bsns_year", "corp_code"], ascending=[False, True])
    out = []
    for r in p.itertuples(index=False):
        for ds, ep in APIS.items():
            if (ep, r.corp_code, int(r.bsns_year)) not in done:
                out.append((ds, r.corp_code, r.symbol, int(r.bsns_year), r.period, r.first_filed))
    return out


def company_todo(codes: pd.DataFrame, company: pd.DataFrame, now: pd.Timestamp, refresh_days: int) -> list[str]:
    """기업개황을 받을 corp_code: 처음이거나 마지막으로 받은 지 refresh_days 지난 회사."""
    if company.empty:
        return codes["corp_code"].tolist()
    last = pd.to_datetime(company.groupby("corp_code")["fetched"].max(), utc=True)
    stale = now - pd.Timedelta(days=refresh_days)
    return [c for c in codes["corp_code"] if c not in last.index or last[c] < stale]


# ── 응답 → 행 ──
def report_rows(corp_code: str, symbol: str, year: int, period: str | None, first: pd.Timestamp | None,
                rows: list[dict], fetched: pd.Timestamp) -> pd.DataFrame:
    """보고서 API 응답 list → 원문 행. 응답 필드는 문자열 그대로 둔다 (줄바꿈·'-'·쉼표 숫자 포함)."""
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).astype("string")
    stlm = d["stlm_dt"].dropna().iloc[0] if "stlm_dt" in d and d["stlm_dt"].notna().any() else None
    if stlm:
        ts = pd.Timestamp(stlm, tz=KST)
    elif period:
        ts = (pd.Timestamp(period.replace(".", "-") + "-01") + pd.offsets.MonthEnd(0)).tz_localize(KST)
    else:
        ts = pd.Timestamp(f"{year}-12-31", tz=KST)
    rcept = d["rcept_no"].fillna("") if "rcept_no" in d else pd.Series("", index=d.index)
    meta = pd.DataFrame({
        "ts": ts.tz_convert("UTC"), "corp_code": corp_code, "symbol": symbol, "bsns_year": year,
        "reprt_code": "11011", "seq": range(len(d)),
        "filed": pd.to_datetime(rcept.str[:8], format="%Y%m%d", errors="coerce").dt.tz_localize(KST)
        .dt.tz_convert("UTC"),
        "first_filed": pd.Timestamp(first) if first is not None and not pd.isna(first) else pd.NaT,
        "fetched": fetched,
    }, index=d.index)
    meta["first_filed"] = pd.to_datetime(meta["first_filed"], utc=True).astype("datetime64[ns, UTC]")  # 비어도 UTC 날짜형 (파일 병합 때 형 충돌 방지)
    return pd.concat([meta, d.drop(columns=[c for c in ("corp_code",) if c in d])], axis=1)


def company_row(corp_code: str, j: dict, fetched: pd.Timestamp) -> dict:
    """기업개황 응답(평평한 dict) → 1행. ts = 받은 날(UTC 자정). 응답 필드는 문자열 그대로.
    '없음'(013)도 status만 있는 행으로 남겨 refresh_days 동안 다시 묻지 않는다."""
    row = {k: (None if v is None else str(v)) for k, v in j.items() if k != "message"}
    return {"ts": fetched.normalize(), "fetched": fetched, **row, "corp_code": corp_code}


# ── 수집 ──
def collect_company(codes: pd.DataFrame, budget: list[int], now: pd.Timestamp, refresh_days: int) -> None:
    todo = company_todo(codes, store.read(COMPANY_DS, "ALL"), now, refresh_days)
    buf: list[dict] = []

    def flush() -> None:
        if buf:
            store.write(COMPANY_DS, "ALL", pd.DataFrame(buf), key=["corp_code", "ts"])
            buf.clear()

    n = 0
    try:
        for corp in todo:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            j = _call("company", {"corp_code": corp})
            n += 1
            buf.append(company_row(corp, j, now))
            if len(buf) >= 500:
                flush()
    finally:
        flush()
        logger.info(f"{COMPANY_DS} {n:,}회 호출, 남은 회사 {len(todo) - n:,}")


def collect_reports(codes: pd.DataFrame, budget: list[int], now: pd.Timestamp, start_year: int) -> None:
    pairs = report_pairs(codes, store.read("dart_dividend", "ALL"), first_filed(), start_year)
    todo = todo_calls(pairs, store.read(CALLS_DS, "ALL"), now)
    bufs: dict[str, list[pd.DataFrame]] = {ds: [] for ds in APIS}
    log: list[dict] = []

    def flush() -> None:  # 데이터를 먼저 쓰고 진행 기록을 쓴다 (중간에 죽어도 빠지는 행이 없게)
        for ds, parts in bufs.items():
            parts = [p for p in parts if not p.empty]
            if parts:
                store.write(ds, "ALL", pd.concat(parts, ignore_index=True), key=REPORT_KEY)
            bufs[ds] = []
        if log:
            store.write(CALLS_DS, "ALL", pd.DataFrame(log), key=CALLS_KEY)
            log.clear()

    n = 0
    try:
        for ds, corp, sym, year, period, first in todo:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            j = _call(APIS[ds], {"corp_code": corp, "bsns_year": str(year), "reprt_code": "11011"})
            n += 1
            rows = (j.get("list") or []) if j.get("status") == "000" else []
            df = report_rows(corp, sym, year, period, first, rows, now)
            bufs[ds].append(df)
            log.append({"ts": df["ts"].iloc[0] if not df.empty else pd.Timestamp(f"{year}-12-31", tz=KST).tz_convert("UTC"),
                        "api": APIS[ds], "corp_code": corp, "symbol": sym, "bsns_year": year,
                        "status": j.get("status"), "n_rows": len(rows),
                        "rcept_no": rows[0].get("rcept_no") if rows else None, "fetched": now})
            if len(log) >= 600:
                flush()
    finally:
        flush()
        left = len(todo) - n
        store.set_state(CALLS_DS, "ALL", backfill_done=left == 0)
        logger.info(f"dart 연고표 보고서 {n:,}회 호출, 남은 호출 {left:,} (회사·연도 {len(pairs):,}쌍 × 3종 기준)")


def collect_dart_profile(budget: list[int]) -> None:
    """연고표 수집. budget은 dividends.run()이 넘기는 실행당 DART 호출 예산(앞선 백필이 쓰고 남은 것)."""
    cfg = config().get("dividends", {})
    now = pd.Timestamp.now(tz="UTC")
    try:
        if budget[0] <= 0:
            logger.info("dart 연고표: 이번 실행 예산 없음 → 다음 실행")
            return
        codes = _corp_codes()
        collect_company(codes, budget, now, int(cfg.get("profile_company_refresh_days", 90)))
        collect_reports(codes, budget, now, int(cfg.get("profile_start_year", 2015)))
    except DartLimit:
        budget[0] = 0
        logger.warning("dart 연고표: DART 하루 호출 한도 도달 → 다음 실행에 이어서")
    except Exception as e:
        store.set_state(CALLS_DS, "ALL", status="error", error=str(e)[:500])
        logger.error(f"dart 연고표 실패: {e}")
