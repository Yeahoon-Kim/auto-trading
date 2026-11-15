"""국내 공공 API 수집: KRX 전 종목 일별 시세, DART 공시 목록, 한국은행 ECOS 거시 지표.

- krx_1d (symbol=ALL, 종목별 행) : KOSPI·KOSDAQ 전 종목 원본 일봉 + 시가총액 + 상장주식수. 2010~.
      그날 상장돼 있던 종목이 전부 들어 있으므로 생존편향이 없다. KRX는 다음 영업일에 공개한다.
- dart_disclosures (symbol=ALL)  : 전체 상장사 공시 목록. 접수일(rcept_dt)만 있으므로
      일봉 전략에서는 "접수일 다음 날부터 알 수 있음"으로 쓴다. 하루 호출 한도(약 2만) 때문에
      백필은 실행마다 최대 MAX_DART_CALLS 만큼 진행한다.
- dart_fin (symbol=ALL)          : 상장사 주요계정 재무(매출·영업이익·순이익·자산·부채·자본), 2015~.
      손익은 3개월 값과 _cum(누적). 4분기는 보고서가 없으므로 FY − Q3 누적으로 계산해서 쓴다.
      filed = 접수번호 앞 8자리(실제 접수일) → 시점 규칙은 "filed 다음 날부터 알 수 있음".
- ecos (symbol=시계열 이름)      : 기준금리, 국고채 3·10년, 회사채 AA-, CD91, 원/달러.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd

from .. import store
from .common import RateLimiter, get_json, logger

KST = "Asia/Seoul"
KRX_API = "https://data-dbg.krx.co.kr/svc/apis/sto"
KRX_START, DART_START = "2010-01-01", "2024-01-01"
MAX_DART_CALLS = 15_000
ECOS = {  # 이름: (통계코드, 주기, 항목코드)
    "base_rate": ("722Y001", "D", "0101000"),
    "ktb_3y": ("817Y002", "D", "010200000"),
    "ktb_10y": ("817Y002", "D", "010210000"),
    "corp_aa_3y": ("817Y002", "D", "010300000"),
    "cd_91d": ("817Y002", "D", "010502000"),
    "usdkrw_ecos": ("731Y001", "D", "0000001"),
}
_http = httpx.Client(timeout=60)
_krx_limit, _dart_limit = RateLimiter(5), RateLimiter(8)


def _kst_ts(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, format="%Y%m%d").dt.tz_localize(KST).dt.tz_convert("UTC")


# ── KRX ──
def _krx_day(day: str) -> pd.DataFrame:
    rows = []
    for path in ("stk_bydd_trd", "ksq_bydd_trd"):
        _krx_limit.wait()
        r = get_json(_http, f"{KRX_API}/{path}", {"basDd": day}, headers={"AUTH_KEY": os.environ["KRX_API_KEY"]})
        rows += r.get("OutBlock_1", [])
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    num = lambda c: pd.to_numeric(d[c].str.replace(",", ""), errors="coerce")  # noqa: E731
    return pd.DataFrame({
        "ts": _kst_ts(d["BAS_DD"]), "symbol": d["ISU_CD"], "name": d["ISU_NM"], "market": d["MKT_NM"],
        "section": d["SECT_TP_NM"], "open": num("TDD_OPNPRC"), "high": num("TDD_HGPRC"), "low": num("TDD_LWPRC"),
        "close": num("TDD_CLSPRC"), "volume": num("ACC_TRDVOL"), "value": num("ACC_TRDVAL"),
        "market_cap": num("MKTCAP"), "shares": num("LIST_SHRS"),
    })


def collect_krx() -> None:
    ds = "krx_1d"
    st = store.get_state(ds, "ALL")
    start = (st["last_ts"].tz_convert(KST).normalize() + pd.Timedelta(days=1)) if st["last_ts"] is not None \
        else pd.Timestamp(KRX_START, tz=KST)
    days = pd.bdate_range(start, pd.Timestamp.now(tz=KST).normalize() - pd.Timedelta(days=1))
    buf, n = [], 0
    try:
        for d in days:
            df = _krx_day(d.strftime("%Y%m%d"))  # 휴장일·미공개일은 빈 값
            if not df.empty:
                buf.append(df)
            if len(buf) >= 60:
                n += store.write(ds, "ALL", pd.concat(buf), key=["ts", "symbol"])
                buf = []
        if buf:
            n += store.write(ds, "ALL", pd.concat(buf), key=["ts", "symbol"])
        store.set_state(ds, "ALL", backfill_done=True)
        logger.info(f"{ds} {len(days)}영업일 확인, {n:,}행")
    except Exception as e:
        if buf:
            store.write(ds, "ALL", pd.concat(buf), key=["ts", "symbol"])
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


# ── DART ──
def _dart_day(day: str, budget: list[int]) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        _dart_limit.wait()
        budget[0] -= 1
        j = get_json(_http, "https://opendart.fss.or.kr/api/list.json",
                     {"crtfc_key": os.environ["DART_API_KEY"], "bgn_de": day, "end_de": day,
                      "page_no": page, "page_count": 100})
        if j.get("status") == "013":  # 조회 결과 없음
            break
        if j.get("status") != "000":
            raise RuntimeError(f"DART {j.get('status')}: {j.get('message')}")
        rows += j["list"]
        if page >= int(j["total_page"]):
            break
        page += 1
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return pd.DataFrame({
        "ts": _kst_ts(d["rcept_dt"]), "rcept_no": d["rcept_no"], "symbol": d["stock_code"].str.strip(),
        "corp_code": d["corp_code"], "corp_name": d["corp_name"], "corp_cls": d["corp_cls"],
        "report_nm": d["report_nm"].str.strip(), "filer": d["flr_nm"], "remark": d["rm"],
    })


def collect_dart() -> None:
    ds = "dart_disclosures"
    st = store.get_state(ds, "ALL")
    start = (st["last_ts"].tz_convert(KST).normalize() + pd.Timedelta(days=1)) if st["last_ts"] is not None \
        else pd.Timestamp(DART_START, tz=KST)
    days = pd.date_range(start, pd.Timestamp.now(tz=KST).normalize() - pd.Timedelta(days=1))
    budget, n, done = [MAX_DART_CALLS], 0, True
    try:
        for d in days:
            if budget[0] <= 30:  # 오늘 한도 소진 → 다음 실행에 이어서
                done = False
                break
            df = _dart_day(d.strftime("%Y%m%d"), budget)
            n += store.write(ds, "ALL", df, key=["rcept_no"])
            if df.empty:  # 공시 없는 날(주말·휴일)도 진행 위치를 남긴다
                store.set_state(ds, "ALL", last_ts=d.tz_convert("UTC"))
        store.set_state(ds, "ALL", backfill_done=done)
        logger.info(f"{ds} {n:,}건 (남은 호출 {budget[0]})")
    except Exception as e:
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


# ── DART 재무 (주요계정) ──
FIN_START = 2015  # fnlttMultiAcnt 제공 시작 연도
REPORTS = {"11013": "Q1", "11012": "H1", "11014": "Q3", "11011": "FY"}  # 4분기 보고서는 제도상 없음
ACCOUNTS = {  # DART account_nm → 컬럼
    "매출액": "revenue", "영업이익": "operating_income", "영업이익(손실)": "operating_income",
    "법인세차감전 순이익": "pretax_income", "당기순이익(손실)": "net_income", "총포괄손익": "comprehensive_income",
    "유동자산": "current_assets", "비유동자산": "noncurrent_assets", "자산총계": "total_assets",
    "유동부채": "current_liabilities", "비유동부채": "noncurrent_liabilities", "부채총계": "total_liabilities",
    "자본금": "capital_stock", "이익잉여금": "retained_earnings", "자본총계": "total_equity",
}
FLOW = ("revenue", "operating_income", "pretax_income", "net_income", "comprehensive_income")


def _corp_codes() -> pd.DataFrame:
    """2015년 이후 KRX에 상장돼 있던 종목(상장폐지 포함)의 DART 고유번호."""
    import io
    import xml.etree.ElementTree as ET
    import zipfile

    _dart_limit.wait()
    r = _http.get("https://opendart.fss.or.kr/api/corpCode.xml", params={"crtfc_key": os.environ["DART_API_KEY"]})
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    root = ET.fromstring(z.read(z.namelist()[0]))
    m = pd.DataFrame([{"corp_code": e.findtext("corp_code").strip(), "symbol": e.findtext("stock_code").strip()}
                      for e in root.iter("list")])
    m = m[m["symbol"] != ""]
    krx = store.query(f"SELECT DISTINCT symbol FROM '{store.DATA_DIR.as_posix()}/krx_1d/ALL/*.parquet' "
                      f"WHERE ts >= '{FIN_START}-01-01'")
    return m[m["symbol"].isin(krx["symbol"])].drop_duplicates("symbol").reset_index(drop=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip().replace({"": None, "-": None}), errors="coerce")


def _dart_fin(codes: list[str], year: int, reprt: str) -> pd.DataFrame:
    """100개사 묶음 × 한 보고서 → 회사당 1행 (연결 우선, 없으면 별도). 손익은 3개월 값 + _cum(누적)."""
    _dart_limit.wait()
    j = get_json(_http, "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json",
                 {"crtfc_key": os.environ["DART_API_KEY"], "corp_code": ",".join(codes),
                  "bsns_year": str(year), "reprt_code": reprt})
    if j.get("status") == "013":
        return pd.DataFrame()
    if j.get("status") != "000":
        raise RuntimeError(f"DART fin {j.get('status')}: {j.get('message')}")
    d = pd.DataFrame(j["list"])
    d["col"] = d["account_nm"].str.strip().map(ACCOUNTS)
    d = d.dropna(subset=["col"])
    d = d[d["fs_div"] == d.groupby("corp_code")["fs_div"].transform("min")]  # 'CFS' < 'OFS' → 연결 우선
    d["ord"] = pd.to_numeric(d["ord"], errors="coerce")
    d = d.sort_values("ord").drop_duplicates(["corp_code", "col"], keep="first")  # 당기순이익 2행 중 첫 행
    d["v"] = _num(d["thstrm_amount"])
    d["v_cum"] = _num(d["thstrm_add_amount"]) if "thstrm_add_amount" in d else float("nan")
    wide = d.pivot(index="corp_code", columns="col", values="v")
    cum = d[d["col"].isin(FLOW)].pivot(index="corp_code", columns="col", values="v_cum").add_suffix("_cum")
    meta = d.groupby("corp_code").agg(symbol=("stock_code", "first"), rcept_no=("rcept_no", "first"),
                                      fs_div=("fs_div", "first"), currency=("currency", "first"),
                                      period_end=("thstrm_dt",  # "2025.01.01 ~ 2025.06.30", "2025.06.30 현재"
                                                  lambda s: s.str.findall(r"\d{4}\.\d{2}\.\d{2}").str[-1].max()))
    out = meta.join(wide).join(cum).reset_index()
    if reprt == "11011":  # 사업보고서: 당기 = 연간 = 누적
        for c in FLOW:
            if c in out:
                out[f"{c}_cum"] = out[c]
    out["ts"] = _kst_ts(out["period_end"].str.replace(".", "", regex=False))
    out["filed"] = _kst_ts(out["rcept_no"].str[:8])  # 접수일 (정정 공시면 정정일)
    out["bsns_year"], out["reprt_code"], out["period"] = year, reprt, REPORTS[reprt]
    return out.drop(columns="period_end")


def collect_dart_fin() -> None:
    """dart_fin (symbol=ALL): 상장사 주요계정 재무. 첫 실행은 2015~ 전체, 이후엔 최근 2개 사업연도만 다시 받는다
    (새 보고서·정정 반영). 호출 수는 (종목수/100) × 4 × 연도수 — 전체 백필 약 1,600회."""
    ds = "dart_fin"
    try:
        codes = _corp_codes()["corp_code"].tolist()
        st = store.get_state(ds, "ALL")
        now = pd.Timestamp.now(tz=KST).year
        years = range(now - 1 if st["backfill_done"] else FIN_START, now + 1)
        n = 0
        for y in years:
            for reprt in REPORTS:
                parts = [_dart_fin(codes[i:i + 100], y, reprt) for i in range(0, len(codes), 100)]
                df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
                    not p.empty for p in parts) else pd.DataFrame()
                n += store.write(ds, "ALL", df, key=["symbol", "bsns_year", "reprt_code"])
        store.set_state(ds, "ALL", backfill_done=True)
        logger.info(f"{ds} {len(codes)}개사 × {len(years)}년, {n:,}행")
    except Exception as e:
        store.set_state(ds, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{ds} 실패: {e}")


# ── ECOS ──
def collect_ecos() -> None:
    key = os.environ["ECOS_API_KEY"]
    end = pd.Timestamp.now(tz=KST).strftime("%Y%m%d")
    for name, (stat, cycle, item) in ECOS.items():
        try:
            j = get_json(_http, f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/"
                                f"{stat}/{cycle}/20050101/{end}/{item}")
            rows = j.get("StatisticSearch", {}).get("row", [])
            if not rows:
                raise RuntimeError(str(j)[:200])
            d = pd.DataFrame(rows)
            df = pd.DataFrame({"ts": pd.to_datetime(d["TIME"], format="%Y%m%d").dt.tz_localize("UTC"),
                               "value": pd.to_numeric(d["DATA_VALUE"], errors="coerce")}).dropna()
            store.write("ecos", name, df)
            store.set_state("ecos", name, backfill_done=True)
        except Exception as e:
            store.set_state("ecos", name, status="error", error=str(e)[:500])
            logger.error(f"ecos {name} 실패: {e}")
    logger.info(f"ecos {len(ECOS)}개 갱신")


def run(symbols: list[str] | None = None) -> None:
    collect_ecos()
    collect_krx()
    collect_dart_fin()  # 호출 수가 적으니 공시 목록 백필(한도 소진형)보다 먼저
    collect_dart()
