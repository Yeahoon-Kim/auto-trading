"""국내 공공 API 수집: KRX 전 종목 일별 시세, DART 공시 목록, 한국은행 ECOS 거시 지표.

- krx_1d (symbol=ALL, 종목별 행) : KOSPI·KOSDAQ 전 종목 원본 일봉 + 시가총액 + 상장주식수. 2010~.
      그날 상장돼 있던 종목이 전부 들어 있으므로 생존편향이 없다. KRX는 다음 영업일에 공개한다.
- dart_disclosures (symbol=ALL)  : 전체 상장사 공시 목록. 접수일(rcept_dt)만 있으므로
      일봉 전략에서는 "접수일 다음 날부터 알 수 있음"으로 쓴다. 하루 호출 한도(약 2만) 때문에
      백필은 실행마다 최대 MAX_DART_CALLS 만큼 진행한다.
- dart_fin (symbol=ALL)          : 상장사 주요계정 재무(매출·영업이익·순이익·자산·부채·자본), 2015~.
      손익은 3개월 값과 _cum(누적). 4분기는 보고서가 없으므로 FY − Q3 누적으로 계산해서 쓴다.
      filed = 접수번호 앞 8자리(실제 접수일) → 시점 규칙은 "filed 다음 날부터 알 수 있음".
      net_income_controlling(+_cum, nic_source): 지배주주 순이익. 연결은 dart_fin_owner(fnlttSinglAcntAll)에서,
      별도는 순이익 그대로. 매 실행 끝에 patch_dart_fin()이 dart_fin 전체에 다시 붙인다.
- dart_fin_owner (symbol=ALL)    : 연결 보고서별 지배주주 순이익 원본 (회사·보고서당 1회 호출, 실행당 예산만큼 백필).
- ecos (symbol=시계열 이름)      : 기준금리, 국고채 3·10년, 회사채 AA-, CD91, 원/달러.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd

from .. import store
from .common import RateLimiter, config, get_json, logger

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


# ── DART 지배주주 순이익 (단일회사 전체 재무제표) ──
# fnlttMultiAcnt(주요계정)에는 "지배기업 소유주지분 순이익"이 없다 (당기순이익 2행 모두 비지배 포함 전체).
# fnlttSinglAcntAll은 회사·보고서·연결구분마다 1회 호출이라, dart_fin의 연결(CFS) 행만 골라 예산 안에서 백필한다.
OWNER_DS = "dart_fin_owner"
OWNER_IDS = {"ifrs-full_ProfitLossAttributableToOwnersOfParent", "ifrs_ProfitLossAttributableToOwnersOfParent"}
NIC = ["net_income_controlling", "net_income_controlling_cum", "nic_source"]


class DartLimitReached(RuntimeError):
    """OpenDART 020: 하루 요청 한도 초과 → 이번 실행은 멈추고 다음 실행이 이어받는다."""


def _owner_name(nm: str) -> bool:
    n = nm.replace(" ", "")
    return ("지배" in n and "비지배" not in n and "포괄" not in n
            and not any(k in n for k in ("계속영업", "중단영업"))
            and any(k in n for k in ("순이익", "순손익", "순손실", "당기순")))


def _first(rows: list[dict], pick) -> dict | None:
    """손익계산서(IS) 먼저, 없으면 포괄손익계산서(CIS)에서 조건에 맞는 첫 행."""
    for sj in ("IS", "CIS"):
        hit = [r for r in rows if r.get("sj_div") == sj and pick(r)]
        if hit:
            return hit[0]
    return None


def _amounts(r: dict, reprt: str) -> tuple[float, float]:
    v = float(_num(pd.Series([r.get("thstrm_amount")])).iloc[0])
    cum = float(_num(pd.Series([r.get("thstrm_add_amount")])).iloc[0])
    if reprt in ("11011", "11013"):  # 사업보고서·1분기: 당기 = 누적 (사업보고서는 누적 칸이 비어 있다)
        cum = v
    return v, cum


def parse_owner(rows: list[dict], reprt: str) -> tuple[float, float, str] | None:
    """fnlttSinglAcntAll(CFS) list → (지배주주 순이익 3개월 값(사업보고서는 연간), 누적, 출처). 못 찾으면 None.
    - 'cfs': 지배주주 항목. 표준 계정 ID를 먼저, 없으면 계정명으로 찾는다
      (2016년 무렵 '-표준계정코드 미사용-' 회사가 있다. 예: SK 2016 1분기 '지배기업소유주지분순이익(손실)').
    - 'cfs_signfix': 지배주주 + 비지배 ≠ 당기순이익인데 부호를 바꾸면 맞는 경우(제출 오류. 예: 073540 2016 1분기).
    - 'cfs_no_nci': 지배·비지배 구분이 아예 없는 연결(비지배지분 없음) → 당기순이익 전체.
    비지배 항목은 있는데 지배주주 순이익이 없으면(포괄이익만 나눈 회사 등) None."""
    own = _first(rows, lambda r: r.get("account_id") in OWNER_IDS) or \
        _first(rows, lambda r: _owner_name(r.get("account_nm", "")))
    total = _first(rows, lambda r: r.get("account_id") in ("ifrs-full_ProfitLoss", "ifrs_ProfitLoss"))
    if own is not None:
        v, cum = _amounts(own, reprt)
        nci = _first(rows, lambda r: r.get("account_id", "").endswith("ProfitLossAttributableToNoncontrollingInterests"))
        if total is not None:
            t = _amounts(total, reprt)[0]
            n = _amounts(nci, reprt)[0] if nci is not None else 0.0
            n = 0.0 if pd.isna(n) else n
            tol = max(1.0, abs(t) * 0.01)
            if abs(v + n - t) > tol and abs(-v + n - t) <= tol:
                return -v, -cum, "cfs_signfix"
        return v, cum, "cfs"
    has_nci = any(r.get("sj_div") in ("IS", "CIS") and ("Noncontrolling" in r.get("account_id", "")
                                                       or "비지배" in r.get("account_nm", "")) for r in rows)
    if total is not None and not has_nci:
        v, cum = _amounts(total, reprt)
        return v, cum, "cfs_no_nci"
    return None


def _owner(corp_code: str, year: int, reprt: str) -> tuple[str, list[dict]]:
    _dart_limit.wait()
    j = get_json(_http, "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                 {"crtfc_key": os.environ["DART_API_KEY"], "corp_code": corp_code, "bsns_year": str(year),
                  "reprt_code": reprt, "fs_div": "CFS"})
    s = j.get("status")
    if s == "020":
        raise DartLimitReached(j.get("message", ""))
    if s not in ("000", "013"):
        raise RuntimeError(f"DART owner {s}: {j.get('message')}")
    return s, j.get("list", [])


def owner_todo(fin: pd.DataFrame, owner: pd.DataFrame) -> pd.DataFrame:
    """받아야 할 dart_fin 연결(CFS) 행: 아직 없거나, dart_fin 보고서가 바뀐(정정) 행. 최근 사업연도부터."""
    f = fin[fin["fs_div"] == "CFS"]
    if not owner.empty:
        have = owner[["symbol", "bsns_year", "reprt_code", "fin_rcept_no"]].rename(columns={"fin_rcept_no": "rcept_no"})
        have = have.drop_duplicates()
        f = f.merge(have.assign(_have=True), on=["symbol", "bsns_year", "reprt_code", "rcept_no"], how="left")
        f = f[f["_have"].isna()].drop(columns="_have")
    return f.sort_values(["bsns_year", "reprt_code", "symbol"], ascending=[False, True, True]).reset_index(drop=True)


def attach_controlling(fin: pd.DataFrame, owner: pd.DataFrame) -> pd.DataFrame:
    """dart_fin에 net_income_controlling(3개월 값), _cum(누적), nic_source를 붙인다.
    - 연결(CFS): dart_fin_owner의 값. 같은 보고서(rcept_no)로 받은 값만 쓴다 (정정 뒤 다시 받기 전엔 NaN).
    - 별도(OFS): 연결 대상 종속회사가 없어 비지배지분이 없으므로 당기순이익 전체 = 지배주주 순이익.
    nic_source: parse_owner()의 출처('cfs'·'cfs_signfix'·'cfs_no_nci'), 'ofs'(별도 = 순이익),
    'cfs_missing'(연결인데 항목 없음 또는 아직 미수집)."""
    out = fin.drop(columns=[c for c in NIC if c in fin]).copy()
    key = ["symbol", "bsns_year", "reprt_code"]
    if not owner.empty:
        o = owner.loc[owner["has_data"].astype(bool), key + ["fin_rcept_no", *NIC[:2], "source"]]
        o = o.rename(columns={"fin_rcept_no": "rcept_no", "source": "nic_source"})
        out = out.merge(o, on=key + ["rcept_no"], how="left")
    else:
        out["net_income_controlling"] = out["net_income_controlling_cum"] = float("nan")
        out["nic_source"] = None
    ofs = out["fs_div"] == "OFS"
    out.loc[ofs, "net_income_controlling"] = out.loc[ofs, "net_income"]
    out.loc[ofs, "net_income_controlling_cum"] = out.loc[ofs, "net_income_cum"]
    out["nic_source"] = out["nic_source"].astype(object).where(out["nic_source"].notna(), "cfs_missing")
    out.loc[ofs, "nic_source"] = "ofs"
    return out


def collect_dart_owner(budget: list[int]) -> None:
    """dart_fin_owner (symbol=ALL): 연결 보고서의 지배주주 순이익. 회사·보고서당 1회 호출.
    실행마다 budget 만큼만 받고 다음 실행이 이어받는다 (하루 한도는 dart_disclosures·dart_fin·dart_dividend와 공유).
    항목이 없거나 조회 결과가 없는(013) 보고서도 has_data=False로 남겨 다시 묻지 않는다 (dart_fin 보고서가 바뀌면 다시)."""
    try:
        fin = store.read("dart_fin", "ALL")
        if fin.empty:
            return
        todo = owner_todo(fin, store.read(OWNER_DS, "ALL"))
        rows, n, calls, limit_hit = [], 0, 0, False

        def flush() -> int:
            if not rows:
                return 0
            m = store.write(OWNER_DS, "ALL", pd.DataFrame(rows), key=["symbol", "bsns_year", "reprt_code"])
            rows.clear()
            return m

        now = pd.Timestamp.now(tz="UTC")
        for r in todo.head(max(0, budget[0])).itertuples(index=False):
            try:
                s, lst = _owner(r.corp_code, int(r.bsns_year), r.reprt_code)
            except DartLimitReached:
                limit_hit = True
                break
            budget[0] -= 1
            calls += 1
            got = parse_owner(lst, r.reprt_code) if s == "000" else None
            rows.append({"ts": r.ts, "symbol": r.symbol, "corp_code": r.corp_code, "bsns_year": int(r.bsns_year),
                         "reprt_code": r.reprt_code, "fin_rcept_no": r.rcept_no,
                         "rcept_no": lst[0].get("rcept_no") if lst else None, "status": s, "has_data": got is not None,
                         "net_income_controlling": got[0] if got else float("nan"),
                         "net_income_controlling_cum": got[1] if got else float("nan"),
                         "source": got[2] if got else None, "fetched": now})
            if len(rows) >= 500:
                n += flush()
        n += flush()
        left = len(todo) - calls
        store.set_state(OWNER_DS, "ALL", backfill_done=left == 0)
        logger.info(f"{OWNER_DS} {calls:,}회 호출, 남은 보고서 {left:,}" + (" (DART 하루 한도 도달)" if limit_hit else ""))
    except Exception as e:
        store.set_state(OWNER_DS, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{OWNER_DS} 실패: {e}")


def patch_dart_fin() -> None:
    """dart_fin 전체에 지배주주 순이익 열을 다시 붙여 저장한다 (collect_dart_fin이 최근 2년을 덮어쓰면 열이 비므로 매번)."""
    try:
        fin = store.read("dart_fin", "ALL")
        if fin.empty:
            return
        out = attach_controlling(fin, store.read(OWNER_DS, "ALL"))
        store.write("dart_fin", "ALL", out, key=["symbol", "bsns_year", "reprt_code"])
        logger.info("dart_fin 지배주주 순이익: " +", ".join(f"{k} {v:,}" for k, v in out["nic_source"].value_counts().items()))
    except Exception as e:
        logger.error(f"dart_fin 지배주주 순이익 붙이기 실패: {e}")


# ── ECOS ──
def collect_ecos() -> None:
    key = os.environ["ECOS_API_KEY"]
    end = pd.Timestamp.now(tz=KST).strftime("%Y%m%d")
    ok = 0
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
            ok += 1
        except Exception as e:
            store.set_state("ecos", name, status="error", error=str(e)[:500])
            logger.error(f"ecos {name} 실패: {e}")
    logger.info(f"ecos {ok}개 갱신, {len(ECOS) - ok}개 실패")


def run(symbols: list[str] | None = None) -> None:
    collect_ecos()
    collect_krx()
    collect_dart_fin()  # 호출 수가 적으니 공시 목록 백필(한도 소진형)보다 먼저
    collect_dart()
    collect_dart_owner([int(config().get("kr_public", {}).get("dart_owner_max_calls", 8000))])
    patch_dart_fin()
