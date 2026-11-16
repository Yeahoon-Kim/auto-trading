"""미국 재무: SEC EDGAR XBRL companyfacts (무료, 키 없음 — User-Agent만 필요).

- edgar_facts (symbol=ALL): us_1d 유니버스(S&P500·NASDAQ100 현재 구성종목) 재무 사실(fact)을 긴 표로.
      한 행 = (종목, 항목, 태그, 단위, 기간, 공시). 같은 기간 값이 여러 공시에 다시 나오면(다음 해 비교 재무제표,
      10-K/A 정정) 공시마다 따로 남는다 → 최초 공시값과 정정·재작성 값을 모두 갖고, 시점 규칙으로 고른다.
      열: ts(기간 끝), symbol, cik, field, tag, tag_rank, unit, start(기간 시작, 시점 항목은 NaT), end, val,
          accn(접수번호), fy, fp, form(10-K, 10-Q, 10-K/A …), filed(공시일), frame.
      시점 규칙: filed 다음 거래일부터 알 수 있음. 같은 (field, start, end)에서 asof 이전 filed 중
          최초(first) 또는 최신(last)을 고른다. 태그가 여러 개인 항목은 tag_rank가 작은 것을 먼저 쓴다.
- edgar_companies (symbol=ALL): 종목 → CIK, 회사명, 마지막 companyfacts 수집 시각(ts).
- 첫 백필: 처음 받는 종목이 bulk_min_new(50) 이상이면 일괄 파일 companyfacts.zip 하나를 받아 해당 회사만 꺼내 적재한다.
- 갱신: SEC 일별 공시 색인(daily-index form.YYYYMMDD.idx)에서 10-K·10-Q(정정 포함)를 낸 CIK만 다시 받는다.
      그 밖에도 refresh_days가 지난 회사는 다시 받는다 (8-K 등으로 사실이 추가되는 경우).
- SEC 규정: 초당 10회 이하(여기선 5회), User-Agent에 연락처 필수. User-Agent는 환경변수 또는 .env의
      SEC_USER_AGENT에서만 읽는다(코드에 개인정보를 두지 않는다). 없으면 "키 필요"로 건너뛴다.
- 한계: 현재 구성종목만(생존편향). us-gaap 태그만 받으므로 20-F·40-F(IFRS) 외국 회사는 비어 있다.
      XBRL 의무화 이전(대형사 2009-06, 그 외 2011-06 무렵) 기간은 그 뒤 공시의 비교 재무제표 값만 있다.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import zipfile
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import pandas as pd

from .. import store
from .common import ROOT, RateLimiter, config, logger

NY = "America/New_York"
DS, CO_DS, IDX = "edgar_facts", "edgar_companies", "edgar_index"
KEY = ["symbol", "tag", "unit", "start", "end", "accn"]
FIELDS = {  # 항목: [(분류, 태그)] — 앞에 있을수록 tag_rank가 작다
    "eps_diluted": [("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasicAndDiluted")],
    "eps_basic": [("us-gaap", "EarningsPerShareBasic"), ("us-gaap", "EarningsPerShareBasicAndDiluted")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
                   ("us-gaap", "ProfitLoss")],
    "revenue": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"), ("us-gaap", "SalesRevenueNet"),
                ("us-gaap", "RevenuesNetOfInterestExpense")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "equity": [("us-gaap", "StockholdersEquity"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "shares_outstanding": [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")],
    "shares_diluted_wavg": [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")],
}
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A"}  # 색인에서 다시 받을 계기가 되는 공시
BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
_limit = RateLimiter(5)


def user_agent() -> str | None:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua and (ROOT / ".env").exists():
        from dotenv import dotenv_values
        ua = dotenv_values(ROOT / ".env").get("SEC_USER_AGENT")
    return ua.strip() if ua and ua.strip() else None


def _ny_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(NY).dt.tz_convert("UTC")


def parse_facts(j: dict, symbol: str) -> pd.DataFrame:
    """companyfacts JSON → FIELDS에 있는 태그만 긴 표로."""
    facts, rows = j.get("facts", {}), []
    for field, tags in FIELDS.items():
        for rank, (ns, tag) in enumerate(tags):
            for unit, items in facts.get(ns, {}).get(tag, {}).get("units", {}).items():
                for x in items:
                    rows.append({"field": field, "tag": tag, "tag_rank": rank, "unit": unit, "start": x.get("start"),
                                 "end": x.get("end"), "val": x.get("val"), "accn": x.get("accn"), "fy": x.get("fy"),
                                 "fp": x.get("fp"), "form": x.get("form"), "filed": x.get("filed"),
                                 "frame": x.get("frame")})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["start"], d["end"], d["filed"] = _ny_ts(d["start"]), _ny_ts(d["end"]), _ny_ts(d["filed"])
    d["val"] = pd.to_numeric(d["val"], errors="coerce")
    d["fy"] = pd.to_numeric(d["fy"], errors="coerce").astype("Int64")
    d.insert(0, "cik", int(j.get("cik", 0)))
    d.insert(0, "symbol", symbol)
    d.insert(0, "ts", d["end"])
    d = d.dropna(subset=["end", "filed", "val"])
    return d.drop_duplicates(KEY + ["field"], keep="last").reset_index(drop=True)


_IDX_LINE = re.compile(r"^(?P<form>\S+(?: \S+)*?)\s{2,}.*\s(?P<cik>\d+)\s+(?P<date>\d{8})\s+edgar/")


def parse_index(text: str) -> set[int]:
    """daily-index form.YYYYMMDD.idx → FORMS 공시를 낸 CIK."""
    out = set()
    for line in text.splitlines():
        m = _IDX_LINE.match(line)
        if m and m["form"] in FORMS:
            out.add(int(m["cik"]))
    return out


def _get(client: httpx.Client, url: str) -> httpx.Response:
    for i in range(5):
        _limit.wait()
        try:
            r = client.get(url)
        except httpx.TransportError as e:
            logger.warning(f"{url} 재시도 {i + 1}/5 ({e})")
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** i)
            continue
        return r
    raise RuntimeError(f"{url} 요청 실패")


def cik_map(client: httpx.Client, tickers: list[str]) -> pd.DataFrame:
    """SEC company_tickers.json으로 티커 → CIK. 표기 차이(BRK.B·BRK-B)는 하이픈으로 맞춘다."""
    r = _get(client, "https://www.sec.gov/files/company_tickers.json")
    r.raise_for_status()
    m = pd.DataFrame(r.json().values())
    m["ticker"] = m["ticker"].str.upper().str.replace(".", "-", regex=False)
    m = m.drop_duplicates("ticker").set_index("ticker")
    out = pd.DataFrame({"symbol": tickers})
    out["cik"] = out["symbol"].str.upper().map(m["cik_str"]).astype("Int64")
    out["title"] = out["symbol"].str.upper().map(m["title"])
    return out


def todo(companies: pd.DataFrame, fetched: pd.DataFrame, filers: set[int], now: pd.Timestamp,
         refresh_days: int) -> list[str]:
    """다시 받을 종목: 처음 보는 종목, 색인에 10-K·10-Q가 잡힌 CIK, refresh_days가 지난 종목. 오래된 것부터."""
    c = companies.dropna(subset=["cik"])
    last = fetched.set_index("symbol")["ts"] if not fetched.empty else pd.Series(dtype="datetime64[ns, UTC]")
    t = c["symbol"].map(last)
    need = t.isna() | c["cik"].astype(int).isin(filers) | (t < now - pd.Timedelta(days=refresh_days))
    c = c.assign(_t=t)[need]
    return c.sort_values("_t", na_position="first")["symbol"].tolist()


def _index_filers(client: httpx.Client, start: pd.Timestamp, end: pd.Timestamp) -> tuple[set[int], pd.Timestamp | None]:
    """start~end(NY 날짜) 일별 색인을 읽어 CIK를 모은다. 색인이 없는 날(주말·휴일)은 건너뛴다.
    반환: (CIK, 마지막으로 확인한 날) — 중간에 실패하면 거기서 멈추고 다음 실행이 이어받는다."""
    ciks, done = set(), None
    for d in pd.date_range(start, end):
        q = (d.month - 1) // 3 + 1
        r = _get(client, f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx")
        if r.status_code == 200:
            ciks |= parse_index(r.text)
        elif r.status_code not in (403, 404):
            break
        done = d
    return ciks, done


def bulk_load(client: httpx.Client, targets: pd.DataFrame, emit) -> None:
    """SEC 일괄 파일 companyfacts.zip(전 기업, 매일 밤 재생성, 약 1.4GB)을 데이터 루트 아래 임시 폴더에 받아
    targets(index=symbol, cik 열)의 회사 파일만 꺼내 parse_facts → emit(symbol, df, 수집 시각).
    수집 시각은 zip의 Last-Modified(생성 시각)로 남긴다 → 그 뒤 공시는 일별 색인·refresh_days로 API가 다시 받는다.
    zip은 끝나면 지운다."""
    with tempfile.TemporaryDirectory(dir=store.DATA_ROOT) as tmp:
        path = Path(tmp) / "companyfacts.zip"
        _limit.wait()
        with client.stream("GET", BULK_URL, timeout=1800) as r:
            r.raise_for_status()
            lm = r.headers.get("last-modified")
            gen = pd.Timestamp(parsedate_to_datetime(lm)).tz_convert("UTC") if lm else pd.Timestamp.now(tz="UTC")
            with open(path, "wb") as f:
                f.writelines(r.iter_bytes(1 << 20))
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for s, c in targets["cik"].items():
                name = f"CIK{int(c):010d}.json"
                emit(s, parse_facts(json.loads(z.read(name)), s) if name in names else pd.DataFrame(), gen)


def run(symbols: list[str] | None = None) -> None:
    ua = user_agent()
    if not ua:
        logger.warning(f"{DS}: SEC_USER_AGENT 없음 → 건너뜀 (키 필요: 환경변수 또는 .env에 '이름 연락처' 형식으로 설정)")
        store.set_state(DS, "ALL", status="skipped", error="SEC_USER_AGENT 없음 (사용자 설정 필요)")
        return
    from .us_stock import universe

    cfg = config().get("edgar", {})
    client = httpx.Client(timeout=120, headers={"User-Agent": ua},
                          follow_redirects=True)
    try:
        tickers = symbols or universe()["yahoo"].tolist()
        companies = cik_map(client, tickers)
        missing = companies[companies["cik"].isna()]["symbol"].tolist()
        if missing:
            logger.warning(f"{DS}: CIK 못 찾음 {len(missing)}개 {missing[:10]}")
        fetched = store.read(CO_DS, "ALL")

        # 색인: 마지막 확인일 다음 날 ~ 어제(NY). 처음이면 색인 없이 전부 받는다.
        now = pd.Timestamp.now(tz="UTC")
        yday = pd.Timestamp.now(tz=NY).normalize().tz_localize(None) - pd.Timedelta(days=1)
        st = store.get_state(IDX, "ALL")
        filers: set[int] = set()
        if st["last_ts"] is not None:
            start = st["last_ts"].tz_convert(NY).tz_localize(None).normalize() + pd.Timedelta(days=1)
            start = max(start, yday - pd.Timedelta(days=60))  # 오래 멈춰 있었으면 refresh_days 규칙에 맡긴다
            filers, done = _index_filers(client, start, yday)
            if done is not None:
                store.set_state(IDX, "ALL", last_ts=done.tz_localize(NY).tz_convert("UTC"))
        else:
            store.set_state(IDX, "ALL", last_ts=yday.tz_localize(NY).tz_convert("UTC"))

        syms = todo(companies, fetched, filers, now, int(cfg.get("refresh_days", 30)))
        syms = syms[:int(cfg.get("max_companies_per_run", 600))]
        buf, co, cnt, fail = [], [], [0], 0
        cik = companies.set_index("symbol")

        def flush() -> None:
            if buf:
                cnt[0] += store.write(DS, "ALL", pd.concat(buf, ignore_index=True), key=KEY + ["field"])
            if co:
                store.write(CO_DS, "ALL", pd.DataFrame(co), key=["symbol"])
            buf.clear()
            co.clear()

        def emit(s: str, df: pd.DataFrame, ts: pd.Timestamp) -> None:
            if not df.empty:
                buf.append(df)
            co.append({"ts": ts, "symbol": s, "cik": int(cik.loc[s, "cik"]), "title": cik.loc[s, "title"],
                       "n_facts": len(df)})
            if len(co) >= 50:
                flush()

        # 처음 받는 종목이 많으면(첫 백필, 유니버스 대량 추가) 일괄 zip 하나로 적재
        seen = set(fetched["symbol"]) if not fetched.empty else set()
        new = [s for s in syms if s not in seen]
        how = "API"
        if len(new) >= int(cfg.get("bulk_min_new", 50)):
            bulk_load(client, cik.loc[new], emit)
            flush()
            syms, how = [s for s in syms if s in seen], f"일괄 zip {len(new)}개사 + API"

        for s in syms:
            c = int(cik.loc[s, "cik"])
            r = _get(client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json")
            if r.status_code == 404:  # XBRL 사실이 없는 회사
                df = pd.DataFrame()
            elif r.status_code != 200:
                fail += 1
                logger.warning(f"{DS} {s} HTTP {r.status_code}")
                continue
            else:
                df = parse_facts(r.json(), s)
            emit(s, df, pd.Timestamp.now(tz="UTC"))
        flush()
        n = cnt[0]
        done = set(store.read(CO_DS, "ALL").get("symbol", pd.Series(dtype=str)))
        left = len(set(companies.dropna(subset=["cik"])["symbol"]) - done)  # 아직 한 번도 못 받은 종목
        store.set_state(DS, "ALL", backfill_done=left == 0, status="ok" if not fail else "error",
                        error=f"companyfacts 실패 {fail}개" if fail else None)
        logger.info(f"{DS} {how}: API {len(syms)}개사 갱신(색인 대상 {len(filers)} CIK), {n:,}행, 미수집 {left}개, 실패 {fail}개")
    except Exception as e:
        store.set_state(DS, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{DS} 실패: {e}")
    finally:
        client.close()
