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
- 옛 CIK (config edgar.predecessors): 지주사 재편 등으로 CIK가 바뀐 종목은 옛 CIK 사실도 그 종목 이름으로 넣는다
      (새 CIK 첫 공시 전 공시분만, cik 열 = 옛 CIK). 상태는 edgar_predecessors.
- edgar_class_shares (symbol=ALL): 클래스가 여럿이라 표지 주식수가 차원(클래스)별로만 있어 companyfacts에 안 나오는
      회사(BRK·V 등)만 공시별 xbrl.zip에서 클래스별 표지 주식수를 읽는다. 분할 보정·환산은 data/shares.py.
- FIELDS를 바꾸면 FIELDS_VERSION을 올린다 → 예전 버전으로 받은 회사를 다시 받는다(일괄 zip 경로).
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
    "shares_basic_wavg": [("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")],
}
FIELDS_VERSION = 2  # FIELDS를 바꾸면 올린다 → 이전 버전으로 받은 회사는 전부 다시 받는다 (edgar_companies.fields_v)
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
    """다시 받을 종목: 처음 보는 종목, 색인에 10-K·10-Q가 잡힌 CIK, refresh_days가 지난 종목,
    예전 FIELDS 버전으로 받은 종목. 오래된 것부터."""
    c = companies.dropna(subset=["cik"])
    last = fetched.set_index("symbol")["ts"] if not fetched.empty else pd.Series(dtype="datetime64[ns, UTC]")
    t = c["symbol"].map(last)
    need = (t.isna() | c["cik"].astype(int).isin(filers) | (t < now - pd.Timedelta(days=refresh_days))
            | c["symbol"].isin(outdated(fetched)))
    c = c.assign(_t=t)[need]
    return c.sort_values("_t", na_position="first")["symbol"].tolist()


def outdated(fetched: pd.DataFrame) -> set[str]:
    """예전 FIELDS 버전(fields_v 없음 포함)으로 받은 종목."""
    if fetched.empty:
        return set()
    v = fetched["fields_v"] if "fields_v" in fetched else pd.Series(float("nan"), index=fetched.index)
    return set(fetched.loc[v.fillna(1) < FIELDS_VERSION, "symbol"])


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
                       "n_facts": len(df), "fields_v": FIELDS_VERSION})
            if len(co) >= 50:
                flush()

        # 처음 받는 종목이 많으면(첫 백필, 유니버스 대량 추가) 일괄 zip 하나로 적재
        seen = (set(fetched["symbol"]) if not fetched.empty else set()) - outdated(fetched)
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
        # 지주사 재편 등으로 CIK가 바뀐 종목의 옛 CIK 사실 (config edgar.predecessors), 클래스별 표지 주식수
        n += collect_predecessors(client, cfg.get("predecessors") or {})
        collect_class_shares(client, cfg)
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


# ── 옛 CIK (지주사 재편·재상장으로 CIK가 바뀐 종목) ──
PRED_DS = "edgar_predecessors"  # 수집 상태만 (symbol = "종목:옛CIK")


def cut_predecessor(df: pd.DataFrame, successor_first_filed: pd.Timestamp | None) -> pd.DataFrame:
    """옛 CIK 사실은 새 CIK가 처음 공시하기 전 공시분만 쓴다 (옛 회사가 자회사로 계속 공시하는 경우 중복 방지)."""
    if df.empty or successor_first_filed is None or pd.isna(successor_first_filed):
        return df
    return df[df["filed"] < successor_first_filed].reset_index(drop=True)


def _first_filed(symbol: str, cik: int) -> pd.Timestamp | None:
    folder = store.DATA_DIR / DS / "ALL"
    if not folder.exists():
        return None
    v = store.query(f"SELECT min(filed) f0 FROM '{folder.as_posix()}/*.parquet' "
                    f"WHERE symbol = '{symbol}' AND cik = {int(cik)}")["f0"].iloc[0]
    return None if pd.isna(v) else store._utc(v)


def collect_predecessors(client: httpx.Client, preds: dict) -> int:
    """config edgar.predecessors {종목: [옛 CIK, ...]} → 옛 CIK companyfacts를 그 종목 이름으로 edgar_facts에 넣는다
    (cik 열은 옛 CIK). FIELDS 버전이 같으면 다시 받지 않는다. 새 CIK 사실을 먼저 받아야 자를 수 있어 그 뒤에 받는다."""
    n = 0
    cur = store.read(CO_DS, "ALL")
    if not preds or cur.empty:
        return 0
    cur = cur.drop_duplicates("symbol", keep="last").set_index("symbol")
    for sym, ciks in preds.items():
        if sym not in cur.index:
            continue
        first = _first_filed(sym, int(cur.loc[sym, "cik"]))
        if first is None:
            continue
        for c in ciks:
            key = f"{sym}:{int(c)}"
            st = store.get_state(PRED_DS, key)
            if st["backfill_done"] and st["error"] == f"v{FIELDS_VERSION}":
                continue
            r = _get(client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(c):010d}.json")
            if r.status_code != 200:
                logger.warning(f"{DS} 옛 CIK {key} HTTP {r.status_code}")
                store.set_state(PRED_DS, key, status="error", error=f"HTTP {r.status_code}")
                continue
            df = cut_predecessor(parse_facts(r.json(), sym), first)
            n += store.write(DS, "ALL", df, key=KEY + ["field"])
            store.set_state(PRED_DS, key, backfill_done=True, error=f"v{FIELDS_VERSION}")  # error 칸 = 받은 버전
            logger.info(f"{DS} 옛 CIK {key}: {len(df):,}행 (새 CIK 첫 공시 {first:%Y-%m-%d} 전까지)")
    return n


# ── 클래스별 표지 주식수 (edgar_class_shares) ──
# companyfacts API는 차원(dimension)이 붙은 사실을 빼고 준다. 클래스가 여럿인 회사(BRK A/B, V A/B/C …)는 표지의
# dei:EntityCommonStockSharesOutstanding을 클래스별로만 내므로 API에 주식수가 거의 없다.
# → 그런 회사만 공시별 XBRL 묶음({접수번호}-xbrl.zip, 약 0.5MB)을 받아 표지 주식수를 클래스별로 읽는다.
CLS_DS = "edgar_class_shares"
CLS_KEY = ["cik", "accn", "member", "asof"]
CLS_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A"}
_CTX = re.compile(r"<(?:xbrli:)?context\b[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>", re.DOTALL)
_MEM = re.compile(r"explicitMember[^>]*dimension=\"([^\"]+)\"[^>]*>\s*([^<\s]+)")
_INST = re.compile(r"<(?:xbrli:)?instant>\s*([^<\s]+)\s*<")
_FACT = re.compile(r"<(ix:nonFraction|dei:EntityCommonStockSharesOutstanding)\b([^>]*)>(.*?)</\1>", re.DOTALL)


def parse_cover_shares(text: str) -> list[dict]:
    """XBRL 인스턴스(.xml) 또는 인라인 XBRL(.htm) 본문 → 표지 주식수 [{member, asof, val}].
    member: 'axis=member' (여럿이면 ';'로 잇는다), 차원 없는 회사 전체 값은 ''."""
    if "EntityCommonStockSharesOutstanding" not in text:
        return []
    ctx = {}
    for m in _CTX.finditer(text):
        mem = ";".join(f"{a}={v}" for a, v in _MEM.findall(m.group(2)))
        inst = _INST.findall(m.group(2))
        ctx[m.group(1)] = (mem, inst[0] if inst else None)
    out = []
    for m in _FACT.finditer(text):
        attrs = m.group(2)
        if m.group(1) == "ix:nonFraction" and "EntityCommonStockSharesOutstanding" not in attrs:
            continue
        ref = re.search(r"contextRef=\"([^\"]+)\"", attrs)
        raw = re.sub(r"<[^>]+>", "", m.group(3)).replace(",", "").strip()
        try:
            val = float(raw)
        except ValueError:
            continue
        sc = re.search(r"\bscale=\"(-?\d+)\"", attrs)
        if sc:
            val *= 10 ** int(sc.group(1))
        mem, inst = ctx.get(ref.group(1), ("", None)) if ref else ("", None)
        out.append({"member": mem, "asof": inst, "val": val})
    return out


def parse_xbrl_zip(content: bytes) -> list[dict]:
    """{접수번호}-xbrl.zip → 표지 주식수. 인스턴스(.xml, _cal·_def·_lab·_pre 제외)나 인라인 본문(.htm)에서 찾는다."""
    import io
    out: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if (n.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$", n)
                                              and "FilingSummary" not in n) or n.endswith((".htm", ".html"))]
        for n in sorted(names, key=lambda x: not x.endswith(".xml")):  # 인스턴스 먼저
            out = parse_cover_shares(z.read(n).decode("utf-8", "ignore"))
            if out:
                break
    seen, uniq = set(), []
    for r in out:  # 같은 값이 인라인 본문에 두 번 나오는 경우
        k = (r["member"], r["asof"], r["val"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def class_targets(facts: pd.DataFrame, min_coverage: float, force: list[int]) -> list[int]:
    """클래스별 수집 대상 CIK: 2009-06 이후 10-K·10-Q 공시 중 차원 없는 주식수 사실(shares_* 항목)이 있는
    공시 비율이 min_coverage 미만인 CIK + force."""
    if facts.empty:
        return sorted({int(c) for c in force})
    f = facts[facts["form"].isin(CLS_FORMS) & (facts["filed"] >= pd.Timestamp("2009-06-01", tz="UTC"))]
    acc = f.groupby("cik")["accn"].nunique()
    sh = f[f["field"].str.startswith("shares")].groupby("cik")["accn"].nunique()
    cov = sh.reindex(acc.index).fillna(0) / acc
    return sorted({int(c) for c in cov[cov < min_coverage].index} | {int(c) for c in force})


def _filings(client: httpx.Client, cik: int) -> pd.DataFrame:
    """submissions API → 10-K·10-Q(정정 포함) 목록 (accn, form, filed). 오래된 공시는 추가 파일에 있다."""
    r = _get(client, f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    r.raise_for_status()
    j = r.json()
    parts = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        if f.get("filingTo", "9999") >= "2009-06-01":
            rr = _get(client, f"https://data.sec.gov/submissions/{f['name']}")
            if rr.status_code == 200:
                parts.append(rr.json())
    d = pd.concat([pd.DataFrame({"accn": p["accessionNumber"], "form": p["form"], "filed": p["filingDate"]})
                   for p in parts], ignore_index=True)
    d = d[d["form"].isin(CLS_FORMS) & (d["filed"] >= "2009-06-01")].drop_duplicates("accn")
    d["filed"] = _ny_ts(d["filed"])
    return d.reset_index(drop=True)


def fetch_class_shares(client: httpx.Client, cik: int, skip: set[str], budget: int) -> tuple[pd.DataFrame, int]:
    """한 CIK의 공시별 표지 주식수 (skip 접수번호 제외, 최대 budget건). 반환: (행, 실패 수).
    표지 주식수가 없는 공시도 member='∅'·val=NaN 한 행으로 남겨 다시 받지 않는다."""
    fl = _filings(client, cik)
    fl = fl[~fl["accn"].isin(skip)].head(budget)
    rows, fail = [], 0
    for _, f in fl.iterrows():
        a = f["accn"]
        r = _get(client, f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/{a}-xbrl.zip")
        if r.status_code == 200:
            got = parse_xbrl_zip(r.content)
        elif r.status_code == 404:  # XBRL 없는 공시 (2009~2011 소형사 등)
            got = []
        else:
            fail += 1
            continue
        for g in got or [{"member": "∅", "asof": None, "val": float("nan")}]:
            rows.append({"cik": cik, "accn": a, "form": f["form"], "filed": f["filed"], **g})
    if not rows:
        return pd.DataFrame(), fail
    d = pd.DataFrame(rows)
    d["asof"] = _ny_ts(d["asof"])
    d["ts"] = d["asof"].fillna(d["filed"])
    return d, fail


def collect_class_shares(client: httpx.Client, cfg: dict) -> None:
    """대상 CIK(class_targets)의 공시마다 xbrl.zip을 받아 edgar_class_shares에 적재. 실행당 class_max_filings 상한."""
    try:
        folder = store.DATA_DIR / DS / "ALL"
        if not folder.exists():
            return
        facts = store.query(f"SELECT DISTINCT cik, field, accn, form, filed FROM '{folder.as_posix()}/*.parquet'")
        facts["filed"] = pd.to_datetime(facts["filed"], utc=True)
        targets = class_targets(facts, float(cfg.get("class_min_coverage", 0.5)), cfg.get("class_force_ciks") or [])
        have = store.read(CLS_DS, "ALL")
        done = set(have["accn"]) if not have.empty else set()
        budget, n, fail = int(cfg.get("class_max_filings", 1500)), 0, 0
        for c in targets:
            if budget <= 0:
                break
            d, f = fetch_class_shares(client, c, done, budget)
            fail += f
            budget -= d["accn"].nunique() + f if not d.empty else f
            n += store.write(CLS_DS, "ALL", d, key=CLS_KEY)
        store.set_state(CLS_DS, "ALL", backfill_done=budget > 0, status="ok" if not fail else "error",
                        error=f"xbrl.zip 실패 {fail}건" if fail else None)
        logger.info(f"{CLS_DS}: 대상 {len(targets)}개 CIK, {n}행, 실패 {fail}")
    except Exception as e:
        store.set_state(CLS_DS, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{CLS_DS} 실패: {e}")
