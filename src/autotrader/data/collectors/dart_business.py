"""DART 사업보고서 'II. 사업의 내용' 본문 (L004 국내 테마주 lab 요청, 사양: labs/L004-kr-theme-events/data-request.md 7절).

- dart_business (symbol=ALL): 한 행 = 회사 × 사업보고서 기간(결산 'YYYY.MM').
      business_text = 'II. 사업의 내용' 절을 태그를 지운 평문으로 (표는 칸을 탭, 줄을 줄바꿈으로). 키워드 매칭·분류는 lab에서.
      products_text = 같은 절 안에 '주요 제품' 소제목이 따로 있으면 그 부분만 (2020년 무렵 이후 서식). 없으면 빈 문자열.
      원본 zip·전체 문서는 저장하지 않는다.

출처: OpenDART 공시서류원본파일 `document.xml` (접수번호 하나당 zip 1개, 호출 1회). zip 안의 `{접수번호}.xml`이 본문이고
  나머지(`_00760.xml` 등)는 감사보고서 같은 첨부다. 머리말은 encoding="utf-8"이라 적혀 있지만 실제는 대개 EUC-KR(CP949)이다
  → utf-8로 엄격하게 풀어 보고 안 되면 cp949로 푼다 (`encoding` 열에 기록).

정정: 같은 기간에 보고서가 여러 개(정정·첨부추가)면 **최초 접수본 텍스트**를 쓴다. 정정본은 접수번호만 남긴다
  (`last_rcept_no`, `correction_rcept_nos`). 최초 접수본이 목록 시작(2015-01) 전이라 정정본만 보이면
  가장 이른 것을 쓰고 `original_found=False`로 표시한다.
시점: 사업보고서 **최초 접수일(first_filed) 다음 날**부터 쓴다 (연고표·배당과 같은 규칙).

대상: dart_periodic_filings(2015~2023 접수) + dart_disclosures(2024~ 접수)의 사업보고서, 결산 연도 business_start_year(2014)~,
  2015년 이후 KRX 상장 종목(상장폐지 포함). **최근 사업연도부터** 받는다 (2025 사업연도가 먼저).
호출 예산: dividends.dart_max_calls 하나를 정기보고서 목록 → 배당 → 연고표 → 이 수집 순서로 나눠 쓴다 (남은 몫만).
"""

from __future__ import annotations

import html
import io
import os
import re
import time
import zipfile

import pandas as pd

from .. import store
from .common import config, logger
from .dividends import FILINGS_DS, KST, _dart_limit, _http, parse_filings
from .kr_public import _corp_codes

DS = "dart_business"
KEY = ["corp_code", "period"]
FINAL = {"ok", "no_section"}  # 다시 묻지 않는 상태. 그 밖(013·014·parse_error 등)은 14일 뒤 다시
_URL = "https://opendart.fss.or.kr/api/document.xml"

_TITLE = re.compile(r"<TITLE\b[^>]*>(.*?)</TITLE>", re.DOTALL | re.IGNORECASE)
_TOP = re.compile(r"^[IVX]+\s*\.")  # 'II. 사업의 내용', 'III. 재무에 관한 사항' 같은 큰 절
_SUB = re.compile(r"^\d+\s*\.")     # '2. 주요 제품 및 서비스' 같은 소절
_EDITOR = re.compile(r"^\d+#\*_.*\.dsl$")  # 예: '11011#*_수주상황.dsl'


class DartLimit(RuntimeError):
    """OpenDART 020: 하루 요청 한도 초과."""


# ── 문서 → 텍스트 ──
def decode(b: bytes) -> tuple[str, str]:
    """머리말과 상관없이 utf-8(엄격) → cp949 순서로 푼다."""
    try:
        return b.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return b.decode("cp949", errors="replace"), "cp949"


def to_text(xml: str) -> str:
    """DART 문서 XML 조각 → 평문. 표 칸은 탭, 줄·문단은 줄바꿈. 빈 줄은 하나로."""
    s = re.sub(r"<(TD|TH|TE|TU)\b[^>]*>", "\t", xml, flags=re.IGNORECASE)
    s = re.sub(r"<(/P|/TR|BR|/TITLE|/TABLE|P)\b[^>]*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s).replace("&cr;", "\n")  # &cr; = DART 편집기 줄바꿈
    s = html.unescape(s).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", lambda m: "\t" if "\t" in m.group() else " ", ln).strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if "◆click◆" not in ln and not _EDITOR.match(ln)]  # 서식 편집기 표시 줄
    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def _titles(xml: str) -> list[tuple[int, int, str]]:
    """(TITLE 시작 위치, 끝 위치, 공백 정리한 제목)."""
    return [(m.start(), m.end(), re.sub(r"\s+", " ", to_text(m.group(1)))) for m in _TITLE.finditer(xml)]


def business_section(xml: str) -> tuple[str, str] | None:
    """'II. 사업의 내용' 절 → (평문, 주요 제품 소절 평문). 절이 없으면 None.
    절의 끝 = 다음 큰 절 제목(III. …). 목차 표의 '사업의 내용' 글자는 TITLE이 아니라서 잡히지 않는다."""
    ts = _titles(xml)
    start = next((i for i, (_, _, t) in enumerate(ts)
                  if _TOP.match(t) and re.sub(r"\s", "", t).endswith("사업의내용")), None)
    if start is None:
        return None
    end_pos = next((s for s, _, t in ts[start + 1:] if _TOP.match(t)), len(xml))
    body = xml[ts[start][1]:end_pos]
    products = ""
    sub = [(s, e, t) for s, e, t in ts[start + 1:] if s < end_pos]
    for i, (_, e, t) in enumerate(sub):
        if _SUB.match(t) and "주요" in t and "제품" in t:
            nxt = next((s for s, _, t2 in sub[i + 1:] if _SUB.match(t2)), end_pos)
            products = to_text(xml[e:nxt])
            break
    return to_text(body), products


def parse_zip(content: bytes, rcept_no: str) -> dict:
    """document.xml 응답(zip) → {status, business_text, products_text, encoding, n_files}."""
    z = zipfile.ZipFile(io.BytesIO(content))
    names = z.namelist()
    main = f"{rcept_no}.xml"
    order = ([main] if main in names else []) + sorted((n for n in names if n != main), key=lambda n: -z.getinfo(n).file_size)
    enc = ""
    for n in order:
        xml, enc = decode(z.read(n))
        sec = business_section(xml)
        if sec is not None:
            return {"status": "ok", "business_text": sec[0], "products_text": sec[1], "encoding": enc, "n_files": len(names)}
    return {"status": "no_section", "business_text": "", "products_text": "", "encoding": enc, "n_files": len(names)}


def _download(rcept_no: str) -> tuple[str, bytes]:
    """(상태, zip 바이트). zip이 아니면 오류 응답(XML/JSON)의 status를 돌려준다."""
    _dart_limit.wait()
    for i in range(3):
        try:
            r = _http.get(_URL, params={"crtfc_key": os.environ["DART_API_KEY"], "rcept_no": rcept_no}, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            break
        except Exception:
            if i == 2:
                raise
            time.sleep(5 * (i + 1))
    if r.content[:2] == b"PK":
        return "000", r.content
    m = re.search(rb"<status>\s*(\d+)\s*</status>|\"status\"\s*:\s*\"(\d+)\"", r.content)
    s = (m.group(1) or m.group(2)).decode() if m else "unknown"
    if s == "020":
        raise DartLimit(rcept_no)
    return s, b""


# ── 무엇을 받을지 ──
def targets(filings: pd.DataFrame, codes: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """정기보고서 목록(parse_filings 형태) → (회사, 기간)별 1행: 최초 접수본 rcept_no·first_filed, 정정본 접수번호들.
    최근 사업연도부터 정렬."""
    cols = ["corp_code", "symbol", "period", "bsns_year", "rcept_no", "first_filed", "original_found",
            "last_rcept_no", "correction_rcept_nos", "n_filings"]
    f = filings[filings["kind"] == "사업보고서"] if not filings.empty else filings
    if f.empty:
        return pd.DataFrame(columns=cols)
    f = f[f["corp_code"].isin(codes["corp_code"]) & (f["period"].str[:4].astype(int) >= start_year)]
    f = f.sort_values(["corp_code", "period", "ts", "rcept_no"])
    rows = []
    for (corp, period), g in f.groupby(["corp_code", "period"], sort=False):
        orig = g[~g["correction"].astype(bool)]
        first = (orig if not orig.empty else g).iloc[0]
        rows.append({"corp_code": corp, "period": period, "bsns_year": int(period[:4]),
                     "rcept_no": first["rcept_no"], "first_filed": first["ts"], "original_found": not orig.empty,
                     "last_rcept_no": g["rcept_no"].iloc[-1],
                     "correction_rcept_nos": ",".join(r for r in g["rcept_no"] if r != first["rcept_no"]),
                     "n_filings": len(g)})
    t = pd.DataFrame(rows).merge(codes[["corp_code", "symbol"]].drop_duplicates("corp_code"), on="corp_code")
    return t.sort_values(["bsns_year", "period", "corp_code"], ascending=[False, False, True])[cols].reset_index(drop=True)


def todo(t: pd.DataFrame, done: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """아직 받을 대상: 처음이거나, 최초 접수본이 바뀌었거나(목록이 늦게 채워짐), 실패 상태로 14일 지난 것."""
    if done.empty:
        return t
    d = done[["corp_code", "period", "rcept_no", "status", "collected_at"]].copy()
    d["collected_at"] = pd.to_datetime(d["collected_at"], utc=True)
    ok = d["status"].isin(FINAL) | (d["collected_at"] > now - pd.Timedelta(days=14))
    have = set(zip(d.loc[ok, "corp_code"], d.loc[ok, "period"], d.loc[ok, "rcept_no"]))
    keep = [(c, p, r) not in have for c, p, r in zip(t["corp_code"], t["period"], t["rcept_no"])]
    return t[keep].reset_index(drop=True)


def _row(tg, res: dict, zip_bytes: int, now: pd.Timestamp) -> dict:
    end = (pd.Timestamp(tg.period.replace(".", "-") + "-01") + pd.offsets.MonthEnd(0)).tz_localize(KST)
    return {"ts": end.tz_convert("UTC"), "corp_code": tg.corp_code, "symbol": tg.symbol, "bsns_year": int(tg.bsns_year),
            "period": tg.period, "rcept_no": tg.rcept_no, "first_filed": tg.first_filed,
            "original_found": bool(tg.original_found), "last_rcept_no": tg.last_rcept_no,
            "correction_rcept_nos": tg.correction_rcept_nos, "n_filings": int(tg.n_filings),
            "status": res["status"], "business_text": res.get("business_text", ""),
            "products_text": res.get("products_text", ""), "n_chars": len(res.get("business_text", "")),
            "encoding": res.get("encoding", ""), "n_files": res.get("n_files", 0), "zip_bytes": zip_bytes,
            "collected_at": now}


def collect_dart_business(budget: list[int]) -> None:
    """budget = dividends.run()이 넘기는 실행당 DART 호출 예산의 남은 몫. 문서 1개 = 호출 1회."""
    cfg = config().get("dividends", {})
    now = pd.Timestamp.now(tz="UTC")
    buf: list[dict] = []

    def flush() -> None:
        if buf:
            store.write(DS, "ALL", pd.DataFrame(buf), key=KEY)
            buf.clear()

    n, left = 0, None
    try:
        if budget[0] <= 0:
            logger.info(f"{DS}: 이번 실행 예산 없음 → 다음 실행")
            return
        filings = pd.concat([store.read(FILINGS_DS, "ALL"), parse_filings(store.read("dart_disclosures", "ALL"))],
                            ignore_index=True)
        t = todo(targets(filings, _corp_codes(), int(cfg.get("business_start_year", 2014))),
                 store.read(DS, "ALL"), now)
        left = len(t)
        for tg in t.itertuples(index=False):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            status, content = _download(tg.rcept_no)
            n += 1
            if status == "000":
                try:
                    res = parse_zip(content, tg.rcept_no)
                except Exception as e:  # 깨진 zip·XML 하나 때문에 전체가 멈추지 않게
                    res = {"status": "parse_error", "business_text": "", "error": str(e)[:200]}
            else:
                res = {"status": status}
            buf.append(_row(tg, res, len(content), now))
            if len(buf) >= 300:
                flush()
        flush()
        store.set_state(DS, "ALL", backfill_done=n >= left)
    except DartLimit:
        flush()
        budget[0] = 0
        logger.warning(f"{DS}: DART 하루 호출 한도 도달 → 다음 실행에 이어서")
    except Exception as e:
        flush()
        store.set_state(DS, "ALL", status="error", error=str(e)[:500])
        logger.error(f"{DS} 실패: {e}")
    finally:
        if left is not None:
            logger.info(f"{DS} {n:,}건 받음, 남은 {left - n:,}건")
