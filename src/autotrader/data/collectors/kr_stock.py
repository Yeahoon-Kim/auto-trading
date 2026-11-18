"""국내 주식 수집 (KIS 실전 키, 조회 전용).

- kr_1d        : 전 종목(KOSPI·KOSDAQ, 상장폐지 포함) 원본 일봉
- kr_adj_factor: 수정계수 = 수정종가 / 원본종가. 분할·증자가 생기면 과거분까지 다시 받는다.
- kr_1m        : 분봉 유니버스(시총 상위 KOSPI 200 + KOSDAQ 150) 1분봉. KIS가 주는 만큼(약 1년) 백필.
ts는 봉 시작 시각(UTC). 일봉은 09:00 KST.
"""

from __future__ import annotations

import shutil
import time

import FinanceDataReader as fdr
import pandas as pd

from ...brokers.kis import KISClient
from .. import store
from .common import RateLimiter, config, logger

KST = "Asia/Seoul"
_kis: KISClient | None = None
_limit: RateLimiter | None = None


def kis() -> KISClient:
    global _kis, _limit
    if _kis is None:
        _kis = KISClient("live")
        _limit = RateLimiter(config()["kr"]["requests_per_sec"])
    return _kis


# OPSQ0002(KIS 서버 일시 오류)는 몇 분짜리 장애(11-17 10:06~10:09)에도 나온다 → 길게 재시도.
# 단, 장애가 OUTAGE_MAX초 넘게 이어지면(모든 스레드 공통) 더 기다리지 않고 바로 실패시켜 수집 전체 시간을 묶는다.
OUTAGE_CODES = ("OPSQ0002",)
OUTAGE_MAX = 600.0  # 장애로 보고 기다려 줄 최대 시간(초). 넘으면 이후 호출은 짧은 재시도만
_outage_since: float | None = None  # 장애 연속 시작 시각(time.monotonic). 성공 한 번이면 초기화


def _sleep(sec: float) -> None:  # 테스트에서 바꿔 끼우는 지점
    time.sleep(sec)


def _get(path: str, tr_id: str, params: dict) -> dict:
    global _outage_since
    c = kis()
    last = ""  # 마지막 실패 메시지(KIS msg_cd·msg1 또는 예외 종류). 키·토큰·계좌번호는 들어가지 않는다
    i = 0      # 일반 재시도 횟수
    while True:
        _limit.wait()
        try:
            out = c.get(path, tr_id, params)
            _outage_since = None
            return out
        except RuntimeError as e:
            last = str(e)[:300]
            # EGW00201: 초당 한도 초과, OPSQ1002: SESSION FULL(동시 세션 초과) → 길게 쉬고 재시도
            if "EGW00201" in last or "OPSQ1002" in last:
                if i < 6:
                    _sleep(min(60, 5 * 2 ** i))
                    i += 1
                    continue
                break
            if any(code in last for code in OUTAGE_CODES):
                now = time.monotonic()
                if _outage_since is None:
                    _outage_since = now
                elapsed = now - _outage_since
                if elapsed < OUTAGE_MAX:
                    # 지수 대기 2·4·8·16·30·30…초, 장애 상한까지
                    _sleep(min(30, 2 * 2 ** i, OUTAGE_MAX - elapsed + 1))
                    i += 1
                    continue
            if i < 2:
                _sleep(1 + i)
                i += 1
                continue
            break
        except Exception as e:
            # 네트워크·HTTP 오류: 응답 본문 대신 예외 종류와 메시지만 (요청 헤더는 담지 않는다)
            last = f"{type(e).__name__}: {str(e)[:200]}"
            if i < 6:
                _sleep(2 ** i)
                i += 1
                continue
            break
    logger.warning(f"KIS {tr_id} 최종 실패 ({i + 1}회 시도) 마지막 응답: {last}")
    raise RuntimeError(f"KIS {tr_id} 반복 실패 — 마지막 응답: {last}")


# ── 유니버스 ──
def build_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config()["kr"]["start_1d"])
    cur = fdr.StockListing("KRX")
    cur = cur[cur["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
    daily = pd.DataFrame({"code": cur["Code"], "name": cur["Name"], "market": cur["Market"],
                          "delisted": pd.NaT})
    dl = fdr.StockListing("KRX-DELISTING")
    dl["DelistingDate"] = pd.to_datetime(dl["DelistingDate"])
    dl = dl[(dl["DelistingDate"] >= start) & (dl["SecuGroup"] == "주권") & dl["Market"].isin(["KOSPI", "KOSDAQ"])]
    daily = pd.concat([daily, pd.DataFrame({"code": dl["Symbol"], "name": dl["Name"], "market": dl["Market"],
                                            "delisted": dl["DelistingDate"]})], ignore_index=True)
    daily = daily.drop_duplicates("code", keep="first")
    store.save_universe("kr_daily", daily)

    # 지수 구성종목 목록은 KRX 로그인이 필요해서, 시총 상위 보통주로 근사
    common = cur[cur["Code"].str.endswith("0")]
    kospi = common[common["Market"] == "KOSPI"].nlargest(200, "Marcap")
    kosdaq = common[common["Market"].str.startswith("KOSDAQ")].nlargest(150, "Marcap")
    minute = pd.concat([kospi, kosdaq])[["Code", "Name", "Market", "Marcap"]]
    minute.columns = ["code", "name", "market", "marcap"]
    # 실제 KOSPI200·KOSDAQ150 구성종목(기존 stock-analysis-site DB에서 가져온 목록)이 있으면 합친다
    idx_file = store.DATA_DIR / "legacy" / "index_constituents.parquet"
    if idx_file.exists():
        idx = pd.read_parquet(idx_file)
        real = idx[idx["index_code"].isin(["KOSPI200", "KOSDAQ150"])]
        extra = real[~real["symbol"].isin(minute["code"])]
        info = cur.set_index("Code")
        minute = pd.concat([minute, pd.DataFrame({
            "code": extra["symbol"], "name": extra["symbol"].map(info["Name"]),
            "market": extra["symbol"].map(info["Market"]), "marcap": extra["symbol"].map(info["Marcap"]),
        })], ignore_index=True)
        minute["in_index"] = minute["code"].isin(real["symbol"])
    store.save_universe("kr_minute", minute)
    logger.info(f"KR 유니버스: 일봉 {len(daily)}개 (상장폐지 {daily['delisted'].notna().sum()}개), 분봉 {len(minute)}개")
    return daily, minute


def universes() -> tuple[pd.DataFrame, pd.DataFrame]:
    d, m = store.load_universe("kr_daily"), store.load_universe("kr_minute")
    return (d, m) if not d.empty and not m.empty else build_universe()


# ── 일봉 ──
def _daily_chunk(code: str, d1: str, d2: str, adjusted: bool) -> pd.DataFrame:
    rows = _get("/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice", "FHKST03010100",
                {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": d1,
                 "FID_INPUT_DATE_2": d2, "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0" if adjusted else "1"})
    rows = [r for r in rows.get("output2", []) if r.get("stck_bsop_date")]
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    df = pd.DataFrame({
        "ts": pd.to_datetime(d["stck_bsop_date"] + "0900").dt.tz_localize(KST).dt.tz_convert("UTC"),
        "open": d["stck_oprc"].astype(float), "high": d["stck_hgpr"].astype(float),
        "low": d["stck_lwpr"].astype(float), "close": d["stck_clpr"].astype(float),
        "volume": d["acml_vol"].astype(float), "value": d["acml_tr_pbmn"].astype(float),
    })
    # 오늘 봉은 받지 않는다: 장중엔 미완성이고, 장 마감 뒤에도 종가가 KRX 공식 종가와 1~4% 다른 경우가 많다
    # (2026-11-13 수집분 79% 불일치, 거래량은 같음 — 시간외 가격 반영 추정). 다음 날 수집 때 공식 종가로 받는다.
    today = pd.Timestamp.now(tz=KST).normalize()
    return df[df["ts"].dt.tz_convert(KST).dt.normalize() < today]


def _daily_range(code: str, start: pd.Timestamp, adjusted: bool) -> pd.DataFrame:
    """오늘부터 start까지 100일 단위로 거꾸로."""
    out, d2 = [], pd.Timestamp.now(tz=KST).normalize()
    start = start.tz_localize(KST) if start.tzinfo is None else start.tz_convert(KST)
    while d2 >= start:
        df = _daily_chunk(code, start.strftime("%Y%m%d"), d2.strftime("%Y%m%d"), adjusted)
        if df.empty:
            break
        out.append(df)
        d2 = df["ts"].min().tz_convert(KST).normalize() - pd.Timedelta(days=1)
        if len(df) < 100:
            break
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _factor(raw: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    m = raw[["ts", "close"]].merge(adj[["ts", "close"]], on="ts", suffixes=("_raw", "_adj"))
    m = m[m["close_raw"] > 0]
    return pd.DataFrame({"ts": m["ts"], "factor": (m["close_adj"] / m["close_raw"]).round(8)})


def collect_daily(code: str, delisted=None) -> None:
    st = store.get_state("kr_1d", code)
    try:
        if st["backfill_done"] and pd.notna(delisted):
            return  # 상장폐지 종목은 한 번만 받으면 끝
        if st["last_ts"] is None or not st["backfill_done"]:
            start = pd.Timestamp(config()["kr"]["start_1d"])
            raw, adj = _daily_range(code, start, False), _daily_range(code, start, True)
            store.write("kr_1d", code, raw)
            _rewrite_factor(code, _factor(raw, adj))
            store.set_state("kr_1d", code, backfill_done=True)
            logger.info(f"kr_1d {code} 백필 {len(raw)}행")
            return
        # 최신분: 마지막 100일 원본+수정 → 겹치는 구간의 계수가 바뀌었으면 수정계수 전체 재수집
        d1 = (st["last_ts"].tz_convert(KST) - pd.Timedelta(days=150)).strftime("%Y%m%d")
        d2 = pd.Timestamp.now(tz=KST).strftime("%Y%m%d")
        raw, adj = _daily_chunk(code, d1, d2, False), _daily_chunk(code, d1, d2, True)
        store.write("kr_1d", code, raw)
        new_f = _factor(raw, adj)
        old_f = store.read("kr_adj_factor", code, start=new_f["ts"].min()) if not new_f.empty else pd.DataFrame()
        if not old_f.empty:
            chk = old_f.merge(new_f, on="ts", suffixes=("_old", "_new"))
            if ((chk["factor_old"] - chk["factor_new"]).abs() > 1e-6).any():
                logger.info(f"kr_1d {code} 수정계수 변경 감지 → 전체 재수집")
                adj_all = _daily_range(code, pd.Timestamp(config()["kr"]["start_1d"]), True)
                _rewrite_factor(code, _factor(store.read("kr_1d", code), adj_all))
                return
        store.write("kr_adj_factor", code, new_f)
    except Exception as e:
        store.set_state("kr_1d", code, status="error", error=str(e)[:500])
        logger.error(f"kr_1d {code} 실패: {e}")


def _rewrite_factor(code: str, f: pd.DataFrame) -> None:
    shutil.rmtree(store.DATA_DIR / "kr_adj_factor" / code, ignore_errors=True)
    store.write("kr_adj_factor", code, f)


# ── 분봉 ──
def _minute_day(code: str, day: str) -> pd.DataFrame:
    out, hour = [], "153000"
    while True:
        rows = _get("/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice", "FHKST03010230",
                    {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_INPUT_HOUR_1": hour,
                     "FID_INPUT_DATE_1": day, "FID_PW_DATA_INCU_YN": "N", "FID_FAKE_TICK_INCU_YN": ""})
        rows = [r for r in rows.get("output2", []) if r.get("stck_bsop_date") == day]
        if not rows:
            break
        out.extend(rows)
        first = min(r["stck_cntg_hour"] for r in rows)
        if first <= "090000" or len(rows) < 120:
            break
        hour = (pd.Timestamp(f"{day} {first}") - pd.Timedelta(minutes=1)).strftime("%H%M%S")
    if not out:
        return pd.DataFrame()
    d = pd.DataFrame(out)
    return pd.DataFrame({
        "ts": pd.to_datetime(d["stck_bsop_date"] + d["stck_cntg_hour"]).dt.tz_localize(KST).dt.tz_convert("UTC"),
        "open": d["stck_oprc"].astype(float), "high": d["stck_hgpr"].astype(float),
        "low": d["stck_lwpr"].astype(float), "close": d["stck_prpr"].astype(float),
        "volume": d["cntg_vol"].astype(float),
    })


def trading_days() -> list[str]:
    """삼성전자 일봉으로 거래일 달력 생성 (최근 2년)."""
    df = _daily_range("005930", pd.Timestamp.now() - pd.Timedelta(days=730), False)
    return sorted(df["ts"].dt.tz_convert(KST).dt.strftime("%Y%m%d").unique())


def collect_minute(code: str, days: list[str], phase: str = "all") -> None:
    """phase: latest(마지막 수집일 이후) / backfill(과거분) / all."""
    st = store.get_state("kr_1m", code)
    today = pd.Timestamp.now(tz=KST)
    # 장중이면 오늘은 제외 (장 마감 후 수집)
    closed = [d for d in days if d < today.strftime("%Y%m%d") or today.hour >= 16]
    try:
        if phase in ("latest", "all") and st["last_ts"] is not None:  # 최신분
            last = st["last_ts"].tz_convert(KST).strftime("%Y%m%d")
            for d in [d for d in closed if d > last]:
                store.write("kr_1m", code, _minute_day(code, d))
        if phase in ("backfill", "all") and not st["backfill_done"]:  # 과거분: 5거래일 연속 비면 KIS 보관 한도로 보고 종료
            first = st["first_ts"].tz_convert(KST).strftime("%Y%m%d") if st["first_ts"] is not None else "99999999"
            empty, buf = 0, []
            for d in reversed([d for d in closed if d < first]):
                df = _minute_day(code, d)
                empty = empty + 1 if df.empty else 0
                if not df.empty:
                    buf.append(df)
                if len(buf) >= 20:
                    store.write("kr_1m", code, pd.concat(buf))
                    buf = []
                if empty >= 5:
                    break
            if buf:
                store.write("kr_1m", code, pd.concat(buf))
            store.set_state("kr_1m", code, backfill_done=True)
            logger.info(f"kr_1m {code} 백필 완료")
    except Exception as e:
        store.set_state("kr_1m", code, status="error", error=str(e)[:500])
        logger.error(f"kr_1m {code} 실패: {e}")


WORKERS = 5  # KIS 응답이 느려(건당 ~0.2초) 순차로는 초당 5건 수준 → 병렬로 돌리고 총량은 RateLimiter(15/s)가 제한


def run(what: str = "all", symbols: list[str] | None = None) -> None:
    from concurrent.futures import ThreadPoolExecutor

    daily, minute = universes()
    kis().token()  # 스레드 시작 전에 토큰 한 번만 발급
    days = trading_days() if what in ("all", "minute") else []
    codes = symbols or minute["code"].tolist()
    d = daily[daily["code"].isin(symbols)] if symbols else daily
    done_daily = d["code"].map(lambda c: store.get_state("kr_1d", c)["backfill_done"])
    # 순서: 최신 데이터를 먼저 맞추고(분봉 → 일봉), 과거분은 그 뒤에 채운다.
    # 분봉 백필을 일봉 백필보다 앞에 두는 이유: KIS 분봉은 약 1년만 보관해 매일 하루치씩 사라진다.
    with ThreadPoolExecutor(WORKERS) as pool:
        if what in ("all", "minute"):
            list(pool.map(lambda c: collect_minute(c, days, "latest"), codes))
            logger.info("kr_1m 최신분 완료")
        if what in ("all", "daily"):
            latest = d[done_daily]
            list(pool.map(lambda r: collect_daily(r[0], r[1]), zip(latest["code"], latest["delisted"])))
            logger.info(f"kr_1d 최신분 완료 ({len(latest)}종목)")
        if what in ("all", "minute"):
            list(pool.map(lambda c: collect_minute(c, days, "backfill"), codes))
        if what in ("all", "daily"):
            rest = d[~done_daily]
            list(pool.map(lambda r: collect_daily(r[0], r[1]), zip(rest["code"], rest["delisted"])))
