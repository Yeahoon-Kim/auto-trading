"""데이터 수집 상태 점검.

수집기가 조용히 죽어도(강제 종료, 절전, 네트워크) 알 수 있게, 저장된 데이터의 신선도를 직접 본다.
- 신선도: 데이터셋별 마지막 봉이 허용 지연보다 오래된 종목 수
- 오류: 마지막 수집에서 오류가 난 종목
- 활동: 데이터셋 전체가 오랫동안 갱신되지 않음 (수집기 미실행 의심)
- 디스크 여유 공간
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import store

# 허용 지연. hours = 시간 기준(24시간 시장), bdays = 영업일 기준(주식, 공휴일은 감안 안 함 → 연휴엔 경고 가능)
RULES = {
    "binance_spot_1m": {"hours": 16},
    "binance_um_1m": {"hours": 16},
    "binance_um_funding": {"hours": 20},
    "binance_um_oi": {"hours": 20},
    "upbit_1m": {"hours": 16},
    "crypto_fear_greed": {"hours": 50},
    "news": {"hours": 36},  # 기사 시각 기준. 주말·연휴엔 기사가 적어 여유를 둔다
    "kr_1d": {"bdays": 2, "tz": "Asia/Seoul", "close_hour": 16},
    "kr_1m": {"bdays": 2, "tz": "Asia/Seoul", "close_hour": 16},
    "us_1d": {"bdays": 2, "tz": "America/New_York", "close_hour": 17},
    "us_1m": {"bdays": 2, "tz": "America/New_York", "close_hour": 17},
    "us_1m_sip": {"bdays": 2, "tz": "America/New_York", "close_hour": 17},
    "macro_1d": {"bdays": 3, "tz": "America/New_York", "close_hour": 17},
    "krx_1d": {"bdays": 3, "tz": "Asia/Seoul", "close_hour": 16},  # KRX는 다음 영업일 공개
    "dart_disclosures": {"bdays": 3, "tz": "Asia/Seoul", "close_hour": 20},
    "ecos": {"bdays": 5, "tz": "Asia/Seoul", "close_hour": 16},
    "fred": {"bdays": 10, "tz": "America/New_York", "close_hour": 17},  # 주간 지표(NFCI 등)는 발표가 늦음
}
RECENT_ERROR_HOURS = 6  # 이보다 최근에 난 오류는 데이터가 신선해도 아직 진행 중일 수 있어 미해결로 본다
INACTIVE_HOURS = 30   # 데이터셋 전체가 이 시간 동안 한 번도 갱신 안 되면 수집기 미실행 의심
MIN_FREE_GB = 10


@dataclass
class Issue:
    level: str  # "error" | "warn"
    dataset: str
    message: str


def _state() -> pd.DataFrame:
    with sqlite3.connect(store.STATE_DB, timeout=60) as con:
        df = pd.read_sql("SELECT * FROM state", con)
    for c in ("last_ts", "updated_at"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df


def _expected_session(rule: dict, now: pd.Timestamp) -> pd.Timestamp:
    """지금 시점에 받아 있어야 할 가장 최근 거래일 (장 마감 후면 오늘, 아니면 직전 평일)."""
    local = now.tz_convert(rule["tz"])
    day = local.normalize()
    if local.hour < rule["close_hour"] or day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def _stale(g: pd.DataFrame, rule: dict, now: pd.Timestamp) -> pd.DataFrame:
    """허용 지연보다 오래된 행."""
    if "hours" in rule:
        return g[g["last_ts"] < now - pd.Timedelta(hours=rule["hours"])]
    exp = _expected_session(rule, now).date()
    last_local = g["last_ts"].dt.tz_convert(rule["tz"]).dt.tz_localize(None).values.astype("datetime64[D]")
    return g[np.busday_count(last_local, np.datetime64(exp)) > rule["bdays"]]


def split_errors(st: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """status=error 행을 (미해결, 복구됨)으로 나눈다.

    수집기는 성공해야만 status를 ok로 덮어쓴다. 그런데 이미 최신이라 다시 받을 게 없는 종목은 성공 기록이
    없어서 error가 남는다. 그래서 다음을 모두 만족하면 복구된 것으로 본다.
    - 오류가 RECENT_ERROR_HOURS보다 오래됨 (updated_at = 오류 시각)
    - 데이터가 신선함 (마지막 봉이 신선도 허용 지연 안)
    매일 실패하는 종목은 오류 시각이 계속 갱신되고 데이터도 낡아가므로 계속 미해결로 남는다.
    """
    errs = st[st["status"] == "error"]
    fixed = pd.Series(False, index=errs.index)
    for ds, g in errs.groupby("dataset"):
        rule = RULES.get(ds)
        if rule is None:
            continue  # 신선도 기준이 없는 데이터셋은 판단 못 함 → 미해결 유지
        g = g[g["last_ts"].notna() & (g["updated_at"] < now - pd.Timedelta(hours=RECENT_ERROR_HOURS))]
        if len(g):
            fixed[g.index.difference(_stale(g, rule, now).index)] = True
    return errs[~fixed], errs[fixed]


def check(now: pd.Timestamp | None = None) -> list[Issue]:
    now = now or pd.Timestamp.now(tz="UTC")
    st = _state()
    open_errs, _ = split_errors(st, now)
    issues: list[Issue] = []
    for ds, g in st.groupby("dataset"):
        rule = RULES.get(ds)
        if rule is None:
            continue
        g = g[g["backfill_done"] == 1]  # 백필 중인 종목은 신선도 점검에서 제외
        if ds == "kr_1d":  # 상장폐지 종목은 갱신 대상 아님
            u = store.load_universe("kr_daily", max_age_days=10**6)
            if not u.empty:
                g = g[~g["symbol"].isin(u.loc[u["delisted"].notna(), "code"])]
        stale = _stale(g, rule, now)
        if len(stale):
            worst = stale.nsmallest(3, "last_ts")
            sample = ", ".join(f"{s}({t:%m-%d %H:%M})" for s, t in zip(worst["symbol"], worst["last_ts"]))
            issues.append(Issue("warn", ds, f"오래된 데이터 {len(stale)}/{len(g)}종목 — 예: {sample}"))
        errs = open_errs[open_errs["dataset"] == ds]  # 복구된 과거 오류는 빼고, 지금도 실패 중인 것만
        if len(errs):
            issues.append(Issue("error", ds, f"수집 오류 {len(errs)}종목 — 예: {errs['symbol'].iloc[0]}: "
                                             f"{str(errs['error'].iloc[0])[:120]}"))
        last_activity = st.loc[st["dataset"] == ds, "updated_at"].max()
        if last_activity < now - pd.Timedelta(hours=INACTIVE_HOURS):
            issues.append(Issue("error", ds, f"{INACTIVE_HOURS}시간 넘게 갱신 없음 (마지막 {last_activity:%m-%d %H:%M} UTC) "
                                             "— 수집기가 실행되지 않고 있을 수 있음"))
    free_gb = shutil.disk_usage(store.DATA_DIR).free / 1e9
    if free_gb < MIN_FREE_GB:
        issues.append(Issue("error", "disk", f"디스크 여유 {free_gb:.1f}GB"))
    return issues


def progress() -> str:
    """백필 진행 현황 한 줄 요약."""
    st = _state()
    parts = [f"{ds} {int(g['backfill_done'].sum())}/{len(g)}" for ds, g in st.groupby("dataset")
             if g["backfill_done"].sum() < len(g) and ds in RULES]
    return "백필 진행: " + (", ".join(parts) if parts else "모두 완료")


def report(issues: list[Issue]) -> str:
    if not issues:
        return "✅ 데이터 수집 정상"
    lines = ["⚠️ 데이터 수집 점검"]
    for i in sorted(issues, key=lambda x: x.level != "error"):
        lines.append(f"{'🔴' if i.level == 'error' else '🟡'} {i.dataset}: {i.message}")
    return "\n".join(lines)
