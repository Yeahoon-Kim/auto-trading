"""시장정보팀 market-analyst: 국면 지표·스캐너·공시를 읽고 3~5줄 해석을 붙인다.

LLM은 해석만 한다. 숫자는 모두 코드가 계산해서 넘기고, LLM은 받은 숫자 밖의 사실을 만들지 않는다.
매수·매도 권유나 가격 예측은 하지 않는다. 실패하면 None → 브리핑은 해석 없이 나간다.
"""

from __future__ import annotations

import re

import pandas as pd

from ..data import store
from ..llm import claude_cli

# 주가에 영향을 주는 경우가 많은 공시 유형 (보고서명 부분 일치)
KEY_DISCLOSURES = ("유상증자", "무상증자", "전환사채", "신주인수권", "최대주주변경", "자기주식",
                   "단일판매", "공급계약", "영업(잠정)실적", "합병", "분할", "감자", "상장폐지",
                   "거래정지", "불성실공시", "횡령", "배임", "소송")

SYSTEM = """너는 헤지펀드 시장정보팀의 애널리스트다. 매일 아침 운용역에게 보내는 브리핑에 해석 문단을 붙인다.
규칙:
- 입력에 있는 숫자와 사실만 쓴다. 입력에 없는 뉴스·사건·수치를 지어내지 않는다.
- 매수·매도 권유, 가격 예측, 목표가를 쓰지 않는다. "어떤 상황인지"와 "오늘 무엇을 주의해 볼지"만 쓴다.
- 지난 1~4주 대비 무엇이 달라졌는지를 우선한다. 신호끼리 엇갈리면 엇갈린다고 쓴다.
- 한국어, 3~5줄, 각 줄은 "- "로 시작. 다른 머리말·맺음말 없이 그 줄들만 출력한다."""


def _changes(regime: pd.DataFrame) -> str:
    """주요 지표의 현재값과 5·20거래일 전 값."""
    cols = ["risk_regime", "risk_off_signals", "risk_off_available", "vix", "vix_term", "hy_spread", "t10y2y", "usdkrw", "dxy",
            "fng", "funding_avg_7d_ann", "kimchi_premium_btc", "spx_trend", "kospi_trend", "btc_trend",
            "spx_ret_20d", "kospi_ret_20d", "btc_ret_20d", "spx_rv_pctile", "kospi_rv_pctile", "btc_rv_pctile"]
    r = regime[[c for c in cols if c in regime]].dropna(how="all")
    pick = r.iloc[[-21, -6, -1]] if len(r) > 21 else r.tail(3)
    pick.index = [f"{i:%m-%d}" for i in pick.index]
    return pick.T.to_string(float_format=lambda v: f"{v:.3g}")


def _disclosures(days: int = 3, limit: int = 25) -> str:
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    d = store.read("dart_disclosures", "ALL", start=since)
    if d.empty:
        return "(없음)"
    d = d[d["corp_cls"].isin(["Y", "K"]) & d["report_nm"].str.contains("|".join(map(re.escape, KEY_DISCLOSURES)), regex=True)]
    d = d.sort_values("ts").tail(limit)
    return "\n".join(f"{t:%m-%d} {n} — {r.strip()}" for t, n, r in zip(d["ts"].dt.tz_convert("Asia/Seoul"), d["corp_name"], d["report_nm"])) or "(없음)"


def commentary(regime: pd.DataFrame, brief_text: str, scan_text: str = "") -> str | None:
    prompt = (
        f"[오늘 브리핑 본문]\n{brief_text}\n\n"
        f"[주요 지표 변화: 열 = 20거래일 전 / 5거래일 전 / 최근]\n{_changes(regime)}\n\n"
        f"[스캐너]\n{scan_text or '(없음)'}\n\n"
        f"[최근 3일 주요 공시 (KOSPI·KOSDAQ)]\n{_disclosures()}\n"
    )
    out = claude_cli.ask(prompt, SYSTEM, model=claude_cli.FAST, tag="market-analyst", timeout=240)
    if not out:
        return None
    lines = [l.rstrip() for l in str(out).strip().splitlines() if l.strip()]
    return "\n".join(lines[:6])
