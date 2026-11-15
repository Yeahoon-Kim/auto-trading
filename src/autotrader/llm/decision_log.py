"""LLM 판단 기록과 사후 채점 (TradingAgents의 memory log + settle에서 가져온 아이디어).

llm lab의 판단(관점 JSON)은 전부 여기에 남기고, 보유 기간이 지나면 실제 가격으로 채점한다.
→ LLM 전략의 "전진 검증" 기록이 이 표다 (모델 지식 = 미래참조라 과거 백테스트를 믿을 수 없기 때문).

규칙
- 진입은 판단 시각 "이후 첫 종가" (판단이 그날 장 마감 뒤 뉴스를 봤을 수 있으므로).
- 청산은 진입 후 horizon 거래일째 종가. 그 종가가 아직 없으면 채점하지 않고 대기.
- 성과는 같은 기간 벤치마크 대비 초과수익(alpha)으로도 기록한다.
- 기록마다 모델·프롬프트 해시·입력 데이터 기준 시각을 남겨 재현할 수 있게 한다.
- 채점 결과를 다음 판단의 입력(반성)으로 넣을지는 lab마다 정한다. 넣는다면 이미 채점된 것만.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "data" / "llm" / "decisions.jsonl"


@dataclass
class Decision:
    lab: str
    decided_at: str          # ISO UTC
    market: str              # kr | us | crypto
    symbol: str
    stance: str              # long | short | flat
    conviction: float        # 0~1
    horizon_days: int
    benchmark: str           # 같은 데이터셋 안의 벤치마크 심볼 (예: 069500, SPY, BTCUSDT)
    model: str
    prompt_sha: str
    data_asof: str
    rationale: str = ""
    settled: bool = False
    entry_ts: str | None = None
    exit_ts: str | None = None
    ret: float | None = None
    bench_ret: float | None = None
    alpha: float | None = None


def append(d: Decision, path: Path = LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def load(path: Path = LOG) -> list[Decision]:
    if not path.exists():
        return []
    return [Decision(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _window(px: pd.Series, after: pd.Timestamp, horizon: int):
    """after 이후 첫 종가(진입)와 그로부터 horizon번째 종가(청산). 아직 없으면 None."""
    fut = px[px.index > after]
    if len(fut) <= horizon:
        return None
    return fut.index[0], fut.iloc[0], fut.index[horizon], fut.iloc[horizon]


def settle_all(prices, path: Path = LOG) -> list[Decision]:
    """prices(market, symbol) -> 종가 Series(index=ts UTC). 채점 가능한 기록을 채점하고 파일을 다시 쓴다.
    반환: 이번에 새로 채점된 기록."""
    decs, done = load(path), []
    for d in decs:
        if d.settled:
            continue
        after = pd.Timestamp(d.decided_at)
        w, b = _window(prices(d.market, d.symbol), after, d.horizon_days), None
        if w is None:
            continue
        b = _window(prices(d.market, d.benchmark), after, d.horizon_days)
        sign = {"long": 1, "short": -1, "flat": 0}[d.stance]
        d.entry_ts, d.exit_ts = w[0].isoformat(), w[2].isoformat()
        d.ret = sign * (w[3] / w[1] - 1)
        if b is not None:
            d.bench_ret = b[3] / b[1] - 1
            d.alpha = d.ret - sign * d.bench_ret if sign else 0.0
        d.settled = True
        done.append(d)
    if done:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(asdict(x), ensure_ascii=False) + "\n" for x in decs), encoding="utf-8")
        tmp.replace(path)
    return done
