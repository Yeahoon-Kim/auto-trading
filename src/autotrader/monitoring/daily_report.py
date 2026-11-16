"""운영팀 일일 마감 보고 (매일 23:00).

사실은 전부 코드가 모은다 — 수집 상태, 신선도 점검, 수집 로그 ERROR, LLM 호출 실패율, 스캐너 결과,
LLM 판단 채점 현황, 데이터 용량 증가, 리서치 진행 상태(labs/). LLM(ops-reporter, 빠른 모델)은 그 사실 표를 10줄 안팎으로
요약만 한다. LLM이 실패하면 사실 표만 보낸다.

    uv run python -m autotrader.monitoring.daily_report [--notify] [--no-llm] [--date YYYY-MM-DD]

결과: reports/ops/YYYY-MM-DD.md (+ 용량 기록 reports/ops/sizes.json)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from ..data import store
from ..llm import claude_cli, decision_log

ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "logs"
SCAN_DIR = ROOT / "reports" / "scan"
REPORT_DIR = ROOT / "reports" / "ops"
LABS_DIR = ROOT / "labs"
TZ = "Asia/Seoul"

SYSTEM = """너는 헤지펀드 운영팀(ops-reporter)이다. 아래 '사실 표'만 보고 일일 마감 보고를 쓴다.
- 숫자는 표에서 그대로 옮긴다. 표에 없는 숫자를 만들지 않는다. 추정하면 "추정"이라고 쓴다.
- 백필완료가 "-"인 데이터셋은 파생 표·신선도 점검 비대상이므로 백필 문제로 보지 않는다.
- 문제는 숨기지 않고 맨 위에 쓴다. 문제가 없으면 "문제 없음"이라고 쓴다.
- '문제'는 표의 "미해결"만이다. "복구됨"과 수집 로그의 과거 ERROR는 이미 해결됐으므로 문제로 세지 않는다
  (필요하면 데이터 상태에 "복구됨 N건"으로만 쓴다). 미해결이 0건이고 신선도 점검도 0건이면 "문제 없음"이다.
- 리서치 진행 상태의 판정 대기(validator 의견서 없음 / 사용자 판정 대기)는 '다음 할 일'에 lab id와 함께 쓴다.
- 형식(순서 고정): 문제 → 데이터 상태 → 전략·판단 기록(리서치 진행 포함) → 다음 할 일. 전체 10줄 안팎(최대 12줄), 한 줄에 한 사실, 한국어.
- 주문·설정 변경을 권하지 않는다. 매수·매도 의견을 내지 않는다."""

# 로그 메시지에 섞일 수 있는 키·토큰 가리기 (URL 쿼리의 key=..., 20자 이상 영숫자 덩어리)
_SECRET = re.compile(r"((?:key|token|secret|apikey|appkey)=)[^&\s]+|[A-Za-z0-9_\-]{20,}", re.IGNORECASE)


def _mask(s: str) -> str:
    return _SECRET.sub(lambda m: (m.group(1) or "") + "***", s)


def _today() -> str:
    return f"{pd.Timestamp.now(tz=TZ):%Y-%m-%d}"


# ── 사실 수집 ────────────────────────────────────────────────────────────────

def _collect_status() -> list[dict]:
    try:
        df = store.summary()
    except Exception as e:  # noqa: BLE001
        return [{"dataset": "(상태DB)", "error": f"{type(e).__name__}: {e}"[:200]}]
    return df.to_dict("records")


def _health() -> list[str]:
    from . import health
    try:
        return [f"[{i.level}] {i.dataset}: {i.message}" for i in health.check()]
    except Exception as e:  # noqa: BLE001
        return [f"[error] health: 점검 실패 {type(e).__name__}: {e}"[:200]]


def _error_state() -> dict:
    """수집 상태DB의 오류 행을 미해결/복구됨으로 나눈다 (분류 기준은 health.split_errors)."""
    from . import health
    try:
        o, r = health.split_errors(health._state(), pd.Timestamp.now(tz="UTC"))
    except Exception as e:  # noqa: BLE001
        return {"open": [], "recovered": [], "failed": f"{type(e).__name__}: {e}"[:200]}

    def rows(df: pd.DataFrame) -> list[tuple[str, str, str]]:
        return [(d, s, _mask(str(m or ""))[:100]) for d, s, m in zip(df["dataset"], df["symbol"], df["error"])]
    return {"open": rows(o), "recovered": rows(r), "failed": None}


# 로그 오류 줄 "kr_1m 017670 실패: ..." 에서 (데이터셋, 종목)을 뽑아 상태DB와 맞춰 본다
_LOG_KEY = re.compile(r"^(\S+) (\S+) 실패")


def split_log_error(msg: str, open_keys: set[tuple[str, str]]) -> bool:
    """로그 ERROR 한 줄이 지금도 미해결인 (데이터셋, 종목)의 것이면 True. 못 알아보면 False(과거 발생으로 센다)."""
    m = _LOG_KEY.match(msg)
    return bool(m) and (m[1], m[2]) in open_keys


def _collect_log(day: str, open_keys: set[tuple[str, str]] | None = None) -> dict:
    path = LOG_DIR / f"collect_{day}.log"
    if not path.exists():
        return {"exists": False, "errors": 0, "open": 0, "where": []}
    where: Counter = Counter()
    first: dict[str, str] = {}
    n_open = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(" | ", 2)
        if len(parts) < 3 or parts[1].strip() != "ERROR":
            continue
        loc, _, msg = parts[2].partition(" - ")
        where[loc] += 1
        first.setdefault(loc, _mask(msg)[:100])
        n_open += split_log_error(msg, open_keys or set())
    return {"exists": True, "errors": sum(where.values()), "open": n_open,
            "where": [(loc, n, first[loc]) for loc, n in where.most_common(3)]}


def _llm_calls(day: str) -> dict:
    path = LOG_DIR / "llm_calls.jsonl"
    recs = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if f"{pd.Timestamp(r['at']).tz_convert(TZ):%Y-%m-%d}" == day:
                recs.append(r)
    fails = [r for r in recs if not r.get("ok")]
    return {"calls": len(recs), "fails": len(fails),
            "fail_rate": len(fails) / len(recs) if recs else None,
            "by_tag": dict(Counter(r.get("tag") or "-" for r in recs)),
            "fail_tags": dict(Counter(r.get("tag") or "-" for r in fails))}


def _scan(day: str) -> dict | None:
    path = SCAN_DIR / f"{day}.json"
    if not path.exists():
        return None
    return {m: len(rows) for m, rows in json.loads(path.read_text(encoding="utf-8")).items()}


def _decisions() -> dict:
    decs = decision_log.load(decision_log.LOG)
    settled = [d for d in decs if d.settled]
    alphas = [d.alpha for d in settled if d.alpha is not None]
    return {"total": len(decs), "settled": len(settled), "pending": len(decs) - len(settled),
            "alpha_mean": sum(alphas) / len(alphas) if alphas else None,
            "hit": sum(a > 0 for a in alphas) / len(alphas) if alphas else None}


def _sizes() -> dict[str, int]:
    out: dict[str, int] = {}
    if store.DATA_DIR.exists():
        for d in store.DATA_DIR.iterdir():
            if d.is_dir():
                out[d.name] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return out


def _size_growth(day: str, now: dict[str, int]) -> dict:
    """reports/ops/sizes.json 의 직전 기록과 비교. 기준이 없으면 증가분 None."""
    hist = _load_sizes()
    prev_days = sorted(k for k in hist if k < day)
    prev = hist[prev_days[-1]] if prev_days else None
    return {"total": sum(now.values()), "since": prev_days[-1] if prev_days else None,
            "delta": sum(now.values()) - sum(prev.values()) if prev else None,
            "top": sorted(((k, v - prev.get(k, 0)) for k, v in now.items()), key=lambda x: -x[1])[:3] if prev else []}


def _load_sizes() -> dict:
    p = REPORT_DIR / "sizes.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_sizes(day: str, now: dict[str, int]) -> None:
    hist = _load_sizes()
    hist[day] = now
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "sizes.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")


# 판정 전 단계: 이 상태에서 의견서 유무로 누가 할 차례인지 보인다
_PRE_VERDICT = {"backtesting", "validating"}


def _labs() -> list[dict]:
    """labs/L*/lab.yaml 마다 id·name·version·status, 판정 대기, 마지막 결과 실행 시각."""
    out = []
    for d in sorted(LABS_DIR.glob("L*")):
        y = d / "lab.yaml"
        if not d.is_dir() or not y.exists():
            continue
        try:
            meta = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001
            out.append({"dir": d.name, "error": f"{type(e).__name__}: {e}"[:120]})
            continue
        status = str(meta.get("status") or "-")
        if status in _PRE_VERDICT:
            waiting = "사용자 판정 대기" if (d / "validation.md").exists() else "validator 의견서 없음"
        else:
            waiting = "-"
        res = d / "results"
        runs = sorted(p.name for p in res.iterdir() if p.is_dir()) if res.is_dir() else []
        out.append({"dir": d.name, "id": meta.get("id") or d.name, "name": meta.get("name") or "-",
                    "version": meta.get("version"), "status": status, "waiting": waiting,
                    "last_run": runs[-1] if runs else None})
    return out


def _backlog(section: str = "후보") -> int | None:
    """labs/BACKLOG.md '## {section}' 표의 데이터 행 수. 파일이 없으면 None."""
    p = LABS_DIR / "BACKLOG.md"
    if not p.exists():
        return None
    n, in_sec, seen_head = 0, False, False
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t.startswith("## "):
            in_sec, seen_head = t[3:].strip() == section, False
            continue
        if not in_sec or not t.startswith("|"):
            continue
        if not seen_head:  # 첫 줄은 머리글
            seen_head = True
        elif not re.fullmatch(r"\|[\s:|-]+\|", t):  # 구분선 제외
            n += 1
    return n


def collect(day: str | None = None) -> dict:
    day = day or _today()
    sizes = _sizes()
    errors = _error_state()
    open_keys = {(d, s) for d, s, _ in errors["open"]}
    return {"day": day, "status": _collect_status(), "health": _health(), "errors": errors,
            "log": _collect_log(day, open_keys),
            "llm": _llm_calls(day), "scan": _scan(day), "decisions": _decisions(),
            "sizes": sizes, "growth": _size_growth(day, sizes), "labs": _labs(), "backlog": _backlog()}


# ── 표·보고서 ────────────────────────────────────────────────────────────────

def _mb(b: int | None) -> str:
    return "-" if b is None else f"{b / 1e6:+,.1f}MB" if b < 0 else f"{b / 1e6:,.1f}MB"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def table(f: dict) -> str:
    from . import health
    errs = f.get("errors") or {"open": [], "recovered": [], "failed": None}
    open_by = Counter(d for d, _, _ in errs["open"])
    L = [f"# 운영 사실 표 {f['day']} (KST)", "", "## 수집 상태 (store.summary)",
         "dataset | 종목 | 마지막 봉(UTC) | 백필완료 | 미해결 오류"]
    for r in f["status"]:
        if "error" in r and "symbols" not in r:
            L.append(f"{r['dataset']} | 읽기 실패: {r['error']}")
            continue
        last = str(r.get("last_ts") or "-")[:16]
        # 신선도 점검 대상이 아닌 데이터셋(파생 표 등)은 백필 개념이 없다 → "-"
        bf = int(r["backfilled"] or 0) if r["dataset"] in health.RULES else "-"
        L.append(f"{r['dataset']} | {r['symbols']} | {last} | {bf} | {open_by.get(r['dataset'], 0) if 'errors' in f else int(r['errors'] or 0)}")
    L += ["", f"## 신선도 점검 (health.check): {len(f['health'])}건"] + [f"- {h}" for h in f["health"]]
    L += ["", "## 수집 오류 (상태DB)"]
    if errs.get("failed"):
        L.append(f"- 상태DB 읽기 실패: {errs['failed']}")
    L.append(f"- 미해결 {len(errs['open'])}건")
    L += [f"  - {d} {s}: {m}" for d, s, m in errs["open"][:5]]
    L.append(f"- 복구됨 {len(errs['recovered'])}건 (오류가 났지만 지금은 데이터가 최신 — 문제 아님)")
    L += [f"  - {d} {s}" for d, s, _ in errs["recovered"][:5]]
    lg = f["log"]
    L += ["", "## 수집 로그"]
    if not lg["exists"]:
        L.append(f"- logs/collect_{f['day']}.log 없음 (오늘 수집 미실행 의심)")
    else:
        L.append(f"- ERROR {lg['errors']}건 중 미해결 종목 {lg.get('open', 0)}건, "
                 f"나머지 {lg['errors'] - lg.get('open', 0)}건은 발생 후 복구됨·종목 불명 (아래 예시는 과거 기록, 문제 아님)")
        L += [f"  - {loc} ×{n}: {msg}" for loc, n, msg in lg["where"]]
    m = f["llm"]
    L += ["", "## LLM 호출 (logs/llm_calls.jsonl, 오늘)",
          f"- 호출 {m['calls']}건, 실패 {m['fails']}건 (실패율 {_pct(m['fail_rate'])})"]
    if m["by_tag"]:
        L.append("- 태그별: " + ", ".join(f"{k} {v}" for k, v in m["by_tag"].items()))
    if m["fail_tags"]:
        L.append("- 실패 태그: " + ", ".join(f"{k} {v}" for k, v in m["fail_tags"].items()))
    s = f["scan"]
    L += ["", "## 스캐너 (reports/scan, 오늘)",
          "- 결과 없음" if s is None else "- " + ", ".join(f"{k} {v}건" for k, v in s.items()) + f" (합계 {sum(s.values())})"]
    d = f["decisions"]
    L += ["", "## LLM 판단 기록 (data/llm/decisions.jsonl)",
          f"- 전체 {d['total']}건, 채점 {d['settled']}건, 대기 {d['pending']}건"
          + (f", 평균 초과수익 {_pct(d['alpha_mean'])}, 적중률 {_pct(d['hit'])}" if d["alpha_mean"] is not None else "")]
    g = f["growth"]
    L += ["", "## 데이터 용량 (data/processed)", f"- 합계 {_mb(g['total'])}"
          + (f", {g['since']} 대비 {_mb(g['delta'])}" if g["since"] else " (비교 기준 없음 — 첫 기록)")]
    if g["top"]:
        L.append("- 증가 상위: " + ", ".join(f"{k} {_mb(v)}" for k, v in g["top"]))
    L += ["", "## 리서치 진행 상태 (labs/)"]
    labs = f.get("labs") or []
    if labs:
        L.append("lab | 이름 | 버전 | 상태 | 판정 대기 | 마지막 결과")
        for r in labs:
            if "error" in r:
                L.append(f"{r['dir']} | lab.yaml 읽기 실패: {r['error']}")
                continue
            ver = "-" if r["version"] is None else f"v{r['version']}"
            L.append(f"{r['id']} | {r['name']} | {ver} | {r['status']} | {r['waiting']} | {r['last_run'] or '결과 없음'}")
    else:
        L.append("- lab 없음")
    if f.get("backlog") is not None:
        L.append(f"- 대기열 후보 (labs/BACKLOG.md): {f['backlog']}건")
    return "\n".join(L)


def summarize(tbl: str) -> str | None:
    out = claude_cli.ask(tbl, SYSTEM, model=claude_cli.FAST, tag="ops-reporter", timeout=180)
    return out.strip() if isinstance(out, str) and out.strip() else None


def build(f: dict, use_llm: bool = True) -> str:
    tbl = table(f)
    note = summarize(tbl) if use_llm else None
    head = f"📋 운영 마감 보고 {f['day']}"
    if note:
        return f"{head}\n\n■ 요약 (ops-reporter · LLM)\n{note}\n\n---\n{tbl}"
    tail = "" if not use_llm else "\n\n(LLM 요약 생성 실패 — logs/llm_calls.jsonl 확인)"
    return f"{head}\n\n{tbl}{tail}"


def main() -> None:
    p = argparse.ArgumentParser(prog="autotrader.monitoring.daily_report")
    p.add_argument("--notify", action="store_true", help="텔레그램으로 보내기")
    p.add_argument("--no-llm", action="store_true", help="LLM 요약 없이 사실 표만")
    p.add_argument("--date", help="보고 날짜 YYYY-MM-DD (KST, 기본 오늘)")
    a = p.parse_args()
    f = collect(a.date)
    text = build(f, use_llm=not a.no_llm)
    _save_sizes(f["day"], f["sizes"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{f['day']}.md").write_text(text, encoding="utf-8")
    print(text)
    if a.notify:
        from . import notify
        notify.send(text)


if __name__ == "__main__":
    main()
