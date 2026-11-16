"""운영팀 주간 회의 자료 (매주 토요일 10:00 KST — 미국 금요일 장 마감 후).

운용 모델(docs/operating-model.md) 고리 ④. 사실은 전부 코드가 모은다 — 지난 7일의
① 데이터 상태(수집 로그 오류·경고, 용량 증가, 상태 점검 🔴 이력) ② LLM 호출(태그별 건수·실패·평균 시간)
③ 리서치(lab 상태, 이번 주 lab.yaml 변경, 대기열 건수, 판정 대기) ④ 운용(paper 장부의 전략별 손익 vs 기대치,
리스크 발동, 퇴출 조건 근접). LLM(ops-reporter, 빠른 모델)은 사실 표를 15줄 안팎으로 요약만 한다.
LLM이 실패하면 사실 표만 남긴다. 사실 수집 함수는 daily_report의 것을 그대로 쓴다.

    uv run python -m autotrader.monitoring.weekly_report [--notify] [--no-llm] [--week YYYY-Www]

기간: 기본은 오늘(KST)까지 7일, 이름은 오늘이 속한 ISO 주. --week를 주면 그 주 토요일까지 7일(일~토)
→ 토요일 예약 실행의 기본값과 같은 기간이 된다.
결과: reports/weekly/YYYY-Www.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from ..llm import claude_cli
from . import daily_report as dr

ROOT = dr.ROOT
REPORT_DIR = ROOT / "reports" / "weekly"
PAPER_DIR = ROOT / "results" / "paper"
TZ = dr.TZ
NEAR = 0.7   # 퇴출 조건 한도의 이 비율을 넘으면 "근접"

SYSTEM = """너는 헤지펀드 운영팀(ops-reporter)이다. 아래 '주간 사실 표'만 보고 주간 회의 자료를 쓴다.
- 숫자는 표에서 그대로 옮긴다. 표에 없는 숫자를 만들지 않는다. 추정하면 "추정"이라고 쓴다.
- 형식(순서·머리말 고정):
  ■ 지난주 요약 — 데이터·LLM·리서치·운용을 한 줄씩
  ■ 문제 — 표의 "현재 미해결" 수집 오류, 현재 신선도 점검, LLM 실패, 리스크 발동. 없으면 "문제 없음".
    기간 중 발생 후 복구된 오류·🔴 이력·로그 ERROR는 이미 해결됐으므로 문제가 아니라 '지난주 요약'에 건수로만 쓴다
  ■ 퇴출 검토 대상 — 표의 '퇴출 조건' 칸이 근접·도달·걸림인 전략. 운용 중 전략이 없거나 해당이 없으면 "없음"
  ■ 다음 주 HQ 배정 제안 — 담당 영역(데이터 수집 / 리서치(quant-researcher·validator) / 리스크·스키마 / 시장정보·운영)과
    할 일 한 줄씩. 판정 대기 lab은 lab id와 함께, 착수 후보는 표의 '대기열 우선순위 상' 항목에서 1~2건 제안.
    영역에 할 일이 없으면 "없음"이라고 쓴다
- rejected·retired lab을 다시 하자고 제안하지 않는다. 커밋 목록은 지난 기록일 뿐 할 일이 아니다.
- 전체 15줄 안팎(최대 18줄), 한 줄에 한 사실, 한국어.
- 주문·설정 변경을 권하지 않는다. 매수·매도 의견을 내지 않는다. 퇴출 여부는 사람이 정한다고 적는다."""


# ── 기간 ─────────────────────────────────────────────────────────────────────

def period(week: str | None = None, today: date | None = None) -> tuple[str, date, date]:
    """(이름 YYYY-Www, 시작일, 끝일) — 둘 다 포함, KST 날짜."""
    if week:
        m = re.fullmatch(r"(\d{4})-W(\d{2})", week)
        if not m:
            raise SystemExit(f"--week 형식은 YYYY-Www (예: 2026-W47): {week}")
        end = date.fromisocalendar(int(m[1]), int(m[2]), 6)  # 그 주 토요일
    else:
        end = today or pd.Timestamp.now(tz=TZ).date()
    y, w, _ = end.isocalendar()
    return f"{y}-W{w:02d}", end - timedelta(days=6), end


def _days(start: date, end: date) -> list[str]:
    return [f"{start + timedelta(days=i):%Y-%m-%d}" for i in range((end - start).days + 1)]


# ── ① 데이터 상태 ─────────────────────────────────────────────────────────────

def _collect_logs(days: list[str], open_keys: set[tuple[str, str]] | None = None) -> dict:
    """logs/collect_{날짜}.log 의 ERROR·WARNING 건수와 위치별 상위 예시. 로그 없는 날도 센다."""
    cnt: dict[str, Counter] = {"ERROR": Counter(), "WARNING": Counter()}
    first: dict[tuple, str] = {}
    per_day, missing, n_open = {}, [], 0
    for day in days:
        p = dr.LOG_DIR / f"collect_{day}.log"
        if not p.exists():
            missing.append(day)
            continue
        e = w = 0
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split(" | ", 2)
            if len(parts) < 3 or (lv := parts[1].strip()) not in cnt:
                continue
            loc, _, msg = parts[2].partition(" - ")
            cnt[lv][loc] += 1
            first.setdefault((lv, loc), dr._mask(msg)[:100])
            e, w = e + (lv == "ERROR"), w + (lv == "WARNING")
            n_open += lv == "ERROR" and dr.split_log_error(msg, open_keys or set())
        per_day[day] = (e, w)
    top = {lv: [(loc, n, first[(lv, loc)]) for loc, n in c.most_common(3)] for lv, c in cnt.items()}
    return {"errors": sum(cnt["ERROR"].values()), "open": n_open, "warnings": sum(cnt["WARNING"].values()),
            "per_day": per_day, "missing": missing, "top": top}


def _health_history(days: list[str]) -> dict:
    """12:00·20:00 상태 점검 출력(logs/task-cli-YYYYMMDD.log)의 🔴·🟡 줄. 데이터셋별로 묶는다."""
    red: Counter = Counter()
    sample: dict[str, str] = {}
    red_days: dict[str, int] = {}
    yellow, ran = 0, 0
    for day in days:
        p = dr.LOG_DIR / f"task-cli-{day.replace('-', '')}.log"
        if not p.exists():
            continue
        ran += 1
        n = 0
        for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            t = line.strip()
            if t.startswith("🔴"):
                ds, _, msg = t[1:].strip().partition(":")
                red[ds] += 1
                sample.setdefault(ds, dr._mask(msg.strip())[:100])
                n += 1
            elif t.startswith("🟡"):
                yellow += 1
        if n:
            red_days[day] = n
    return {"ran_days": ran, "red": sum(red.values()), "yellow": yellow, "red_days": red_days,
            "red_by": [(ds, n, sample[ds]) for ds, n in red.most_common(5)]}


def _size_week(start: str, end: str) -> dict:
    """reports/ops/sizes.json 기록으로 주간 증가. 끝 기록이 없으면 지금 용량을 쓴다."""
    hist = dr._load_sizes()
    ends = sorted(k for k in hist if k <= end)
    if ends and ends[-1] >= start:
        end_key, now = ends[-1], hist[ends[-1]]
    else:
        end_key, now = "현재", dr._sizes()
    before = sorted(k for k in hist if k < start)
    inside = sorted(k for k in hist if start <= k <= end and k != end_key)
    base_key = before[-1] if before else (inside[0] if inside else None)
    base = hist.get(base_key) if base_key else None
    return {"total": sum(now.values()), "end": end_key, "since": base_key,
            "delta": sum(now.values()) - sum(base.values()) if base else None,
            "top": sorted(((k, v - base.get(k, 0)) for k, v in now.items() if v != base.get(k, 0)),
                          key=lambda x: -x[1])[:3] if base else []}


# ── ② LLM 호출 ───────────────────────────────────────────────────────────────

def _llm_week(start: str, end: str) -> dict:
    path = dr.LOG_DIR / "llm_calls.jsonl"
    by: dict[str, list] = defaultdict(list)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if start <= f"{pd.Timestamp(r['at']).tz_convert(TZ):%Y-%m-%d}" <= end:
                by[r.get("tag") or "-"].append(r)
    rows = [{"tag": t, "calls": len(rs), "fails": sum(not r.get("ok") for r in rs),
             "avg_sec": sum(float(r.get("sec") or 0) for r in rs) / len(rs)}
            for t, rs in sorted(by.items(), key=lambda x: -len(x[1]))]
    n, f = sum(r["calls"] for r in rows), sum(r["fails"] for r in rows)
    return {"calls": n, "fails": f, "fail_rate": f / n if n else None, "by_tag": rows}


# ── ③ 리서치 ─────────────────────────────────────────────────────────────────

def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def _status_at(path: str, rev: str) -> str | None:
    try:
        return str((yaml.safe_load(_git("show", f"{rev}:{path}")) or {}).get("status") or "-")
    except Exception:  # noqa: BLE001 — 그 시점에 파일이 없었음
        return None


def _lab_changes(start: str, end: str) -> dict:
    """이번 주 labs/L*/lab.yaml 을 바꾼 커밋과 lab별 status 변화(주 시작 전 → 지금)."""
    since = f"{start} 00:00:00 +0900"
    until = f"{(date.fromisoformat(end) + timedelta(days=1)):%Y-%m-%d} 00:00:00 +0900"
    try:
        out = _git("log", f"--since={since}", f"--until={until}", "--format=@@%h\t%cd\t%s",
                   "--date=format-local:%m-%d", "--name-only", "--", "labs")
    except Exception as e:  # noqa: BLE001
        return {"error": f"git log 실패: {type(e).__name__}"[:120], "commits": [], "labs": []}
    commits, paths, cur = [], set(), None
    for line in out.splitlines():
        if line.startswith("@@"):
            h, d, s = (line[2:].split("\t", 2) + ["", ""])[:3]
            cur = {"hash": h, "date": d, "subject": s[:80], "labs": []}
        elif cur is not None and re.fullmatch(r"labs/L[^/]+/lab\.yaml", line.strip()):
            p = line.strip()
            paths.add(p)
            if not cur["labs"]:
                commits.append(cur)
            cur["labs"].append(p.split("/")[1])
    labs = []
    for p in sorted(paths):
        try:
            before = _git("log", "-1", "--format=%H", f"--until={since}", "--", p).strip()
        except Exception:  # noqa: BLE001
            before = ""
        was = _status_at(p, before) if before else None
        now_path = dr.LABS_DIR / p.split("/", 1)[1]
        now = None
        if now_path.exists():
            now = str((yaml.safe_load(now_path.read_text(encoding="utf-8")) or {}).get("status") or "-")
        labs.append({"dir": p.split("/")[1], "before": was or "(신규)", "now": now or "(삭제)"})
    return {"commits": commits, "labs": labs}


def _backlog_top(n: int = 3) -> list[str]:
    """labs/BACKLOG.md '## 후보' 표에서 우선순위(마지막 칸)가 '상'인 아이디어 앞 n건."""
    p = dr.LABS_DIR / "BACKLOG.md"
    if not p.exists():
        return []
    out, in_sec = [], False
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t.startswith("## "):
            in_sec = t[3:].strip() == "후보"
        elif in_sec and t.startswith("|"):
            cells = [c.strip() for c in t.strip("|").split("|")]
            if len(cells) >= 2 and cells[-1].startswith("상"):
                out.append(f"{cells[0][:60]} [{cells[1]}]")
    return out[:n]


def _research(start: str, end: str) -> dict:
    labs = dr._labs()
    return {"labs": labs, "changes": _lab_changes(start, end),
            "backlog": {s: dr._backlog(s) for s in ("후보", "진행", "기각 기록")}, "top": _backlog_top(),
            "waiting": [r for r in labs if r.get("waiting", "-") != "-"]}


# ── ④ 운용 (paper) ────────────────────────────────────────────────────────────

def _expected(lab_dir: Path) -> dict | None:
    """최신 백테스트 결과(results/*/summary.json)의 시험 구간 CAGR·MDD."""
    runs = sorted((lab_dir / "results").glob("2*/summary.json"))
    if not runs:
        return None
    t = json.loads(runs[-1].read_text(encoding="utf-8")).get("test") or {}
    return {"run": runs[-1].parent.name, "cagr": t.get("cagr"), "mdd": t.get("mdd")}


def _near(value: float, limit) -> str:
    if limit in (None, ""):
        return "한도 없음"
    r = value / float(limit) if float(limit) else 0.0
    return f"{'도달' if r >= 1 else '근접' if r >= NEAR else '여유'} ({value:.1f}/{float(limit):g})"


def _strategy(d: Path, start: str, end: str) -> dict:
    book = json.loads((d / "book.json").read_text(encoding="utf-8")) if (d / "book.json").exists() else {}
    ev = []
    if (d / "events.jsonl").exists():
        ev = [json.loads(x) for x in (d / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    eq = sorted((str(e["bar"])[:10], float(e["equity"])) for e in ev if e.get("kind") == "equity")
    wk = [x for x in eq if start <= x[0] <= end]
    prior = [x for x in eq if x[0] < start]
    base = prior[-1][1] if prior else (wk[0][1] if wk else None)
    last = wk[-1][1] if wk else None
    week_ev = [e for e in ev if start <= str(e.get("bar", ""))[:10] <= end]
    k = Counter(e.get("kind") for e in week_ev)
    fills = [e for e in week_ev if e.get("kind") == "fill"]

    lab_dir = dr.LABS_DIR / d.name
    meta = {}
    if (lab_dir / "lab.yaml").exists():
        meta = yaml.safe_load((lab_dir / "lab.yaml").read_text(encoding="utf-8")) or {}
    exp = _expected(lab_dir)
    rules = meta.get("retirement") or {}
    equity, peak = float(book.get("equity") or 0), float(book.get("peak") or 0)
    dd = (equity / peak - 1) * 100 if peak > 0 else 0.0
    exp_week = None
    if exp and exp.get("cagr") is not None:
        exp_week = (1 + float(exp["cagr"])) ** (7 / 365) - 1
    return {
        "dir": d.name, "id": meta.get("id") or d.name, "status": meta.get("status") or "-",
        "last_bar": book.get("last_bar"), "equity": equity, "bars": len(wk),
        "week_ret": (last / base - 1) if base and last else None,
        "total_ret": (eq[-1][1] / eq[0][1] - 1) if len(eq) > 1 and eq[0][1] else None,
        "dd_pct": dd, "losing_streak": int(book.get("losing_streak") or 0),
        "expected": exp, "exp_week": exp_week,
        "fills": len(fills), "cost": sum(float(e.get("fee") or 0) + float(e.get("tax") or 0) for e in fills),
        "risk": {x: k[x] for x in ("risk_block", "reduce", "cancel", "unmeasurable_exit", "corporate_action") if k[x]},
        "retirement_hits": list(book.get("retirement") or []),
        "near": {"max_drawdown_pct": _near(abs(min(dd, 0.0)), rules.get("max_drawdown_pct")),
                 "max_losing_days": _near(float(book.get("losing_streak") or 0), rules.get("max_losing_days"))},
    }


def _operations(start: str, end: str) -> dict:
    dirs = sorted(p for p in PAPER_DIR.iterdir() if p.is_dir() and
                  ((p / "book.json").exists() or (p / "events.jsonl").exists())) if PAPER_DIR.is_dir() else []
    strategies = []
    for d in dirs:
        try:
            strategies.append(_strategy(d, start, end))
        except Exception as e:  # noqa: BLE001
            strategies.append({"dir": d.name, "error": f"{type(e).__name__}: {e}"[:120]})
    recorded = {s["dir"] for s in strategies}
    # 운용 상태인데 장부가 없는 lab → 운용데스크가 안 돌고 있을 수 있음
    orphan = [r["id"] for r in dr._labs() if r.get("status") in ("paper", "live") and r["dir"] not in recorded]
    return {"strategies": strategies, "orphan": orphan}


def collect(week: str | None = None, today: date | None = None) -> dict:
    label, s, e = period(week, today)
    start, end = f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}"
    days = _days(s, e)
    errors = dr._error_state()  # 지금 시점의 상태DB 기준 (지난 주를 다시 만들어도 "현재" 미해결)
    open_keys = {(d, sym) for d, sym, _ in errors["open"]}
    return {"week": label, "start": start, "end": end, "logs": _collect_logs(days, open_keys), "errors": errors,
            "health": _health_history(days), "growth": _size_week(start, end),
            "llm": _llm_week(start, end), "research": _research(start, end), "ops": _operations(start, end)}


# ── 표·보고서 ────────────────────────────────────────────────────────────────

def _r(x: float | None) -> str:
    return "-" if x is None else f"{x:+.2%}"


def table(f: dict) -> str:
    L = [f"# 주간 사실 표 {f['week']} ({f['start']} ~ {f['end']}, KST)", "", "## ① 데이터 상태"]
    errs = f.get("errors") or {"open": [], "recovered": [], "failed": None}
    if errs.get("failed"):
        L.append(f"- 수집 상태DB 읽기 실패: {errs['failed']}")
    L.append(f"- 수집 오류 현재 미해결 {len(errs['open'])}건 / 기간 중 발생 후 복구됨 {len(errs['recovered'])}건 (상태DB 지금 기준)")
    L += [f"  - 미해결 {d} {sym}: {m}" for d, sym, m in errs["open"][:5]]
    lg = f["logs"]
    L.append(f"- 수집 로그(기간 중 발생 기록, 현재 문제 아님): ERROR {lg['errors']}건"
             f"(그중 지금 미해결 종목 {lg.get('open', 0)}건), WARNING {lg['warnings']}건"
             + (f" · 로그 없는 날 {len(lg['missing'])}일 ({', '.join(d[5:] for d in lg['missing'])})" if lg["missing"] else ""))
    bad = [f"{d[5:]} E{e}/W{w}" for d, (e, w) in lg["per_day"].items() if e or w]
    if bad:
        L.append("- 날짜별: " + ", ".join(bad))
    for lv in ("ERROR", "WARNING"):
        L += [f"  - {lv} {loc} ×{n}: {msg}" for loc, n, msg in lg["top"][lv]]
    h = f["health"]
    L.append(f"- 상태 점검 이력 (logs/task-cli-*.log, {h['ran_days']}일 기록, 당시 알림 — 이미 복구된 것 포함): 🔴 {h['red']}건, 🟡 {h['yellow']}건"
             + (" — 날짜별 " + ", ".join(f"{d[5:]} {n}" for d, n in h["red_days"].items()) if h["red_days"] else ""))
    L += [f"  - 🔴 {ds} ×{n}: {msg}" for ds, n, msg in h["red_by"]]
    g = f["growth"]
    L.append(f"- 데이터 용량 (data/processed): {dr._mb(g['total'])} ({g['end']})"
             + (f", {g['since']} 대비 {dr._mb(g['delta'])}" if g["since"] else " (비교 기준 없음)"))
    if g["top"]:
        L.append("  - 증가 상위: " + ", ".join(f"{k} {dr._mb(v)}" for k, v in g["top"]))

    m = f["llm"]
    L += ["", "## ② LLM 호출 (logs/llm_calls.jsonl)",
          f"- 호출 {m['calls']}건, 실패 {m['fails']}건 (실패율 {dr._pct(m['fail_rate'])})"]
    if m["by_tag"]:
        L.append("태그 | 건수 | 실패 | 평균 시간(초)")
        L += [f"{r['tag']} | {r['calls']} | {r['fails']} | {r['avg_sec']:.1f}" for r in m["by_tag"]]

    rs = f["research"]
    L += ["", "## ③ 리서치 (labs/)"]
    if rs["labs"]:
        L.append("lab | 이름 | 버전 | 상태 | 판정 대기 | 마지막 결과")
        for r in rs["labs"]:
            if "error" in r:
                L.append(f"{r['dir']} | lab.yaml 읽기 실패: {r['error']}")
                continue
            ver = "-" if r["version"] is None else f"v{r['version']}"
            L.append(f"{r['id']} | {r['name']} | {ver} | {r['status']} | {r['waiting']} | {r['last_run'] or '결과 없음'}")
    else:
        L.append("- lab 없음")
    ch = rs["changes"]
    if ch.get("error"):
        L.append(f"- 이번 주 lab.yaml 변경: 확인 실패 ({ch['error']})")
    elif ch["labs"]:
        L.append("- 이번 주 status 변화: " + ", ".join(
            f"{x['dir']} {x['before']}→{x['now']}" if x["before"] != x["now"] else f"{x['dir']} {x['now']}(그대로)"
            for x in ch["labs"]))
        L += [f"  - {c['date']} {c['hash']} {c['subject']}" for c in ch["commits"][:6]]
    else:
        L.append("- 이번 주 lab.yaml 변경 없음")
    b = rs["backlog"]
    if any(v is not None for v in b.values()):
        L.append("- 대기열 (labs/BACKLOG.md): " + ", ".join(f"{k} {v if v is not None else '-'}건" for k, v in b.items()))
    if rs.get("top"):
        L.append("- 대기열 우선순위 상: " + " / ".join(rs["top"]))
    L.append("- 판정 대기: " + (", ".join(f"{r['id']}({r['waiting']})" for r in rs["waiting"]) or "없음"))

    op = f["ops"]
    L += ["", "## ④ 운용 (results/paper)"]
    if not op["strategies"]:
        L.append("- 운용 중 전략 없음 (paper 장부 기록 없음)")
    else:
        L.append("전략 | 상태 | 마지막 봉 | 주간 수익 | 기대(주간 환산·추정) | 누적 | 낙폭 | 체결 | 비용 | 리스크 발동 | 퇴출 조건")
        for s in op["strategies"]:
            if "error" in s:
                L.append(f"{s['dir']} | 기록 읽기 실패: {s['error']}")
                continue
            risk = ", ".join(f"{k} {v}" for k, v in s["risk"].items()) or "없음"
            if s["retirement_hits"]:
                ret = "걸림: " + "; ".join(s["retirement_hits"])
            else:
                ret = f"낙폭 {s['near']['max_drawdown_pct']}, 연속손실 {s['near']['max_losing_days']}"
            L.append(f"{s['id']} | {s['status']} | {s['last_bar'] or '-'} | {_r(s['week_ret'])} ({s['bars']}봉) | "
                     f"{_r(s['exp_week'])} | {_r(s['total_ret'])} | {s['dd_pct']:.1f}% | {s['fills']} | "
                     f"{s['cost']:,.0f} | {risk} | {ret}")
            e = s["expected"]
            L.append("  - 기대치 출처: " + (f"results/{e['run']} 시험 구간 CAGR {_r(e['cagr'])}, MDD {_r(e['mdd'])}"
                                          if e else "백테스트 결과 없음"))
    if op["orphan"]:
        L.append(f"- ⚠️ status가 paper/live인데 장부 없음: {', '.join(op['orphan'])}")
    return "\n".join(L)


def summarize(tbl: str) -> str | None:
    out = claude_cli.ask(tbl, SYSTEM, model=claude_cli.FAST, tag="ops-reporter-weekly", timeout=240)
    return out.strip() if isinstance(out, str) and out.strip() else None


def build(f: dict, use_llm: bool = True) -> str:
    tbl = table(f)
    note = summarize(tbl) if use_llm else None
    head = f"🗓️ 주간 회의 자료 {f['week']} ({f['start']} ~ {f['end']})"
    if note:
        return f"{head}\n\n■ 요약 (ops-reporter · LLM)\n{note}\n\n---\n{tbl}"
    tail = "" if not use_llm else "\n\n(LLM 요약 생성 실패 — logs/llm_calls.jsonl 확인)"
    return f"{head}\n\n{tbl}{tail}"


def main() -> None:
    p = argparse.ArgumentParser(prog="autotrader.monitoring.weekly_report")
    p.add_argument("--notify", action="store_true", help="텔레그램으로 보내기")
    p.add_argument("--no-llm", action="store_true", help="LLM 요약 없이 사실 표만")
    p.add_argument("--week", help="ISO 주 YYYY-Www (그 주 토요일까지 7일). 기본은 오늘까지 7일")
    a = p.parse_args()
    f = collect(a.week)
    text = build(f, use_llm=not a.no_llm)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{f['week']}.md").write_text(text, encoding="utf-8")
    print(text)
    if a.notify:
        from . import notify
        notify.send(text)


if __name__ == "__main__":
    main()
