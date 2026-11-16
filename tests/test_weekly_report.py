"""주간 회의 자료: 네트워크·LLM 없이 임시 폴더에서 사실 수집과 대체 동작을 확인한다.
운용 기록(results/paper)이 없는 경우와 있는 경우 두 가지."""

import json
import os
import subprocess
from datetime import date

import pytest

from autotrader.data import store
from autotrader.llm import claude_cli
from autotrader.monitoring import daily_report as dr
from autotrader.monitoring import weekly_report as wr

TODAY = date(2026, 11, 21)  # 토요일 → 2026-W47, 11-15(일) ~ 11-21(토)


def _git(root, *args, when=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    if when:
        env |= {"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)


def _lab(labs, name, status, retirement=""):
    d = labs / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "lab.yaml").write_text(f"id: {name[:4]}\nname: {name[5:]}\nversion: 1\nstatus: {status}\n{retirement}",
                                encoding="utf-8")
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    for name in ("LOG_DIR", "REPORT_DIR"):
        monkeypatch.setattr(dr, name, tmp_path / name.lower())
    monkeypatch.setattr(dr, "LABS_DIR", tmp_path / "labs")
    monkeypatch.setattr(wr, "ROOT", tmp_path)
    monkeypatch.setattr(wr, "PAPER_DIR", tmp_path / "results" / "paper")
    (tmp_path / "log_dir").mkdir()
    (tmp_path / "labs").mkdir()
    _git(tmp_path, "init", "-q")
    return tmp_path


def test_period():
    assert wr.period(today=TODAY) == ("2026-W47", date(2026, 11, 15), date(2026, 11, 21))
    assert wr.period("2026-W47") == ("2026-W47", date(2026, 11, 15), date(2026, 11, 21))
    with pytest.raises(SystemExit):
        wr.period("2026-47")


def test_weekly_without_operations(env, monkeypatch):
    logs, labs = env / "log_dir", env / "labs"
    key = "B" * 32
    (logs / "collect_2026-11-15.log").write_text(
        f"2026-11-15 07:30:00.000 +09:00 | ERROR    | m:f:1 - 실패 https://x?key={key}\n"
        "2026-11-15 07:30:01.000 +09:00 | WARNING  | c:g:2 - 재시도 1/5\n"
        "2026-11-15 07:30:02.000 +09:00 | INFO     | m:f:3 - 완료\n", encoding="utf-8")
    (logs / "collect_2026-11-14.log").write_text(  # 기간 밖
        "2026-11-14 07:30:00.000 +09:00 | ERROR    | m:f:1 - 옛날\n", encoding="utf-8")
    (logs / "collect_2026-11-16.log").write_text(
        "2026-11-16 07:30:01.000 +09:00 | WARNING  | c:g:2 - 재시도 2/5\n", encoding="utf-8")
    (logs / "task-cli-20261115.log").write_text(
        "﻿⚠️ 데이터 수집 점검\n🔴 kr_1m: 수집 오류 5종목 — 예: 017670: SESSION FULL\n🟡 news: 오래된 데이터\n"
        "백필 진행: kr_1m 345/350\n✅ 데이터 수집 정상\n", encoding="utf-8")
    calls = [{"at": "2026-11-15T01:00:00+00:00", "tag": "ops-reporter", "ok": True, "sec": 30.0},
             {"at": "2026-11-16T01:00:00+00:00", "tag": "ops-reporter", "ok": False, "sec": 10.0},
             {"at": "2026-11-17T01:00:00+00:00", "tag": "market-analyst", "ok": True, "sec": 5.0},
             {"at": "2026-11-14T10:00:00+00:00", "tag": "x", "ok": True, "sec": 1.0}]  # KST 11-14 → 제외
    (logs / "llm_calls.jsonl").write_text("\n".join(map(json.dumps, calls)), encoding="utf-8")
    (env / "report_dir").mkdir()
    (env / "report_dir" / "sizes.json").write_text(json.dumps(
        {"2026-11-14": {"a": 1_000_000}, "2026-11-20": {"a": 3_000_000, "b": 500_000}}), encoding="utf-8")

    # lab: L001은 주 시작 전부터 backtesting → 이번 주 rejected, L002는 이번 주 신규
    _lab(labs, "L001-mom", "backtesting")
    _git(env, "add", "-A")
    _git(env, "commit", "-qm", "L001 착수", when="2026-11-10T12:00:00+09:00")
    _lab(labs, "L001-mom", "rejected")
    (labs / "L001-mom" / "validation.md").write_text("의견서", encoding="utf-8")
    _git(env, "add", "-A")
    _git(env, "commit", "-qm", "L001 판정", when="2026-11-16T12:00:00+09:00")
    _lab(labs, "L002-val", "validating")
    _git(env, "add", "-A")
    _git(env, "commit", "-qm", "L002 착수", when="2026-11-17T12:00:00+09:00")
    (labs / "BACKLOG.md").write_text(
        "# 대기열\n\n## 후보\n\n| 아이디어 | 시장 | 우선순위 |\n|---|---|---|\n| x | kr | 상 (제안) |\n| y | us | 중 |\n\n"
        "## 진행\n\n| lab | 아이디어 |\n|---|---|\n| L002 | z |\n\n"
        "## 기각 기록\n\n| lab | 판정 |\n|---|---|\n| L001 | rejected |\n", encoding="utf-8")

    f = wr.collect(today=TODAY)
    lg = f["logs"]
    assert (lg["errors"], lg["warnings"]) == (1, 2)
    assert "2026-11-17" in lg["missing"] and key not in lg["top"]["ERROR"][0][2]
    assert (f["health"]["red"], f["health"]["yellow"]) == (1, 1) and f["health"]["red_by"][0][0] == "kr_1m"
    assert (f["growth"]["since"], f["growth"]["end"], f["growth"]["delta"]) == ("2026-11-14", "2026-11-20", 2_500_000)
    tags = {r["tag"]: r for r in f["llm"]["by_tag"]}
    assert f["llm"]["calls"] == 3 and tags["ops-reporter"]["fails"] == 1 and tags["ops-reporter"]["avg_sec"] == 20.0
    rs = f["research"]
    assert {x["dir"]: (x["before"], x["now"]) for x in rs["changes"]["labs"]} == {
        "L001-mom": ("backtesting", "rejected"), "L002-val": ("(신규)", "validating")}
    assert [c["subject"] for c in rs["changes"]["commits"]] == ["L002 착수", "L001 판정"]
    assert rs["backlog"] == {"후보": 2, "진행": 1, "기각 기록": 1} and rs["top"] == ["x [kr]"]
    assert [r["id"] for r in rs["waiting"]] == ["L002"]
    assert f["ops"] == {"strategies": [], "orphan": []}

    tbl = wr.table(f)
    assert "운용 중 전략 없음" in tbl and "L001-mom backtesting→rejected" in tbl
    assert "ops-reporter | 2 | 1 | 20.0" in tbl and "🔴 1건" in tbl

    monkeypatch.setattr(claude_cli, "ask", lambda *a, **k: None)  # LLM 실패 → 사실 표만
    text = wr.build(f)
    assert "주간 사실 표 2026-W47" in text and "■ 요약" not in text and "LLM 요약 생성 실패" in text
    monkeypatch.setattr(claude_cli, "ask", lambda *a, **k: "■ 지난주 요약\n문제 없음")
    assert "■ 요약 (ops-reporter · LLM)" in wr.build(f)


def test_weekly_with_paper_records(env, monkeypatch):
    labs = env / "labs"
    d = _lab(labs, "L005-live", "paper", "retirement:\n  max_drawdown_pct: 10\n  max_losing_days: 5\n")
    run = d / "results" / "20261101-000000"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"test": {"cagr": 0.2, "mdd": -0.15}}), encoding="utf-8")
    _lab(labs, "L006-orphan", "paper")  # 장부 없음

    out = env / "results" / "paper" / "L005-live"
    out.mkdir(parents=True)
    (out / "book.json").write_text(json.dumps({
        "cash": 100.0, "shares": {"A": 1.0}, "last_bar": "2026-11-20", "equity": 920.0, "peak": 1000.0,
        "losing_streak": 2, "retirement": []}), encoding="utf-8")
    ev = [{"kind": "equity", "bar": "2026-11-13", "equity": 1000.0},   # 주 시작 전 → 기준
          {"kind": "fill", "bar": "2026-11-16", "fee": 1.5, "tax": 0.5},
          {"kind": "risk_block", "bar": "2026-11-16", "rule": "weights_invalid"},
          {"kind": "cancel", "bar": "2026-11-17"},
          {"kind": "equity", "bar": "2026-11-16", "equity": 980.0},
          {"kind": "equity", "bar": "2026-11-20", "equity": 920.0}]
    (out / "events.jsonl").write_text("\n".join(map(json.dumps, ev)), encoding="utf-8")

    f = wr.collect(today=TODAY)
    s, = f["ops"]["strategies"]
    assert s["id"] == "L005" and s["bars"] == 2
    assert s["week_ret"] == pytest.approx(-0.08) and s["dd_pct"] == pytest.approx(-8.0)
    assert s["exp_week"] == pytest.approx(1.2 ** (7 / 365) - 1)
    assert (s["fills"], s["cost"]) == (1, 2.0) and s["risk"] == {"risk_block": 1, "cancel": 1}
    assert s["near"]["max_drawdown_pct"].startswith("근접") and s["near"]["max_losing_days"].startswith("여유")
    assert f["ops"]["orphan"] == ["L006"]

    tbl = wr.table(f)
    assert "운용 중 전략 없음" not in tbl and "L005 | paper | 2026-11-20 | -8.00% (2봉)" in tbl
    assert "risk_block 1, cancel 1" in tbl and "장부 없음: L006" in tbl and "CAGR +20.00%" in tbl

    seen = {}
    monkeypatch.setattr(claude_cli, "ask", lambda prompt, *a, **k: seen.setdefault("p", prompt) and None)
    wr.build(f)
    assert "근접" in seen["p"]  # 퇴출 근접 사실이 LLM 입력에 들어간다
