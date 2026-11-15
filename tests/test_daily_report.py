"""운영 마감 보고: 네트워크·LLM 없이 사실 수집과 실패 시 대체 동작을 확인한다."""

import json

import pandas as pd

from autotrader.data import store
from autotrader.llm import claude_cli, decision_log
from autotrader.monitoring import daily_report as dr
from autotrader.monitoring import health


def test_daily_report_facts_and_llm_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(decision_log, "LOG", tmp_path / "decisions.jsonl")
    for name in ("LOG_DIR", "SCAN_DIR", "REPORT_DIR"):
        monkeypatch.setattr(dr, name, tmp_path / name.lower())
    monkeypatch.setattr(health, "check", list)
    day = "2026-11-15"

    ts = pd.date_range("2026-11-14", periods=3, freq="1min", tz="UTC")
    store.write("x_1m", "AAA", pd.DataFrame({"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                                             "close": 1.0, "volume": 1.0}))
    store.set_state("x_1m", "AAA", first_ts=ts[0], last_ts=ts[-1], backfill_done=True)

    (tmp_path / "log_dir").mkdir()
    key = "A" * 32
    (tmp_path / "log_dir" / f"collect_{day}.log").write_text(
        f"2026-11-15 01:00:00.000 +09:00 | ERROR    | m:f:1 - 실패 https://x?key={key}\n"
        "2026-11-15 01:00:01.000 +09:00 | INFO     | m:f:2 - 완료\n", encoding="utf-8")
    calls = [{"at": "2026-11-15T01:00:00+00:00", "tag": "a", "ok": True},
             {"at": "2026-11-15T02:00:00+00:00", "tag": "a", "ok": False},
             {"at": "2026-11-15T16:00:00+00:00", "tag": "b", "ok": True}]  # KST 11-16 → 제외
    (tmp_path / "log_dir" / "llm_calls.jsonl").write_text("\n".join(map(json.dumps, calls)), encoding="utf-8")
    (tmp_path / "scan_dir").mkdir()
    (tmp_path / "scan_dir" / f"{day}.json").write_text(json.dumps({"kr": [{}, {}], "us": [{}]}), encoding="utf-8")

    f = dr.collect(day)
    assert f["log"]["errors"] == 1 and key not in f["log"]["where"][0][2]
    assert (f["llm"]["calls"], f["llm"]["fails"]) == (2, 1)
    assert f["scan"] == {"kr": 2, "us": 1}
    assert f["decisions"]["total"] == 0
    assert f["growth"]["since"] is None and f["growth"]["total"] > 0

    monkeypatch.setattr(claude_cli, "ask", lambda *a, **k: None)  # LLM 실패 → 사실 표만
    text = dr.build(f)
    assert "운영 사실 표" in text and "■ 요약" not in text and "실패율 50.0%" in text
    monkeypatch.setattr(claude_cli, "ask", lambda *a, **k: "문제 없음")
    assert "■ 요약" in dr.build(f)


def test_daily_report_research_status(tmp_path, monkeypatch):
    labs = tmp_path / "labs"
    monkeypatch.setattr(dr, "LABS_DIR", labs)

    def lab(name, status, validation=False, runs=()):
        d = labs / name
        d.mkdir(parents=True)
        (d / "lab.yaml").write_text(f"id: {name[:4]}\nname: {name[5:]}\nversion: 2\nstatus: {status}\n",
                                    encoding="utf-8")
        if validation:
            (d / "validation.md").write_text("의견서", encoding="utf-8")
        for r in runs:
            (d / "results" / r).mkdir(parents=True)

    lab("L001-mom", "backtesting", runs=["20261114-090000", "20261115-191551"])
    lab("L002-val", "validating", validation=True)
    lab("L003-old", "rejected", validation=True, runs=["20261101-000000"])
    (labs / "_template").mkdir()
    (labs / "_template" / "lab.yaml").write_text("id: LXXX\nstatus: draft\n", encoding="utf-8")

    rows = {r["id"]: r for r in dr._labs()}
    assert set(rows) == {"L001", "L002", "L003"}  # _template 제외
    assert rows["L001"]["waiting"] == "validator 의견서 없음"
    assert rows["L001"]["last_run"] == "20261115-191551"
    assert rows["L002"]["waiting"] == "사용자 판정 대기" and rows["L002"]["last_run"] is None
    assert rows["L003"]["waiting"] == "-"

    assert dr._backlog() is None  # BACKLOG.md 없음 → 생략
    (labs / "BACKLOG.md").write_text(
        "# 대기열\n\n## 형식\n\n| 칸 | 내용 |\n|---|---|\n| a | b |\n\n"
        "## 후보\n\n| 아이디어 | 시장 |\n|---|---|\n| x | kr |\n| y | us |\n\n"
        "## 진행\n\n| lab | 아이디어 |\n|---|---|\n| L9 | z |\n", encoding="utf-8")
    assert dr._backlog() == 2

    f = {"day": "2026-11-15", "status": [], "health": [], "log": {"exists": False},
         "llm": {"calls": 0, "fails": 0, "fail_rate": None, "by_tag": {}, "fail_tags": {}},
         "scan": None, "decisions": {"total": 0, "settled": 0, "pending": 0, "alpha_mean": None},
         "growth": {"total": 0, "since": None, "delta": None, "top": []},
         "labs": dr._labs(), "backlog": dr._backlog()}
    tbl = dr.table(f)
    assert "## 리서치 진행 상태" in tbl and "L002 | val | v2 | validating | 사용자 판정 대기 | 결과 없음" in tbl
    assert "대기열 후보 (labs/BACKLOG.md): 2건" in tbl

    seen = {}
    monkeypatch.setattr(claude_cli, "ask", lambda prompt, *a, **k: seen.setdefault("p", prompt) and None)
    text = dr.build(f)  # LLM 실패 → 사실 표만, 그래도 리서치 절 포함
    assert "validator 의견서 없음" in seen["p"] and "■ 요약" not in text and "## 리서치 진행 상태" in text
