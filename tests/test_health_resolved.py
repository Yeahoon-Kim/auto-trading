"""복구된 수집 오류가 알림·보고에 계속 나오지 않는지 확인한다 (미해결 / 복구됨 분리)."""

import pandas as pd
import pytest

from autotrader.data import store
from autotrader.monitoring import daily_report as dr
from autotrader.monitoring import health
from autotrader.monitoring import weekly_report as wr

NOW = pd.Timestamp("2026-11-16 03:00", tz="UTC")  # KST 12:00 월요일 (장 마감 전 → 기대 거래일 = 금 11-13)
FRESH = pd.Timestamp("2026-11-13 06:30", tz="UTC")
OLD = pd.Timestamp("2026-11-05 06:30", tz="UTC")


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    for name in ("LOG_DIR", "SCAN_DIR", "REPORT_DIR"):
        monkeypatch.setattr(dr, name, tmp_path / name.lower())
    (tmp_path / "log_dir").mkdir()
    (tmp_path / "processed").mkdir()

    def put(ds, sym, last_ts, status="ok", error=None, updated="2026-11-15T13:00:00+00:00"):
        store.set_state(ds, sym, last_ts=last_ts, backfill_done=True, status=status, error=error)
        import sqlite3
        with sqlite3.connect(store.STATE_DB) as con:
            con.execute("UPDATE state SET updated_at=? WHERE dataset=? AND symbol=?", (updated, ds, sym))
    return put


def test_recovered_error_hidden_open_error_kept(state):
    state("kr_1m", "017670", FRESH, "error", "KIS 오류 OPSQ1002: SESSION FULL")   # 오래된 오류 + 데이터 최신 → 복구됨
    state("kr_1m", "000001", OLD, "error", "KIS 반복 실패")                        # 데이터 낡음 → 미해결
    state("kr_1m", "000002", FRESH, "error", "방금 실패", updated="2026-11-16T01:00:00+00:00")  # 2시간 전 → 미해결
    state("kr_1m", "000003", FRESH)
    o, r = health.split_errors(health._state(), NOW)
    assert sorted(o["symbol"]) == ["000001", "000002"] and list(r["symbol"]) == ["017670"]
    msgs = [i.message for i in health.check(NOW) if i.level == "error" and i.dataset == "kr_1m"]
    assert len(msgs) == 1 and msgs[0].startswith("수집 오류 2종목")
    assert "017670" not in msgs[0] and "SESSION FULL" not in msgs[0]


def test_only_recovered_errors_means_healthy(state):
    state("kr_1m", "017670", FRESH, "error", "KIS 오류 OPSQ1002: SESSION FULL")
    assert not [i for i in health.check(NOW) if i.dataset == "kr_1m"]
    assert health.report([i for i in health.check(NOW) if i.dataset == "kr_1m"]) == "✅ 데이터 수집 정상"


def test_dataset_without_rule_stays_open(state):
    state("mystery", "A", FRESH, "error", "x")
    o, r = health.split_errors(health._state(), NOW)
    assert len(o) == 1 and r.empty


def test_daily_report_splits_open_and_recovered(state, tmp_path):
    state("kr_1m", "017670", FRESH, "error", "KIS 오류 OPSQ1002: SESSION FULL")
    state("kr_1m", "000001", OLD, "error", "KIS 반복 실패")
    (tmp_path / "log_dir" / "collect_2026-11-16.log").write_text(
        "2026-11-16 01:00:00.000 +09:00 | ERROR    | m:f:1 - kr_1m 017670 실패: KIS 오류 OPSQ1002: SESSION FULL\n"
        "2026-11-16 01:00:01.000 +09:00 | ERROR    | m:f:1 - kr_1m 000001 실패: KIS 반복 실패\n"
        "2026-11-16 01:00:02.000 +09:00 | ERROR    | m:g:1 - ecos base_rate 실패: 옛날\n", encoding="utf-8")
    es = dr._error_state()
    assert [k[1] for k in es["open"]] == ["000001"] and [k[1] for k in es["recovered"]] == ["017670"]
    f = dr.collect("2026-11-16")
    assert (f["log"]["errors"], f["log"]["open"]) == (3, 1)
    tbl = dr.table(f)
    assert "미해결 1건" in tbl and "복구됨 1건" in tbl and "kr_1m 000001: KIS 반복 실패" in tbl
    assert "kr_1m 017670: " not in tbl  # 복구된 종목의 오류 메시지는 반복해서 찍지 않는다
    assert "미해결 종목 1건" in tbl


def test_weekly_table_splits(state, tmp_path, monkeypatch):
    state("kr_1m", "017670", FRESH, "error", "KIS 오류 OPSQ1002: SESSION FULL")
    monkeypatch.setattr(wr, "ROOT", tmp_path)
    es = dr._error_state()
    f = {"errors": es}
    assert len(es["open"]) == 0 and len(es["recovered"]) == 1
    assert wr._collect_logs([], {("kr_1m", "x")})["open"] == 0
