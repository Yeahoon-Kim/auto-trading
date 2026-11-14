"""네트워크 없이 도는 핵심 로직 테스트. 실행: uv run pytest -q"""

import numpy as np
import pandas as pd
import pytest

from autotrader.data import store
from autotrader.intel import market
from autotrader.monitoring import health, notify


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    return tmp_path


def _bars(start, n, freq="1min", price=100.0):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"ts": ts, "open": price, "high": price, "low": price, "close": price, "volume": 1.0})


def test_write_merges_and_dedupes(tmp_store):
    store.write("x_1m", "AAA", _bars("2026-01-31 23:58", 4))           # 월 경계를 걸침
    store.write("x_1m", "AAA", _bars("2026-02-01 00:00", 3, price=101))  # 겹치는 구간은 새 값 우선
    df = store.read("x_1m", "AAA")
    assert len(df) == 5
    assert df["ts"].is_monotonic_increasing and df["ts"].dt.tz is not None
    assert df.loc[df["ts"] == pd.Timestamp("2026-02-01 00:00", tz="UTC"), "close"].item() == 101
    assert {p.stem for p in (store.DATA_DIR / "x_1m" / "AAA").glob("*.parquet")} == {"2026-01", "2026-02"}


def test_state_keeps_min_first_and_max_last(tmp_store):
    store.write("x_1m", "AAA", _bars("2026-03-01", 2))
    store.write("x_1m", "AAA", _bars("2026-02-01", 2))
    st = store.get_state("x_1m", "AAA")
    assert st["first_ts"] == pd.Timestamp("2026-02-01", tz="UTC")
    assert st["last_ts"] == pd.Timestamp("2026-03-01 00:01", tz="UTC")


def test_read_accepts_naive_and_aware_bounds(tmp_store):
    store.write("x_1m", "AAA", _bars("2026-01-01", 10))
    assert len(store.read("x_1m", "AAA", start="2026-01-01 00:05")) == 5
    assert len(store.read("x_1m", "AAA", start=pd.Timestamp("2026-01-01 00:05", tz="UTC"))) == 5


def test_expected_session_weekend_and_before_close():
    rule = {"tz": "Asia/Seoul", "close_hour": 16}
    sat = pd.Timestamp("2026-11-14 10:00", tz="Asia/Seoul").tz_convert("UTC")
    assert health._expected_session(rule, sat).strftime("%Y-%m-%d") == "2026-11-13"
    mon_morning = pd.Timestamp("2026-11-16 10:00", tz="Asia/Seoul").tz_convert("UTC")
    assert health._expected_session(rule, mon_morning).strftime("%Y-%m-%d") == "2026-11-13"
    mon_evening = pd.Timestamp("2026-11-16 17:00", tz="Asia/Seoul").tz_convert("UTC")
    assert health._expected_session(rule, mon_evening).strftime("%Y-%m-%d") == "2026-11-16"


def test_health_flags_stale_and_errors(tmp_store, monkeypatch):
    now = pd.Timestamp("2026-11-14 12:00", tz="UTC")
    store.write("binance_spot_1m", "FRESH", _bars(now - pd.Timedelta(hours=1), 2))
    store.write("binance_spot_1m", "STALE", _bars(now - pd.Timedelta(hours=40), 2))
    for s in ("FRESH", "STALE"):
        store.set_state("binance_spot_1m", s, backfill_done=True)
    store.set_state("binance_spot_1m", "BROKEN", status="error", error="boom")
    monkeypatch.setattr(health, "MIN_FREE_GB", 0)
    msgs = {(i.level, i.dataset): i.message for i in health.check(now)}
    assert "STALE" in msgs[("warn", "binance_spot_1m")]
    assert "BROKEN" in msgs[("error", "binance_spot_1m")]


def test_regime_labels():
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, tz="UTC")
    df = pd.DataFrame(index=idx)
    for p in ("spx", "kospi", "btc"):
        df[f"{p}_above_200d"] = True
        df[f"{p}_ret_20d"] = 0.05
        df[f"{p}_rv_pctile"] = 0.5
    df["vix_term"], df["vix"], df["hy_spread_chg_20d"] = 0.9, 15.0, 0.0
    out = market._labels(df.copy())
    assert (out["spx_trend"] == "상승").all() and (out["risk_regime"] == "위험선호").all()
    df["vix_term"], df["vix"], df["spx_above_200d"] = 1.2, 40.0, False
    out = market._labels(df.copy())
    assert (out["risk_regime"] == "위험회피").all() and (out["spx_trend"] == "혼조").all()


def test_notify_without_token_logs_only(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(notify, "ALERT_LOG", tmp_path / "alerts.log")
    assert notify.send("테스트 알림") is False
    assert "테스트 알림" in (tmp_path / "alerts.log").read_text(encoding="utf-8")
