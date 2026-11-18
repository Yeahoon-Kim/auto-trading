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


def test_cross_section_key_keeps_all_symbols(tmp_store):
    a, b = _bars("2026-01-02", 2, freq="1D"), _bars("2026-01-02", 2, freq="1D", price=5)
    a["symbol"], b["symbol"] = "AAA", "BBB"
    store.write("x_1d", "ALL", pd.concat([a, b]), key=["ts", "symbol"])
    store.write("x_1d", "ALL", a.assign(close=7.0), key=["ts", "symbol"])  # AAA만 갱신
    df = store.read("x_1d", "ALL")
    assert len(df) == 4 and set(df.loc[df.symbol == "AAA", "close"]) == {7.0}


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


def test_regime_warmup_labels_blank():
    """200일선·변동성 백분위 워밍업 구간은 값·라벨 모두 비어야 한다 (가짜 '하락'·'보통' 금지)."""
    idx = pd.date_range("2010-01-01", periods=400, freq="B")
    px = pd.Series(np.linspace(200, 100, 400), index=idx)  # 계속 하락
    blk = market._trend_block(px, "spx")
    assert blk["spx_above_200d"].iloc[:199].isna().all()
    assert blk["spx_above_200d"].iloc[199:].notna().all()
    df = blk.copy()
    for p in ("kospi", "btc"):
        df[f"{p}_above_200d"], df[f"{p}_ret_20d"], df[f"{p}_rv_pctile"] = np.nan, np.nan, np.nan
    df["vix_term"], df["vix"], df["hy_spread_chg_20d"] = np.nan, np.nan, np.nan
    out = market._labels(df)
    assert out["spx_trend"].iloc[:199].isna().all()
    assert (out["spx_trend"].iloc[199:] == "하락").all()
    assert out["spx_vol"][out["spx_rv_pctile"].isna()].isna().all()
    assert out["btc_trend"].isna().all() and out["btc_vol"].isna().all()
    # 위험회피 신호: 셀 수 있는 것이 없으면 NaN, 200일선만 있으면 available=1
    assert out["risk_off_signals"].iloc[:199].isna().all() and out["risk_regime"].iloc[:199].isna().all()
    assert (out["risk_off_available"].iloc[199:] == 1).all()
    assert (out["risk_off_signals"].iloc[199:] == 1).all() and (out["risk_regime"].iloc[199:] == "중립").all()


def test_regime_risk_off_partial_sources():
    """HY 스프레드가 없는 날은 그 신호를 '없음'으로 세지 않고 available에서 뺀다."""
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(index=idx)
    for p in ("spx", "kospi", "btc"):
        df[f"{p}_above_200d"], df[f"{p}_ret_20d"], df[f"{p}_rv_pctile"] = False, -0.1, 0.9
    df["vix_term"], df["vix"] = 1.2, 30.0
    df["hy_spread_chg_20d"] = [np.nan, 1.0, 0.0]
    out = market._labels(df)
    assert out["risk_off_available"].tolist() == [3, 4, 4]
    assert out["risk_off_signals"].tolist() == [3, 4, 3]
    assert (out["risk_regime"] == "위험회피").all()
    assert "3/4, 셀 수 있는 신호 3개" not in market.brief(out.assign(
        spx_observed=True, kospi_observed=True, btc_observed=True, spx_close=1.0, kospi_close=1.0,
        btc_close=1.0, spx_rv_20d=0.1, kospi_rv_20d=0.1, btc_rv_20d=0.1, hy_spread=3.0, t10y2y=0.1,
        usdkrw=1300.0, usdkrw_chg_20d=0.0))
    first = market.brief(out.iloc[:1].assign(
        spx_observed=True, kospi_observed=True, btc_observed=True, spx_close=1.0, kospi_close=1.0,
        btc_close=1.0, spx_rv_20d=0.1, kospi_rv_20d=0.1, btc_rv_20d=0.1, hy_spread=np.nan, t10y2y=0.1,
        usdkrw=1300.0, usdkrw_chg_20d=0.0))
    assert "위험회피 신호 3/4, 셀 수 있는 신호 3개" in first


def test_notify_without_token_logs_only(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(notify, "ALERT_LOG", tmp_path / "alerts.log")
    assert notify.send("테스트 알림") is False
    assert "테스트 알림" in (tmp_path / "alerts.log").read_text(encoding="utf-8")


# ── 스캐너·타임머신·사이징·판단 기록 ──
from autotrader.backtest import replay  # noqa: E402
from autotrader.intel import scanner  # noqa: E402
from autotrader.llm import decision_log  # noqa: E402
from autotrader.risk import sizing  # noqa: E402


def _daily(closes):
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=len(c), tz="UTC"),
                         "open": c, "high": c, "low": c, "close": c})


def test_scanner_setups():
    flat = [100.0] * 40
    assert scanner.setups(_daily(flat + [110]))["breakout_20d"]
    assert not scanner.setups(_daily(flat + [100]))["breakout_20d"]
    falling = list(np.linspace(200, 100, 41))
    assert scanner.setups(_daily(falling))["oversold"]
    flag = [100.0] * 25 + list(np.linspace(100, 120, 10)) + [120, 121, 120, 119, 120]
    assert scanner.setups(_daily(flag))["bull_flag"]


def test_sizing_strictest_constraint_and_floor():
    r = sizing.size(sizing.SizingInput(equity=10_000_000, entry=50_000, stop=48_000, risk_pct=1.0))
    assert r.qty == 50 and r.binding == "risk"            # 100,000 / 2,000
    r = sizing.size(sizing.SizingInput(equity=10_000_000, entry=50_000, stop=48_000, max_position_pct=5))
    assert r.qty == 10 and r.binding == "max_position"   # 500,000 / 50,000
    r = sizing.size(sizing.SizingInput(equity=1_000_000, entry=333, stop=300, risk_pct=1.0))
    assert r.qty == 303 and r.risk_amount <= 10_000       # 내림
    r = sizing.size(sizing.SizingInput(equity=1_000, entry=60_000, stop=59_000, min_notional=5))
    assert r.blocked and r.qty == 0                       # 0주 → BLOCK
    r = sizing.size(sizing.SizingInput(equity=1e6, entry=100, stop=90, win_rate=0.3, payoff=1.0))
    assert r.blocked and r.binding == "kelly"             # 기대값 음수 → 자본 없음
    with pytest.raises(ValueError):
        sizing.size(sizing.SizingInput(equity=1e6, entry=100, stop=110))


def _intraday(n_days=30, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    rows, px = [], 100.0
    for d in pd.bdate_range("2026-09-01", periods=n_days):
        for i in range(78):
            ts = pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(minutes=570 + 5 * i)
            o = px
            px = px * (1 + drift + rng.normal(0, 0.002))
            rows.append((ts.tz_convert("UTC"), o, max(o, px) * 1.0005, min(o, px) * 0.9995, px))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])


def test_replay_reproducible_and_drift_direction():
    bars = _intraday()
    last_day = bars["ts"].dt.tz_convert("America/New_York").dt.date.max()
    day = bars[bars["ts"].dt.tz_convert("America/New_York").dt.date == last_day]
    e = day.iloc[8]
    plan = replay.Plan(entry=e.close, stop=e.close * 0.99, target=e.close * 1.01, entry_ts=e.ts)
    a = replay.simulate(bars, plan, "America/New_York", n=2000, min_sessions=20)
    b = replay.simulate(bars, plan, "America/New_York", n=2000, min_sessions=20)
    assert a.p_target == b.p_target and a.seed == 19                       # 같은 시드 = 같은 결과
    assert abs(a.p_target + a.p_stop + a.p_neither - 1) < 1e-9
    up = replay.simulate(_intraday(drift=0.001), replay.Plan(entry=e.close, stop=e.close * 0.99,
                         target=e.close * 1.01, entry_ts=e.ts), "America/New_York", n=2000)
    assert up.p_target > up.p_stop                                          # 상승 표본이면 목표 쪽
    assert "기울어짐" in up.verdict
    with pytest.raises(ValueError):
        replay.simulate(bars, replay.Plan(entry=100, stop=101, target=102, entry_ts=e.ts), "America/New_York")


def test_replay_verdict_uses_breakeven():
    assert replay.verdict(0.241, 0.505, 2.0).startswith("손익분기 근처")   # 원본 예시: 2R에서 32% vs 33%
    assert replay.verdict(0.30, 0.30, 1.0).startswith("동전")
    assert replay.verdict(0.60, 0.20, 1.0).startswith("강하게 목표")


def test_decision_log_settles_after_horizon(tmp_path):
    path = tmp_path / "d.jsonl"
    idx = pd.date_range("2026-01-01", periods=10, tz="UTC")
    px = {"AAA": pd.Series(np.arange(100, 110, dtype=float), idx), "BM": pd.Series(100.0, idx)}
    base = dict(lab="L900", market="us", stance="long", conviction=0.6, benchmark="BM",
                model="haiku", prompt_sha="x", data_asof="2026-01-02")
    decision_log.append(decision_log.Decision(symbol="AAA", decided_at="2026-01-02T12:00:00+00:00",
                                              horizon_days=3, **base), path)
    decision_log.append(decision_log.Decision(symbol="AAA", decided_at="2026-01-08T12:00:00+00:00",
                                              horizon_days=5, **base), path)
    done = decision_log.settle_all(lambda m, s: px[s], path)
    assert len(done) == 1                                   # 두 번째는 아직 기간이 안 지남
    d = decision_log.load(path)[0]
    assert d.entry_ts.startswith("2026-01-03") and d.ret == pytest.approx(105 / 102 - 1) and d.alpha == d.ret


def test_alpaca_pages_and_skip_without_keys(monkeypatch, tmp_path):
    """Alpaca 분봉: 페이지를 끝까지 따라가고, 키가 없으면 아무것도 하지 않는다."""
    from autotrader.data import store
    from autotrader.data.collectors import us_alpaca

    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(us_alpaca._limit, "wait", lambda: None)
    pages = {
        None: {"bars": {"BRK.B": [{"t": "2016-01-04T14:30:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10,
                                   "n": 3, "vw": 1.2}]}, "next_page_token": "p2"},
        "p2": {"bars": {"BRK.B": [{"t": "2016-01-04T14:31:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 5,
                                   "n": 2, "vw": 1.6}]}, "next_page_token": None},
    }
    seen = []

    def fake(client, url, params, **kw):
        seen.append(params["symbols"])
        return pages[params.get("page_token")]

    monkeypatch.setattr(us_alpaca, "get_json", fake)
    us_alpaca.collect("BRK-B", {"APCA-API-KEY-ID": "x", "APCA-API-SECRET-KEY": "y"})
    df = store.read("us_1m_sip", "BRK-B")
    assert len(df) == 2 and seen == ["BRK.B", "BRK.B"]
    assert store.get_state("us_1m_sip", "BRK-B")["last_ts"] == pd.Timestamp("2016-01-04T14:31:00Z")

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    seen.clear()
    us_alpaca.run(["BRK-B"])
    assert seen == []


def test_news_resumes_and_dedupes_by_id(monkeypatch, tmp_path):
    """뉴스: 페이지를 따라가고, 다음 실행은 마지막 기사 시각부터 이어받으며, 같은 id는 한 번만 남는다."""
    from autotrader.data import store
    from autotrader.data.collectors import news

    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.sqlite")
    monkeypatch.setattr(news._limit, "wait", lambda: None)

    def art(i, t):
        return {"id": str(i), "created_at": t, "updated_at": t, "symbols": ["AAPL", "BTCUSD"], "headline": f"h{i}",
                "summary": "", "source": "benzinga", "url": f"https://x/{i}", "author": "a", "images": []}

    pages = {None: {"news": [art(1, "2015-01-01T13:00:00Z")], "next_page_token": "p2"},
             "p2": {"news": [art(2, "2015-01-02T13:00:00Z")], "next_page_token": None}}
    starts = []

    def fake(client, url, params, **kw):
        starts.append(params["start"])
        return pages[params.get("page_token")]

    monkeypatch.setattr(news, "get_json", fake)
    h = {"APCA-API-KEY-ID": "x", "APCA-API-SECRET-KEY": "y"}
    news.collect(h)
    assert store.get_state("news", "ALL")["last_ts"] == pd.Timestamp("2015-01-02T13:00:00Z")

    pages[None] = {"news": [art(2, "2015-01-02T13:00:00Z"), art(3, "2015-01-03T13:00:00Z")], "next_page_token": None}
    news.collect(h)
    assert starts[-1] == "2015-01-02T13:00:00Z"            # 이어받기: 마지막 기사 시각부터
    df = store.read("news", "ALL")
    assert df["id"].tolist() == [1, 2, 3] and df["symbols"].iloc[0] == "AAPL,BTCUSD"
    assert str(df["ts"].dt.tz) == "UTC"
