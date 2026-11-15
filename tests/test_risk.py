"""리스크 계층(core/schemas, risk/guards) 테스트. 네트워크 없음. 실행: uv run pytest -q tests/test_risk.py"""

import copy
from datetime import UTC, datetime, timedelta, timezone

import pytest

from autotrader.core.schemas import Fill, Order, Position, Signal
from autotrader.risk import guards

NOW = datetime(2026, 11, 16, 5, 0, tzinfo=UTC)   # KST 14:00
CFG = guards.load_config()


def cfg(**over):
    c = copy.deepcopy(CFG)
    for path, v in over.items():
        d = c
        *keys, last = path.split("__")
        for k in keys:
            d = d[k]
        d[last] = v
    return c


def sig(**kw):
    base = dict(lab="L001", symbol="005930", market="kr_stock", side="long", price=50_000,
                stop=48_000, reason="12-1 모멘텀 상위", ts=NOW, quote_ts=NOW - timedelta(seconds=5))
    return Signal(**{**base, **kw})


# --- 스키마 ---

def test_schema_requires_aware_ts_and_converts_to_utc():
    with pytest.raises(ValueError):
        sig(ts=datetime(2026, 11, 16, 14, 0))
    kst = timezone(timedelta(hours=9))
    s = sig(ts=datetime(2026, 11, 16, 14, 0, tzinfo=kst))
    assert s.ts == NOW and s.ts.tzinfo == UTC
    with pytest.raises(ValueError):
        sig(side="buy")


def test_chain_carries_ids_lab_reason():
    p = guards.make_plan(sig(), guards.Account(equity=100_000_000), NOW, CFG)
    assert p.approved and p.signal_id and p.lab == "L001" and p.reason == "12-1 모멘텀 상위"
    o = guards.plan_to_order(p, NOW, CFG)
    assert o.plan_id == p.id and o.side == "buy" and o.order_type == "limit" and o.limit_price == 50_000
    assert o.mode == "paper"
    f = Fill(order_id=o.id, lab=o.lab, symbol=o.symbol, side=o.side, qty=o.qty, price=50_050, ts=NOW, fee=7.5)
    assert f.notional == o.qty * 50_050
    with pytest.raises(ValueError):
        Order(plan_id="x", lab="L", symbol="A", market="kr_stock", side="buy", qty=1,
              order_type="limit", limit_price=None, reason="", ts=NOW)


# --- §9-3 데이터 신선도 ---

def test_staleness_blocks_new_orders():
    assert guards.check_staleness("kr_stock", NOW - timedelta(seconds=29), NOW, CFG) is None
    assert "한도 30초" in guards.check_staleness("kr_stock", NOW - timedelta(seconds=31), NOW, CFG)
    assert guards.check_staleness("crypto_spot", NOW - timedelta(seconds=11), NOW, CFG)   # crypto 10초
    assert guards.check_staleness("kr_stock", None, NOW, CFG) == "시세 시각 없음"
    assert "기준 없음" in guards.check_staleness("forex", NOW, NOW, CFG)
    assert "미래" in guards.check_staleness("kr_stock", NOW + timedelta(seconds=5), NOW, CFG)

    p = guards.make_plan(sig(quote_ts=NOW - timedelta(minutes=5)), guards.Account(equity=1e8), NOW, CFG)
    assert not p.approved and p.qty == 0 and p.sizing is None and p.blocked.startswith("데이터 신선도")


def test_unmeasurable_after_retries():
    assert not guards.is_unmeasurable(2, CFG) and guards.is_unmeasurable(3, CFG)


# --- §9-4 Plan: sizing 재사용·차단 사유 ---

def test_plan_uses_sizing_and_records_binding():
    # 1억, 위험 1% = 100만 / 주당 2000 → 500주, 비중 5% = 500만 / 5만 → 100주, 주문 상한 100만 → 20주
    p = guards.make_plan(sig(), guards.Account(equity=100_000_000), NOW, CFG)
    assert p.qty == 20 and p.sizing.binding == "max_order_value"
    assert any("sizing ok" in c for c in p.checks)


def test_plan_converts_order_cap_for_foreign_account():
    s = sig(market="us_stock", symbol="AAPL", price=200, stop=190)
    with pytest.raises(ValueError):
        guards.make_plan(s, guards.Account(equity=100_000, currency="USD"), NOW, CFG)
    p = guards.make_plan(s, guards.Account(equity=100_000, currency="USD", krw_per_unit=1_400), NOW, CFG)
    assert p.qty == 3 and p.sizing.binding == "max_order_value"   # 100만원/1400 = 714달러 → 3주


def test_plan_blocks_kill_switch_heat_and_positions():
    acc = guards.Account(equity=1e8)
    assert "kill_switch" in guards.make_plan(sig(), acc, NOW, cfg(global__kill_switch=True)).blocked
    p = guards.make_plan(sig(), guards.Account(equity=1e8, open_heat_pct=6.0), NOW, CFG)
    assert not p.approved and "max_heat" in p.blocked and p.sizing is not None
    p = guards.make_plan(sig(), guards.Account(equity=1e8, open_positions=10), NOW, CFG)
    assert "보유 종목" in p.blocked
    with pytest.raises(ValueError):
        guards.plan_to_order(p, NOW, CFG)


# --- §9-7 최소 주문 금액 ---

def test_min_notional_blocks_with_reason():
    acc = guards.Account(equity=1_000_000)   # 비중 5% = 5만원 → 1주(5만원)
    p = guards.make_plan(sig(), acc, NOW, CFG, min_notional=100_000)
    assert not p.approved and "최소 주문" in p.blocked
    p = guards.make_plan(sig(), acc, NOW, cfg(orders__min_notional_check=False), min_notional=100_000)
    assert p.approved and p.qty == 1
    assert guards.check_min_notional(5_000, 10_000, CFG).startswith("주문 금액")
    assert guards.check_min_notional(5_000, 10_000, cfg(orders__min_notional_check=False)) is None


# --- §9-5 보유 포지션 보호 ---

def pos(**kw):
    base = dict(lab="L001", symbol="005930", market="kr_stock", side="long", qty=10,
                entry=50_000, stop=48_000, target=56_000)
    return Position(**{**base, **kw})


def test_position_guard_stop_target_and_priority():
    assert guards.check_position(pos(), 50_500, NOW, CFG) is None
    assert guards.check_position(pos(), 48_000, NOW, CFG).kind == "stop_loss"
    assert guards.check_position(pos(), 56_100, NOW, CFG).kind == "take_profit"
    short = pos(side="short", stop=52_000, target=45_000)
    assert guards.check_position(short, 52_500, NOW, CFG).kind == "stop_loss"
    assert guards.check_position(short, 44_000, NOW, CFG).kind == "take_profit"
    assert guards.check_position(pos(), 48_000, NOW, cfg(position_guard__enforce_stop_loss=False)) is None
    # 시세 실패: 한도 전엔 보류, 한도에 닿으면 측정 불가가 손절보다 먼저
    assert guards.check_position(pos(), None, NOW, CFG, missed_quotes=2) is None
    assert guards.check_position(pos(), 40_000, NOW, CFG, missed_quotes=3).kind == "unmeasurable"


def test_flat_before_close_only_intraday():
    close = NOW + timedelta(minutes=8)   # 마감 8분 전, 기준 10분
    assert guards.check_position(pos(intraday=True), 50_500, NOW, CFG, close_at=close).kind == "flat_before_close"
    assert guards.check_position(pos(intraday=True), 50_500, NOW, CFG, close_at=NOW + timedelta(minutes=30)) is None
    assert guards.check_position(pos(), 50_500, NOW, CFG, close_at=close) is None
    coin = pos(market="crypto_spot", intraday=True)
    assert guards.check_position(coin, 50_500, NOW, CFG, close_at=close) is None   # 마감 규칙 없는 시장
