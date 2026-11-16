"""비중형 paper 데스크의 배당락·분할 처리 (합성 데이터).
수정주가(close)에는 kr_adj_factor(분할·병합) + 세후 배당 재투자 + 주식수 기반 병합 보정이 들어 있다.
paper 장부는 원본가라서, 원본가·수정주가 수익 차이를 보고 무엇이 일어났는지 나눠 처리해야 한다.
실행: uv run pytest -q tests/test_paper_dividend.py"""

import pandas as pd
import pytest

from autotrader.backtest import engine
from autotrader.backtest.data import Panel
from autotrader.risk.guards import load_config
from autotrader.runner.paper import _no_limits, replay

CAP = 1_000_000
KW = {"lab": "T", "market": "kr_stock", "lot": 1.0, "limit_band_pct": None}


def _panel(adj, raw, factor=None) -> Panel:
    idx = pd.bdate_range("2024-01-01", periods=len(adj))
    c = pd.DataFrame({"A": adj}, index=idx, dtype=float)
    f = {"open": c.copy(), "close": c, "raw_close": pd.DataFrame({"A": raw}, index=idx, dtype=float)}
    if factor is not None:
        f["_f"] = pd.DataFrame({"A": factor}, index=idx, dtype=float)     # kr_adj_factor (load fields에 "_f")
    return Panel(f, "kr_stock")


def _hold(p):
    """첫 봉에 100% 매수 → 이후 신호 없음 (on_signal: 다시 맞추지 않음)."""
    W = engine.align_weights(pd.DataFrame({"A": [1.0]}, index=p.index[:1]), p)
    book, ev = replay(p, W, 0, CAP, engine.Costs(), _no_limits(load_config()), rebalance="on_signal", **KW)
    q0 = next(e for e in ev if e["kind"] == "fill")["qty"]
    return book, ev, q0


def _kinds(ev, *k):
    return [e for e in ev if e["kind"] in k]


def test_ex_dividend_over_1pct_is_cash_not_split():
    # 수정주가 평평(세후 배당 재투자), 원본가 102 → 100: 세후 배당 2원 (2%)
    p = _panel([100] * 5, [102, 102, 100, 100, 100])
    book, ev, q0 = _hold(p)
    assert not _kinds(ev, "corporate_action")
    assert book.shares["A"] == q0                                  # 수량 그대로
    (d,) = _kinds(ev, "dividend")
    assert d["cash"] == pytest.approx(q0 * 2.0)
    assert book.equity == pytest.approx(book.cash + q0 * 100)
    assert book.equity == pytest.approx(CAP)                       # 평가액 보존 (배당 포함)


def test_ex_dividend_under_1pct_is_not_ignored():
    p = _panel([100] * 5, [100.5, 100.5, 100, 100, 100])          # 세후 배당 0.5원 (0.5%)
    book, ev, q0 = _hold(p)
    (d,) = _kinds(ev, "dividend")
    assert d["cash"] == pytest.approx(q0 * 0.5)
    assert book.equity == pytest.approx(CAP)                       # 원본가 하락분이 현금으로


def test_float_noise_is_ignored_and_shares_stay_whole():
    p = _panel([100] * 5, [100, 100, 100.0004, 100, 100])          # 수정주가 계산 오차 수준 (±4e-6)
    book, ev, _ = _hold(p)
    assert not _kinds(ev, "corporate_action", "dividend")
    assert float(book.shares["A"]).is_integer()


def test_split_from_adj_factor_with_dividend_same_day():
    # 2:1 분할(kr_adj_factor 0.5 → 1)과 같은 날 세후 배당 1원: 원본가 200 → 99, 수정주가 100 → 100
    p = _panel([100] * 4, [200, 200, 99, 99], factor=[0.5, 0.5, 1.0, 1.0])
    book, ev, q0 = _hold(p)
    assert book.shares["A"] == q0 * 2
    (d,) = _kinds(ev, "dividend")
    assert d["cash"] == pytest.approx(q0 * 2 * 1.0)
    assert book.equity == pytest.approx(CAP - q0 * 200 + q0 * 2 * 99 + q0 * 2 * 1.0)


def test_merge_outside_adj_factor_adjusts_shares():
    # 주식수 기반 병합 보정 (kr_adj_factor에 없음): 10:1 병합, 원본가 10 → 100, 수정주가 100 평평
    p = _panel([100] * 4, [10, 10, 100, 100], factor=[1.0] * 4)
    book, ev, q0 = _hold(p)
    assert book.shares["A"] == q0 / 10 or book.shares["A"] == q0 // 10
    assert _kinds(ev, "corporate_action")
    assert not _kinds(ev, "dividend")
    assert book.equity == pytest.approx(CAP, rel=1e-4)


def test_fractional_split_pays_cash_in_lieu():
    # 3:2 분할: 원본가 150 → 100. 홀수 주식이면 0.5주는 현금으로
    p = _panel([100] * 4, [150, 150, 100, 100], factor=[2 / 3, 2 / 3, 1.0, 1.0])
    W = engine.align_weights(pd.DataFrame({"A": [1.0]}, index=p.index[:1]), p)
    book, ev = replay(p, W, 0, 1700, engine.Costs(), _no_limits(load_config()), rebalance="on_signal", **KW)
    q0 = next(e for e in ev if e["kind"] == "fill")["qty"]
    assert q0 % 2 == 1
    assert float(book.shares["A"]).is_integer() and book.shares["A"] == (q0 * 3) // 2
    assert book.equity == pytest.approx(1700)
