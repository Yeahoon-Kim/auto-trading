"""lab 판정 Sharpe rf 통일(기준 1·2·3), KR 시가총액 검사 미확인 표시, 거래정지 잔량 비중 상한 (2026-11-19).
합성 데이터, 네트워크·저장소 없음."""

import json

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest import benchmark, engine, lab
from autotrader.backtest.data import Panel, check_mcap_causality

RF = 0.0003          # 하루 무위험수익률 (연 7.6% — 차이가 잘 보이게 크게)

STRATEGY = '''
import pandas as pd

def generate_signals(data, params):
    c = data.close
    mom = c / c.shift(int(params["n"])) - 1
    w = (mom > 0).astype(float)
    return w.div(w.sum(axis=1).where(w.sum(axis=1) > 0), axis=0).fillna(0.0)
'''


def _synthetic_panel() -> Panel:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=1300)
    r = pd.DataFrame(rng.normal(0.0006, 0.012, (len(idx), 3)), index=idx, columns=["A", "B", "C"])
    c = 100 * (1 + r).cumprod()
    return Panel({"open": c.copy(), "close": c}, "us_stock",
                 meta={"source": "합성", "survivorship": "해당 없음", "dev_end": str(idx[-1].date())})


def _run(tmp_path, monkeypatch, rf: float | None) -> tuple[dict, pd.Series]:
    d = tmp_path / ("rf" if rf else "rf0")
    d.mkdir()
    (d / "lab.yaml").write_text("id: T\nversion: 1\nmarket: us_stock\nparams:\n  n: 20\n"
                                "walk_forward:\n  train: 24M\n  test: 6M\n", encoding="utf-8")
    (d / "strategy.py").write_text(STRATEGY, encoding="utf-8")
    panel = _synthetic_panel()
    monkeypatch.setattr(lab, "_panel", lambda cfg, lockbox: panel)
    monkeypatch.setattr(lab, "_rf", lambda p, index: None if rf is None else pd.Series(rf, index=index))
    s = lab.develop(d)
    net = pd.read_parquet(next((d / "results").glob("*/test_returns.parquet")))["net"]
    return s, net


def test_judge_sharpe_matches_metrics_definition(monkeypatch):
    p = _synthetic_panel()
    r = p.close["A"].pct_change().dropna()
    monkeypatch.setattr(lab, "_rf", lambda panel, index: pd.Series(RF, index=index))
    s = lab.judge_sharpe(r, p)
    assert s["sharpe"] == pytest.approx(benchmark.metrics(r, pd.Series(RF, index=r.index), year=252)["sharpe"])
    assert s["sharpe_rf0"] == pytest.approx(engine.sharpe(r, 252))
    assert s["sharpe"] < s["sharpe_rf0"]
    monkeypatch.setattr(lab, "_rf", lambda panel, index: None)
    assert lab.judge_sharpe(r, p)["sharpe"] == pytest.approx(engine.sharpe(r, 252))


def test_criteria_1_2_3_use_same_rf_definition(tmp_path, monkeypatch):
    s, net = _run(tmp_path, monkeypatch, RF)
    ex = lambda x: engine.sharpe(x - RF, 252)
    c = {x["no"]: x for x in s["checks"]}
    # 기준 1: 시험 구간 전체, rf 차감 = summary test.sharpe
    assert c[1]["value"] == pytest.approx(ex(net))
    assert c[1]["value"] == pytest.approx(s["test"]["sharpe"])
    assert c[1]["value_rf0"] == pytest.approx(engine.sharpe(net, 252))
    # 기준 2: 창별 시험 Sharpe도 같은 정의 (시험 수익을 창 날짜로 잘라 다시 계산)
    for w in s["windows"]:
        part = net[(net.index >= pd.Timestamp(w["test"][0])) & (net.index < pd.Timestamp(w["test"][1]))]
        assert w["test_sharpe"] == pytest.approx(ex(part))
        assert w["test_sharpe_rf0"] == pytest.approx(engine.sharpe(part, 252))
        assert w["train_sharpe"] < w["train_sharpe_rf0"]
    assert s["train_sharpe_mean"] == pytest.approx(np.mean([w["train_sharpe"] for w in s["windows"]]))
    # 기준 3: 민감도도 rf 차감
    assert s["sensitivity"] and all(x["test_sharpe"] < x["test_sharpe_rf0"] for x in s["sensitivity"])
    assert c[3]["value"] == pytest.approx(min(x["test_sharpe"] for x in s["sensitivity"]))
    # ① summary의 sharpe 키는 어디서나 rf 차감, sharpe_rf0는 참고
    assert s["test_no_cost"]["sharpe"] < s["test_no_cost"]["sharpe_rf0"]
    assert s["test_cost_x2"]["sharpe"] < s["test_cost_x2"]["sharpe_rf0"]
    assert "rf 차감" in s["sharpe_definition"]
    rep = lab.report(json.loads(json.dumps(s, default=str)))
    assert "rf=0:" in rep and "Sharpe 정의" in rep


def test_rf_zero_reproduces_previous_values(tmp_path, monkeypatch):
    """rf가 없으면(0) 새 정의 = 예전 engine.sharpe 값."""
    s, net = _run(tmp_path, monkeypatch, None)
    assert s["checks"][0]["value"] == pytest.approx(engine.sharpe(net, 252))
    for w in s["windows"]:
        assert w["test_sharpe"] == w["test_sharpe_rf0"] and w["train_sharpe"] == w["train_sharpe_rf0"]
    assert all(x["test_sharpe"] == x["test_sharpe_rf0"] for x in s["sensitivity"])
    assert s["test_no_cost"]["sharpe"] == pytest.approx(s["test_no_cost"]["sharpe_rf0"])


# ── ② KR 시가총액: 재구성하지 않으면 통과로 적지 않는다 ──
def test_kr_mcap_causality_is_unchecked_not_passed():
    idx = pd.bdate_range("2024-01-01", periods=10)
    c = pd.DataFrame({"A": 100.0}, index=idx)
    p = Panel({"open": c, "close": c, "market_cap": c * 1e6}, "kr_stock")
    r = check_mcap_causality(p)
    assert r["passed"] is None and r["fields"] == ["market_cap"] and r["cuts"] == [] and "krx_1d" in r["reason"]
    # 미확인은 실패가 아니다 (기준 7 전체 결과는 그대로 신호 검사가 정한다)
    causal = {"passed": True, "cuts": [], "dividend_fields": {"passed": True}, "fin_fields": {"passed": True},
              "mcap_fields": r}
    assert lab._all_passed(causal) is True
    assert lab._all_passed({**causal, "fin_fields": {"passed": False}}) is False
    # 필드가 없으면 예전처럼 통과 (검사할 것이 없음)
    assert check_mcap_causality(Panel({"open": c, "close": c}, "kr_stock"))["passed"] is True


# ── ③ 거래정지 잔량: 비중 합이 1을 넘지 않게 ──
def _halt_panel():
    idx = pd.bdate_range("2024-01-01", periods=6)
    c = pd.DataFrame({"A": [100, 100, 100, 100, 110, 110], "B": [50, 51, 52, 53, 54, 55],
                      "C": [20, 20, 21, 21, 22, 22]}, index=idx, dtype=float)
    p = Panel({"open": c.copy(), "close": c}, "kr_stock")
    t = pd.DataFrame(True, index=idx, columns=c.columns)
    t.loc[idx[2]:idx[3], "A"] = False           # A가 둘째·셋째 체결 봉에 거래정지
    p["tradable"] = t
    return p


def test_frozen_holding_does_not_lever_up():
    p = _halt_panel()
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 0.5], "C": [0.0, 0.5]}, index=p.index[[0, 1]])
    res = engine.run(p, w, engine.Costs(), fill="next_open")
    tot = res.weights.sum(axis=1)
    assert tot.max() <= 1 + 1e-9                                 # 예전: A 1.0 + B·C 1.0 = 2.0
    assert res.meta["frozen_capped_bars"] >= 1
    assert res.weights["A"].iloc[1] == pytest.approx(1.0)        # 팔 수 없어 그대로
    assert res.weights["A"].iloc[-1] == pytest.approx(0.0)       # 정지 풀린 뒤 팔림
    assert res.weights[["B", "C"]].iloc[-1].sum() == pytest.approx(1.0)


def test_frozen_holding_with_spare_cash_is_untouched():
    """목표 합이 1보다 작아 현금으로 살 수 있으면 줄이지 않는다."""
    p = _halt_panel()
    w = pd.DataFrame({"A": [0.4, 0.0], "B": [0.0, 0.3], "C": [0.0, 0.3]}, index=p.index[[0, 1]])
    res = engine.run(p, w, engine.Costs(), fill="next_open")
    assert res.meta["frozen_capped_bars"] == 0
    assert res.weights[["B", "C"]].iloc[2].sum() == pytest.approx(0.6, rel=0.05)
