"""하네스 단위 검사 (합성 데이터, 실제 데이터 불필요).

    uv run pytest labs/L003-chart-patterns/study/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H  # noqa: E402


def _wide(n_sym=40, start="2016-01-01", end="2019-12-31", seed=0, kr=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    syms = [f"S{i:03d}" for i in range(n_sym)]
    r = rng.normal(0.0003, 0.02, (len(idx), n_sym))
    c = 100 * np.exp(np.cumsum(r, 0))
    o = c * np.exp(rng.normal(0, 0.005, c.shape))
    hi = np.maximum(o, c) * 1.01
    lo = np.minimum(o, c) * 0.99
    w = {k: pd.DataFrame(v, idx, syms) for k, v in (("open", o), ("high", hi), ("low", lo), ("close", c))}
    # 상장폐지 1종목, 늦게 상장 1종목
    for k in w:
        w[k].loc[idx[min(600, len(idx) // 2)]:, "S000"] = np.nan
        w[k].loc[:idx[min(100, len(idx) // 4)], "S001"] = np.nan
    if kr:
        w["market_cap"] = pd.DataFrame(rng.uniform(1e10, 1e12, c.shape), idx, syms)
        w["value"] = pd.DataFrame(rng.uniform(5e8, 5e9, c.shape), idx, syms)
        w["tradable"] = pd.DataFrame(rng.random(c.shape) > 0.02, idx, syms)
        w["value"] = w["value"].where(w["close"].notna())
        w["market_cap"] = w["market_cap"].where(w["close"].notna())
    return w


def test_weekly_bars():
    w = _wide(3, "2017-01-02", "2017-03-31", kr=False)
    s = H.stocks_from_wide(w)[2]
    df, end = H.bars_of(s, "1w")
    daily = pd.DataFrame({"open": s.o, "high": s.h, "low": s.l, "close": s.c}, index=pd.DatetimeIndex(s.dates))
    ref = daily.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    last_day = daily.index.to_series().resample("W-FRI").last().dropna()
    assert (df.index == pd.DatetimeIndex(last_day.values)).all()
    np.testing.assert_allclose(df.values, ref.values)
    assert (pd.DatetimeIndex(s.dates[end]) == df.index).all()


def test_returns_entry_exit_delist_censor():
    w = _wide(40, kr=True)
    costs = H.Costs("kr_stock")
    stocks = H.stocks_from_wide(w)
    bt = H.build_bars("kr_stock", "1d", stocks, None, w["close"].index[-1], costs)
    df = bt.df
    s = next(x for x in stocks if x.sym == "S005")
    sid = bt.syms.index("S005")
    row = df[(df["sym"] == sid) & (df["k"] == 300)].iloc[0]
    assert row["ret20"] == pytest.approx(s.c[320] / s.o[301] - 1)
    assert row["ret5"] == pytest.approx(s.c[305] / s.o[301] - 1)
    # 살아 있는 종목의 끝 부분: h봉이 모자라면 NaN (버림)
    K = len(s.c)
    tail = df[(df["sym"] == sid) & (df["k"] == K - 10)].iloc[0]
    assert np.isnan(tail["ret20"]) and not np.isnan(tail["ret5"])
    # 상장폐지 종목: 마지막 종가로 청산하고 표시
    d0 = bt.syms.index("S000")
    s0 = stocks[d0]
    t = df[(df["sym"] == d0) & (df["k"] == len(s0.c) - 10)].iloc[0]
    assert t["ret20"] == pytest.approx(s0.c[-1] / s0.o[len(s0.c) - 9] - 1)
    assert bool(t["delist20"]) and not bool(t["delist5"])
    # 주봉: 진입 = 다음 일봉 시가, 청산 = (신호 주 + h)주 마지막 거래일 종가
    bw = H.build_bars("kr_stock", "1w", stocks, None, w["close"].index[-1], costs)
    _, end = H.bars_of(s, "1w")
    r = bw.df[(bw.df["sym"] == sid) & (bw.df["k"] == 50)].iloc[0]
    assert r["ret20"] == pytest.approx(s.c[end[70]] / s.o[end[50] + 1] - 1)


def test_costs_kr_tax_by_exit_date():
    c = H.Costs("kr_stock")
    v = c.roundtrip(np.array(["2017-06-01", "2019-07-01", "2025-03-01", "2026-02-01"], dtype="datetime64[ns]"))
    base = 2 * (0.015 + 0.05) / 100
    np.testing.assert_allclose(v, base + np.array([0.30, 0.25, 0.15, 0.20]) / 100)
    u = H.Costs("us_stock")
    np.testing.assert_allclose(u.roundtrip(np.array(["2017-06-01"], dtype="datetime64[ns]")), 2 * (0.25 + 0.05) / 100)


def test_period_split_no_overlap():
    w = _wide(30, kr=False)
    bt = H.build_bars("us_stock", "1d", H.stocks_from_wide(w), None, w["close"].index[-1], H.Costs("us_stock"))
    df = bt.df
    b = np.datetime64(H.BOUNDARY)
    ex = df[df["period"] == 0]
    assert (ex["date"] < b).all() and (ex["exit20"] < b).all()
    assert not (ex["ret60"].notna() & (ex["exit60"] >= b)).any()
    assert (df[df["period"] == 1]["date"] >= b).all()
    gap = df[(df["date"] < b) & (df["exit20"] >= b)]
    assert len(gap) and (gap["period"] == -1).all()


def test_universe_kr_rules():
    w = _wide(40, kr=True)
    u = H.universe_wide("kr_stock", w)
    # 첫 19일은 20일 평균 거래대금이 없어 유니버스 없음
    assert not u.iloc[:19].any().any()
    w2 = {k: v.copy() for k, v in w.items()}
    w2["value"].loc[:, "S010"] = 1e8                       # 거래대금 부족
    w2["tradable"].iloc[200, 11] = False                   # 거래 정지일
    u2 = H.universe_wide("kr_stock", w2)
    assert not u2["S010"].any()
    assert not u2.iloc[200, 11]


def test_placebo_rules():
    w = _wide(60, kr=False, seed=3)
    stocks = H.stocks_from_wide(w)
    bt = H.build_bars("us_stock", "1d", stocks, None, w["close"].index[-1], H.Costs("us_stock"))
    rng = np.random.default_rng(1)
    ev = []
    for s in stocks[2:30]:
        k = np.sort(rng.choice(np.arange(70, len(s.c) - 70), 4, replace=False))
        ev.append(pd.DataFrame({"date": s.dates[k], "direction": 1, "tool": "t", "variant": "v", "sym": s.sym}))
    ev = pd.concat(ev, ignore_index=True)
    ev, blocked, _ = H.attach_events(ev, bt, (0, 1))
    for pool, matched in [(p, m) for p in ("year", "period") for m in (True, False)]:
        idx = H.sample_placebos(ev, bt, blocked, seed=0, matched=matched, pool=pool)
        assert (idx >= 0).all(1).mean() > 0.9
        df = bt.df
        for i in range(len(ev)):
            p = idx[i][idx[i] >= 0]
            pr = df.iloc[p]
            e = ev.iloc[i]
            assert (pr["sym"] == e["sid"]).all() and (pr["period"] == e["period"]).all()
            if pool == "year":
                assert (pr["year"] == e["year"]).all()
            bk = blocked[(e["sid"], "t")]
            dk = pr["k"].values[:, None] - bk[None, :]          # 위약 − 신호 (봉)
            assert not ((dk >= 0) & (dk <= H.BLOCK_BARS)).any()  # 신호 날짜·신호 뒤 20봉은 빠진다
            if matched:
                assert (pr["quint"] == e["quint"]).all()
    # xsec: 같은 날짜·같은 분위의 다른 종목
    idx = H.sample_placebos(ev, bt, blocked, seed=0, matched=True, pool="xsec")
    df = bt.df
    for i in range(0, len(ev), 7):
        p = idx[i][idx[i] >= 0]
        pr, e = df.iloc[p], ev.iloc[i]
        assert len(p) and (pr["date"] == e["date"]).all() and (pr["sym"] != e["sid"]).all() and (pr["quint"] == e["quint"]).all()
    # 같은 시드면 같은 결과, 다른 시드면 다른 결과
    a = H.sample_placebos(ev, bt, blocked, seed=5, matched=True)
    assert (a == H.sample_placebos(ev, bt, blocked, seed=5, matched=True)).all()
    assert (a != H.sample_placebos(ev, bt, blocked, seed=6, matched=True)).any()


def test_shuffle_preserves_daily_tuples():
    w = _wide(3, kr=False)
    s = H.stocks_from_wide(w)[1]
    z = H.shuffle_stock(s, 1, method="perm")
    def tup(x):
        return np.column_stack([x.o[1:] / x.c[:-1], x.h[1:] / x.c[1:], x.l[1:] / x.c[1:], x.c[1:] / x.c[:-1]])
    a, b = tup(s), tup(z)
    seg = s.dates[1:] < np.datetime64(H.BOUNDARY)
    for m in (seg, ~seg):   # 구간 안에서만 섞인다
        np.testing.assert_allclose(np.sort(a[m], 0), np.sort(b[m], 0), rtol=1e-9)
    assert z.c[0] == s.c[0] and not np.allclose(a, b)
    assert (z.h >= np.maximum(z.o, z.c) - 1e-9).all() and (z.l <= np.minimum(z.o, z.c) + 1e-9).all()
    # boot: 같은 구간의 원래 묶음 중에서 복원 추출
    zb = H.shuffle_stock(s, 1, method="boot")
    bb = tup(zb)
    for m in (seg, ~seg):
        orig = {tuple(np.round(x, 9)) for x in a[m]}
        assert all(tuple(np.round(x, 9)) in orig for x in bb[m])
    assert len({tuple(np.round(x, 9)) for x in bb}) < len(bb)   # 중복이 생긴다 (복원)


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    adj = H.holm(p)
    # 정렬: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 → 단조 → 0.06
    np.testing.assert_allclose(adj[:4], [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(adj[4])


def test_bootstrap_null_uniform():
    rng = np.random.default_rng(0)
    ps = []
    for i in range(300):
        cl = rng.integers(0, 150, 600)
        shock = rng.normal(0, 1, 150)[cl]               # 같은 날 상관
        d = shock + rng.normal(0, 1, 600)
        ps.append(H.cluster_bootstrap(d, cl, seed=i, n_boot=400)["p"])
    ps = np.array(ps)
    assert 0.02 < (ps < 0.05).mean() < 0.10
    assert 0.40 < (ps < 0.5).mean() < 0.60


def test_bootstrap_power():
    rng = np.random.default_rng(1)
    cl = rng.integers(0, 200, 800)
    d = 0.3 + rng.normal(0, 1, 800)
    assert H.cluster_bootstrap(d, cl, n_boot=500)["p"] < 1e-6


def test_universe_quintile_causality():
    w = _wide(60, kr=True, seed=4)
    for tf in ("1d", "1w"):
        idx = w["close"].index
        wk = H.week_key(idx.values)
        ends = idx[np.r_[wk[1:] != wk[:-1], False]]
        res = H.check_universe_causality("kr_stock", w, tf, [ends[150], ends[160]], H.Costs("kr_stock"))
        assert res["ok"], res
        assert res["compared"] > 1000


def test_causality_check_catches_leak(monkeypatch):
    """잘라내기 검사가 미래 값을 쓰는 유니버스를 실제로 잡아내는지 (검사의 검사)."""
    w = _wide(60, kr=True, seed=4)
    orig = H.universe_wide

    def leaky(market, wide):
        u = orig(market, wide)
        return u & u.shift(-5, fill_value=True)           # 5일 뒤에도 유니버스인 종목만 → 미래참조
    monkeypatch.setattr(H, "universe_wide", leaky)
    idx = w["close"].index
    wk = H.week_key(idx.values)
    ends = idx[np.r_[wk[1:] != wk[:-1], False]]
    res = H.check_universe_causality("kr_stock", w, "1d", [ends[150]], H.Costs("kr_stock"))
    assert not res["ok"]
