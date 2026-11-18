"""L003 R2 하네스 단위 검사 (합성 데이터, 실제 데이터 불필요).

    uv run pytest labs/L003-chart-patterns/model/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import dataset as D  # noqa: E402
import walkforward as W  # noqa: E402

H = D.H


# ───────────────────────── 합성 데이터셋 (특징·목표값 직접 생성) ─────────────────────────
def synth_ds(n_days=1600, per_day=12, seed=0, leak_corr=None, weekly_frac=0.25):
    """2010-01 ~ 2016 영업일마다 신호 per_day개. y는 특징과 독립 (leak_corr면 y+잡음 'leak' 특징 추가)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-04", periods=n_days)
    date = np.repeat(days.values, per_day)
    n = len(date)
    wk = rng.random(n) < weekly_frac
    exit_ = date + np.where(wk, np.timedelta64(140, "D"), np.timedelta64(28, "D"))
    df = pd.DataFrame({"date": date, "exit20": exit_, "is_weekly": wk.astype(float), "period": 0,
                       "market": np.where(rng.random(n) < 0.5, "kr_stock", "us_stock"),
                       "timeframe": np.where(wk, "1w", "1d"), "sym": rng.integers(0, 300, n).astype(str),
                       "tool": "t", "variant": rng.choice(["a", "b", "c"], n)})
    for f in D.NUM_FEATURES:
        if f not in ("is_weekly",):
            df[f] = rng.normal(size=n)
    df.loc[rng.random(n) < 0.1, "ret_250"] = np.nan
    df["y"] = rng.standard_t(4, n) * 0.08
    df["ret20"] = df["y"]
    df["cost20"] = 0.004
    df[D.CAT_FEATURE] = df["tool"] + ":" + df["variant"]
    feats = list(D.NUM_FEATURES)
    if leak_corr:
        sd = df["y"].std() * np.sqrt(1 / leak_corr ** 2 - 1)
        df["leak"] = df["y"] + rng.normal(0, sd, n)
        feats.append("leak")
    return df, feats


# ───────────────────────── (d) purge ─────────────────────────
def test_fold_purge_and_gap():
    df, _ = synth_ds()
    folds = W.fold_masks(df)
    assert [f[0] for f in folds] == [str(y) for y in range(2013, 2017)]
    for name, start, train, test in folds:
        assert train.sum() > 0 and test.sum() > 0
        tr, te = df[train], df[test]
        # 학습 행의 목표값은 예측 시작 전에 다 정해진다
        assert tr["exit20"].max() < te["date"].min()
        assert (te["date"] >= start).all() and (te["date"] < start + pd.DateOffset(years=1)).all()
        # 봉 단위로 20봉 이상 비움: 일봉 신호는 20거래일(28일 달력) 이상, 주봉 신호는 20주 이상 전
        d_gap = (start - tr.loc[tr["is_weekly"] == 0, "date"].max()).days
        w_gap = (start - tr.loc[tr["is_weekly"] == 1, "date"].max()).days
        assert d_gap > 28 and w_gap > 140
        # 내부 검증도 같은 규칙
        itr, val = W.inner_masks(df, train, start)
        assert df.loc[itr, "exit20"].max() < df.loc[val, "date"].min()
        assert not (itr & val).any() and not (val & ~train).any()


def test_fold_excludes_non_explore_rows():
    df, _ = synth_ds()
    df.loc[df["date"] > "2015-06-01", "period"] = -1
    for _, _, train, test in W.fold_masks(df):
        assert (df.loc[train | test, "period"] == 0).all()


def test_confirm_mode_trains_on_explore_only():
    df, _ = synth_ds(n_days=2600)
    df["period"] = np.where(df["date"] >= H.BOUNDARY, 1, 0)
    df.loc[(df["period"] == 0) & (df["exit20"] >= H.BOUNDARY), "period"] = -1
    (name, start, train, test), = W.fold_masks(df, mode="confirm")
    assert df.loc[train, "exit20"].max() < H.BOUNDARY <= df.loc[test, "date"].min()


# ───────────────────────── (a)(b) 합성 헛발견·검출력 ─────────────────────────
@pytest.mark.parametrize("seed", [1, 2])
def test_null_synthetic_no_pass(seed):
    df, feats = synth_ds(seed=seed)
    preds, summ, folds, _ = W.walk_forward(df, ("zero", "ridge", "hgb"), (0,), feats, log=lambda *_: None)
    for _, r in summ.iterrows():
        assert not r["passed"], r.to_dict()
        assert abs(r["ic"]) < 0.03
    # 합성 y 평균 0, 비용 0.4% → 상위 20% 비용 후는 음수 근처
    assert (summ["top_net"] < 0.01).all()


def test_leak_synthetic_passes():
    df, feats = synth_ds(seed=3, leak_corr=0.3)
    df["y"] += 0.0   # 그대로
    preds, summ, folds, _ = W.walk_forward(df, ("ridge", "hgb"), (0,), feats, log=lambda *_: None)
    for _, r in summ.iterrows():
        assert r["ic"] > 0.15, r.to_dict()
        assert r["passed"], r.to_dict()


def test_evaluate_zero_model_is_all_signals():
    df, _ = synth_ds(n_days=300)
    r = W.evaluate(df, np.zeros(len(df)), np.full(len(df), "x"))
    assert r["top_n"] == len(df) and r["ic"] == 0 and not r["passed"]
    assert np.isclose(r["top_net"], (df["y"] - df["cost20"]).mean())


def test_top20_is_within_fold():
    df, _ = synth_ds(n_days=600)
    fold = np.where(df["date"] < "2011-01-01", "a", "b")
    pred = np.where(fold == "a", 10.0, 0.0) + np.random.default_rng(0).random(len(df))
    r = W.evaluate(df, pred, fold)
    assert abs(r["top_n"] - 0.2 * len(df)) <= 2


def test_date_ic_perfect_and_reverse():
    d = np.repeat(pd.bdate_range("2012-01-02", periods=50).values, 10)
    y = np.random.default_rng(0).normal(size=len(d))
    assert np.isclose(W.date_ic(d, y, y)["ic"].mean(), 1.0)
    assert np.isclose(W.date_ic(d, -y, y)["ic"].mean(), -1.0)


# ───────────────────────── (c) 특징 잘라내기 (합성 종목) ─────────────────────────
def _stock(n=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n).values.astype("datetime64[ns]")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * np.exp(rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + rng.random(n) * 0.01)
    l = np.minimum(o, c) * (1 - rng.random(n) * 0.01)
    v = rng.uniform(1e8, 1e9, n)
    v[rng.random(n) < 0.03] = np.nan
    return H.Stock("S", dates, o, h, l, c), v


@pytest.mark.parametrize("tf", ["1d", "1w"])
def test_stock_features_truncation(tf):
    s, v = _stock()
    full = D.stock_bar_features(s, v, tf)
    _, end = H.bars_of(s, tf)
    rng = np.random.default_rng(1)
    for k in rng.choice(np.arange(5, len(end)), 40, replace=False):
        cut = end[k] + 1
        st = H.Stock(s.sym, s.dates[:cut], s.o[:cut], s.h[:cut], s.l[:cut], s.c[:cut])
        tr = D.stock_bar_features(st, v[:cut], tf)
        assert len(tr) == k + 1
        a = full.iloc[k][D.STOCK_FEATS + ["atr", "close"]].to_numpy(float)
        b = tr.iloc[-1][D.STOCK_FEATS + ["atr", "close"]].to_numpy(float)
        assert D._same(a, b).all(), (k, a, b)


def test_index_lookup_asof_and_stale():
    full = pd.Series(np.arange(300, dtype=float) + 100, index=pd.bdate_range("2012-01-02", periods=300))
    idx = full.drop(full.index[251])                                  # 지수 휴장일
    tab = D.index_table(idx)
    q = np.array([full.index[251], idx.index[-1] + pd.Timedelta(days=30)], dtype="datetime64[ns]")
    out = D.lookup_index(tab, q)
    assert out["idx_ret60"].iloc[0] == tab["idx_ret60"].iloc[250]   # 휴일 → 직전 값
    assert np.isnan(out["idx_ret60"].iloc[1])                       # 오래된 값은 쓰지 않음


def test_remove_drift_zero_mean_and_shapes():
    s, _ = _stock(seed=4)
    s2 = D.remove_drift(s)
    g = s2.c[1:] / s2.c[:-1]
    assert abs(g.mean() - 1) < 1e-12
    assert np.allclose(s2.h / s2.c, s.h / s.c) and np.allclose(s2.o[1:] / s2.c[:-1], s.o[1:] / s.c[:-1])
    n = D.null_stock(s, 5)
    assert len(n.c) == len(s.c) and (n.c > 0).all()


# ───────────────────────── 조립 (합성 시장, 실제 탐지기) ─────────────────────────
def _wide(n_sym=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2011-01-03", "2016-12-30")
    syms = [f"S{i:03d}" for i in range(n_sym)]
    r = rng.normal(0.0003, 0.02, (len(idx), n_sym))
    c = 100 * np.exp(np.cumsum(r, 0))
    o = c * np.exp(rng.normal(0, 0.005, c.shape))
    w = {"open": o, "close": c, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99}
    w = {k: pd.DataFrame(v, idx, syms) for k, v in w.items()}
    w["market_cap"] = pd.DataFrame(rng.uniform(1e10, 1e12, c.shape), idx, syms)
    w["value"] = pd.DataFrame(rng.uniform(2e9, 5e9, c.shape), idx, syms)
    w["tradable"] = pd.DataFrame(True, idx, syms)
    return w


def test_build_market_target_is_xs_demeaned():
    w = _wide()
    idx = w["close"].mean(1)
    df, info = D.build_market("kr_stock", w, idx, (0,), workers=1, timeframes=("1d",))
    assert len(df) > 50
    assert set(D.FEATURES) <= set(df.columns)
    # y = ret20 − 그날 횡단면 평균 → 같은 날 행끼리 y − ret20 이 같다
    g = (df["ret20"] - df["y"]).groupby(df["date"]).agg(lambda x: x.max() - x.min())
    assert (g < 1e-12).all()
    assert np.allclose(df["xs_mean"], df["ret20"] - df["y"])
    assert (df["xs_n"] >= D.MIN_XS).all()
    # 잘라내기 검사 (합성)
    res = D.check_truncation("kr_stock", w, idx, df, n_events=60, n_cuts=2)
    assert res["ok"], res
