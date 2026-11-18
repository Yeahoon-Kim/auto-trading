"""L003 R3 국면 표 단위 검사 (합성 원천, 실제 데이터 불필요).

    uv run pytest labs/L003-chart-patterns/regime/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import regime as RG  # noqa: E402
from autotrader.intel import market as M  # noqa: E402


def synth_src(seed=0, start="2010-01-04", end="2015-12-31"):
    rng = np.random.default_rng(seed)
    us = pd.bdate_range(start, end)
    kr = us[rng.random(len(us)) > 0.04]                     # 한국 휴장일이 다르게
    def px(idx, s):
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, s, len(idx)))), idx)
    fri = us[us.dayofweek == 4]
    return {"SPX": px(us, 0.01), "KOSPI": px(kr, 0.012), "VIX": px(us, 0.05) / 5, "VIX3M": px(us, 0.03) / 5,
            "USDKRW": px(us, 0.005) * 11, "DXY": px(us, 0.004), "T10Y2Y": pd.Series(rng.normal(1, 0.3, len(us)), us),
            "NFCI": pd.Series(rng.normal(-0.5, 0.2, len(fri)), fri)}


def test_uses_market_definitions():
    src = synth_src()
    t = RG.table(None, src)
    ref = M._trend_block(src["KOSPI"], "kospi")
    obs = t["kospi_observed"].to_numpy(bool)
    for c in ref.columns:
        a = t.loc[obs, c].astype(float).to_numpy()
        b = ref.reindex(t.index[obs])[c].astype(float).to_numpy()
        assert np.allclose(a, b, equal_nan=True), c
    # 라벨은 _labels 그대로 (관측일·입력 정의된 행)
    ok = obs & t["kospi_trend_ok"].to_numpy(bool)
    up = (src["KOSPI"] > src["KOSPI"].rolling(200).mean()) & (src["KOSPI"].pct_change(20) > 0)
    assert (t.loc[ok, "kospi_trend"].eq("상승").to_numpy() == up.reindex(t.index[ok]).to_numpy()).all()


def test_lookup_is_shift1_and_nfci_lag():
    src = synth_src()
    t = RG.table(None, src)
    rows = t[t["spx_observed"]]
    d = rows.index[400:420]
    lk = RG.lookup(t, d.values, "us_stock")
    # 신호일 t의 값 = 직전 관측일(t−1 거래일)의 값
    prev = rows.index[399:419]
    assert np.allclose(lk["rv_pctile"].to_numpy(float), rows.loc[prev, "spx_rv_pctile"].to_numpy(float), equal_nan=True)
    assert np.allclose(lk["vix"].to_numpy(float), src["VIX"].reindex(prev).to_numpy())
    # NFCI: 금요일 관측 → 다음 주 목요일(관측 + 6일)부터 보인다
    nf = src["NFCI"]
    fri = nf.index[100]
    q = np.array([fri + pd.Timedelta(days=k) for k in (1, 3, 5, 6, 7)], dtype="datetime64[ns]")
    v = RG.lookup(t, q, "us_stock")["nfci"].to_numpy(float)
    prev_fri = nf.index[99]
    assert np.allclose(v[:3], nf[prev_fri]) and np.allclose(v[3:], nf[fri])   # 토·월·수 = 지난주 값, 목·금 = 이번 값


def test_lookup_never_uses_same_day():
    """같은 날 값을 바꿔도 그날 조회값은 그대로 (관측일 < 신호일)."""
    src = synth_src()
    t1 = RG.table(None, src)
    day = src["SPX"].index[700]
    src2 = {k: v.copy() for k, v in src.items()}
    for k in ("SPX", "KOSPI", "VIX", "VIX3M", "USDKRW", "DXY", "T10Y2Y"):
        if day in src2[k].index:
            src2[k].loc[day] *= 1.5
    t2 = RG.table(None, src2)
    for mk in ("kr_stock", "us_stock"):
        a, b = RG.lookup(t1, np.array([day], dtype="datetime64[ns]"), mk), RG.lookup(t2, np.array([day], dtype="datetime64[ns]"), mk)
        for c in RG.LOOKUP_COLS:
            assert RG._eq(pd.Series(a[c]), pd.Series(b[c])).all(), (mk, c)


def test_unknown_labels_are_nan_and_stale():
    src = synth_src()
    t = RG.table(None, src)
    q = pd.bdate_range("2010-01-05", "2010-06-30").values      # 200일선 전
    lk = RG.lookup(t, q, "kr_stock")
    assert pd.isna(lk["trend"]).all() and pd.isna(lk["vol"]).all()
    far = np.array([src["SPX"].index[-1] + pd.Timedelta(days=30)], dtype="datetime64[ns]")
    lk2 = RG.lookup(t, far, "us_stock")
    assert all(pd.isna(lk2[c]).all() for c in RG.LOOKUP_COLS)


def test_truncation_check_synthetic():
    src = synth_src()
    cuts = [pd.Timestamp("2012-03-07"), pd.Timestamp("2014-11-20")]
    r = RG.check_truncation(src, cuts, end=None)
    assert r["ok"] and r["table_cells"] > 0 and r["lookup_cells"] > 0, r


def test_truncation_check_catches_leak(monkeypatch):
    """미래를 보는 정의(중심 이동평균)를 넣으면 잘라내기 검사가 잡아야 한다."""
    src = synth_src()
    orig = M._trend_block

    def leaky(px, prefix):
        out = orig(px, prefix)
        out[f"{prefix}_ret_20d"] = px.shift(-5) / px - 1
        return out
    monkeypatch.setattr(M, "_trend_block", leaky)
    r = RG.check_truncation(src, [pd.Timestamp("2013-06-05")], end=None)
    assert not r["ok"] and r["table_mismatch"] > 0


def test_features_encoding():
    src = synth_src()
    t = RG.table(None, src)
    q = pd.bdate_range("2012-01-02", "2015-12-30").values
    F = RG.features(t, q, "kr_stock")
    assert list(F.columns) == RG.FEATURES
    assert set(np.unique(F["rg_trend"].dropna())) <= {-1.0, 0.0, 1.0}
    assert set(np.unique(F["rg_vol"].dropna())) <= {0.0, 1.0, 2.0}
    assert F.notna().mean().min() > 0.9
