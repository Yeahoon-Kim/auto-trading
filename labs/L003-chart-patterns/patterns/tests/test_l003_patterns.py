"""L003 A0 탐지기 빠른 검사 (pytest). 전체 verifier는 patterns/verify.py.

    uv run pytest labs/L003-chart-patterns/patterns/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patterns import COLUMNS, TOOLS, detect, to_weekly  # noqa: E402
from patterns import synth  # noqa: E402

PLANTS = [
    ("head_shoulders", "ihs", "ihs", False), ("head_shoulders", "hs", "ihs", True),
    ("double_triple", "double_bottom", "double_bottom", False),
    ("double_triple", "double_top", "double_bottom", True),
    ("double_triple", "triple_bottom", "triple_bottom", False),
    ("double_triple", "triple_top", "triple_bottom", True),
    ("fib_retrace", "bull_retrace", "bull_retrace", False),
    ("fib_retrace", "bear_retrace", "bull_retrace", True),
    ("trendline", "up_break", "up_break", False), ("trendline", "down_break", "up_break", True),
]


@pytest.mark.parametrize("tool", TOOLS)
def test_interface(tool):
    d = synth.random_walk(1200, seed=1)
    ev = detect(d, tool, "1d")
    assert list(ev.columns) == COLUMNS
    assert set(ev["direction"].unique()) <= {1, -1}
    assert ev["date"].isin(d.index).all()
    assert (ev["pattern_start"] <= ev["date"]).all()
    empty = detect(d.iloc[:20], tool, "1d")
    assert list(empty.columns) == COLUMNS and len(empty) == 0


@pytest.mark.parametrize("tf", ["1d", "1w"])
@pytest.mark.parametrize("tool,variant,kind,mir", PLANTS)
def test_recall(tool, variant, kind, mir, tf):
    hit = tot = 0
    for s in range(25):
        r = synth.plant(kind, tf, seed=10_000 + 31 * s, for_mirror=mir)
        if r is None:
            continue
        df, comp, _ = r
        if mir:
            df = synth.mirror(df)
        ev = detect(df, tool, tf)
        pos = np.searchsorted(df.index.to_numpy(), ev.loc[ev.variant == variant, "date"].to_numpy())
        tot += 1
        hit += bool(np.any(np.abs(pos - comp) <= 3))
    assert tot >= 20 and hit / tot >= 0.9, (hit, tot)


@pytest.mark.parametrize("tf", ["1d", "1w"])
@pytest.mark.parametrize("tool", TOOLS)
def test_truncation(tool, tf):
    rng = np.random.default_rng(5)
    d = synth.random_walk(2500, seed=2)
    bars = d if tf == "1d" else to_weekly(d)
    full = detect(bars, tool, tf)
    pos = np.searchsorted(bars.index.to_numpy(), full["date"].to_numpy())
    ts = list(rng.integers(30, len(bars) - 1, 8)) + list(pos[:6]) + list(pos[:6] - 1)
    for t in ts:
        cut = bars.index[t]
        tr = detect(bars.iloc[:t + 1], tool, tf)
        a = tr[tr.date <= cut].reset_index(drop=True)
        b = full[full.date <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_weekly():
    d = synth.random_walk(30, seed=3)
    w = to_weekly(d)
    assert w.index.isin(d.index).all()
    assert w["high"].iloc[0] == d["high"].iloc[:len(d.loc[:w.index[0]])].max()
