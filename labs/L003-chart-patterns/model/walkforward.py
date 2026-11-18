"""L003 라운드 2 워크포워드 학습·평가.

창과 purge
  - 탐색: 연 단위 확장 창. 예측 연도 Y(2013~)마다 학습 = 20봉 목표값의 청산일(exit20)이 Y년 1월 1일 **전**인 신호.
    → 일봉 신호는 예측 시작 20거래일 이상 전, 주봉 신호는 20주 이상 전 (각자 봉 단위로 '학습 끝과 예측 시작 사이 20봉 이상').
    목표값이 예측 구간 안으로 넘어오는 학습 행은 하나도 없다 (`fold_masks`, tests의 purge 검사).
  - 하이퍼파라미터: 학습 창 안에서만. 내부 검증 = 예측 시작 직전 1년의 신호, 내부 학습 = 청산일이 그 1년 시작 전인 신호.
    내부 검증 날짜별 IC 평균이 가장 큰 값을 고르고 학습 창 전체로 다시 학습한다.
  - 확인(confirm): 학습 = 탐색 구간 전체(period 0), 예측 = 확인 구간(period 1) 한 번. 모델 1개·시드 0 (사전등록).

모델 (경쟁)
  zero  : 예측 0 = 고르지 않음. IC는 정의상 0, '상위 20%' = 전체 신호.
  ridge : 수치 특징을 학습 창 1~99% 분위로 자르고, 중앙값으로 채우고(결측 표시 열 추가), 표준화 + 도구·변형 원핫. 결정적이라 시드 영향 없음.
  hgb   : HistGradientBoostingRegressor (결측 그대로, 도구·변형은 범주). max_features 0.8 → 시드가 결과를 바꾼다.
  목표값은 학습 창 1~99% 분위로 잘라 학습한다 (꼬리 몇 개가 적합을 지배하지 않게). 평가는 자르지 않은 y.

지표 (예측 구간 전체를 이어 붙여서)
  - IC: 날짜별(달력 날짜, 시장·봉 합침) 스피어만 순위 상관, 신호 ≥ 5개인 날의 평균.
  - 상위 20%: 예측 연도(fold)마다 예측값 백분위 > 0.8 인 신호. 초과수익 = y 평균, 비용 후 = (y − 왕복 비용) 평균.
    비용 = study 하네스 Costs (청산일 기준 KR 매도세 포함) — 데이터셋 cost20.
  - p: 날짜 묶음(같은 달력 날짜) / 20봉 블록(일봉 28일, 주봉 140일 달력 구간) 부트스트랩 (H.cluster_bootstrap, 양측 정규 p).
  - 통과(탐색 verifier용): 상위 20% 표본 ≥ 100 ∧ 비용 후 > 0 ∧ 두 p < 0.05 ∧ IC > 0.
    확인 구간 통과는 여기에 'IC ≥ 탐색 워크포워드 IC의 50%'가 더해진다 (README 3절).

라운드 3 H4 (horizon h = 60): 모든 함수가 horizon 인자를 받는다 (기본 20 = 라운드 2 그대로).
  학습 행 = exit{h} < 예측 시작 (60봉 purge), 비용 = cost{h}, 블록 = 일봉 28·h/20일, 주봉 140·h/20일 (60봉 → 84일·420일).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset as D  # noqa: E402

H = D.H
FIRST_TEST_YEAR = 2013
VAL_DAYS = 365
TOP_Q = 0.8
MIN_IC_N = 5
N_BOOT = 2000
MIN_TOP_N = 100
ALPHA = 0.05
RIDGE_GRID = [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0, 10000.0)]
HGB_BASE = dict(loss="squared_error", learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                max_features=0.8, early_stopping=False)
HGB_GRID = [{"max_leaf_nodes": 7, "min_samples_leaf": 500},
            {"max_leaf_nodes": 15, "min_samples_leaf": 1000},
            {"max_leaf_nodes": 31, "min_samples_leaf": 2000}]


# ───────────────────────── 창 ─────────────────────────
def train_mask_before(df: pd.DataFrame, start, horizon: int = 20) -> np.ndarray:
    """목표값이 start 전에 다 정해진 행 (h봉 청산일 < start)."""
    return (df[f"exit{horizon}"].values < np.datetime64(pd.Timestamp(start))) & (df["period"].values == 0)


def fold_masks(df: pd.DataFrame, first_year: int = FIRST_TEST_YEAR, mode: str = "explore", horizon: int = 20):
    """(이름, 예측 시작, 학습 mask, 예측 mask) 목록."""
    out = []
    if mode == "explore":
        yr = pd.DatetimeIndex(df["date"]).year.values
        ex = df["period"].values == 0
        for Y in range(first_year, int(yr[ex].max()) + 1):
            start = pd.Timestamp(f"{Y}-01-01")
            test = ex & (yr == Y)
            train = train_mask_before(df, start, horizon)
            if test.sum() and train.sum():
                out.append((str(Y), start, train, test))
    elif mode == "confirm":
        out.append(("confirm", H.BOUNDARY, train_mask_before(df, H.BOUNDARY, horizon), df["period"].values == 1))
    else:
        raise ValueError(mode)
    return out


def inner_masks(df: pd.DataFrame, train: np.ndarray, start, horizon: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """학습 창 안의 내부 검증: 예측 시작 직전 VAL_DAYS일의 신호 / 내부 학습: 청산일이 그 전인 신호."""
    vs = pd.Timestamp(start) - pd.Timedelta(days=VAL_DAYS)
    val = train & (df["date"].values >= np.datetime64(vs))
    itr = train & (df[f"exit{horizon}"].values < np.datetime64(vs))
    return itr, val


# ───────────────────────── 특징 행렬 ─────────────────────────
@dataclass
class Prep:
    lo: np.ndarray
    hi: np.ndarray
    med: np.ndarray
    miss_cols: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    cats: list
    ylo: float
    yhi: float


def fit_prep(df: pd.DataFrame, features: list[str], cats: list) -> Prep:
    X = df[features].to_numpy(float)
    with np.errstate(all="ignore"):
        lo, hi = np.nanquantile(X, 0.01, axis=0), np.nanquantile(X, 0.99, axis=0)
        med = np.nanmedian(np.clip(X, lo, hi), axis=0)
    lo, hi, med = (np.nan_to_num(v, nan=0.0) for v in (lo, hi, med))
    miss = np.flatnonzero(np.isnan(X).any(0))
    Z = _ridge_x(X, df[D.CAT_FEATURE].values, lo, hi, med, miss, None, None, cats)
    mu, sd = Z.mean(0), Z.std(0)
    sd[sd == 0] = 1.0
    y = df["y"].to_numpy(float)
    return Prep(lo, hi, med, miss, mu, sd, cats, *np.quantile(y, [0.01, 0.99]))


def _ridge_x(X, cat, lo, hi, med, miss, mu, sd, cats):
    Xc = np.clip(X, lo, hi)
    M = np.isnan(Xc[:, miss]).astype(float)
    Xc = np.where(np.isnan(Xc), med, Xc)
    oh = (np.asarray(cat)[:, None] == np.asarray(cats)[None, :]).astype(float)
    Z = np.hstack([Xc, M, oh])
    if mu is not None:
        Z = (Z - mu) / sd
    return Z


def ridge_x(df, features, p: Prep):
    return _ridge_x(df[features].to_numpy(float), df[D.CAT_FEATURE].values, p.lo, p.hi, p.med, p.miss_cols, p.mu, p.sd, p.cats)


def hgb_x(df, features, cats):
    X = df[features].to_numpy(float)
    code = pd.Categorical(df[D.CAT_FEATURE].values, categories=cats).codes.astype(float)
    code[code < 0] = np.nan
    return np.hstack([X, code[:, None]])


def fit_predict(kind: str, hp: dict, tr: pd.DataFrame, te: pd.DataFrame, features: list[str], cats: list, seed: int):
    if kind == "zero":
        return np.zeros(len(te))
    p = fit_prep(tr, features, cats)
    y = np.clip(tr["y"].to_numpy(float), p.ylo, p.yhi)
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=hp["alpha"]).fit(ridge_x(tr, features, p), y)
        return m.predict(ridge_x(te, features, p))
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        cat_mask = np.r_[np.zeros(len(features), bool), True]
        m = HistGradientBoostingRegressor(**HGB_BASE, **hp, categorical_features=cat_mask, random_state=seed)
        m.fit(hgb_x(tr, features, cats), y)
        return m.predict(hgb_x(te, features, cats))
    raise ValueError(kind)


GRIDS = {"zero": [{}], "ridge": RIDGE_GRID, "hgb": HGB_GRID}


# ───────────────────────── 지표 ─────────────────────────
def date_ic(date: np.ndarray, pred: np.ndarray, y: np.ndarray, min_n: int = MIN_IC_N) -> pd.DataFrame:
    """날짜별 스피어만 IC (신호 ≥ min_n, 예측이 상수가 아닌 날)."""
    d = pd.DataFrame({"date": pd.DatetimeIndex(date).normalize(), "p": pred, "y": y})
    d = d[np.isfinite(d["p"]) & np.isfinite(d["y"])]
    g = d.groupby("date")
    n = g["p"].transform("size")
    d = d[n >= min_n].copy()
    if d.empty:
        return pd.DataFrame(columns=["date", "ic", "n"])
    gd = d.groupby("date")
    rp = gd["p"].rank().to_numpy()
    ry = gd["y"].rank().to_numpy()
    # 순위의 피어슨 상관 = 스피어만 (동순위는 평균 순위). 날짜별 합으로 한 번에 계산
    t = pd.DataFrame({"date": d["date"].values, "a": rp, "b": ry, "aa": rp * rp, "bb": ry * ry, "ab": rp * ry})
    s = t.groupby("date").agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"), aa=("aa", "sum"), bb=("bb", "sum"), ab=("ab", "sum"))
    n = s["n"].to_numpy(float)
    cov = s["ab"] - s["a"] * s["b"] / n
    va = s["aa"] - s["a"] ** 2 / n
    vb = s["bb"] - s["b"] ** 2 / n
    with np.errstate(all="ignore"):
        ic = cov / np.sqrt(va * vb)
    ic = ic.where((va > 1e-12) & (vb > 1e-12))
    out = pd.DataFrame({"date": s.index.values, "ic": ic.to_numpy(float), "n": s["n"].to_numpy()})
    return out.dropna(subset=["ic"]).reset_index(drop=True)


def block_key(date: np.ndarray, weekly: np.ndarray, horizon: int = 20) -> np.ndarray:
    """h봉 블록: 일봉 = 28·h/20일 달력 구간, 주봉 = 140·h/20일 달력 구간 (시장은 합친다). 20봉 → 28·140일, 60봉 → 84·420일."""
    day = np.asarray(date, dtype="datetime64[D]").astype(np.int64)
    bd, bw = 28 * horizon // 20, 140 * horizon // 20
    return np.where(np.asarray(weekly) > 0.5, 10_000_000 + day // bw, day // bd)


def evaluate(te: pd.DataFrame, pred: np.ndarray, fold: np.ndarray, seed: int = 0, horizon: int = 20) -> dict:
    """te: 예측 구간 행(y, cost{h}, date, is_weekly). fold: 행마다 fold 이름 (상위 20% 기준을 fold 안에서 정한다)."""
    y = te["y"].to_numpy(float)
    cost = te[f"cost{horizon}"].to_numpy(float)
    date = te["date"].values
    wk = te["is_weekly"].to_numpy(float)
    r = {"n_test": len(te)}
    if np.nanstd(pred) == 0:
        top = np.ones(len(te), bool)          # 예측 0 = 고르지 않음
        r |= {"ic": 0.0, "ic_dates": 0, "ic_p_date": np.nan, "ic_p_blk": np.nan}
    else:
        pct = pd.Series(pred).groupby(np.asarray(fold)).rank(pct=True).to_numpy()
        top = pct > TOP_Q
        ic = date_ic(date, pred, y)
        r["ic"] = float(ic["ic"].mean()) if len(ic) else np.nan
        r["ic_dates"] = len(ic)
        if len(ic) > 2:
            icd = ic["date"].values
            icw = np.zeros(len(ic))   # IC 날짜는 시장·봉을 합친 날이라 일봉 블록(28일)으로 묶는다
            r["ic_p_date"] = H.cluster_bootstrap(ic["ic"].to_numpy(float), icd.astype("datetime64[D]").astype(np.int64), seed, N_BOOT)["p"]
            r["ic_p_blk"] = H.cluster_bootstrap(ic["ic"].to_numpy(float), block_key(icd, icw, horizon), seed, N_BOOT)["p"]
        else:
            r["ic_p_date"] = r["ic_p_blk"] = np.nan
    d = y[top] - cost[top]
    r["top_n"] = int(top.sum())
    r["top_exc"] = float(np.mean(y[top])) if top.any() else np.nan
    r["top_cost"] = float(np.mean(cost[top])) if top.any() else np.nan
    r["top_net"] = float(np.mean(d)) if top.any() else np.nan
    if top.sum() > 2:
        dc = np.asarray(date[top], dtype="datetime64[D]").astype(np.int64)
        r["top_p_date"] = H.cluster_bootstrap(d, dc, seed, N_BOOT)["p"]
        r["top_p_blk"] = H.cluster_bootstrap(d, block_key(date[top], wk[top], horizon), seed, N_BOOT)["p"]
    else:
        r["top_p_date"] = r["top_p_blk"] = np.nan
    r["all_exc"] = float(np.mean(y))
    r["all_net"] = float(np.mean(y - cost))
    r["passed"] = bool(r["top_n"] >= MIN_TOP_N and r["top_net"] > 0 and r["top_p_date"] < ALPHA
                       and r["top_p_blk"] < ALPHA and r["ic"] > 0)
    return r


# ───────────────────────── 실행 ─────────────────────────
def select_hp(kind, df, train, start, features, cats, seed, horizon: int = 20):
    grid = GRIDS[kind]
    if len(grid) == 1:
        return grid[0], []
    itr, val = inner_masks(df, train, start, horizon)
    if itr.sum() < 500 or val.sum() < 100:
        return grid[0], [{"note": f"내부 검증 표본 부족 (학습 {int(itr.sum())}, 검증 {int(val.sum())}) → 첫 값"}]
    tr, va = df[itr], df[val]
    scores = []
    for hp in grid:
        p = fit_predict(kind, hp, tr, va, features, cats, seed)
        ic = date_ic(va["date"].values, p, va["y"].to_numpy(float))
        scores.append(float(ic["ic"].mean()) if len(ic) else -np.inf)
    s = np.nan_to_num(np.array(scores), nan=-np.inf)
    return grid[int(np.argmax(s))], [{"hp": hp, "val_ic": sc} for hp, sc in zip(grid, scores)]


def walk_forward(df: pd.DataFrame, models=("zero", "ridge", "hgb"), seeds=(0, 1, 2, 3, 4), features=None,
                 mode: str = "explore", first_year: int = FIRST_TEST_YEAR, log=print, horizon: int = 20):
    """반환: (예측 DataFrame, 요약 DataFrame, fold별 DataFrame, 하이퍼파라미터 기록)."""
    features = list(features or D.NUM_FEATURES)
    hz = horizon
    df = df.sort_values(["date", "market", "timeframe", "sym", "tool", "variant"]).reset_index(drop=True)
    cats = sorted(df[D.CAT_FEATURE].unique())
    folds = fold_masks(df, first_year, mode, hz)
    pred_cols, hp_log, fold_rows = {}, [], []
    test_any = np.zeros(len(df), bool)
    fold_of = np.full(len(df), "", dtype=object)
    for name, start, train, test in folds:
        test_any |= test
        fold_of[test] = name
    for kind in models:
        for seed in (seeds if kind == "hgb" else seeds[:1]):
            t0 = time.time()
            pred = np.full(len(df), np.nan)
            for name, start, train, test in folds:
                hp, tried = select_hp(kind, df, train, start, features, cats, seed, hz)
                tr, te = df[train], df[test]
                pred[test] = fit_predict(kind, hp, tr, te, features, cats, seed)
                hp_log.append({"model": kind, "seed": seed, "fold": name, "n_train": int(train.sum()),
                               "n_test": int(test.sum()), "train_last_exit": str(df.loc[train, f"exit{hz}"].max()),
                               "test_first_date": str(df.loc[test, "date"].min()), "chosen": hp, "tried": tried})
                fr = evaluate(te, pred[test], np.full(test.sum(), name), seed, hz)
                fold_rows.append({"model": kind, "seed": seed, "fold": name, **fr})
            pred_cols[(kind, seed)] = pred
            log(f"  {kind} seed {seed}: {time.time() - t0:.1f}초")
    te = df[test_any]
    rows = []
    for (kind, seed), pred in pred_cols.items():
        r = evaluate(te, pred[test_any], fold_of[test_any], seed, hz)
        rows.append({"model": kind, "seed": seed, **r})
    summ = pd.DataFrame(rows)
    # 결정적 모델(zero·ridge)은 시드 1개만 돌린다 → 다른 시드 행은 같은 값 (noise floor 0)
    preds = df.loc[test_any, ["market", "timeframe", "sym", "date", "tool", "variant", "direction", "y", f"cost{hz}",
                              f"ret{hz}", f"exit{hz}", "is_weekly"]].copy()
    preds["fold"] = fold_of[test_any]
    for (kind, seed), pred in pred_cols.items():
        preds[f"pred_{kind}_s{seed}"] = pred[test_any]
    return preds, summ, pd.DataFrame(fold_rows), hp_log


def noise_floor(summ: pd.DataFrame) -> pd.DataFrame:
    g = summ.groupby("model")
    return pd.DataFrame({"seeds": g.size(), "ic_mean": g["ic"].mean(), "ic_sd": g["ic"].std(ddof=1),
                         "top_net_mean": g["top_net"].mean(), "top_net_sd": g["top_net"].std(ddof=1),
                         "passed": g["passed"].sum()}).reset_index()


def breakdown(preds: pd.DataFrame, col: str, horizon: int = 20) -> pd.DataFrame:
    """시장·봉별 참고 지표 (seed 0)."""
    rows = []
    for (m, tf), g in preds.groupby(["market", "timeframe"]):
        r = evaluate(g, g[col].to_numpy(float), g["fold"].values, horizon=horizon)
        rows.append({"market": m, "timeframe": tf, **{k: r[k] for k in ("n_test", "ic", "ic_dates", "top_n", "top_exc", "top_net", "top_p_date", "top_p_blk")}})
    return pd.DataFrame(rows)
