"""US 분할 보정 주식수·시가총액 (시점 정보 포함). 수집기가 아니라 저장된 데이터를 읽어 계산하는 도우미.

입력: edgar_facts(shares_* 항목), edgar_class_shares(클래스별 표지 주식수), us_splits(분할 이력), us_1d(종가).
출력:
  share_events()  공시(접수번호)마다 주식수 하나: symbol, cik, accn, form, filed, asof, source, shares_raw,
                  factor(분할 보정 배수), shares(= us_1d 가격과 같은 오늘 기준 주식수), flag.
  daily_shares()  us_1d 봉마다 그날 알 수 있던 주식수 + close, market_cap.
  quality()       종목별 결측·이상치 비율 (메타).

규칙 (docs/data.md "US 주식수·시가총액" 참고):
- 시점: 공시일(filed) 다음 날부터 쓴다 → 그날 장 마감 뒤 공시도 미래참조 없음. 공시가 max_age_days보다 오래되면 NaN.
- 후보 우선순위(공시 하나 안에서): 클래스 합계(edgar_class_shares 대상 회사만, 그 회사는 이것만 씀) >
  dei 표지 주식수(EntityCommonStockSharesOutstanding) > 재무상태표 발행주식수(CommonStockSharesOutstanding) >
  기본 가중평균(…SharesOutstandingBasic) > 희석 가중평균(…DilutedSharesOutstanding). 같은 공시에 기간이 여럿이면 가장 최근 기간.
- 분할 보정: 표지 값(dei·클래스)은 그 기준일(asof) 뒤 분할, 재무제표 값은 공시일 뒤 분할만큼 곱한다
  (재무제표는 공시 전 분할을 소급 반영해 내는 것이 규정 — SAB Topic 4C). 공시 직후 분할을 미리 반영한 회사가 있어
  (ETN 2011) 공시일+15일 기준 값도 만들고 직전 값에 가까운 쪽을 쓴다. 가격(us_1d)도 오늘 기준 분할 반영이라
  close × shares = 그날 실제 시가총액. 분할 보정 배수는 미래 분할을 쓰지만 가격과 상쇄되므로 미래참조가 아니다.
- 분사형 배수: Yahoo는 분사(spin-off)도 소수 비율 '분할'로 적고 가격을 그만큼 소급 보정한다(ABT 2013 2.084).
  실제 주식수는 그대로이므로 공시 기준이 아니라 봉 날짜 기준(그 날 뒤 배수)으로 곱한다. 실제 분할인지는
  classify_splits가 EDGAR 원래 주식수 변화로 정한다.
- 이상치 (직전 값·그날 가격만 보므로 미래참조 없음):
  ① 공시일 종가 × 주식수가 MCAP_RANGE(5천만~10조 달러) 밖인 후보는 틀린 값 → 10^3·10^6·10^9로 나눠 범위에 들면 고쳐 쓴다
     (여럿이면 시총이 300억 달러에 가까운 쪽).
  ② 직전 채택값 3개 중앙값(R) 대비 0.5~2배면 채택. 벗어나도 같은 공시의 다른 태그와 15% 안에서 맞으면 채택
     (합병·대량 발행 같은 실제 변화). 직전 공시에서 버린 값과 15% 안에서 같으면 새 수준이 두 번 확인된 것 → 채택(level_change).
  ③ R 대비 10^±3·10^±6 배 단위 오류면 고쳐 채택(scale_fixed), 아니면 버린다(dropped).
- 클래스가 여럿인 회사: 회사 전체 주식수(상장 클래스 환산)를 쓴다 → GOOG·GOOGL은 둘 다 회사 전체 시총이 나온다.
  순위를 매길 때는 cik로 묶어 하나만 고른다 (엔진·전략의 몫).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from . import store

NY = "America/New_York"
SOURCES = ["class", "dei", "bs", "basic", "diluted"]  # 우선순위
TAGS = {"EntityCommonStockSharesOutstanding": "dei", "CommonStockSharesOutstanding": "bs",
        "WeightedAverageNumberOfSharesOutstandingBasic": "basic",
        "WeightedAverageNumberOfDilutedSharesOutstanding": "diluted"}
INSTANT_SOURCES = {"class", "dei"}  # 분할 보정 기준일 = asof (나머지는 filed)
ALT_DAYS = pd.Timedelta(days=15)  # 대안 기준일: 공시일 + 15일 (공시 직후 분할을 표지·재무제표에 미리 반영한 회사)


@lru_cache(maxsize=1)
def _cfg() -> dict:
    from .collectors.common import config
    return config().get("us_shares", {}) or {}


def _day(s: pd.Series) -> pd.Series:
    """UTC 시각 → 뉴욕 날짜(시간대 없는 자정)."""
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()


# ── 분할 ──
def load_splits(splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """us_splits → (symbol, day, ratio)."""
    sp = store.read("us_splits", "ALL") if splits is None else splits
    if sp is None or sp.empty:
        return pd.DataFrame({"symbol": pd.Series(dtype=str), "day": pd.Series(dtype="datetime64[ns]"),
                             "ratio": pd.Series(dtype=float)})
    return pd.DataFrame({"symbol": sp["symbol"].values, "day": _day(sp["ts"]).values,
                         "ratio": sp["ratio"].astype(float).values})


def is_standard(ratio: float, tol: float = 0.0015) -> bool:
    """보통 분할·병합 비율(p/q, q ≤ 4)인가. 아니면 Yahoo가 분사(spin-off)를 가격 보정 배수로 적은 것으로 본다
    (예: A 2014 1.398 = Keysight 분사, ABT 2013 2.084 = AbbVie 분사)."""
    for r in (ratio, 1 / ratio):
        for q in (1, 2, 3, 4):
            p = round(r * q)
            if p and abs(r * q / p - 1) < tol:
                return True
    return False


def classify_splits(splits: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """분할마다 실제 분할(true=True: 주식수가 ratio배로 바뀜)인지 분사형 가격 보정(False: 주식수 그대로)인지.
    raw(symbol, filed, asof, raw) = 공시된 원래 주식수. 분할 45일 전까지 공시된 마지막 값과 분할 뒤 기준일의 첫 값의
    비율이 ratio와 1 중 어디에 가까운지로 정한다 (발표 뒤 미리 반영한 공시를 피하려고 45일 전). 자료가 없거나
    비율이 1에 가까워(0.9~1.1) 구분이 안 되면 is_standard로 정한다. 분할 종류는 효력일 전에 공시되는 사실이라
    뒤 공시로 정해도 미래참조가 아니다."""
    sp = splits.copy()
    sp["true"] = [is_standard(r) for r in sp["ratio"]]
    if sp.empty or raw.empty:
        return sp
    by = {k: g for k, g in raw.groupby("symbol")}
    for i, r in sp.iterrows():
        g = by.get(r["symbol"])
        if g is None or 0.9 <= r["ratio"] <= 1.1:
            continue
        b = g[_day(g["filed"]) < r["day"] - pd.Timedelta(days=45)]
        a = g[_day(g["asof"]) >= r["day"]]
        if b.empty or a.empty:
            continue
        x = a.sort_values("filed")["raw"].iloc[0] / b.sort_values("filed")["raw"].iloc[-1]
        sp.at[i, "true"] = abs(math.log(x / r["ratio"])) < abs(math.log(x))
    return sp


def split_factor(symbols: pd.Series, ref_days: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """ref_day 뒤(초과) 효력이 난 분할 비율의 곱 → 그 날짜 기준 주식수를 오늘 기준으로 바꾸는 배수."""
    out = np.ones(len(symbols))
    if splits.empty:
        return out
    by = {s: g for s, g in splits.groupby("symbol")}
    sym, ref = symbols.to_numpy(), pd.to_datetime(ref_days).to_numpy()
    for i in range(len(out)):
        g = by.get(sym[i])
        if g is not None and not pd.isna(ref[i]):
            out[i] = float(np.prod(g["ratio"].to_numpy()[g["day"].to_numpy() > ref[i]]))
    return out


# ── 후보 ──
def _facts_candidates(facts: pd.DataFrame) -> pd.DataFrame:
    """edgar_facts → 공시(symbol, accn)·출처마다 후보 하나: 가장 최근 기간(end), 가중평균은 가장 짧은 기간."""
    f = facts[facts["tag"].isin(TAGS) & (facts["unit"] == "shares")].copy()
    if f.empty:
        return pd.DataFrame(columns=["symbol", "cik", "accn", "form", "filed", "asof", "source", "raw"])
    f["source"] = f["tag"].map(TAGS)
    for c in ("start", "end", "filed"):  # duckdb(시간대 없음, UTC)·parquet(UTC) 섞여도 되게
        f[c] = pd.to_datetime(f[c], utc=True)
    f["dur"] = (f["end"] - f["start"]).dt.days.fillna(0)
    f = f.sort_values(["end", "dur"], ascending=[False, True])
    f = f.drop_duplicates(["symbol", "accn", "source"])
    return pd.DataFrame({"symbol": f["symbol"], "cik": f["cik"].astype("int64"), "accn": f["accn"], "form": f["form"],
                         "filed": pd.to_datetime(f["filed"], utc=True), "asof": pd.to_datetime(f["end"], utc=True),
                         "source": f["source"], "raw": f["val"].astype(float)})


def _ratio(symbol: str, member: str, table: dict) -> tuple[float, bool]:
    """(환산비, 오늘 기준 단위인지). 키가 member 이름에 들어 있으면 그 비율(분할 보정 안 함), 없으면 (1, False)."""
    for k, v in (table.get(symbol) or {}).items():
        if k in member:
            return float(v), True
    return 1.0, False


def _class_candidates(cls: pd.DataFrame, cik_symbols: pd.DataFrame, splits: pd.DataFrame,
                      ratios: dict) -> pd.DataFrame:
    """edgar_class_shares → 공시·종목마다 클래스 합계 (상장 클래스 환산, 오늘 기준). raw = 환산 전 단순 합,
    shares = 환산·분할 보정 합 (factor = shares / raw)."""
    cols = ["symbol", "cik", "accn", "form", "filed", "asof", "source", "raw", "shares", "alt"]
    c = cls[cls["member"] != "∅"].dropna(subset=["val"]) if not cls.empty else cls
    if c is None or c.empty:
        return pd.DataFrame(columns=cols)
    c = c.merge(cik_symbols, on="cik")
    by_class = c["member"].str.contains("ClassOfStockAxis", regex=False)
    has_class = by_class.groupby([c["symbol"], c["accn"]]).transform("any")
    c = c[by_class | (~has_class & (c["member"] == ""))].copy()  # 클래스별 값이 있으면 그것만, 없으면 회사 전체 값
    rv = [_ratio(s, m, ratios) for s, m in zip(c["symbol"], c["member"])]
    c["ratio"], c["today_unit"] = [r for r, _ in rv], [t for _, t in rv]
    c["asof"] = pd.to_datetime(c["asof"], utc=True).fillna(pd.to_datetime(c["filed"], utc=True))
    f = split_factor(c["symbol"], _day(c["asof"]), splits)
    f2 = split_factor(c["symbol"], _day(c["filed"]) + ALT_DAYS, splits)
    c["adj"] = c["val"] * c["ratio"] * np.where(c["today_unit"], 1.0, f)
    c["adj2"] = c["val"] * c["ratio"] * np.where(c["today_unit"], 1.0, f2)
    g = c.groupby(["symbol", "accn"]).agg(cik=("cik", "first"), form=("form", "first"), filed=("filed", "first"),
                                          asof=("asof", "max"), raw=("val", "sum"), shares=("adj", "sum"),
                                          alt=("adj2", "sum"))
    g = g.reset_index().assign(source="class")
    g["filed"] = pd.to_datetime(g["filed"], utc=True)
    return g[cols]


# ── 선택 ──
def _choose_variant(cands: list[tuple], ref: float | None) -> list[tuple[str, float]]:
    """후보 (출처, 기본값[, 대안값]) → (출처, 값). 기본값과 대안값(분할 보정 기준일이 다른 값)이 다르면
    직전 채택값(ref)에 가까운 쪽, ref가 없으면 다른 후보들 기본값의 중앙값에 가까운 쪽, 그것도 없으면 기본값."""
    out = []
    prim = [c[1] for c in cands]
    for i, c in enumerate(cands):
        v, alt = c[1], (c[2] if len(c) > 2 else float("nan"))
        if not (alt > 0) or abs(alt / v - 1) < 1e-9:
            out.append((c[0], v))
            continue
        target = ref if ref is not None else (float(np.median(prim[:i] + prim[i + 1:])) if len(prim) > 1 else None)
        if target is not None and abs(math.log(alt / target)) < abs(math.log(v / target)):
            v = alt
        out.append((c[0], v))
    return out


MCAP_RANGE = (5e7, 1e13)  # 공시일 종가 × 주식수가 이 밖이면 그 후보는 틀린 값 (S&P500·NASDAQ100 종목 기준 여유 있게)
MCAP_TYPICAL = 3e10       # 직전 값이 없을 때 단위 보정 배수가 여럿 가능하면 시총이 이 값에 가장 가까운 쪽


def _pick(cands: list[tuple], ref: float | None, px: float = float("nan"),
          pending: float | None = None) -> tuple[str, float, str] | None:
    """한 공시의 후보 [(출처, 오늘 기준 값)] (우선순위 순) → (출처, 값, flag) 또는 None(버림).
    ref = 직전 채택값 3개 중앙값, px = 공시일(또는 그 전) 종가(오늘 기준, 없으면 NaN),
    pending = 직전 공시에서 버린 1순위 값 (같은 새 수준이 두 번 연속이면 실제 변화로 본다)."""
    lo, hi = MCAP_RANGE
    cands = _choose_variant(cands, ref)

    def plausible(v: float) -> bool:
        return math.isnan(px) or lo <= v * px <= hi

    def near(x: float, y: float, tol: float) -> bool:
        return abs(math.log(x / y)) <= math.log(1 + tol)

    ok, fixed = [], []
    for src, v in cands:
        if plausible(v):
            ok.append((src, v))
            continue
        ks = [k for k in (3, 6, 9) + ((-3, -6) if ref is not None else ())  # 기준값 없으면 줄이는 쪽만 (껍데기 회사 1주 등)
              if plausible(v / 10 ** k) and (ref is None or 0.8 <= v / 10 ** k / ref <= 1.25)]
        if ks:
            k = min(ks, key=lambda k: abs(math.log(v / 10 ** k * px / MCAP_TYPICAL)) if not math.isnan(px) else 0)
            fixed.append((src, v / 10 ** k))
    vals = [v for _, v in ok]
    for i, (src, v) in enumerate(ok):
        agree = any(j != i and near(v, vals[j], 0.15) for j in range(len(vals)))
        if ref is None:
            if len(ok) == 1 or agree or any(j != i and near(v, vals[j], 1.0) for j in range(len(vals))):
                return src, v, "unchecked" if len(ok) == 1 else "ok"
            continue
        if 0.5 <= v / ref <= 2 or agree:
            return src, v, "ok"
        if i == 0 and pending is not None and near(v, pending, 0.15):
            return src, v, "level_change"
    if ref is not None:
        for src, v in ok:
            k = round(math.log10(v / ref) / 3) * 3
            if k and 0.8 <= v / ref / 10 ** k <= 1.25 and plausible(v / 10 ** k):
                return src, v / 10 ** k, "scale_fixed"
    if fixed:
        return fixed[0][0], fixed[0][1], "scale_fixed"
    if ref is None and ok:  # 첫 공시인데 태그끼리 2배 넘게 다름 → 우선순위 1순위
        return ok[0][0], ok[0][1], "unchecked"
    return None


def share_events(symbols: list[str] | None = None, facts: pd.DataFrame | None = None,
                 class_shares: pd.DataFrame | None = None, splits: pd.DataFrame | None = None,
                 ratios: dict | None = None, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """공시마다 채택한 주식수 (오늘 기준). 버린 공시도 flag='dropped', shares=NaN으로 남긴다(품질 집계용).
    facts·class_shares·splits·bars(us_1d 종가, 시총 상식 범위 점검용)를 주지 않으면 저장소에서 읽는다
    (테스트·분석은 직접 넘긴다. bars가 빈 표면 가격 점검 없이 고른다)."""
    if facts is None:
        folder = store.DATA_DIR / "edgar_facts" / "ALL"
        if not folder.exists():
            return pd.DataFrame()
        where = f" AND symbol IN ({','.join(repr(s) for s in symbols)})" if symbols else ""
        facts = store.query(f"SELECT symbol, cik, tag, unit, start, \"end\", val, accn, form, filed FROM "
                            f"'{folder.as_posix()}/*.parquet' WHERE field LIKE 'shares%'{where}")
    elif symbols:
        facts = facts[facts["symbol"].isin(symbols)]
    if class_shares is None:
        class_shares = store.read("edgar_class_shares", "ALL")
    if bars is None:
        bars = _bars(symbols)
    sp = load_splits(splits)
    ratios = _cfg().get("class_ratio", {}) if ratios is None else ratios

    fc = _facts_candidates(facts)
    cik_symbols = fc[["cik", "symbol"]].drop_duplicates()
    sp = classify_splits(sp, _raw_series(fc, class_shares, cik_symbols))
    sp_all, sp = sp, sp[sp["true"]]  # 주식수 보정은 실제 분할만. 분사형 배수는 daily_shares가 봉 날짜 기준으로 곱한다
    cc = _class_candidates(class_shares, cik_symbols, sp, ratios)
    class_ciks = set(cc["cik"])
    fc = fc[~fc["cik"].isin(class_ciks)]  # 클래스 대상 회사는 클래스 합계만 (차원 없는 값은 한 클래스뿐일 수 있음)
    inst = fc["source"].isin(INSTANT_SOURCES).to_numpy()
    f_asof = split_factor(fc["symbol"], _day(fc["asof"]), sp)
    f_filed = split_factor(fc["symbol"], _day(fc["filed"]), sp)
    f_late = split_factor(fc["symbol"], _day(fc["filed"]) + ALT_DAYS, sp)
    # 기본: 표지 값은 asof, 재무제표 값은 filed 기준. 대안: 공시일+15일 기준 (표지 값), asof 기준 (재무제표 값)
    fc = fc.assign(shares=fc["raw"] * np.where(inst, f_asof, f_filed),
                   alt=fc["raw"] * np.where(inst, f_late, f_asof))
    c = pd.concat([x for x in (cc, fc) if not x.empty], ignore_index=True) if len(cc) or len(fc) else fc
    c = c[c["shares"] > 0]
    if c.empty:
        return pd.DataFrame()
    c["prio"] = c["source"].map({s: i for i, s in enumerate(SOURCES)})
    c["fday"] = _day(c["filed"])
    if bars is not None and not bars.empty:  # 공시일(또는 그 전 마지막 거래일) 종가
        px = pd.DataFrame({"symbol": bars["symbol"].values, "fday": _day(bars["ts"]).values,
                           "px": bars["close"].astype(float).values}).sort_values("fday")
        c = pd.merge_asof(c.sort_values("fday"), px, on="fday", by="symbol", direction="backward")
        # 가격 이력 시작 전 공시: 그 값이 처음 쓰이는 날(첫 거래일) 종가로 점검 → 쓰는 시점 기준이라 미래참조 아님
        fwd = pd.merge_asof(c[["fday", "symbol"]], px, on="fday", by="symbol", direction="forward")["px"]
        c["px"] = c["px"].fillna(fwd)
    else:
        c["px"] = np.nan
    c = c.sort_values(["symbol", "filed", "accn", "prio"])

    rows = []
    for sym, g in c.groupby("symbol", sort=False):
        hist: list[float] = []
        pending = None
        for accn, a in g.groupby("accn", sort=False):
            cands = list(zip(a["source"], a["shares"], a["alt"]))
            ref = float(np.median(hist[-3:])) if hist else None
            p = _pick(cands, ref, float(a["px"].iloc[0]), pending)
            first = a.iloc[0]
            if p is None:
                rows.append((sym, first["cik"], accn, first["form"], first["filed"], first["asof"], first["source"],
                             first["raw"], np.nan, np.nan, "dropped"))
                pending = float(first["shares"])
                continue
            src, v, flag = p
            r = a[a["source"] == src].iloc[0]
            hist, pending = (hist + [v]) if flag != "level_change" else [v], None
            rows.append((sym, r["cik"], accn, r["form"], r["filed"], r["asof"], src, r["raw"], v / r["raw"], v, flag))
    out = pd.DataFrame(rows, columns=["symbol", "cik", "accn", "form", "filed", "asof", "source", "shares_raw",
                                      "factor", "shares", "flag"])
    out = out.sort_values(["symbol", "filed"]).reset_index(drop=True)
    out.attrs["splits"] = splits_to_attrs(sp_all)
    return out


def splits_to_attrs(sp: pd.DataFrame) -> dict:
    """판정한 분할 표 → JSON으로 저장되는 dict (DataFrame.attrs에 넣어 parquet에도 남게)."""
    return {"symbol": list(sp["symbol"]), "day": [str(pd.Timestamp(d).date()) for d in sp["day"]],
            "ratio": [float(x) for x in sp["ratio"]], "true": [bool(x) for x in sp["true"]]}


def splits_from_attrs(a: dict | None) -> pd.DataFrame | None:
    if not a:
        return None
    return pd.DataFrame({"symbol": a["symbol"], "day": pd.to_datetime(a["day"]), "ratio": a["ratio"],
                         "true": a["true"]})


def _raw_series(fc: pd.DataFrame, cls: pd.DataFrame, cik_symbols: pd.DataFrame) -> pd.DataFrame:
    """분할 종류 판정용 원래 주식수 (symbol, filed, asof, raw): 클래스 합계(단순 합) 또는 표지 값, 없으면 다른 태그."""
    parts = []
    if cls is not None and not cls.empty:
        c = cls[cls["member"] != "∅"].dropna(subset=["val"]).merge(cik_symbols, on="cik")
        fc = fc[~fc["cik"].isin(set(c["cik"]))]  # 클래스 대상 회사는 클래스 합계만
        c = c.groupby(["symbol", "accn"]).agg(filed=("filed", "first"), asof=("asof", "max"), raw=("val", "sum"))
        parts.append(c.reset_index()[["symbol", "filed", "asof", "raw"]])
    f = fc.assign(p=fc["source"].map({"dei": 0, "bs": 1, "basic": 2, "diluted": 3})).sort_values("p")
    f = f.drop_duplicates(["symbol", "accn"])
    parts.append(f[["symbol", "filed", "asof", "raw"]])
    r = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(len(p) for p in parts) else parts[-1]
    r["filed"], r["asof"] = pd.to_datetime(r["filed"], utc=True), pd.to_datetime(r["asof"], utc=True)
    return r[r["raw"] > 0]


def _bars(symbols: list[str] | None, start=None, end=None) -> pd.DataFrame:
    folder = store.DATA_DIR / "us_1d"
    if not any(folder.glob("*/*.parquet")):
        return pd.DataFrame(columns=["ts", "symbol", "close"])
    files = (f"'{folder.as_posix()}/*/*.parquet'" if not symbols else
             "[" + ",".join(f"'{(folder / store._safe(s)).as_posix()}/*.parquet'" for s in symbols
                            if (folder / store._safe(s)).exists()) + "]")
    if files == "[]":
        return pd.DataFrame(columns=["ts", "symbol", "close"])
    w = []
    if start is not None:
        w.append(f"ts >= '{store._utc(start).isoformat()}'")
    if end is not None:
        w.append(f"ts < '{store._utc(end).isoformat()}'")
    b = store.query(f"SELECT ts, regexp_extract(replace(filename, '\\', '/'), 'us_1d/([^/]+)/', 1) symbol, close "
                    f"FROM read_parquet({files}, filename=true){' WHERE ' + ' AND '.join(w) if w else ''}")
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    return b


def daily_shares(symbols: list[str] | None = None, start=None, end=None, events: pd.DataFrame | None = None,
                 bars: pd.DataFrame | None = None, max_age_days: int | None = None) -> pd.DataFrame:
    """us_1d 봉마다 그날 알 수 있던 주식수와 시가총액.
    열: ts, symbol, close, shares(오늘 기준, close와 같은 단위), market_cap(USD), cik, filed, asof, source, age_days.
    그날 = 뉴욕 날짜. 공시일 다음 날부터 사용, asof가 max_age_days보다 오래되면 shares·market_cap NaN."""
    b = _bars(symbols, start, end) if bars is None else bars.copy()
    ev = share_events(symbols, bars=_bars(symbols) if start is not None else b) if events is None else events
    if b.empty:
        return b
    max_age = int(max_age_days or _cfg().get("max_age_days", 400))
    b["day"] = _day(b["ts"])
    ev = ev[ev["flag"] != "dropped"] if not ev.empty else ev
    if ev.empty:
        return b.drop(columns="day").assign(shares=np.nan, market_cap=np.nan)
    e = ev.assign(day=_day(ev["filed"]) + pd.Timedelta(days=1))[
        ["symbol", "day", "shares", "cik", "filed", "asof", "source"]].sort_values("day")
    out = pd.merge_asof(b.sort_values("day"), e, on="day", by="symbol", direction="backward")
    sp = splits_from_attrs(ev.attrs.get("splits"))
    if sp is None:  # 파일에서 읽은 events 등: 저장된 분할로 다시 판정
        sp = classify_splits(load_splits(), ev.rename(columns={"shares_raw": "raw"})[["symbol", "filed", "asof", "raw"]])
    pseudo = sp[~sp["true"]] if "true" in sp else sp.iloc[:0]
    if len(pseudo):  # 분사형 가격 보정: 그 봉 날짜 뒤 배수를 곱해야 보정된 종가와 맞는다 (실제 주식수는 그대로)
        out["shares"] = out["shares"] * split_factor(out["symbol"], out["day"], pseudo)
    age = (out["day"] - _day(out["asof"])).dt.days
    stale = age > max_age
    out.loc[stale, "shares"] = np.nan
    out["age_days"] = age
    out["market_cap"] = out["close"] * out["shares"]
    out = out.drop(columns="day").sort_values(["symbol", "ts"]).reset_index(drop=True)
    out.attrs["splits"] = splits_to_attrs(sp)
    return out


def quality(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """종목별 메타: 공시 수, 버림·단위 수정 비율, 봉 기준 결측률(주식수 없는 날), 주식수 첫 날짜,
    전 기간 중앙값 대비 10배 넘게 벗어난 날 비율(단위 이상치)."""
    e = events.groupby("symbol").agg(n_filings=("accn", "size"),
                                     dropped=("flag", lambda s: (s == "dropped").mean()),
                                     scale_fixed=("flag", lambda s: (s == "scale_fixed").mean()),
                                     source=("source", lambda s: s.value_counts().index[0]))
    d = daily.copy()
    med = d.groupby("symbol")["shares"].transform("median")
    d["far"] = ((d["shares"] / med) > 10) | ((d["shares"] / med) < 0.1)
    q = d.groupby("symbol").agg(bars=("ts", "size"), missing=("shares", lambda s: s.isna().mean()),
                                unit_outlier=("far", "mean"),
                                first_shares=("ts", lambda s: s[d.loc[s.index, "shares"].notna()].min()))
    return q.join(e, how="left").reset_index()
