"""US 분할 이력·클래스별 주식수·분할 보정 주식수(shares.py)·시총 verifier 테스트 (네트워크 없이)."""

import io
import zipfile

import numpy as np
import pandas as pd

from autotrader.data import shares as sh
from autotrader.data import shares_check as ck
from autotrader.data.collectors import edgar, us_splits

NY = "America/New_York"


def _utc(d: str) -> pd.Timestamp:
    return pd.Timestamp(d, tz=NY).tz_convert("UTC")


# ── us_splits ──
def test_parse_yahoo_and_declared():
    s = pd.Series([7.0, 4.0, 2.0], index=pd.DatetimeIndex(["2005-02-28", "2014-06-09", "2020-08-31"]).tz_localize(NY))
    d = us_splits.parse_yahoo("AAPL", s, "2010-01-01")
    assert list(d["ratio"]) == [4.0, 2.0] and (d["ts"] == [_utc("2014-06-09"), _utc("2020-08-31")]).all()
    ann = us_splits.parse_alpaca_ann([{"initiating_symbol": "", "target_symbol": "AAPL", "ex_date": "2020-08-31",
                                       "effective_date": "2020-07-30", "old_rate": "1", "new_rate": "4"}])
    assert ann["alpaca_date"].iloc[0] == pd.Timestamp("2020-07-30")
    out = us_splits.attach_declared(d, ann)
    assert out["alpaca_date"].isna().tolist() == [True, False] and out["alpaca_ratio"].iloc[1] == 4
    assert us_splits.parse_yahoo("X", pd.Series(dtype=float), "2010-01-01").empty


# ── edgar: 클래스별 표지 주식수 ──
INLINE = """<html><ix:header><ix:resources>
<xbrli:context id="c-1"><xbrli:entity><xbrli:identifier>1</xbrli:identifier><xbrli:segment>
<xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">us-gaap:CommonClassAMember</xbrldi:explicitMember>
</xbrli:segment></xbrli:entity><xbrli:period><xbrli:instant>2026-10-28</xbrli:instant></xbrli:period></xbrli:context>
<xbrli:context id="c-2"><xbrli:entity><xbrli:identifier>1</xbrli:identifier><xbrli:segment>
<xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">us-gaap:CommonClassBMember</xbrldi:explicitMember>
</xbrli:segment></xbrli:entity><xbrli:period><xbrli:instant>2026-10-28</xbrli:instant></xbrli:period></xbrli:context>
</ix:resources></ix:header>
<ix:nonFraction name="dei:EntityCommonStockSharesOutstanding" contextRef="c-1" unitRef="shares" decimals="INF">480,476</ix:nonFraction>
<ix:nonFraction name="dei:EntityCommonStockSharesOutstanding" contextRef="c-2" unitRef="shares" scale="6"><span>1,403.8</span></ix:nonFraction>
<ix:nonFraction name="us-gaap:Revenues" contextRef="c-1" unitRef="usd">5</ix:nonFraction></html>"""
CLASSIC = """<xbrl><context id="D1"><entity><identifier>1</identifier></entity><period><instant>2017-02-16</instant></period></context>
<dei:EntityCommonStockSharesOutstanding contextRef="D1" unitRef="shares" decimals="0">1304592522</dei:EntityCommonStockSharesOutstanding></xbrl>"""


def test_parse_cover_shares_inline_and_classic():
    r = edgar.parse_cover_shares(INLINE)
    assert [x["val"] for x in r] == [480476.0, 1403.8e6]
    assert r[0]["member"] == "us-gaap:StatementClassOfStockAxis=us-gaap:CommonClassAMember"
    assert r[0]["asof"] == "2026-10-28"
    c = edgar.parse_cover_shares(CLASSIC)
    assert c == [{"member": "", "asof": "2017-02-16", "val": 1304592522.0}]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.xsd", "<schema/>")
        z.writestr("brka-20260930.htm", INLINE)
    assert len(edgar.parse_xbrl_zip(buf.getvalue())) == 2


def test_class_targets_and_predecessor_cut():
    f = pd.DataFrame({"cik": [1, 1, 1, 2, 2], "field": ["revenue", "shares_outstanding", "revenue", "revenue", "eps"],
                      "accn": ["a", "a", "b", "c", "d"], "form": ["10-Q"] * 5,
                      "filed": [_utc("2015-01-01")] * 5})
    assert edgar.class_targets(f, 0.5, [9]) == [2, 9]  # cik 1: 2개 중 1개(50%) → 대상 아님
    d = pd.DataFrame({"filed": [_utc("2015-01-01"), _utc("2016-01-01")]})
    assert len(edgar.cut_predecessor(d, _utc("2015-06-01"))) == 1
    assert len(edgar.cut_predecessor(d, None)) == 2


def test_fields_version_refetch():
    co = pd.DataFrame({"symbol": ["A", "B"], "cik": [1, 2]})
    now = pd.Timestamp("2026-11-17", tz="UTC")
    old = pd.DataFrame({"symbol": ["A", "B"], "ts": [now, now], "fields_v": [np.nan, edgar.FIELDS_VERSION]})
    assert edgar.outdated(old) == {"A"}
    assert edgar.todo(co, old, set(), now, 30) == ["A"]


# ── shares.py ──
def _facts(rows):
    """(symbol, cik, tag, end, filed, val, start) → edgar_facts 형태."""
    d = pd.DataFrame(rows, columns=["symbol", "cik", "tag", "end", "filed", "val", "start"])
    d["accn"] = d["symbol"] + "-" + d["filed"]
    d["form"], d["unit"] = "10-Q", "shares"
    for c in ("end", "filed", "start"):
        d[c] = pd.to_datetime(d[c]).dt.tz_localize(NY).dt.tz_convert("UTC")
    return d


SPLITS = pd.DataFrame({"ts": [_utc("2020-08-31")], "symbol": ["AAA"], "ratio": [4.0]})


def test_split_adjust_and_point_in_time():
    dei, bs = "EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"
    f = _facts([
        ("AAA", 1, dei, "2020-07-20", "2020-07-31", 100.0, None),       # 분할 전 표지 → ×4
        ("AAA", 1, bs, "2020-06-27", "2020-07-31", 101.0, None),
        ("AAA", 1, dei, "2020-10-16", "2020-10-30", 400.0, None),       # 분할 후 → 그대로
        ("AAA", 1, bs, "2020-08-28", "2020-10-30", 404.0, None),        # 기간 끝은 분할 전이지만 공시 전 분할 → 소급 반영값
    ])
    ev = sh.share_events(facts=f, class_shares=pd.DataFrame(), splits=SPLITS, ratios={})
    assert list(ev["source"]) == ["dei", "dei"] and list(ev["shares"]) == [400.0, 400.0]
    assert list(ev["factor"]) == [4.0, 1.0] and (ev["flag"] != "dropped").all()
    bars = pd.DataFrame({"ts": [_utc(d) + pd.Timedelta(hours=9.5) for d in ("2020-07-31", "2020-08-03", "2022-01-03")],
                         "symbol": "AAA", "close": [100.0, 110.0, 120.0]})
    d = sh.daily_shares(events=ev, bars=bars, max_age_days=400)
    assert np.isnan(d["shares"].iloc[0])           # 공시 당일은 아직 모름
    assert d["market_cap"].iloc[1] == 110.0 * 400  # 다음 거래일부터
    assert np.isnan(d["shares"].iloc[2])           # asof 2020-10-16에서 400일 넘음


def test_scale_error_fixed_and_outlier_dropped():
    dei = "EntityCommonStockSharesOutstanding"
    f = _facts([
        ("ON", 1, dei, "2019-01-20", "2019-02-01", 4.1e8, None),
        ("ON", 1, dei, "2019-04-20", "2019-05-01", 4.1e8, None),
        ("ON", 1, dei, "2019-07-20", "2019-08-01", 4.1e14, None),  # 백만 배 단위 오류
        ("ON", 1, dei, "2019-10-20", "2019-11-01", 9.9e10, None),  # 240배: 단위로도 설명 안 됨 → 버림
        ("ON", 1, dei, "2020-01-20", "2020-02-01", 4.0e8, None),
    ])
    ev = sh.share_events(facts=f, class_shares=pd.DataFrame(), splits=SPLITS.iloc[:0], ratios={})
    assert list(ev["flag"]) == ["unchecked", "ok", "scale_fixed", "dropped", "ok"]
    assert ev["shares"].iloc[2] == 4.1e8


def test_level_change_accepted_when_tags_agree():
    dei, basic = "EntityCommonStockSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"
    f = _facts([
        ("M", 1, dei, "2019-01-20", "2019-02-01", 1e8, None),
        ("M", 1, dei, "2019-04-20", "2019-05-01", 3e8, None),             # 합병으로 3배
        ("M", 1, basic, "2019-03-31", "2019-05-01", 2.9e8, "2019-01-01"),
    ])
    ev = sh.share_events(facts=f, class_shares=pd.DataFrame(), splits=SPLITS.iloc[:0], ratios={})
    assert list(ev["flag"]) == ["unchecked", "ok"] and ev["shares"].iloc[1] == 3e8


def test_class_total_with_ratio():
    """BRK: A주 × 1500 + B주 (B주만 분할 보정). 클래스 대상 회사는 차원 없는 값(A주 수)을 쓰지 않는다."""
    f = _facts([("BRK-B", 7, "EntityCommonStockSharesOutstanding", "2010-01-10", "2010-01-15", 950000.0, None)])
    ax = "us-gaap:StatementClassOfStockAxis="
    cls = pd.DataFrame({"cik": 7, "accn": "x1", "form": "10-K", "filed": _utc("2010-01-15"),
                        "member": [ax + "us-gaap:CommonClassAMember", ax + "us-gaap:CommonClassBMember"],
                        "asof": _utc("2010-01-10"), "val": [1000.0, 30000.0]})
    sp = pd.DataFrame({"ts": [_utc("2010-01-21")], "symbol": ["BRK-B"], "ratio": [50.0]})
    ev = sh.share_events(facts=f, class_shares=cls, splits=sp, ratios={"BRK-B": {"CommonClassAMember": 1500}})
    assert len(ev) == 1 and ev["source"].iloc[0] == "class"
    assert ev["shares"].iloc[0] == 1000 * 1500 + 30000 * 50


# ── verifier ──
def test_checks():
    days = pd.bdate_range("2020-08-03", "2020-10-30")
    ts = [_utc(str(d.date())) + pd.Timedelta(hours=9.5) for d in days]
    good = pd.DataFrame({"ts": ts, "symbol": "AAA", "close": 1e6, "cik": 1,
                         "shares": np.where(days < "2020-10-01", 400.0, 401.0)})
    bad = good.assign(shares=np.where(days < "2020-10-01", 100.0, 400.0))  # 분할 미보정
    good["market_cap"], bad["market_cap"] = good["close"] * good["shares"], bad["close"] * bad["shares"]
    assert ck.check_split_jump(good, SPLITS)["ok"].all()
    assert not ck.check_split_jump(bad, SPLITS)["ok"].any()
    assert ck.check_jump(bad)["jumps"].sum() == 1
    r = ck.check_rank(good.assign(symbol="AAPL"), {"2020-10-30": ["AAPL"], "2020-09-30": ["MSFT"]})
    assert r["ok"].tolist() == [True, False]
    cv = ck.check_coverage(good)
    assert cv["coverage"].iloc[0] == 1.0 and cv["absurd"].iloc[0] == 0


def test_mcap_gate_fixes_bad_first_value():
    """첫 공시가 단위 오류(백만 배)면 직전 값이 없어도 공시일 종가로 본 시총 범위로 고친다 (SRE 2009)."""
    dei = "EntityCommonStockSharesOutstanding"
    f = _facts([("SRE", 1, dei, "2009-09-30", "2009-11-09", 2.46e14, None),
                ("SRE", 1, dei, "2009-12-31", "2010-02-26", 2.47e8, None)])
    bars = pd.DataFrame({"ts": [_utc("2010-01-04") + pd.Timedelta(hours=9.5)], "symbol": "SRE", "close": 25.0})
    ev = sh.share_events(facts=f, class_shares=pd.DataFrame(), splits=SPLITS.iloc[:0], ratios={}, bars=bars)
    assert list(ev["flag"]) == ["scale_fixed", "ok"] and ev["shares"].iloc[0] == 2.46e8


def test_spinoff_factor_applied_by_bar_date():
    """Yahoo 분사형 배수(2.084)는 실제 주식수가 그대로 → 공시가 아니라 봉 날짜 기준으로 곱한다 (ABT 2013)."""
    dei = "EntityCommonStockSharesOutstanding"
    f = _facts([("ABT", 1, dei, "2012-07-20", "2012-08-07", 1.57e9, None),
                ("ABT", 1, dei, "2012-10-20", "2012-11-07", 1.58e9, None),
                ("ABT", 1, dei, "2013-04-20", "2013-05-08", 1.56e9, None)])
    sp = pd.DataFrame({"ts": [_utc("2013-01-02")], "symbol": ["ABT"], "ratio": [2.084202]})
    ev = sh.share_events(facts=f, class_shares=pd.DataFrame(), splits=sp, ratios={}, bars=pd.DataFrame())
    assert (ev["factor"] == 1.0).all()  # 주식수 자체는 보정하지 않음
    assert sh.splits_from_attrs(ev.attrs["splits"])["true"].tolist() == [False]
    bars = pd.DataFrame({"ts": [_utc(d) + pd.Timedelta(hours=9.5) for d in ("2012-12-31", "2013-01-02")],
                         "symbol": "ABT", "close": [31.4, 32.0]})
    d = sh.daily_shares(events=ev, bars=bars)
    assert d["shares"].round(-6).tolist() == [round(1.58e9 * 2.084202, -6), 1.58e9]
    assert sh.is_standard(1.5) and sh.is_standard(0.1) and not sh.is_standard(2.084202)
