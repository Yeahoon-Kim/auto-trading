"""배당 수집기 파싱·진행 규칙 테스트 (네트워크 없이)."""

import itertools

import pandas as pd

from autotrader.data.collectors import dividends as dv

# 삼성전자 2024 사업보고서 alotMatter 응답 일부 (실제 응답 형태)
ALOT = [
    {"rcept_no": "20250311001085", "se": "주당액면가액(원)", "stock_knd": "-", "thstrm": "100", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "(연결)당기순이익(백만원)", "stock_knd": "-", "thstrm": "33,621,363",
     "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "(별도)당기순이익(백만원)", "stock_knd": "-", "thstrm": "23,582,565",
     "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "현금배당금총액(백만원)", "stock_knd": "-", "thstrm": "9,810,767",
     "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "주식배당금총액(백만원)", "stock_knd": "-", "thstrm": "-", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "(연결)현금배당성향(%)", "stock_knd": "-", "thstrm": "29.20", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "현금배당수익률(%)", "stock_knd": "보통주", "thstrm": "2.70", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "현금배당수익률(%)", "stock_knd": "우선주", "thstrm": "3.30", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,446", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "주당 현금배당금(원)", "stock_knd": "우선주", "thstrm": "1,447", "stlm_dt": "2024-12-31"},
    {"rcept_no": "20250311001085", "se": "주당 주식배당(주)", "stock_knd": "보통주", "thstrm": "-", "stlm_dt": "2024-12-31"},
]


def test_parse_alot():
    v = dv.parse_alot(ALOT)
    assert v["dps_common"] == 1446 and v["dps_pref"] == 1447
    assert v["cash_yield_common"] == 2.7 and v["cash_yield_pref"] == 3.3
    assert v["payout_ratio"] == 29.2 and v["cash_div_total"] == 9_810_767
    assert v["net_income"] == 33_621_363  # 연결 우선
    assert v["stock_div_total"] is None and v["stock_dps_common"] is None
    assert v["face_value"] == 100


def test_alot_row_point_in_time():
    now = pd.Timestamp("2026-11-16", tz="UTC")
    r = dv._alot_row("00126380", "005930", 2024, ALOT, now)
    assert r["has_data"] and r["dps_common"] == 1446
    assert r["ts"] == pd.Timestamp("2024-12-31", tz="Asia/Seoul")
    assert r["filed"] == pd.Timestamp("2025-03-11", tz="Asia/Seoul")  # 접수일 = 알 수 있는 시점의 기준
    empty = dv._alot_row("00000001", "000001", 2024, [], now)
    assert not empty["has_data"] and pd.isna(empty["filed"]) and empty["dps_common"] is None


def test_todo_pairs():
    codes = pd.DataFrame({"corp_code": ["A", "B", "C"], "symbol": ["1", "2", "3"]})
    now = pd.Timestamp("2026-04-01", tz="UTC")
    done = pd.DataFrame({
        "corp_code": ["A", "B", "C", "A"], "bsns_year": [2024, 2024, 2024, 2025],
        "has_data": [True, False, False, False],
        "fetched": pd.to_datetime(["2025-04-01", "2025-08-01", "2025-04-01", "2026-03-25"], utc=True),
    })
    todo = dv.todo_pairs(codes, range(2024, 2026), done, now)
    # 2024: A 값 있음, B는 마감 뒤 "없음" 확인 → 끝, C는 마감 전 "없음"이고 14일 지남 → 다시
    # 2025: A는 7일 전 확인 → 아직 기다림, B·C는 처음
    assert todo == [("C", "3", 2024), ("B", "2", 2025), ("C", "3", 2025)]
    # 사업보고서가 나오기 전 연도(2026)는 묻지 않는다
    assert all(y < 2026 for _, _, y in dv.todo_pairs(codes, range(2024, 2027), pd.DataFrame(), now))


def test_windows():
    w = dv.windows(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01"), 30)
    assert w[0] == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-30"))
    assert w[-1][1] == pd.Timestamp("2024-03-01")
    assert all(b[0] == a[1] + pd.Timedelta(days=1) for a, b in itertools.pairwise(w))  # 빈틈·겹침 없음


def test_parse_ca_and_ann():
    j = {"corporate_actions": {"cash_dividends": [
        {"cusip": "037833100", "ex_date": "2024-02-09", "foreign": False, "id": "x1", "payable_date": "2024-02-15",
         "process_date": "2024-02-15", "rate": 0.24, "record_date": "2024-02-12", "special": False, "symbol": "AAPL"},
        {"cusip": "", "ex_date": "2016-03-01", "foreign": False, "id": "x2", "process_date": "2016-03-01",
         "rate": 0.11, "special": False, "symbol": "ABAX"}]}}
    d = dv.parse_ca(j)
    assert list(d["symbol"]) == ["AAPL", "ABAX"] and d["rate"].tolist() == [0.24, 0.11]
    assert d["ts"].iloc[0] == pd.Timestamp("2024-02-09", tz="America/New_York")
    assert pd.isna(d["cusip"].iloc[1]) and pd.isna(d["payable_date"].iloc[1])
    assert dv.parse_ca({"corporate_actions": {}}).empty

    a = dv.parse_ann([{"id": "a1", "corporate_action_id": "037833100_AA24", "ca_type": "dividend",
                       "ca_sub_type": "cash", "initiating_symbol": "AAPL", "initiating_original_cusip": "037833100",
                       "target_symbol": "AAPL", "declaration_date": "2024-02-08", "ex_date": "2024-02-09",
                       "record_date": "2024-02-12", "payable_date": "2024-02-15", "cash": "0.24",
                       "old_rate": "1", "new_rate": "1"}])
    assert a["ts"].iloc[0] == pd.Timestamp("2024-02-08", tz="America/New_York")  # 공시일
    assert a["cash"].iloc[0] == 0.24 and a["ex_date"].iloc[0] == pd.Timestamp("2024-02-09")
    assert dv.parse_ann([]).empty


def test_parse_alot_stock_knd_dash():
    """stock_knd가 "-"로 오는 회사(SK하이닉스 2016): 순서대로 보통주, 우선주."""
    rows = [{"se": "(연결)현금배당성향(%)", "stock_knd": "-", "thstrm": "14.30"},
            {"se": "현금배당수익률(%)", "stock_knd": "-", "thstrm": "1.30"},
            {"se": "현금배당수익률(%)", "stock_knd": "-", "thstrm": "-"},
            {"se": "주당 현금배당금(원)", "stock_knd": "-", "thstrm": "600"},
            {"se": "주당 현금배당금(원)", "stock_knd": "-", "thstrm": "-"}]
    v = dv.parse_alot(rows)
    assert v["dps_common"] == 600 and v["dps_pref"] is None and v["cash_yield_common"] == 1.3


def test_todo_pairs_reparse_old_parser():
    codes = pd.DataFrame({"corp_code": ["A", "B"], "symbol": ["1", "2"]})
    done = pd.DataFrame({"corp_code": ["A", "B"], "bsns_year": [2016, 2016], "has_data": [True, True],
                         "fetched": pd.to_datetime(["2026-11-16", "2026-11-16"], utc=True),
                         "dps_common": [None, 500.0], "cash_div_total": [423_601.0, 100.0]})
    # A: 1판 파서(parser 열 없음)에서 배당총액은 있는데 DPS가 빔 → 다시 받는다
    assert dv.todo_pairs(codes, range(2016, 2017), done, pd.Timestamp("2026-11-17", tz="UTC")) == [("A", "1", 2016)]
    done["parser"] = dv.PARSER
    assert dv.todo_pairs(codes, range(2016, 2017), done, pd.Timestamp("2026-11-17", tz="UTC")) == []


def test_parse_filings_first_filed(monkeypatch):
    raw = pd.DataFrame({
        "report_nm": ["사업보고서 (2015.12)", "[기재정정]사업보고서 (2015.12)", "반기보고서 (2016.06)", "주요사항보고서(자기주식취득결정)"],
        "rcept_no": ["20160330000001", "20220216000002", "20160816000003", "20160901000004"],
        "rcept_dt": ["20160330", "20220216", "20160816", "20160901"],
        "corp_code": ["00164742"] * 4, "stock_code": ["005380"] * 4,
    })
    f = dv.parse_filings(raw)
    assert len(f) == 3 and f["correction"].tolist() == [False, True, False]
    assert f["period"].tolist() == ["2015.12", "2015.12", "2016.06"]

    monkeypatch.setattr(dv.store, "read", lambda ds, sym: f if ds == dv.FILINGS_DS else pd.DataFrame())
    ff = dv.first_filed().set_index(["kind", "period"])
    row = ff.loc[("사업보고서", "2015.12")]
    assert row["first_filed"] == pd.Timestamp("2016-03-30", tz="Asia/Seoul") and row["n_filings"] == 2


def test_ex_dates():
    td = pd.DatetimeIndex(["2024-12-26", "2024-12-27", "2024-12-30", "2025-01-02", "2025-03-28", "2025-03-31"])
    rec = pd.Series(pd.to_datetime(["2024-12-31", "2025-03-31", "2024-12-26", "2025-06-30", None]))
    ex = dv.ex_dates(rec, td)
    assert ex[0] == pd.Timestamp("2024-12-27")   # 휴장일 기준일 → 마지막 거래일(12-30)의 전 거래일
    assert ex[1] == pd.Timestamp("2025-03-28")   # 거래일 기준일 → 전 거래일
    assert pd.isna(ex[2])                        # 거래일 범위 앞
    assert pd.isna(ex[3]) and pd.isna(ex[4])     # 아직 모르는 미래 거래일, 빈 값


def test_parse_schedule():
    td = pd.DatetimeIndex(["2025-03-27", "2025-03-28", "2025-03-31"])
    rows = [{"record_date": "20250331", "sht_cd": "005930", "isin_name": "삼성전자", "divi_kind": "분기",
             "face_val": "100", "per_sto_divi_amt": "365", "divi_rate": "365.00", "stk_divi_rate": "0.00",
             "divi_pay_dt": "2025/05/20", "stk_div_pay_dt": "", "odd_pay_dt": "", "stk_kind": "보통",
             "high_divi_gb": ""},
            {"record_date": "20250331", "sht_cd": "000000", "isin_name": "다른종목", "divi_kind": "결산",
             "face_val": "500", "per_sto_divi_amt": "0", "divi_rate": "0", "stk_divi_rate": "0",
             "divi_pay_dt": "", "stk_div_pay_dt": "", "odd_pay_dt": "", "stk_kind": "", "high_divi_gb": ""}]
    d = dv.parse_schedule(rows, "005930", td)
    assert len(d) == 1 and d["dps"].iloc[0] == 365 and d["kind"].iloc[0] == "분기"
    assert d["ts"].iloc[0] == pd.Timestamp("2025-03-31", tz="Asia/Seoul")
    assert d["ex_date"].iloc[0] == pd.Timestamp("2025-03-28") and d["pay_date"].iloc[0] == pd.Timestamp("2025-05-20")
    assert dv.parse_schedule([], "005930", td).empty
