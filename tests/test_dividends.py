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
