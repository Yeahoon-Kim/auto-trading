"""2026-11-19 수집 문제 3건의 회귀 테스트.

1. 로그·예외에 키가 남지 않는다 (redact, get_json, 로거 patcher, set_state)
2. DART fnlttMultiAcnt 302 → 묶음을 반으로 나눠 다시, 1개사도 302면 멈춤
(네이버 start 상한·질의별 실패는 tests/test_naver.py)
"""

from __future__ import annotations

import httpx
import pytest

from autotrader.data import store
from autotrader.data.collectors import common, kr_public
from autotrader.data.redact import redact

FAKE = "FAKEKEY0123456789abcdef"


# ── 1. 키 가리기 ──
@pytest.mark.parametrize("text", [
    f"https://opendart.fss.or.kr/api/fnlttMultiAcnt.json?crtfc_key={FAKE}&corp_code=001",
    f"http://apis.data.go.kr/x?serviceKey={FAKE}&pageNo=1",
    f"https://a.b/c?api_key={FAKE}", f"https://a.b/c?APIKEY={FAKE}&x=1", f"https://a.b/c?x=1&key={FAKE}",
    f"https://a.b/c?token={FAKE}", f"https://a.b/c?appkey={FAKE}&appsecret={FAKE}",
    f"https://ecos.bok.or.kr/api/StatisticSearch/{FAKE}/json/kr/1/100000/722Y001",
    f"Redirect response '302 Found' for url 'https://a.b/c?crtfc_key={FAKE}'",
])
def test_redact_hides_key_params(text):
    out = redact(text)
    assert FAKE not in out and "***" in out


def test_redact_keeps_other_params_and_env_values(monkeypatch):
    s = "https://a.b/c?corp_code=00126380&bsns_year=2025&start=1000"
    assert redact(s) == s
    monkeypatch.setenv("SOME_API_KEY", FAKE)  # 이름이 키 성격인 환경변수 값은 어디에 있든 가린다
    assert redact(f"oops {FAKE} in text") == "oops *** in text"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_get_json_logs_and_errors_have_no_key(monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    lines = []
    sink = common.logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        url = "https://opendart.fss.or.kr/api/x.json"
        # 5xx 재시도 → 실패
        with pytest.raises(RuntimeError) as e5:
            common.get_json(_client(lambda r: httpx.Response(503)), url, {"crtfc_key": FAKE}, retries=2)
        # 4xx → HTTPStatusError (형은 그대로, 문구만 가림)
        with pytest.raises(httpx.HTTPStatusError) as e4:
            common.get_json(_client(lambda r: httpx.Response(404, text="nope")), url, {"crtfc_key": FAKE})
        assert e4.value.response.status_code == 404
        # 네트워크 오류 문구에 URL이 들어가도
        def boom(r):
            raise httpx.ConnectError(f"failed {r.url}", request=r)
        with pytest.raises(RuntimeError) as et:
            common.get_json(_client(boom), url, {"crtfc_key": FAKE}, retries=1)
        # 로거로 직접 찍어도 patcher가 가린다
        common.logger.warning(f"{url}?crtfc_key={FAKE}")
    finally:
        common.logger.remove(sink)
    assert lines and all(FAKE not in s for s in lines)
    for e in (e5, e4, et):
        assert FAKE not in str(e.value)


def test_get_json_redirect_not_retried(monkeypatch):
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    n = []

    def h(r):
        n.append(1)
        return httpx.Response(302, headers={"location": "https://opendart.fss.or.kr/error1.html"})
    with pytest.raises(common.Redirected) as e:
        common.get_json(_client(h), "https://opendart.fss.or.kr/api/x.json", {"crtfc_key": FAKE})
    assert len(n) == 1 and e.value.status == 302 and "error1.html" in e.value.location and FAKE not in str(e.value)


def test_set_state_error_redacted():
    store.set_state("x", "ALL", status="error", error=f"boom https://a.b/c?crtfc_key={FAKE}")
    assert FAKE not in store.get_state("x", "ALL")["error"]


def test_excepthook_redacts(capsys):
    try:
        raise RuntimeError(f"https://a.b/c?crtfc_key={FAKE}")
    except RuntimeError as e:
        common._excepthook(type(e), e, e.__traceback__)
    err = capsys.readouterr().err
    assert "RuntimeError" in err and FAKE not in err


# ── 2. DART 302 → 나눠 다시 ──
def _fin_row(code: str) -> dict:
    return {"corp_code": code, "stock_code": code[-6:], "rcept_no": "20250515000123", "fs_div": "CFS", "currency": "KRW",
            "account_nm": "매출액", "ord": "1", "thstrm_amount": "1,000", "thstrm_add_amount": "1,000",
            "thstrm_dt": "2025.01.01 ~ 2025.03.31"}


def test_dart_fin_splits_batch_on_redirect(monkeypatch):
    monkeypatch.setenv("DART_API_KEY", FAKE)
    monkeypatch.setattr(kr_public._dart_limit, "wait", lambda: None)
    seen = []

    def fake(client, url, params, **kw):
        cs = params["corp_code"].split(",")
        seen.append(len(cs))
        if len(cs) == 100:  # 원래 묶음만 막힘
            raise common.Redirected("302", 302, "https://opendart.fss.or.kr/error1.html")
        return {"status": "000", "list": [_fin_row(c) for c in cs]}
    monkeypatch.setattr(kr_public, "get_json", fake)
    codes = [f"{i:08d}" for i in range(100)]
    out = kr_public._dart_fin(codes, 2025, "11013")
    assert seen == [100, 50, 50]
    assert len(out) == 100 and set(out["corp_code"]) == set(codes)


def test_dart_fin_single_company_redirect_stops(monkeypatch):
    monkeypatch.setenv("DART_API_KEY", FAKE)
    monkeypatch.setattr(kr_public._dart_limit, "wait", lambda: None)
    seen = []

    def fake(client, url, params, **kw):
        seen.append(1)
        raise common.Redirected("302", 302, "https://opendart.fss.or.kr/error1.html")
    monkeypatch.setattr(kr_public, "get_json", fake)
    with pytest.raises(RuntimeError, match="점검·차단"):
        kr_public._dart_fin([f"{i:08d}" for i in range(4)], 2025, "11013")
    assert len(seen) == 3  # 4 → 2 → 1에서 멈춤 (전부 막혀도 호출이 묶음 크기만큼 늘지 않는다)
