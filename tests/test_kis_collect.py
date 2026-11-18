"""KIS 수집 견고성 (2026-11-18): 최종 실패 시 마지막 응답 로깅, OPSQ0002 장애 재시도, ECOS 성공 개수 로그.

실제 KIS·ECOS는 부르지 않는다 — 가짜 클라이언트와 가짜 응답으로만.
"""

from __future__ import annotations

import pytest
from loguru import logger

from autotrader.data.collectors import kr_public, kr_stock


class FakeKIS:
    """정해 둔 결과(예외 또는 dict)를 차례로 돌려준다."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def get(self, path, tr_id, params):
        self.calls += 1
        r = self.results.pop(0) if self.results else self.last
        self.last = r
        if isinstance(r, Exception):
            raise r
        return r


class NoLimit:
    def wait(self):
        pass


@pytest.fixture
def kis(monkeypatch):
    """가짜 KIS + 가짜 시계(sleep이 시계를 앞으로 민다)."""
    clock = {"t": 1000.0}
    slept: list[float] = []

    def fake_sleep(s):
        slept.append(s)
        clock["t"] += s

    monkeypatch.setattr(kr_stock, "_sleep", fake_sleep)
    monkeypatch.setattr(kr_stock.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(kr_stock, "_limit", NoLimit())
    monkeypatch.setattr(kr_stock, "_outage_since", None)

    def install(results):
        fake = FakeKIS(results)
        monkeypatch.setattr(kr_stock, "kis", lambda: fake)
        return fake

    install.slept = slept
    install.clock = clock
    return install


@pytest.fixture
def logs():
    lines: list[str] = []
    hid = logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    yield lines
    logger.remove(hid)


OPSQ0002 = RuntimeError("KIS 오류 OPSQ0002: 일시적 오류")


def test_last_message_logged_on_final_failure(kis, logs):
    kis([RuntimeError("KIS 오류 OPSQ1234: 알 수 없는 오류")] * 10)
    with pytest.raises(RuntimeError, match="OPSQ1234"):
        kr_stock._get("/p", "TR1", {"FID_INPUT_ISCD": "005930"})
    assert any("TR1 최종 실패" in l and "OPSQ1234" in l for l in logs)


def test_network_error_message_kept(kis, logs):
    kis([ConnectionError("connection reset")] * 10)
    with pytest.raises(RuntimeError, match="ConnectionError: connection reset"):
        kr_stock._get("/p", "TR2", {})
    assert any("ConnectionError" in l for l in logs)


def test_short_outage_recovers(kis):
    """3분짜리 OPSQ0002 장애는 넘긴다 (예전엔 1·2초 두 번 재시도 뒤 실패)."""
    ok = {"rt_cd": "0", "output2": []}
    n_fail = 12  # 2+4+8+16+30×8 = 270초 동안 실패
    fake = kis([OPSQ0002] * n_fail + [ok])
    assert kr_stock._get("/p", "TR3", {}) == ok
    assert fake.calls == n_fail + 1
    assert kis.slept[:5] == [2, 4, 8, 16, 30] and max(kis.slept) == 30
    assert kr_stock._outage_since is None  # 성공하면 장애 표시 해제


def test_long_outage_capped_and_fails_fast_after(kis):
    """장애가 OUTAGE_MAX를 넘으면 그 호출은 실패하고, 이후 호출은 기다리지 않고 바로 실패한다."""
    kis([OPSQ0002] * 1000)
    with pytest.raises(RuntimeError, match="OPSQ0002"):
        kr_stock._get("/p", "TR4", {})
    first_wait = sum(kis.slept)
    assert kr_stock.OUTAGE_MAX <= first_wait <= kr_stock.OUTAGE_MAX + 31
    kis.slept.clear()
    with pytest.raises(RuntimeError, match="OPSQ0002"):
        kr_stock._get("/p", "TR4", {})
    assert sum(kis.slept) <= 3  # 짧은 재시도(1·2초)만


def test_rate_limit_codes_unchanged(kis):
    ok = {"rt_cd": "0"}
    kis([RuntimeError("KIS 오류 EGW00201: 초당 거래건수 초과")] * 2 + [ok])
    assert kr_stock._get("/p", "TR5", {}) == ok
    assert kis.slept == [5, 10]


def test_ecos_logs_actual_success_count(monkeypatch, logs):
    monkeypatch.setenv("ECOS_API_KEY", "dummy")
    names = list(kr_public.ECOS)
    good = {"StatisticSearch": {"row": [{"TIME": "20260102", "DATA_VALUE": "1.5"}]}}

    def fake_get_json(client, url, *a, **kw):
        stat = url.split("/")[-5]  # .../{stat}/{cycle}/{start}/{end}/{item}
        return {"RESULT": {"CODE": "ERR"}} if stat == kr_public.ECOS[names[0]][0] else good

    writes = []
    monkeypatch.setattr(kr_public, "get_json", fake_get_json)
    monkeypatch.setattr(kr_public.store, "write", lambda ds, name, df: writes.append(name))
    monkeypatch.setattr(kr_public.store, "set_state", lambda *a, **k: None)
    kr_public.collect_ecos()
    n_bad = sum(1 for n in names if kr_public.ECOS[n][0] == kr_public.ECOS[names[0]][0])
    assert any(f"ecos {len(names) - n_bad}개 갱신, {n_bad}개 실패" in l for l in logs)
