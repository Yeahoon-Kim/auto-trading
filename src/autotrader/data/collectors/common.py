"""수집기 공통: HTTP 재시도, 속도 제한, 설정 로딩, 로거.

로그·예외에 키가 남지 않게: 로거의 모든 메시지와 잡히지 않은 예외의 트레이스백을 `redact()`로 가린다 (../redact.py).
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

import httpx
import yaml
from loguru import logger

from ..redact import redact

ROOT = Path(__file__).resolve().parents[4]

logger.remove()
logger.configure(patcher=lambda r: r.update(message=redact(r["message"])))


def _excepthook(etype, value, tb) -> None:
    """잡히지 않은 예외의 트레이스백(.err 파일로 가는 stderr)에서도 키를 가린다."""
    sys.stderr.write(redact("".join(traceback.format_exception(etype, value, tb))))


sys.excepthook = _excepthook
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} {level:<7} {message}")
logger.add(ROOT / "logs" / "collect_{time:YYYY-MM-DD}.log", level="INFO", rotation="1 day", retention="30 days",
           encoding="utf-8")


def config() -> dict:
    return yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))


class RateLimiter:
    """여러 스레드가 공유해도 초당 per_sec 회를 넘지 않게 한다."""

    def __init__(self, per_sec: float):
        self.interval = 1.0 / per_sec
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


def keep_awake() -> None:
    """수집이 도는 동안 Windows 절전 진입을 막는다 (프로세스가 끝나면 자동 해제)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


class Redirected(RuntimeError):
    """3xx 응답. 리다이렉트는 따라가지 않고 재시도도 하지 않는다 (API가 오류 페이지로 보내는 경우라 같은 요청은 또 실패).
    예: DART가 일부 corp_code 묶음에 302 → error1.html (2026-11-19)."""

    def __init__(self, msg: str, status: int, location: str):
        super().__init__(msg)
        self.status, self.location = status, location


def get_json(client: httpx.Client, url: str, params: dict | None = None, retries: int = 5, **kw):
    """429/5xx/네트워크 오류는 지수 백오프로 재시도. 그 밖의 4xx는 httpx.HTTPStatusError, 3xx는 Redirected.
    로그와 예외 문구에는 URL·오류를 redact()로 가려 넣는다 (쿼리에 든 키가 남지 않게)."""
    last = ""
    for i in range(retries):
        try:
            r = client.get(url, params=params, **kw)
        except httpx.TransportError as e:
            last = redact(f"{type(e).__name__}: {e}")
        else:
            if 300 <= r.status_code < 400:
                loc = redact(r.headers.get("location", ""))
                raise Redirected(f"{redact(url)} HTTP {r.status_code} → {loc}", r.status_code, loc)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            elif r.status_code >= 400:
                raise httpx.HTTPStatusError(redact(f"HTTP {r.status_code} for {r.request.url}: {r.text[:200]}"),
                                            request=r.request, response=r)
            else:
                return r.json()
        wait = 2 ** i
        logger.warning(f"{redact(url)} 재시도 {i + 1}/{retries} ({last}) — {wait}s 대기")
        time.sleep(wait)
    raise RuntimeError(f"{redact(url)} 요청 실패 ({last})")
