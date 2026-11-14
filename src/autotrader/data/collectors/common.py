"""수집기 공통: HTTP 재시도, 속도 제한, 설정 로딩, 로거."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import httpx
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[4]

logger.remove()
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


def get_json(client: httpx.Client, url: str, params: dict | None = None, retries: int = 5, **kw):
    """429/5xx/네트워크 오류는 지수 백오프로 재시도."""
    for i in range(retries):
        try:
            r = client.get(url, params=params, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            wait = 2 ** i
            logger.warning(f"{url} 재시도 {i + 1}/{retries} ({e}) — {wait}s 대기")
            time.sleep(wait)
    raise RuntimeError(f"{url} 요청 실패")
