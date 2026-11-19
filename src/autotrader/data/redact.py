"""로그·예외 메시지에서 비밀 값(API 키·토큰)을 가린다.

수집기는 키를 URL 쿼리(`crtfc_key=` 등)나 경로(ECOS `/api/StatisticSearch/<키>/`)에 넣어 보내는데, httpx 오류 문자열과
재시도 로그에 URL 전체가 들어가 키가 로그 파일에 남았다 (2026-11-19 DART). 그래서 세 겹으로 가린다.
  1. 쿼리 이름이 키 성격이면 값을 `***`로 (`SECRET_PARAMS`, 대소문자 무시)
  2. 값을 경로에 넣는 API의 그 칸 (`_PATH_KEYS`)
  3. 환경변수 중 이름이 키 성격(KEY·SECRET·TOKEN·PASSWORD·ACCOUNT_NO·CLIENT_ID)인 값이 그대로 나오면 어디에 있든
쓰는 곳: collectors/common.py(로그 전체에 patcher, get_json 예외), store.set_state(error), cli의 텔레그램 알림·트레이스백.
"""

from __future__ import annotations

import os
import re

MASK = "***"
SECRET_PARAMS = frozenset({
    "crtfc_key", "servicekey", "service_key", "api_key", "apikey", "key", "token", "access_token", "auth_key",
    "appkey", "app_key", "appsecret", "app_secret", "secret", "client_secret", "client_id", "password", "signature",
})
_QS = re.compile(r"([?&;]|^)([^=&?#\s'\"]+)=([^&#\s'\"<>)]*)")
_PATH_KEYS = [re.compile(r"(ecos\.bok\.or\.kr/api/[A-Za-z]+/)[^/\s'\"?]+")]
_ENV_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|ACCOUNT_NO|CLIENT_ID", re.IGNORECASE)


def _env_secrets() -> list[str]:
    vals = {v.strip() for k, v in os.environ.items() if _ENV_NAME.search(k) and v and len(v.strip()) >= 8}
    return sorted(vals, key=len, reverse=True)  # 긴 값부터 (한 값이 다른 값의 일부일 때)


def redact(text: object) -> str:
    """문자열 안의 비밀 값을 `***`로 바꾼다. 문자열이 아니면 str()로 바꿔서."""
    s = str(text)
    s = _QS.sub(lambda m: f"{m[1]}{m[2]}={MASK}" if m[2].lower() in SECRET_PARAMS and m[3] else m[0], s)
    for p in _PATH_KEYS:
        s = p.sub(rf"\g<1>{MASK}", s)
    for v in _env_secrets():
        if v in s:
            s = s.replace(v, MASK)
    return s
