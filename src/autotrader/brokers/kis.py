"""한국투자증권 OpenAPI 클라이언트 (현재는 시세 조회 전용).

- 키는 환경변수(KIS_LIVE_APP_KEY / KIS_LIVE_APP_SECRET 등)에서 읽는다.
- 접근 토큰은 24시간 유효하고 발급이 분당 1회로 제한되므로 파일에 캐시한다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URLS = {
    "live": "https://openapi.koreainvestment.com:9443",
    "paper": "https://openapivts.koreainvestment.com:29443",
}
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"


class KISClient:
    def __init__(self, env: str = "live"):
        prefix = f"KIS_{env.upper()}_"
        self.env = env
        self.app_key = os.environ[prefix + "APP_KEY"]
        self.app_secret = os.environ[prefix + "APP_SECRET"]
        self.base_url = BASE_URLS[env]
        self._http = httpx.Client(base_url=self.base_url, timeout=10)
        self._token_file = CACHE_DIR / f"kis_token_{env}.json"
        self._token: str | None = None

    # ── 인증 ──
    def token(self) -> str:
        if self._token:
            return self._token
        if self._token_file.exists():
            cached = json.loads(self._token_file.read_text())
            # 만료 10분 전까지만 재사용
            if datetime.fromisoformat(cached["expires_at"]) - timedelta(minutes=10) > datetime.now():
                self._token = cached["access_token"]
                return self._token
        r = self._http.post(
            "/oauth2/tokenP",
            json={"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret},
        )
        r.raise_for_status()
        data = r.json()
        CACHE_DIR.mkdir(exist_ok=True)
        expires_at = datetime.now() + timedelta(seconds=int(data["expires_in"]))
        self._token_file.write_text(
            json.dumps({"access_token": data["access_token"], "expires_at": expires_at.isoformat()})
        )
        self._token = data["access_token"]
        return self._token

    def get(self, path: str, tr_id: str, params: dict) -> dict:
        headers = {
            "authorization": f"Bearer {self.token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        r = self._http.get(path, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        if data.get("rt_cd") != "0":
            raise RuntimeError(f"KIS 오류 {data.get('msg_cd')}: {data.get('msg1')}")
        return data  # 호출 속도 제한은 호출하는 쪽(RateLimiter)에서 관리

    # ── 시세 조회 ──
    def kr_price(self, code: str) -> dict:
        """국내 주식 현재가."""
        return self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )["output"]
