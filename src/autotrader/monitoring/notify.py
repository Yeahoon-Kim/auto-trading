"""텔레그램 알림.

환경변수 TELEGRAM_BOT_TOKEN 이 필요하다. TELEGRAM_CHAT_ID 가 없으면 봇에게 마지막으로
메시지를 보낸 대화방을 찾아 .cache/telegram_chat_id 에 저장해 쓴다.
토큰이 없으면 보내지 않고 logs/alerts.log 에만 남긴다 (알림 실패가 수집을 멈추지 않게).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[3]
CHAT_CACHE = ROOT / ".cache" / "telegram_chat_id"
ALERT_LOG = ROOT / "logs" / "alerts.log"
MAX_LEN = 4000  # 텔레그램 메시지 한도 4096자


def _log(text: str) -> None:
    ALERT_LOG.parent.mkdir(exist_ok=True)
    with ALERT_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n\n")


def _chat_id(token: str) -> str | None:
    if os.environ.get("TELEGRAM_CHAT_ID"):
        return os.environ["TELEGRAM_CHAT_ID"]
    if CHAT_CACHE.exists():
        return CHAT_CACHE.read_text().strip()
    r = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
    chats = [u["message"]["chat"]["id"] for u in r.get("result", []) if "message" in u]
    if not chats:
        return None
    CHAT_CACHE.parent.mkdir(exist_ok=True)
    CHAT_CACHE.write_text(str(chats[-1]))
    return str(chats[-1])


def send(text: str) -> bool:
    """보냈으면 True. 토큰·대화방이 없거나 실패하면 로그에만 남기고 False."""
    _log(text)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return False
    try:
        chat = _chat_id(token)
        if not chat:
            _log("텔레그램: 대화방을 찾지 못함 — 봇에게 아무 메시지나 한 번 보내 주세요")
            return False
        for i in range(0, len(text), MAX_LEN):
            httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                       json={"chat_id": chat, "text": text[i:i + MAX_LEN]}, timeout=20).raise_for_status()
        return True
    except Exception as e:  # 알림 실패는 로그만
        _log(f"텔레그램 전송 실패: {type(e).__name__}")
        return False
