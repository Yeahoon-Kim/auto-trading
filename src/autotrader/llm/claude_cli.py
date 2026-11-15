"""Claude Code CLI(`claude -p`)로 LLM을 부른다. 구독 안에서 돌아가므로 API 키·추가 과금이 없다.

- 도구 없이(--tools "") 입력 텍스트만 보고 답한다. 파일·명령·웹 접근 없음.
- 세션을 남기지 않는다 (--no-session-persistence).
- 모든 호출의 설정(모델, 프롬프트 해시, 소요 시간, 성공 여부)을 logs/llm_calls.jsonl에 남긴다
  → 재현성과 사후 검증용 (TradingAgents의 "보고서마다 실행 설정 기록"에서 가져옴).
- 실패하면 예외 대신 None을 돌려준다. 호출 측은 LLM 없이도 동작해야 한다 (브리핑은 해석 문단만 빠짐).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CALL_LOG = ROOT / "logs" / "llm_calls.jsonl"

# 역할별 모델 (architecture.md §8: 빠른 모델 / 깊은 모델)
FAST, DEEP = "haiku", "opus"


def _exe() -> str | None:
    found = shutil.which("claude")
    if found:
        return found
    p = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".local" / "bin" / "claude.exe"
    return str(p) if p.exists() else None


def ask(prompt: str, system: str, model: str = FAST, schema: dict | None = None,
        timeout: int = 300, tag: str = "") -> str | dict | None:
    exe = _exe()
    if exe is None:
        _log(tag, model, prompt, system, ok=False, err="claude CLI 없음")
        return None
    cmd = [exe, "-p", "--model", model, "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--system-prompt", system]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
    t0 = time.time()
    try:
        out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                             timeout=timeout, cwd=ROOT)
        res = json.loads(out.stdout)
        if res.get("is_error"):
            raise RuntimeError(str(res.get("result"))[:300])
        val = res.get("structured_output") if schema is not None else res.get("result")
        if schema is not None and val is None:
            val = json.loads(res["result"])
        _log(tag, model, prompt, system, ok=True, sec=time.time() - t0, turns=res.get("num_turns"))
        return val
    except Exception as e:  # noqa: BLE001 — LLM 실패는 기록만 하고 넘어간다
        _log(tag, model, prompt, system, ok=False, sec=time.time() - t0, err=f"{type(e).__name__}: {e}"[:300])
        return None


def _log(tag, model, prompt, system, ok, sec=0.0, err=None, turns=None) -> None:
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256((system + "\n---\n" + prompt).encode()).hexdigest()[:16]
    rec = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "tag": tag, "model": model,
           "prompt_sha": h, "prompt_chars": len(prompt), "ok": ok, "sec": round(sec, 1),
           "turns": turns, "error": err}
    with CALL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
