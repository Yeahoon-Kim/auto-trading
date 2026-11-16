"""테스트 격리: 테스트가 실제 logs/, data/, reports/, results/ 를 건드리지 않게 한다.

배경: 테스트가 실행하는 수집기 코드가 logs/collect_*.log 에 "키 없음 → 건너뜀" 같은 줄을 남겨
실제 운영 로그에 섞였다 (2026-11-16). 소스는 고치지 않고 테스트 쪽에서만 막는다.
"""

from __future__ import annotations

import importlib

import pytest
from loguru import logger

# loguru 파일 sink 차단: collectors/common.py 는 import 시점에 logs/collect_*.log 를 연다(파일 생성 포함).
# 그 import 전에 add()가 파일 경로를 받으면 무시하도록 바꾼다. stderr 등 다른 sink 는 그대로.
_orig_add = logger.add


def _add_no_file(sink, *args, **kwargs):
    if isinstance(sink, (str, bytes)) or hasattr(sink, "__fspath__"):
        return -1
    return _orig_add(sink, *args, **kwargs)


logger.add = _add_no_file
importlib.import_module("autotrader.data.collectors.common")
logger.add = _orig_add  # 이후 테스트 안에서 직접 쓰는 add()는 원래대로

# 모듈 상수 → (모듈, 속성, 임시 폴더 아래 상대 경로)
_PATHS = [
    ("autotrader.data.store", "DATA_ROOT", "data"),
    ("autotrader.data.store", "DATA_DIR", "data/processed"),
    ("autotrader.data.store", "STATE_DB", "data/state.sqlite"),
    ("autotrader.data.collectors.legacy_db", "OUT", "data/processed/legacy"),
    ("autotrader.llm.claude_cli", "CALL_LOG", "logs/llm_calls.jsonl"),
    ("autotrader.llm.decision_log", "LOG", "data/llm/decisions.jsonl"),
    ("autotrader.monitoring.notify", "ALERT_LOG", "logs/alerts.log"),
    ("autotrader.monitoring.notify", "CHAT_CACHE", ".cache/telegram_chat_id"),
    ("autotrader.monitoring.daily_report", "LOG_DIR", "logs"),
    ("autotrader.monitoring.daily_report", "SCAN_DIR", "reports/scan"),
    ("autotrader.monitoring.daily_report", "REPORT_DIR", "reports/ops"),
    ("autotrader.monitoring.weekly_report", "REPORT_DIR", "reports/weekly"),
    ("autotrader.monitoring.weekly_report", "PAPER_DIR", "results/paper"),
    ("autotrader.intel.market", "REPORT_DIR", "reports/market"),
    ("autotrader.intel.scanner", "REPORT_DIR", "reports/scan"),
    ("autotrader.backtest.benchmark", "OUT", "results/benchmarks"),
]


@pytest.fixture(autouse=True)
def _isolate_project_paths(tmp_path_factory, monkeypatch):
    """모든 테스트에서 프로젝트 경로 상수를 임시 폴더로 돌린다. 개별 테스트가 다시 patch 해도 된다."""
    base = tmp_path_factory.mktemp("isolated_root")
    # AUTOTRADER_DATA_DIR(메인 data/ 지정)이 켜진 셸·.env 에서 돌려도, 테스트가 띄우는 하위 프로세스까지
    # 임시 data/ 를 보게 한다. 이미 import 된 store 상수는 아래 _PATHS 로 돌린다.
    (base / "data").mkdir()
    monkeypatch.setenv("AUTOTRADER_DATA_DIR", str(base / "data"))
    for mod_name, attr, rel in _PATHS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # 선택적 모듈 import 실패는 격리 대상에서 제외
            continue
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, base / rel if rel else base)
    try:  # paper_dir 은 store.ROOT(설정 파일 읽기에도 씀)를 쓰므로 ROOT 대신 함수만 돌린다
        paper = importlib.import_module("autotrader.runner.paper")
        monkeypatch.setattr(paper, "paper_dir", lambda lab_dir: base / "results" / "paper" / lab_dir.name)
    except Exception:
        pass
    yield base
