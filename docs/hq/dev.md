# 개발·점검 HQ 상태판

> 새 개발·점검 HQ 세션은 이 파일, [operating-model.md](../operating-model.md) 5절, CLAUDE.md 병렬 작업 규칙을 읽고 이어받는다.
> 교대 전에 최신으로 맞춘다. 마지막 갱신: 2026-11-16 (첫 개발 HQ 세션).

## 맡은 일
- 엔진(백테스트·paper), 데이터 수집, 리스크·실행, 시장정보·운영의 개발과 점검을 작업 세션에 배정한다.
- **master 병합·push 창구.** 전략부의 lab 브랜치도 여기서 확인 후 병합한다.
- 결과를 바꾸는 엔진 변경을 [engine-changes.md](../engine-changes.md)에 적는다.
- 요청·미결 결정은 [dev-backlog.md](../dev-backlog.md).

## 진행 중 세션

| 세션 | 브랜치 | 상태 |
|---|---|---|
| 재무 연결 → 펀딩비 (`fin-engine`) | `claude/funding-engine` | 완료, 병합 승인 대기. `trade.py`도 고침 → 공개 함수 유지 확인 후 병합 |
| 신호형 → 비중형 paper 배당 (`trade`) | `claude/paper-dividend` | 작업 중 |

## 예약 작업 (Windows 작업 스케줄러)
`autotrader-collect`(07:30·17:00·로그인), `autotrader-brief`(08:00), `autotrader-health`(12:00·20:00), `autotrader-ops`(23:00), `autotrader-weekly`(토 10:00). 등록 스크립트 `scripts/register_schedule.ps1`.

## 알아 둘 것
- 데이터는 메인 폴더 `data/`에만 있다(git 제외). 작업 폴더에서는 `AUTOTRADER_DATA_DIR` 지정.
- 테스트는 `tests/conftest.py`가 logs/·data/를 임시 경로로 돌린다. 병합 후 메인에서 `uv run pytest` 전후로 `logs/` 변화가 없는지 확인한다.
- 11-16 "뉴스 반복", "Alpaca 키 없음"은 테스트 로그 오염이었다(해결). 로그를 근거로 보고하기 전에 출처를 확인한다.
- DART 배당 과거분은 정기 수집이 이어받는 중 (11-17 저녁 완료 예상).
