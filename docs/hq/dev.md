# 개발·점검 HQ 상태판

> 새 개발·점검 HQ 세션은 이 파일, [operating-model.md](../operating-model.md) 5절, CLAUDE.md 병렬 작업 규칙을 읽고 이어받는다.
> 교대 전에 최신으로 맞춘다. 마지막 갱신: 2026-11-18 (2대 개발 HQ 세션 인수 시점).

## 맡은 일
- 엔진(백테스트·paper), 데이터 수집, 리스크·실행, 시장정보·운영의 개발과 점검을 작업 세션에 배정한다.
- **master 병합·push 창구.** 전략부의 lab 브랜치도 여기서 확인 후 병합한다.
- 결과를 바꾸는 엔진 변경을 [engine-changes.md](../engine-changes.md)에 적는다.
- 요청·미결 결정은 [dev-backlog.md](../dev-backlog.md).

## 진행 중 세션

| 세션 | 브랜치 | 상태 |
|---|---|---|
| (없음) | — | 11-18 기준 개발부 작업 세션 없음. 펀딩비·paper 배당·US 분할/주식수·US `market_cap`까지 모두 병합됨 (`8db99e7`, origin과 같음) |

## 다음 할 일
1. **11-18 17:00 정기 수집 뒤 US 시총 실데이터 확인** — `us_splits`가 채워졌는지, 2024-06·2015-06 시총 상위 10이 맞는지 (병합 전 오류 예: 2024-06 NVDA 1/10, 2015-06 GE 8배). 이번 수집은 EDGAR companyfacts 일괄 파일을 다시 받아 오래 걸린다.
2. 맞으면 **전략부 HQ에 L002 시작 신호**. 전략부에는 "HQ 신호 전까지 L002 재시작 금지"라고 전해 둔 상태.
3. 그 밖의 일은 [dev-backlog.md](../dev-backlog.md) "결정 대기"·"대기" 순.

## 예약 작업 (Windows 작업 스케줄러)
`autotrader-collect`(07:30·17:00·로그인), `autotrader-brief`(08:00), `autotrader-health`(12:00·20:00), `autotrader-ops`(23:00), `autotrader-weekly`(토 10:00). 등록 스크립트 `scripts/register_schedule.ps1`.

## 알아 둘 것
- 데이터는 메인 폴더 `data/`에만 있다(git 제외). 작업 폴더에서는 `AUTOTRADER_DATA_DIR` 지정.
- 테스트는 `tests/conftest.py`가 logs/·data/를 임시 경로로 돌린다. 병합 후 메인에서 `uv run pytest` 전후로 `logs/` 변화가 없는지 확인한다.
- 11-16 "뉴스 반복", "Alpaca 키 없음"은 테스트 로그 오염이었다(해결). 로그를 근거로 보고하기 전에 출처를 확인한다.
- DART 배당 과거분은 정기 수집이 이어받는 중 (11-17 저녁 완료 예상 — 완료 여부 미확인).
- 대시보드(stock-analysis-site) 흡수 검토 결론 (11-17, 사용자 결정): 두 저장소는 합치지 않고 따로 둔다.
