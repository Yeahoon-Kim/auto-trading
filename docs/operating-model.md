# 에이전트 운용 모델 (2026-11-16 결정)

> 조직도·에이전트 정의는 [architecture.md](architecture.md) 2·8절, 실행 순서는 [roadmap.md](roadmap.md).
> 이 문서는 그 에이전트들이 **서로 어떻게 이어져 돌아가는지**를 정한다.

## 1. 원칙

1. **인수인계는 파일로 한다.** 에이전트끼리 대화로 넘기지 않는다. 받는 쪽이 읽는 파일·형식을 아래 표에 고정한다.
2. **HQ(총괄 세션)는 배정·확인만 한다.** 메인 폴더에서 읽고, 일을 영역별 세션에 나누고, 결과를 확인해 사용자에게 올린다. 코드는 고치지 않는다 (CLAUDE.md 병렬 작업 규칙).
3. **관문은 사람이 연다.** lab 판정 반영, master 병합, paper→live 승격은 사용자 승인 뒤에만 한다.
4. **주문 경로에는 LLM이 없다.** LLM은 해석·연구·검토·보고만 한다 (architecture.md 3절).

## 2. 네 개의 고리

### 고리 ① 일일 자동 운영 — 코드 + 빠른 모델, 사람 개입 없음

```
07:30·17:00 수집 ─▶ 08:00 market-analyst 브리핑 ─▶ 12:00·20:00 상태 점검 ─▶ 23:00 ops-reporter 마감
                      │ regime_1d (데이터)                                     │
                      ▼                                                        ▼
             전략이 조건으로 사용                                  다음 날 HQ가 처리할 문제 목록
```

- 예약 작업: `autotrader-collect`, `autotrader-brief`, `autotrader-health`, `autotrader-ops` (`scripts/register_schedule.ps1`).
- 23:00 마감 보고에는 수집·LLM 호출·스캐너 기록과 함께 **lab 진행 상태**(lab별 `lab.yaml`의 `status`, 판정 대기 건)를 넣는다.
- 아침에 HQ가 마감 보고의 🔴 항목을 담당 영역별로 나눠 세션에 배정한다 (예: 수집 오류 → 데이터 수집 세션).

### 고리 ② 리서치 — 깊은 모델, 사용자 승인이 관문

```
아이디어 출처                       HQ + 사용자        quant-researcher       validator          사용자
market-analyst (국면 변화)  ┐                            (작업 폴더)
fundamental-analyst (섹터)  ├─▶ labs/BACKLOG.md ─▶ 우선순위 ─▶ lab 작성·백테스트 ─▶ 판정 의견서 ─▶ 승인/기각
스캐너·기각된 lab·사용자    ┘                                                       │
                                                         rejected → BACKLOG의 "기각 기록"으로
```

- **리서치 대기열**: [`labs/BACKLOG.md`](../labs/BACKLOG.md). 시장정보·펀더멘털 보고서, 기각된 lab의 "다음 단계"가 여기로 들어와 lab 후보가 된다.
- **validator는 판정 전 필수 관문**이다. 의견서는 `labs/<lab>/validation.md` (validator 정의 참고). `lab.yaml`의 `status` 반영은 사용자 승인 뒤에 한다.
- **시도 횟수**: 같은 개발 구간을 보고 나온 아이디어, 파라미터만 바꾼 재시도는 lab에 시도 횟수로 남기고, validator는 시도 횟수가 많을수록 더 엄격하게 본다 (다중 검정 보정).

### 고리 ③ 실행 — 단계 C 이후, LLM 없음

```
validated lab ─▶ 운용데스크(paper, 코드) ─▶ 리스크 한도(PASS/BLOCK) ─▶ 체결·장부 ─▶ ops-reporter 대사·손익
```

- 운용데스크 골격은 `execution/paper.py`, `runner/paper.py` (2026-11-15 병합). 퇴출 조건은 lab의 `retirement`로 감시한다.

### 고리 ④ 주간 회의 — 매주, 반자동

- ops-reporter: 주간 자료 (전략별 성과 vs 기대치, 리스크 발동, 데이터 상태).
- validator: 운용 중인 전략의 퇴출 조건 점검.
- HQ + 사용자: `labs/BACKLOG.md` 재정렬, 다음 주 배정.

## 3. 인수인계 표

| 보내는 쪽 | 파일 | 받는 쪽 |
|---|---|---|
| 수집(코드) | `data/processed/*`, 수집 상태 | 전 팀 |
| market-analyst | `reports/market/{날짜}.md`, `regime_1d` | 전략(조건), HQ, BACKLOG |
| fundamental-analyst | `docs/research/{날짜}-{대상}.md` | HQ, BACKLOG |
| HQ + 사용자 | `labs/BACKLOG.md` | quant-researcher |
| quant-researcher | `labs/L###/` (`lab.yaml`, `README.md`, `results/`) | validator |
| validator | `labs/L###/validation.md` | 사용자(판정), BACKLOG(기각 기록) |
| 운용데스크·리스크(코드) | 주문·체결·리스크 판단 기록 | ops-reporter |
| ops-reporter | 일일·주간 보고 (텔레그램, `logs/task-daily_report-*.log`) | HQ, 사용자 |

## 4. 빈 곳과 진행 (2026-11-16)

| # | 할 일 | 영역 | 상태 |
|---|---|---|---|
| 1 | 리서치 대기열 파일·형식 | 문서 | ✅ `labs/BACKLOG.md` |
| 2 | lab별 시도 횟수 기록, validator가 반영 | `labs/_template`, validator 정의 | 진행 |
| 3 | 23:00 마감 보고에 lab 진행 상태 추가 | `monitoring/` | ✅ "리서치 진행 상태" 절 (2026-11-16) |
| 4 | 주간 회의 예약 작업 + ops-reporter 주간 모드 | `monitoring/` | ✅ `monitoring/weekly_report.py`, `autotrader-weekly` 토 10:00 (2026-11-16, 등록은 병합 후) |
| 5 | paper 운용데스크를 validated lab에 연결 | `execution/`, `runner/` | 대기 (validated lab이 나온 뒤) |
| — | validator 의견서 위치 | validator 정의 | ✅ 이미 `labs/<lab>/validation.md`로 정해져 있음 |
