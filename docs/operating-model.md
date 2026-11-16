# 에이전트 운용 모델 (2026-11-16 결정)

> 조직도·에이전트 정의는 [architecture.md](architecture.md) 2·8절, 실행 순서는 [roadmap.md](roadmap.md).
> 이 문서는 그 에이전트들이 **서로 어떻게 이어져 돌아가는지**를 정한다.

## 1. 원칙

1. **인수인계는 파일로 한다.** 에이전트끼리 대화로 넘기지 않는다. 받는 쪽이 읽는 파일·형식을 아래 표에 고정한다.
2. **HQ(총괄 세션)는 배정·확인만 한다.** 메인 폴더에서 읽고, 일을 영역별 세션에 나누고, 결과를 확인해 사용자에게 올린다. 코드는 고치지 않는다 (CLAUDE.md 병렬 작업 규칙).
   HQ는 개발·점검 HQ와 전략부 HQ 둘이다 — 5절 세션 조직.
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
| 2 | lab별 시도 횟수 기록, validator가 반영 | `labs/_template`, validator 정의 | ✅ `trials` 블록 (2026-11-16) |
| 3 | 23:00 마감 보고에 lab 진행 상태 추가 | `monitoring/` | ✅ "리서치 진행 상태" 절 (2026-11-16) |
| 4 | 주간 회의 예약 작업 + ops-reporter 주간 모드 | `monitoring/` | ✅ `monitoring/weekly_report.py`, `autotrader-weekly` 토 10:00 (2026-11-16, 등록은 병합 후) |
| 5 | paper 운용데스크를 validated lab에 연결 | `execution/`, `runner/` | 비중형·신호형 paper 경로 ✅ (2026-11-16). validated lab이 나오면 연결 |
| — | validator 의견서 위치 | validator 정의 | ✅ 이미 `labs/<lab>/validation.md`로 정해져 있음 |

## 5. 세션 조직 (2026-11-16 결정)

에이전트 조직(2절, architecture.md 2·8절)은 **만들어지는 시스템 안의** 조직이다. 이 절은 그 시스템을 **만들고 굴리는 Claude Code 세션들의** 조직이다.

### 5-1. 왜 전략부와 개발부로 나누나

- **직무 분리**: 전략 결과가 나쁠 때 엔진을 고쳐 결과를 좋게 만들 수 있으면 검증이 무의미하다. 전략부는 엔진을 고치지 않고 요청만 하고, 개발부는 전략 판정에 관여하지 않는다 (validator가 전략 코드를 못 고치는 것과 같은 원리).
- **맥락 분리**: 엔진·데이터·운영 일과 전략 여러 개의 가설·결과를 한 세션에 쌓으면 판단 근거가 묻힌다.
- **파일 경계**: 전략부는 `labs/L###/`·`labs/BACKLOG.md`만, 개발부는 그 밖의 코드를 맡는다.
- 대가: 엔진 기능이 필요하면 요청을 한 번 거쳐야 하고, 엔진 변경이 지난 lab 결과를 바꿀 수 있다 → 5-4의 파일로 줄인다.

### 5-2. 조직도

```
                        사용자 (CIO) ── 최종 승인: 병합, 판정, paper→live, 리스크 값
                 ┌──────────────┴──────────────┐
          개발·점검 HQ                      전략부 HQ
   (엔진·데이터·운영, master 병합 창구)   (대기열, lab 배정, 판정 정리)
        │                                    │
  영역별 작업 세션 (작업 폴더, 1건씩)     lab 세션 (lab 하나당 하나, 작업 폴더)
   · 백테스트 엔진   · 데이터 수집          · quant-researcher 역할
   · 리스크·실행    · 시장정보·운영        validator (lab 세션과 맥락 분리)
  점검 세션 (메인 폴더 읽기 전용)          fundamental / market analyst (필요할 때)
```

### 5-3. 에이전트 조직 ↔ 세션 조직

| 에이전트 조직 | 실제 실행 주체 | 만들고 고치는 세션 | 운영 감독 |
|---|---|---|---|
| ① 시장정보 (`market-analyst`) | 08:00 예약 작업 | 개발부 (`intel/`) | 개발부(보고 품질), 전략부(해석 활용) |
| ② 퀀트리서치 (`quant-researcher`, `fundamental-analyst`) | lab 세션 | 전략부 | 전략부 |
| ③ 검증 (`validator`) | lab 세션과 분리된 맥락 | 정의 파일은 개발부 | 전략부가 호출, 판정은 사용자 |
| ④ 운용데스크 (paper·live) | 예약 작업 (master 코드) | 개발부 (`execution/`, `runner/`) | 무엇을 돌릴지 전략부+사용자, 잘 도는지 개발부 |
| ⑤ 리스크 (거부권) | 코드 | 개발부 (`risk/`) | `config/risk.yaml` 값은 어느 HQ도 단독으로 못 바꾼다 — 사용자 승인 |
| ⑥ 운영 (수집, `ops-reporter`) | 예약 작업 | 개발부 | 개발부 (마감·주간 보고 → 다음 날 배정) |

### 5-4. 두 HQ 사이의 인수인계

| 파일 | 쓰는 쪽 → 읽는 쪽 | 내용 |
|---|---|---|
| [`labs/BACKLOG.md`](../labs/BACKLOG.md) | 전략부 | 전략 대기열 |
| [`docs/dev-backlog.md`](dev-backlog.md) | 전략부·사용자 → 개발부 | 엔진·데이터·운영 요청과 미결 결정 |
| [`docs/engine-changes.md`](engine-changes.md) | 개발부 → 전략부 | **결과를 바꾸는 엔진 변경** (날짜, 커밋, 영향, 다시 돌릴 lab) |
| lab 브랜치 병합 요청 | 전략부 → 개발부 | 개발 HQ는 `labs/` 안 파일만 바뀌었는지, 판정이 났다면 `validation.md`가 있는지만 확인하고 병합한다. 전략 내용은 다시 판단하지 않는다 |
| HQ 상태판 [`docs/hq/dev.md`](hq/dev.md), `docs/hq/strategy.md` | 각 HQ | 진행 중 세션, 대기 결정, 다음 할 일 — 새 HQ 세션이 이 파일 하나로 이어받는다 |

### 5-5. 세션 운영 규칙

1. **master 병합·push는 개발·점검 HQ만** 한다. 메인 폴더에서 git 작업을 하는 주체가 하나여야 꼬이지 않는다.
2. **영역마다 코드를 고치는 세션은 한 번에 하나.** 같은 영역 작업은 순서대로 한다.
3. **작업 세션은 일 하나만** 하고, 병합되면 보관 처리한다.
4. **점검 세션은 메인 폴더에서 읽기만** 하고 가벼운 모델로 돌린다. 고칠 일이 생기면 그때 작업 폴더를 만든다.
5. **HQ도 대화가 길어지면 교대한다.** 교대 전에 상태판을 최신으로 맞춘다.
6. **작업 폴더에서 백테스트할 때는 `AUTOTRADER_DATA_DIR`을 메인 폴더 `data/`로** 지정한다. 수집기는 메인 폴더에서만 돈다 (`docs/data.md`).
7. **운용부(세 번째 HQ)는 지금 두지 않는다.** live를 시작해 체결 대사·리스크 발동·증액/퇴출 판단이 매일 생기면 그때 검토한다.
