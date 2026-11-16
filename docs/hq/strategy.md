# 전략부 HQ 상태판

> 새 전략부 HQ 세션은 이 파일, [operating-model.md](../operating-model.md) 2절 고리 ②·5절, [labs/README.md](../../labs/README.md), CLAUDE.md 병렬 작업 규칙을 읽고 시작한다.
> 교대 전에 최신으로 맞춘다. 마지막 갱신: 2026-11-16 (개발·점검 HQ가 만든 시작판 — 전략부 HQ는 아직 없음).

## 맡은 일
- [labs/BACKLOG.md](../../labs/BACKLOG.md) 우선순위를 제안하고 사용자 확정을 받는다.
- lab 하나당 세션 하나를 작업 폴더에 띄운다(quant-researcher 역할). 판정 전에는 lab 세션과 맥락이 분리된 validator를 거친다.
- 판정은 사용자가 한다. 판정이 나면 lab 브랜치 병합을 개발·점검 HQ에 요청한다 (전략부는 master에 병합하지 않는다).
- 고칠 수 있는 곳: `labs/L###/`, `labs/BACKLOG.md`. 엔진·데이터가 필요하면 [docs/dev-backlog.md](../dev-backlog.md)에 적고 개발·점검 HQ에 알린다.
- [docs/engine-changes.md](../engine-changes.md)를 보고, 영향받는 판정 전 lab을 다시 돌린다.

## 지금 쓸 수 있는 것 (2026-11-16)
- 비중형: KR 주식(상장폐지 포함, 배당 재투자, 재무 `dart_fin` 필드), US 주식(배당 반영, 생존편향 있음), 코인 현물.
- 신호형(`type: trade`): 가능. 단 1회 주문 상한 처리 방식이 미결(dev-backlog 결정 대기).
- 코인 선물 펀딩비: 개발 중(병합 대기).
- 안 되는 것: 시장 간(`cross_market`) 차익거래, llm-desk.
- 작업 폴더에서 백테스트: `AUTOTRADER_DATA_DIR`을 메인 폴더 `data/`로 지정.

## 넘겨받은 미결 사항
- BACKLOG 우선순위 확정 (HQ 제안: 상 = PBR-ROE 상대가치, 국면 필터 단독).
- 사용자 아이디어 "US 12-1 모멘텀"을 BACKLOG에 올리기 (`exposed_from: [L001]`, US는 생존편향 명시).
- 한국 배당 신호는 2020년 이후분이 11-17 저녁쯤 다 찬다. 그 전에는 2024년 이전 배당 시점이 정정일 기준.

## 진행 중 lab
없음. (L001 v1 rejected, 2026-11-15)
