# Auto Trading

AI·알고리즘 기반 자동매매 연구 및 운용 프로젝트.

**데이터 수집 → Lab(시장 선택 → 전략 → 백테스트 → 검증) → paper(모의) / live(실전)** 순서로 진행합니다. 자세한 계획은 [docs/roadmap.md](docs/roadmap.md)에 있습니다.

- 주식 (국내/해외): 한국투자증권 OpenAPI + MCP
- 암호화폐 (현물/선물): Binance API, 김치 프리미엄 비교용으로 Upbit API

## 폴더 구성

```
auto-trading/
├── config/                 설정 (비밀 값 제외)
│   ├── settings.yaml       시장, 수수료, 데이터 소스
│   └── risk.yaml           리스크 한도 + 실전 전환 조건(live_gate)
├── labs/                   전략 실험실: lab 하나 = 독립 실험 하나
│   ├── README.md           진행 과정, 검증 기준, 현황판
│   └── _template/          새 lab 복사용 (lab.yaml, README.md, strategy.py)
├── src/autotrader/
│   ├── core/               설정 로딩, 로깅, 공통 타입
│   ├── data/               수집기(collectors/), 저장, 주기적 갱신
│   ├── backtest/           공용 백테스트 엔진, 비용 모델, 성과 지표
│   ├── strategies/         검증을 통과해 승격된 전략
│   ├── brokers/            KIS, Binance, Upbit 연결 (모의/실전 공통 인터페이스)
│   ├── risk/               주문 전 리스크 검사, 킬 스위치
│   ├── execution/          주문 실행·체결 관리
│   ├── runner/             paper/live 모드 실행기
│   └── monitoring/         텔레그램 알림
├── deploy/                 Docker / Oracle 서버 배포 설정
├── docs/                   roadmap, research(공통 리서치), journal(매매 일지)
├── notebooks/              데이터 탐색용
├── scripts/                실행 스크립트
├── tests/
├── data/                   raw / processed  (git 제외)
├── results/                paper / live 운용 결과 (git 제외)
└── logs/                   (git 제외)
```

## 시작하기

```bash
uv sync --extra research --extra backtest --extra crypto --extra dev
cp .env.example .env   # 그 다음 .env 에 API 키 입력
```

## 데이터 수집

수집 대상, 사용법, 알려진 한계는 [docs/data.md](docs/data.md)에 정리돼 있습니다.

```bash
uv run python -m autotrader.data.cli status
powershell -ExecutionPolicy Bypass -File scripts/collect_all.ps1
```

## 원칙

1. 모든 전략은 lab 검증 → 모의투자 → 소액 실전 → 단계적 증액 순서를 거친다 (`config/risk.yaml`의 `live_gate`).
2. 백테스트에는 수수료·세금·슬리피지를 반드시 포함하고, 생존편향과 미래참조(look-ahead)를 점검한다.
3. API 키는 `.env`에만 둔다. 거래소 키는 출금 권한을 끄고 IP를 제한한다.
4. 실전 주문은 항상 `risk/` 검사를 통과해야 하며, 킬 스위치로 즉시 멈출 수 있어야 한다.
