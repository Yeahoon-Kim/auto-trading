# Labs

전략 실험 하나 = lab 하나. 각 lab은 독립적으로 설계·검증합니다.
공용 데이터(`data/processed`)와 공용 엔진(`src/autotrader/backtest`)만 가져다 쓰고, lab끼리는 서로 의존하지 않습니다.

## 진행 과정

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1. 타겟 시장 | KR 주식 / US 주식 / 코인 현물 / 코인 선물 / 시장 간 | `lab.yaml`의 `market` |
| 2. 전략 수립 | **가설과 규칙을 백테스트 전에 먼저 기록** | `README.md` 2절, `strategy.py` |
| 3. 백테스트 | 비용 포함, 워크포워드 | `results/` |
| 4. 유효성 검증 | 아래 고정 기준으로 판정 → 마지막에 잠금 기간 1회 확인 | `README.md` 4절·판정 |

판정 결과는 `validated`(검증 통과), `rejected`(기각), `needs-more`(추가 검증 필요) 중 하나입니다.

## 데이터 기간 구분

```
|────────── 개발 구간 (워크포워드: 학습 → 시험을 앞으로 굴림) ──────────|── 잠금 구간: 최근 6개월 ──|
```

- **잠금 구간**은 항상 "실행일 기준 최근 6개월"입니다. 시간이 지나면 구간도 따라 이동합니다.
- 전략을 개발하는 동안에는 잠금 구간을 보지 않습니다. 최종 판정할 때 **단 한 번만** 실행하고, 그 결과를 보고 전략을 고쳤다면 그 lab은 새 버전으로 처음부터 다시 검증합니다.
- lab은 판정할 때의 잠금 구간 날짜를 기록합니다 (`lab.yaml`의 `lockbox`).

## 검증 통과 기준 (고정, 모든 lab 공통)
시도 횟수와 상관없이 같은 기준을 적용합니다. lab마다 더 엄격하게 정할 수는 있지만 완화할 수는 없습니다.

1. 수수료, 세금, 슬리피지를 반영한 워크포워드 시험 구간 합산 Sharpe가 1.0 이상이다.
2. 시험 구간 Sharpe가 학습 구간의 50% 이상이다. 학습 구간 Sharpe가 0 이하면 탈락이다 (기준선이 음수가 되면 손실 전략도 통과하므로).
3. 파라미터를 ±20% 바꿔도 Sharpe가 0.5 이상을 유지한다.
4. 비용을 2배로 올려도 수익이 남는다.
5. 거래 횟수가 30회 이상이다.
6. 잠금 구간 수익이 0 이상이고 최대 낙폭(MDD)이 개발 구간 MDD의 1.5배 이내다.
7. 생존편향과 미래참조 점검을 통과한다.

### 시도 횟수 (`lab.yaml`의 `trials`)
같은 개발 구간을 여러 번 보면 우연히 기준을 넘는 결과가 나올 확률이 커진다 (다중 검정).
- 개발 구간 결과를 볼 때마다 실행 **전에** `trials.count`를 1 올리고 `trials.log`에 날짜·바꾼 것을 한 줄 적은 뒤 커밋한다.
  파라미터만 바꾼 재실행, `--quick` 실행, 이전 version의 실행도 센다.
- 다른 lab의 결과나 판정을 보고 나온 아이디어는 그 lab이 이미 같은 개발 구간을 본 것이므로 `trials.exposed_from`에 그 lab ID를 적는다.
- 위 고정 기준은 그대로다. 다만 `count`가 많거나 `exposed_from`이 있으면 validator가 의견서에 그 사실을 적고,
  기준을 조금 넘는 경계선 결과는 보류 쪽으로 판단한다.

## LLM 판단 전략 (category: llm)
LLM이 판단에 들어가는 전략은 과거 백테스트 대신 **앞으로 검증**(모의투자) 트랙을 따른다.
방법과 추가 통과 기준은 [docs/architecture.md](../docs/architecture.md) 7절을 따른다.
이 경우 검증 단계 자체가 paper 단계이므로, 통과하면 바로 live 후보가 된다.

## 검증 이후
`validated` → **paper**(모의투자) → **live**(소액으로 시작해 성과에 따라 증액) → 퇴출 조건에 걸리면 **retired**.
퇴출 조건은 전략마다 다르므로 lab의 `lab.yaml`에서 `retirement`로 정의합니다.

## 새 lab 만들기
`_template`을 `L###-짧은-이름` 형식으로 복사합니다 (예: `L001-btc-funding-arb`).

## 백테스트 실행 (공용 엔진)
```
uv run python -m autotrader.backtest.lab labs/L001-이름            # 개발 구간 워크포워드 + 기준 1~5·7(자동분)
uv run python -m autotrader.backtest.lab labs/L001-이름 --quick    # ±20% 민감도 생략 (개발 중)
uv run python -m autotrader.backtest.lab labs/L001-이름 --lockbox  # 최종 판정 때 1회 (기준 6). lab.yaml lockbox에 자동 기록
```
- **작업 폴더(git worktree)에서 실행할 때**: 작업 폴더의 `data/`는 비어 있다. `AUTOTRADER_DATA_DIR`을 메인 폴더 `data/`로 지정하면
  메인 데이터를 읽고, 결과는 작업 폴더의 `labs/<lab>/results/`에 생긴다. 작업 폴더에서는 수집기를 돌리지 않는다(메인 데이터에 쓰게 된다).
  자세한 것은 `docs/data.md`의 "데이터 위치" 참고.
  ```powershell
  $env:AUTOTRADER_DATA_DIR = "C:\Users\apple\Claude\auto-trading\data"
  uv run python -m autotrader.backtest.lab labs/L001-이름 --quick
  ```
- 결과: `labs/<lab>/results/<실행시각>/report.md`, `summary.json`, 수익률 parquet. 판정은 하지 않는다 (validator 검증 후).
- 데이터: `autotrader.backtest.data.load()` — 잠금 구간은 `--lockbox` 없이는 읽히지 않는다.
  KR은 `krx_1d`(상장폐지 포함) + 수정계수, US는 `us_1d`(현재 종목 → 생존편향 있음), 코인은 1분봉을 `timeframe`으로 묶음.
- 체결: 봉 t 종가에 정한 비중을 봉 t+1 시가(`fill: next_open`)나 종가(`next_close`)에 맞춘다.
- 비용: 회전율(가격 변동으로 흘러간 비중을 되돌리는 것 포함) × (수수료 + 슬리피지), 롱 매도분 × 매도세. 2배 비용도 자동 계산.
  KR 매도세는 `config/settings.yaml`의 연도별 세율표로 매도일마다 적용한다 (2025년 0.15%, 2026년 0.20% 등).
- 금액·실행 방식 (`lab.yaml` `execution`): `capital`(초기 자본), `monthly_contribution`(월초 적립금),
  `whole_shares`(정수 주식, 1주 미만은 현금), `rebalance`:
  `always`(매 봉 목표로), `on_change`(목표가 바뀐 종목만 — 나머지는 불어난 비중 그대로라 합이 1을 넘을 수 있음),
  `on_signal`(월 리밸런싱용: 신호 표에 행이 있는 날마다 전 종목을 목표로, 같은 목표를 다시 내도 맞춤.
  매일 행을 내는 전략이면 목표가 바뀐 날만. 상장폐지는 그 종목만 청산하고 남은 돈은 다음 신호까지 현금).
  자본을 주면 리포트에 투입액·평가액·투입 대비 수익률·XIRR이 붙는다. Sharpe·MDD는 적립금 영향을 뺀 시간가중 수익률 기준.
- US 주식: 배당은 기본으로 원천징수 15%를 떼고 재투자 (`dividend_tax`로 변경), `currency: KRW`면 그날 원/달러(ECOS)로 환산해 환율 변동까지 수익에 넣는다.
- 상장폐지 종목은 마지막 가격 다음 봉에 그 가격으로 강제 청산한다 (낙관적 가정 — 리포트에 횟수 표시).
- 미래참조 자동 검사: 데이터를 3곳에서 잘라 다시 계산한 신호가 원래와 같은지 비교한다. 통과해도 데이터 시점 규칙(거시 `shift(1)`, 공시 다음 날, 재무 +45/+90일)은 validator가 따로 본다.
- KR 주식: 배당은 `kr_dividend_schedule` 배당락일에 배당소득세 15.4%를 떼고 재투자한다 (`dividend_tax`로 변경, 리포트에 "KR 배당 재투자 반영" 표시).
- 배당세: KR 15.4%, US 원천징수 15%를 배당 재투자 때 뗀다.
- 반영 안 하는 것: 공매도 대차 비용, 선물 펀딩비, 시장 충격, 양도소득세, 현금 이자.

## 벤치마크
US 주식 lab은 S&P500 매수 후 보유(`SPY_tr_wht`)와 같은 기간으로 비교합니다.
`autotrader.backtest.benchmark.benchmark_returns()`로 받고, 기준 수치는 [docs/research/benchmark-sp500.md](../docs/research/benchmark-sp500.md)를 봅니다.

## 현황판

| ID | 이름 | 시장 | 유형 | 상태 | 판정 |
|---|---|---|---|---|---|
| L001 | [kr-momentum-regime](L001-kr-momentum-regime/README.md) — KR 12-1 모멘텀 + KOSPI 200일선 | kr_stock | classic | rejected | 2026-11-15 기준 1·3·4 실패 (Sharpe −0.13, MDD −85%) |
