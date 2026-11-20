# L013-us — US 참고 실행 (판정 근거 아님)

- 규칙: 상위 폴더 [`strategy.py`](../strategy.py)를 그대로 쓴다 (`us/strategy.py`는 불러오기만). 가설·진입·청산은 [상위 README](../README.md) 2절.
- 다른 점 (`us/lab.yaml`): 시장 `us_stock`, 이평 5·10·20·50·100·125·200일(사용자 지정 US), 거래대금 ≥ 1,000만 달러, 주가 ≥ 5달러,
  자본 10,000달러, 비교 지수 `SPY_tr_wht`(S&P500 총수익, 배당 원천징수 15%). 학습 비용 근사 0.6%.
- **생존편향**: `us_1d`는 현재 S&P500·Nasdaq100 구성종목만 있다 (상장폐지·편출 종목 없음). L003 validator가 US 효과가 생존편향 구간에서만 나온다고 확인했으므로
  이 결과는 **판정 근거로 쓰지 않고** KR 결과의 방향을 보는 참고로만 쓴다.
- 시도: `us/lab.yaml` `trials`에 따로 센다. `exposed_from`에 L003과 KR lab을 적었다 (같은 규칙을 KR에서 먼저 봄).
- 실행: `uv run python -m autotrader.backtest.lab labs/L013-kr-ma-support/us` + `uv run python labs/L013-kr-ma-support/placebo.py labs/L013-kr-ma-support/us`
- 사전 예측 (lead agent, 2026-11-19): KR보다 Sharpe가 높게 나올 수 있지만(생존편향·미국 장기 상승장) SPY 매수 후 보유보다 낮을 것.
