"""lab 전략 정의.

공용 백테스트 엔진(src/autotrader/backtest)이 이 모듈의 generate_signals를 호출합니다.
    uv run python -m autotrader.backtest.lab labs/L###-이름 [--quick | --lockbox]
검증을 통과하면 이 파일을 src/autotrader/strategies/ 로 옮겨 paper/live 모드에서 사용합니다.
"""

import pandas as pd


def generate_signals(data, params: dict) -> pd.DataFrame:
    """봉 t 종가까지의 데이터로 정한 목표 비중(날짜 × 종목)을 반환한다. 엔진이 봉 t+1에 체결한다.

    data: Panel — data.close, data.open (그리고 lab.yaml fields에 적은 것)이 날짜 × 종목 DataFrame.
          kr_stock 가격은 수정주가, data.raw_close는 원본가. data.market, data.timeframe.
    반환: 1.0 = 자산 100%. 합 < 1이면 나머지 현금, 음수는 공매도. 없는 날짜는 직전 비중 유지.

    미래참조 금지: rolling(center=True), 전체 기간 평균·표준화, bfill, shift(-n) 금지.
    엔진이 데이터를 잘라 다시 계산해 신호가 같은지 자동으로 확인한다.
    """
    raise NotImplementedError


# 선택: 학습 창 데이터로 파라미터를 고르는 경우에만 정의한다. 워크포워드가 창마다 호출한다.
# def fit(train_data, params: dict) -> dict:
#     return params
