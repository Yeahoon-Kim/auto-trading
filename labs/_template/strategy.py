"""lab 전략 정의.

공용 백테스트 엔진(src/autotrader/backtest)이 이 모듈의 generate_signals를 호출합니다.
검증을 통과하면 이 파일을 src/autotrader/strategies/ 로 옮겨 paper/live 모드에서 사용합니다.
"""

import pandas as pd


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """시점 t까지의 데이터만 사용해 목표 포지션(비중)을 반환한다.

    미래참조 방지: t 시점 신호는 t+1 시점에 체결된다고 가정한다.
    """
    raise NotImplementedError
