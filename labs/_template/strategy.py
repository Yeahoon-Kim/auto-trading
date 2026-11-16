"""lab 전략 정의.

공용 백테스트 엔진(src/autotrader/backtest)이 이 모듈을 호출합니다.
    uv run python -m autotrader.backtest.lab labs/L###-이름 [--quick | --lockbox]
lab.yaml type에 맞는 함수 하나만 남기고 나머지는 지웁니다.
    type: portfolio → generate_signals (목표 비중)
    type: trade     → generate_trades  (진입 신호)
검증을 통과하면 이 파일을 src/autotrader/strategies/ 로 옮겨 paper/live 모드에서 사용합니다.
"""

import pandas as pd


def generate_signals(data, params: dict) -> pd.DataFrame:
    """[type: portfolio] 봉 t 종가까지의 데이터로 정한 목표 비중(날짜 × 종목)을 반환한다. 엔진이 봉 t+1에 체결한다.

    data: Panel — data.close, data.open (그리고 lab.yaml fields에 적은 것)이 날짜 × 종목 DataFrame.
          kr_stock 가격은 수정주가, data.raw_close는 원본가. data.market, data.timeframe.
          수정주가(close·open)는 배당 재투자까지 반영돼 과거 가격 수준이 실제보다 낮다 (최근 가격 = 실제).
          가격 수준 필터(최소 주가 등)나 배당수익률 분모에는 raw_close를 쓴다.
          배당 신호는 fields에 dps·div_yield를 적고 data.dps, data.div_yield로 읽는다 (그날 알 수 있는 값만).
    반환: 1.0 = 자산 100%. 합 < 1이면 나머지 현금, 음수는 공매도. 없는 날짜는 직전 비중 유지.

    미래참조 금지: rolling(center=True), 전체 기간 평균·표준화, bfill, shift(-n) 금지.
    엔진이 데이터를 잘라 다시 계산해 신호가 같은지 자동으로 확인한다.
    """
    raise NotImplementedError


def generate_trades(data, params: dict) -> pd.DataFrame:
    """[type: trade] 봉 t 종가까지 보고 낸 진입 신호 목록. 재생기가 데스크와 같은 코드로 처리한다:
    make_plan(가드 + 손절 기반 사이징) → 봉 t+1 체결 → 매 봉 check_position(손절·목표가·장 마감 전 청산).

    data: Panel — open·high·low·close (high·low는 자동으로 읽음).
    반환: DataFrame 또는 dict 목록. 한 행 = 신호 하나.
        ts        판단 봉 (그 봉 종가까지의 데이터만 썼어야 함)
        symbol    종목
        side      long | short
        stop      손절가 (없으면 atr 열 → 진입가 ± atr×2)
        target    목표가 (선택)
        entry     지정가 (선택). 비우면 다음 봉 시가 진입. 주면 다음 봉 하루짜리 지정가 (안 닿으면 버림)
        intraday  true면 당일 청산 (kr_stock·us_stock: 그날 종가에 정리)
        reason    사유 (선택)
    같은 봉에서 손절·목표가가 둘 다 닿으면 손절로 친다. 이미 보유 중인 종목의 신호는 버린다.
    수량은 전략이 정하지 않는다 — lab.yaml trade 블록(거래당 위험·heat·종목 비중·보유 종목 수)으로 정해진다.

    예시 (20일 신고가 돌파, 손절 -5%, 목표 +10%):
        c, hi = data.close, data.close.rolling(params["n"]).max().shift(1)
        hit = (c > hi).stack()
        return [{"ts": ts, "symbol": s, "side": "long", "stop": c.at[ts, s] * 0.95,
                 "target": c.at[ts, s] * 1.10} for (ts, s), v in hit.items() if v]

    미래참조 금지는 generate_signals와 같다. 재생기가 데이터를 잘라 다시 계산해 신호가 같은지 확인한다.
    """
    raise NotImplementedError


def valuation(data) -> dict[str, pd.DataFrame]:
    """[예시] 재무 필드로 만드는 가치 지표 (kr_stock). lab.yaml fields: [market_cap, equity, net_income_ttm].
    모두 그날 알 수 있는 값이라 그대로 신호에 써도 된다. 자본·이익이 0 이하인 칸은 비율이 뜻이 없어 NaN으로 둔다.
        pbr = 시가총액 / 자본총계,  per = 시가총액 / 순이익(TTM),  roe = 순이익(TTM) / 자본총계
    주의: market_cap은 그날 종가 기준, equity는 최근 공개 분기말 값이다 (최대 몇 달 차이).
    연결 재무의 자본·순이익은 비지배지분을 포함한다 (지배주주 몫만 따로 있지 않음).
    필요 없으면 지운다."""
    eq = data.equity.where(data.equity > 0)
    ni = data.net_income_ttm
    out = {"pbr": data.market_cap / eq,
           "per": data.market_cap / ni.where(ni > 0),
           "roe": ni / eq}
    if "eps_ttm" in data:   # 지배주주 기준 PER (fields에 raw_close, eps_ttm). 우선주까지 포함한 회사 전체 이익 대비 주가
        eps = data.eps_ttm
        out["per_controlling"] = data.raw_close / eps.where(eps > 0)
    return out


# 선택: 학습 창 데이터로 파라미터를 고르는 경우에만 정의한다. 워크포워드가 창마다 호출한다.
# def fit(train_data, params: dict) -> dict:
#     return params
