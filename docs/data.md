# 데이터

## 사용법

```bash
uv run python -m autotrader.data.cli status                 # 데이터셋별 수집 현황
uv run python -m autotrader.data.cli collect binance        # 개별 수집 (백필과 갱신은 같은 명령)
powershell -ExecutionPolicy Bypass -File scripts/collect_all.ps1   # 전체 병렬 실행
```

lab에서 읽기:

```python
from autotrader.data import store
df = store.read("kr_1d", "005930", start="2020-01-01")     # 한 종목
f  = store.read("kr_adj_factor", "005930")                 # 수정주가 = close × factor
store.query("SELECT ... FROM 'data/processed/binance_um_funding/*/*.parquet'")  # 여러 종목
```

## 데이터셋

| dataset | 내용 | 소스 | 범위 | 갱신 |
|---|---|---|---|---|
| `binance_spot_1m` | 현물 1분봉 | data.binance.vision + API | 거래대금 상위 30개(무기한 선물 있는 코인), 2022~ | 매일 |
| `binance_um_1m` | USDT-M 무기한 선물 1분봉 | 같음 | 같음 | 매일 |
| `binance_um_funding` | 펀딩비 (8시간 등) + 마크가격 | API | 2020~ | 매일 |
| `upbit_1m` | Upbit KRW 1분봉 | API | Binance 유니버스와 겹치는 코인, 2024~ | 매일 |
| `kr_1d` | 국내 원본 일봉 + 거래대금 | KIS | KOSPI·KOSDAQ 전 종목 + 2010년 이후 상장폐지 종목, 2010~ | 매일 |
| `kr_adj_factor` | 수정계수 (수정종가/원본종가) | KIS | 같음 | 변경 감지 시 전체 재수집 |
| `kr_1m` | 국내 1분봉 | KIS | 시총 상위 KOSPI 200 + KOSDAQ 150, KIS 보관 한도(약 1년)까지 | 매일 |
| `us_1d` | 미국 일봉 + adj_close | Yahoo | S&P500 + Nasdaq100, 2010~ | 매일 |
| `us_1m` | 미국 1분봉 | Yahoo | 같음, 최근 30일부터 누적 | 매일 |

- 공통 컬럼: `ts`(봉 시작 시각, UTC), `open`, `high`, `low`, `close`, `volume`. 미완성 봉은 저장하지 않는다.
- 유니버스는 `data/processed/universe/{name}/{날짜}.parquet`로 날짜별 기록되고, 30일이 지나면 새로 만든다.
- 수집 상태: `data/state.sqlite` (데이터셋·종목별 first/last ts, 백필 완료 여부, 마지막 오류)
- 로그: `logs/collect_YYYY-MM-DD.log`

## 알려진 한계 (lab 결과 해석 시 주의)

- **KR 분봉 유니버스**: 지수 구성종목 목록은 KRX 로그인이 필요해서 시총 상위로 근사했다 (실제 KOSPI200/KOSDAQ150과 일부 다름).
- **US**: 현재 구성종목 기준이라 **생존편향**이 있다. Yahoo OHLC는 이미 분할 반영 값이라 진짜 원본 가격이 아니다.
- **Binance**: 현재 거래대금 상위 코인을 고른 것이라, 과거 시점 기준으로 보면 **생존편향**이 있다 (지금까지 살아남아 커진 코인 위주).
- **Upbit**: 거래가 없던 분은 봉이 없다.
- **KR 종목코드 재사용**: 상장폐지 종목의 코드를 다른 회사가 다시 쓰는 경우가 드물게 있다.
- 데이터 스냅샷(lab 재현용)은 백테스트 엔진을 만들 때 함께 구현한다.
