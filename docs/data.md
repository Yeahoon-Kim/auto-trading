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

| `binance_um_oi` | 미결제약정 + 계정 롱/숏 비율 (1시간) | API | Binance 유니버스, **최근 30일만 제공 → 매일 누적** | 매일 |
| `macro_1d` | VIX·VIX3M·MOVE, S&P500·나스닥·러셀·KOSPI·KOSDAQ, 원/달러·달러·엔, 미 10년 금리, WTI·금·구리 | Yahoo | 2010~ | 매일 |
| `fred` | 미 국채 2·10년, 장단기 금리차, 연방기금금리, 하이일드·투자등급 스프레드, 기대인플레, 금융여건(NFCI), 달러지수 | FRED (키 불필요) | 2005~ (신용 스프레드는 최근 3년만) | 매일 |
| `crypto_fear_greed` | 코인 공포·탐욕 지수 | alternative.me | 2018~ | 매일 |
| `regime_1d` | **국면 지표** (시장정보팀이 계산, 매일 전체 재계산) | 위 데이터에서 계산 | 2015~ | 매일 08:00 |

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
- **거시 지표 시점**: `macro_1d`·`fred`·`regime_1d`의 날짜는 관측일이다. 실제로 알 수 있는 건 다음 날이므로 전략에서는 `shift(1)` 해서 쓴다.
- **FRED 신용 스프레드**(ICE BofA)는 라이선스 문제로 최근 3년만 제공된다. 그 이전 구간의 국면 지표에서는 이 신호가 빠진다.
- **공시·뉴스, 한국 거시 지표**: OpenDART·한국은행 ECOS 키가 필요해서 아직 수집하지 않는다.
- 데이터 스냅샷(lab 재현용)은 백테스트 엔진을 만들 때 함께 구현한다.

## 상태 점검과 알림

```bash
uv run python -m autotrader.data.cli health            # 신선도·오류·디스크 점검
uv run python -m autotrader.intel.market               # 국면 지표 재계산 + 브리핑 출력
```

- 작업 스케줄러: `autotrader-collect`(07:30, 17:00, 로그인), `autotrader-brief`(08:00), `autotrader-health`(12:00, 20:00)
- 수집기 전체가 오류로 죽으면 즉시 텔레그램 알림. 강제 종료처럼 알릴 틈이 없는 경우는 health 점검이 데이터 신선도로 잡아낸다.
- 텔레그램 토큰이 없으면 `logs/alerts.log`에만 남는다.
