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
store.query(f"SELECT ... FROM '{store.DATA_DIR.as_posix()}/binance_um_funding/*/*.parquet'")  # 여러 종목
```

경로는 `'data/processed/...'`처럼 상대 경로로 쓰지 말고 `store.DATA_DIR`에서 만든다 (실행 위치·`AUTOTRADER_DATA_DIR`과 무관하게 맞는 폴더를 본다).

## 데이터 위치 (`AUTOTRADER_DATA_DIR`)

- 데이터 루트는 `processed/`와 `state.sqlite`가 들어 있는 `data/` 폴더다. 코드에서는 `store.DATA_ROOT`, `store.DATA_DIR`(= `processed/`), `store.STATE_DB`.
- 기본값은 코드 위치 기준 `data/` (메인 폴더면 메인 `data/`). 환경변수 또는 `.env`의 `AUTOTRADER_DATA_DIR`이 있으면 그 폴더를 쓴다. 지정한 폴더가 없으면 import 때 바로 오류를 낸다.
- **작업 폴더(git worktree)에서 백테스트할 때**: 작업 폴더의 `data/`는 비어 있으니, 이 변수를 메인 폴더 `data/`로 지정한다.
  ```powershell
  $env:AUTOTRADER_DATA_DIR = "C:\Users\apple\Claude\auto-trading\data"
  uv run python -m autotrader.backtest.lab labs/L001-kr-momentum-regime --quick
  ```
  lab 결과(`labs/<lab>/results/`)는 작업 폴더 쪽에 생긴다. `config/`·`logs/`·`reports/`도 작업 폴더 기준 그대로다.
- **작업 폴더에서는 수집기(`autotrader.data.cli collect`, `scripts/collect_all.ps1`), 국면 재계산(`intel.market`), 스캐너를 돌리지 않는다.**
  이 변수를 켠 채 돌리면 메인 `data/`에 직접 쓰게 되어, 정기 수집(메인 폴더 코드)과 충돌하거나 검토 안 된 코드로 데이터를 바꾸게 된다.
- 테스트(`uv run pytest`)는 변수가 켜져 있어도 `tests/conftest.py`가 임시 폴더로 돌리므로 메인 `data/`를 건드리지 않는다.
- 예외: LLM 판단 기록(`data/llm/decisions.jsonl`)은 데이터 루트와 상관없이 코드 위치 기준이다.

## 데이터셋

| dataset | 내용 | 소스 | 범위 | 갱신 |
|---|---|---|---|---|
| `binance_spot_1m` | 현물 1분봉 | data.binance.vision + API | 거래대금 상위 30개(무기한 선물 있는 코인), 2017-08~ (상장 이후분, 앞쪽은 아카이브 월별 파일로 거꾸로 채움) | 매일 |
| `binance_um_1m` | USDT-M 무기한 선물 1분봉 | 같음 | 같음, 2020-01~ (아카이브 최초) | 매일 |
| `binance_um_funding` | 펀딩비 (8시간 등) + 마크가격 | API | 2020~ | 매일 |
| `upbit_1m` | Upbit KRW 1분봉 | API | Binance 유니버스와 겹치는 코인, 2017-09~ (업비트 개장) | 매일 |
| `kr_1d` | 국내 원본 일봉 + 거래대금 | KIS | KOSPI·KOSDAQ 전 종목 + 2010년 이후 상장폐지 종목, 2010~ | 매일 |
| `kr_adj_factor` | 수정계수 (수정종가/원본종가) | KIS | 같음 | 변경 감지 시 전체 재수집 |
| `kr_1m` | 국내 1분봉 | KIS | 시총 상위 KOSPI 200 + KOSDAQ 150, KIS 보관 한도(약 1년, 2025-11-05 이전 없음 확인)까지 | 매일 |
| `us_1d` | 미국 일봉 + adj_close | Yahoo | S&P500 + Nasdaq100, 2010~ | 매일 |
| `us_1m` | 미국 1분봉 | Yahoo | 같음, 최근 30일부터 누적 (Yahoo·KIS 해외분봉 모두 약 1개월만 제공, 더 과거는 유료·별도 계정 필요) | 매일 |
| `us_1m_sip` | 미국 1분봉 장기 이력 (SIP 전체 거래소, 원본 가격, 프리·애프터 포함) | Alpaca 무료 플랜 (키 필요) | 같음, 2016~ | 매일 |

| `binance_um_oi` | 미결제약정 + 계정 롱/숏 비율 (1시간) | API | Binance 유니버스, **최근 30일만 제공 → 매일 누적** | 매일 |
| `macro_1d` | VIX·VIX3M·MOVE, S&P500·나스닥·러셀·KOSPI·KOSDAQ, 원/달러·달러·엔, 미 10년 금리, WTI·금·구리 | Yahoo | 2010~ | 매일 |
| `fred` | 미 국채 2·10년, 장단기 금리차, 연방기금금리, 하이일드·투자등급 스프레드, 기대인플레, 금융여건(NFCI), 달러지수 | FRED (키 불필요) | 2005~ (신용 스프레드는 최근 3년만) | 매일 |
| `crypto_fear_greed` | 코인 공포·탐욕 지수 | alternative.me | 2018~ | 매일 |
| `krx_1d` (symbol=ALL) | KOSPI·KOSDAQ **전 종목** 원본 일봉 + 시가총액 + 상장주식수 (그날 상장 종목 전부 → 생존편향 없음) | KRX Open API | 2010~ | 매일 (KRX는 다음 영업일 공개) |
| `dart_disclosures` (symbol=ALL) | 전체 상장사 공시 목록 (접수번호, 종목, 보고서명, 제출인) | OpenDART | 2024~ (하루 1.5만 호출씩 백필) | 매일 |
| `dart_fin` (symbol=ALL) | 상장사 **주요계정 재무**: 매출·영업이익·세전이익·순이익·총포괄손익(3개월 값 + `_cum` 누적), 유동·비유동 자산·부채, 자본금·이익잉여금·자본총계. 연결 우선, 없으면 별도. `filed` = 실제 접수일 | OpenDART `fnlttMultiAcnt` | 2015~, 2015년 이후 KRX 상장 종목(상장폐지 포함 중 DART 매핑되는 것) | 매일 (최근 2개 사업연도 재수집) |
| `dart_dividend` (symbol=ALL) | 사업보고서 **배당에 관한 사항**: 주당 현금배당금(`dps_common`·`dps_pref`, 원, 중간·분기배당 포함 연간 합계), 현금배당수익률(%), 현금배당성향(%), 현금·주식배당 총액(백만원), 주당 주식배당, 순이익·EPS(연결 우선). 회사·사업연도당 1행, 배당 기록 없음은 `has_data=False`. `ts` = 결산일, `filed` = 접수일 | OpenDART `alotMatter` | 사업연도 2015~, `dart_fin`과 같은 종목(상장폐지 포함) | 매일 (실행당 8천 호출씩 백필 → 이후 새 사업보고서만) |
| `kr_dividend_schedule` (symbol=ALL) | KR **배당 기준일·배당락일**: `ts` = 기준일, `ex_date` = 배당락일(KRX 거래일로 계산: 기준일 이하 마지막 거래일의 전 거래일), `dps`(주당 현금배당금, 원), `kind`(결산·중간·분기), `stock_kind`(보통·우선), `divi_rate`(액면 대비 %), 주식배당률, `pay_date`. 공시일 없음 → 배당락 가격 보정(총수익)용 | KIS 예탁원정보 배당일정 (`HHKDB669102C0`) | 2010년 이후 KRX 상장 종목(상장폐지 포함), 기준일 2010~ (1년 앞 발표분까지) | 매일 (상장 종목은 7일마다 재조회) |
| `dart_periodic_filings` (symbol=ALL) | 2015~2023 정기보고서(사업·반기·분기) 목록, 정정 포함: `ts`(접수일), `rcept_no`, `kind`, `period`('YYYY.MM'), `correction`. 2024~는 `dart_disclosures`에 있다. `dividends.first_filed()`가 둘을 합쳐 최초 접수일을 준다 | OpenDART 공시 목록 (회사별) | `dart_fin`과 같은 종목 | 회사당 1회 |
| `us_dividends` (symbol=ALL) | 미국 **현금·주식배당**: `ts` = 배당락일, `rate`(주당 금액, 원본·분할 미반영), `record_date`, `payable_date`, `special`, `foreign`, `cusip`. ETF·펀드 포함 | Alpaca corporate actions v1 (키 필요) | **시장 전체**(상장폐지 포함), 배당락 2016~ | 매일 (최근 150일 + 발표된 90일 앞까지 재수집) |
| `us_dividend_announcements` (symbol=ALL) | 미국 배당 발표 기록: `ts` = Alpaca `declaration_date` (**실제 결의일이 아님**, 아래 한계 참고), `cash`, `ex_date`, `record_date`, `payable_date`, `sub_type` | Alpaca corporate action announcements v2 (키 필요) | 시장 전체(2020~21은 뮤추얼펀드 다수), 2020~ | 매일 (최근 30일 재수집) |
| `news` (symbol=ALL) | 미국 주식·코인 뉴스 (Benzinga): `id`, `ts`(기사 작성 시각), `updated_at`, `symbols`(쉼표로 이은 티커, 코인은 `BTCUSD`), `headline`, `summary`, `source`, `url`. 본문 없음 | Alpaca News API 무료 플랜 (키 필요) | 2015-01~ | 매일 |
| `ecos` | 기준금리, 국고채 3·10년, 회사채 AA- 3년, CD 91일, 원/달러 | 한국은행 ECOS | 2005~ | 매일 |
| `legacy/*.parquet` | 기존 stock-analysis-site DB에서 가져온 것: 일봉(KR 3,259·US 5,487종목, 2016-08~ +시가총액), 실제 지수 구성종목, 재무제표(K-IFRS 349사·US GAAP 511사, 2021~), 공시(SEC 1995~, DART 2024-08~), 내부자 거래(SEC Form 4 2000~) | Docker Postgres (1회 가져오기) | 표 참고 | 수동 (`collect legacy`) |
| `scan_hits` (symbol=ALL) | 장 전 스캐너가 규칙에 맞았다고 표시한 종목 (시장·셋업·종가·20일 평균 거래대금) | 위 데이터에서 계산 | 2026-11-15~ (셋업 함수로 과거 재계산 가능) | 매일 08:00 |
| `regime_1d` | **국면 지표** (시장정보팀이 계산, 매일 전체 재계산) | 위 데이터에서 계산 | 2015~ | 매일 08:00 |

- 공통 컬럼: `ts`(봉 시작 시각, UTC), `open`, `high`, `low`, `close`, `volume`. 미완성 봉은 저장하지 않는다.
- 유니버스는 `data/processed/universe/{name}/{날짜}.parquet`로 날짜별 기록되고, 30일이 지나면 새로 만든다.
- 수집 상태: `data/state.sqlite` (데이터셋·종목별 first/last ts, 백필 완료 여부, 마지막 오류)
- 로그: `logs/collect_YYYY-MM-DD.log`

## 알려진 한계 (lab 결과 해석 시 주의)

- **US**: 현재 구성종목 기준이라 **생존편향**이 있다. Yahoo OHLC는 이미 분할 반영 값이라 진짜 원본 가격이 아니다.
- **Binance**: 현재 거래대금 상위 코인을 고른 것이라, 과거 시점 기준으로 보면 **생존편향**이 있다 (지금까지 살아남아 커진 코인 위주).
- **Upbit**: 거래가 없던 분은 봉이 없다.
- **KR 종목코드 재사용**: 상장폐지 종목의 코드를 다른 회사가 다시 쓰는 경우가 드물게 있다.
- **거시 지표 시점**: `macro_1d`·`fred`·`regime_1d`의 날짜는 관측일이다. 실제로 알 수 있는 건 다음 날이므로 전략에서는 `shift(1)` 해서 쓴다.
- **FRED 신용 스프레드**(ICE BofA)는 라이선스 문제로 최근 3년만 제공된다. 그 이전 구간의 국면 지표에서는 이 신호가 빠진다.
- **공시 시점**: DART는 접수일만 있으므로 일봉 전략에서는 다음 날부터 알 수 있는 정보로 쓴다.
- **KR 재무는 `dart_fin`을 쓴다** (legacy는 2026-08 1회 가져오기라 갱신되지 않는다).
  - 시점: `filed` 다음 날부터 알 수 있는 정보로 쓴다. 기존의 +45일 규칙은 실제 접수보다 이를 수 있다. 예: 삼성전자 2025 반기는 추정 8/12, 실제 접수 8/14.
  - 4분기: 국내에는 4분기 보고서가 없다. 4분기는 `FY − Q3의 _cum`으로 계산하고, 알 수 있는 시점은 사업보고서 접수일이다. 4분기 잠정실적은 공시 목록의 "영업(잠정)실적"이 더 빠르다.
  - 정정: DART는 최신(정정) 값을 준다. 정정된 보고서는 `filed`가 정정일이라 시점은 보수적이다. 다만 정정 전 원래 값은 남지 않는다.
  - 업종: 금융업은 계정이 달라 매출액·영업이익이 비는 경우가 있다 (영업수익 등).
- **배당 데이터 시점** (검토: `docs/research/2026-11-16-dividend-data-sources.md`):
  - `dart_dividend`: DART가 최신(정정) 보고서를 주므로 `filed`가 정정일인 행이 많다 (결산일 +120일 넘는 행 22%, +1년 넘는 행 6%. 예: 현대차 2015 → 2022-02). **최초 접수일**(`collectors.dividends.first_filed()`, `kind='사업보고서'`, `period`=결산 'YYYY.MM') 다음 날부터 쓴다. 값은 정정 후 값이라 정정이 배당 수치를 바꿨다면 아주 약한 미래참조가 남는다. 사업보고서(3월)는 배당결정 공시(보통 2월, `dart_disclosures`의 "현금ㆍ현물배당결정")보다 늦다 → 그 점에선 보수적. 같은 문제가 `dart_fin`에도 있으니 재무 전략도 `first_filed()`를 쓰면 된다.
  - `us_dividends`: 공시일이 없다. 배당은 배당락 전에 발표되므로 `ts`(배당락일)부터 쓰면 미래참조는 없지만 늦다 (실제 발표는 보통 1~4주 전).
  - `us_dividend_announcements`: Alpaca의 `declaration_date`는 실제 결의일보다 늦다 (2022~ 대개 배당락 1~4일 전, 2020~21은 배당락 뒤도 많음). `ts` 다음 날부터 쓰면 미래참조는 없지만 **발표 이벤트 시점으로 쓰면 안 된다.** 실제 결의일이 필요하면 다른 소스가 필요하다 (리서치 문서 참고).
  - `kr_dividend_schedule`: 공시일이 없다. 배당락일 가격 보정에만 쓰고, 신호(배당수익률 등)에는 `dart_dividend` + 최초 접수일을 쓴다. 배당락일은 기준일 전에 정해져 있으므로 배당락일 당일 보정은 미래참조가 아니다. 마지막 수집 거래일 뒤 기준일은 `ex_date`가 비어 있다 (다음 갱신 때 채워짐).
  - `kr_adj_factor`는 분할·증자만 반영하고 배당은 반영하지 않는다 (삼성전자 수정계수는 2018-05 액면분할 외 변화 없음).
    대신 백테스트 엔진(`backtest/data.py` `load("kr_stock")`)이 `kr_dividend_schedule`의 배당락일에 배당소득세 15.4%(`config/settings.yaml` `kr_stock.dividend_tax_pct`)를 뗀 배당을 재투자해 **총수익**으로 계산한다.
    세율은 `dividend_tax` 인자(lab.yaml `dividend_tax`)로 바꾸고, `reinvest_dividends=False`면 예전처럼 가격 수익만 계산한다. report에 "KR 배당 재투자 반영" 여부가 찍힌다.
    `store.read("kr_adj_factor")`로 직접 수정주가를 만들면 여전히 배당이 빠진다.
- **legacy 재무제표 (US)**: 회계기간 기준이라 공개일이 아니다. 분기 +45일, 연간 +90일 공개 지연 규칙을 적용해서 쓴다. 같은 기간 중복과 9개월 누적 행이 섞여 있다.
- **legacy US 일봉**: 분할 반영 가격이고 상장폐지 종목이 없다 (생존편향).
- **KR 분봉 유니버스**: 시총 상위 350개 + 실제 KOSPI200·KOSDAQ150 구성종목(2026-08 기준) 합집합, 420개.
- **KRX 지수 API**: 현재 키로는 401(미신청 서비스). 필요하면 KRX Open API에서 지수 서비스를 추가 신청한다.
- **뉴스 시점** (`news`): 기사는 `ts`(작성 시각, UTC) 이후에만 쓸 수 있다. 분봉 전략은 `ts`가 속한 봉의 다음 봉부터, 일봉 전략은 장 마감 뒤 기사면 다음 거래일부터 쓴다. headline·summary는 나중에 고쳐질 수 있고 저장된 건 수집 시점의 최신 값이라, `updated_at > ts`인 기사는 원래 문구와 다를 수 있다.
- **뉴스 범위**: Benzinga 한 곳이고 영문뿐이다. 국내 종목 뉴스는 없다. `symbols`는 Benzinga 태깅이라 빠지거나 넓게 붙은 경우가 있다.
- 데이터 스냅샷(lab 재현용)은 백테스트 엔진을 만들 때 함께 구현한다.

## 상태 점검과 알림

```bash
uv run python -m autotrader.data.cli health            # 신선도·오류·디스크 점검
uv run python -m autotrader.intel.market               # 국면 지표 + 스캐너 + LLM 해석, 브리핑 출력 (--no-llm, --no-scan)
uv run python -m autotrader.intel.scanner              # 스캐너만 (reports/scan/날짜.json)
uv run python -m autotrader.intel.company 005930 --peers 000660 [--asof 2024-06-30]  # 기업 스냅샷 (fundamental-analyst용)
```

- 기업 스냅샷의 재무 출처는 KR이 `dart_fin`, US가 legacy다.
  - KR에서 `dart_fin`에 없는 종목은 legacy로 대신한다.
  - 4분기는 `Q4*`로 계산한다.
  - legacy US의 중복·누적 행은 기간당 1행으로 줄이고, 급변한 값에는 `급변·원문확인` 표시를 붙인다.
- `--asof`는 그 날짜에 알 수 있던 정보만 보여 준다.
  - 가격: 현지 날짜 기준
  - DART 재무·공시: 접수일 다음 날부터
  - legacy 재무: 추정 공개일부터

- 작업 스케줄러: `autotrader-collect`(07:30, 17:00, 로그인), `autotrader-brief`(08:00), `autotrader-health`(12:00, 20:00)
- 수집기 전체가 오류로 죽으면 즉시 텔레그램 알림. 강제 종료처럼 알릴 틈이 없는 경우는 health 점검이 데이터 신선도로 잡아낸다.
- 텔레그램 토큰이 없으면 `logs/alerts.log`에만 남는다.
