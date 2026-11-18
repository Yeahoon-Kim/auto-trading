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
| `dart_fin` 지배주주 순이익 열 | `net_income_controlling`(3개월 값, 사업보고서는 연간) + `_cum`(누적) + `nic_source`. 지주사처럼 비지배지분이 큰 회사의 EPS용. 연결은 아래 `dart_fin_owner` 값, 별도는 `net_income` 그대로 (방침은 아래 한계 참고). 매 `kr-public` 실행 끝에 전체 `dart_fin`에 다시 붙인다 | OpenDART `fnlttSinglAcntAll` | `dart_fin`과 같음 | 매일 |
| `dart_fin_owner` (symbol=ALL) | 연결(CFS) 보고서별 지배주주 순이익 원본: `fin_rcept_no`(받을 때의 `dart_fin` 접수번호), `rcept_no`(응답), `status`, `has_data`, `source`, 값 2개. 키 = (symbol, bsns_year, reprt_code). `dart_fin` 보고서가 정정돼 접수번호가 바뀌면 다시 받는다 | OpenDART `fnlttSinglAcntAll` (회사·보고서당 1회) | `dart_fin` 연결 행 약 7.9만 건 | 매일 (실행당 `kr_public.dart_owner_max_calls` 8천 호출씩 백필, 최근 사업연도부터) |
| `dart_dividend` (symbol=ALL) | 사업보고서 **배당에 관한 사항**: 주당 현금배당금(`dps_common`·`dps_pref`, 원, 중간·분기배당 포함 연간 합계), 현금배당수익률(%), 현금배당성향(%), 현금·주식배당 총액(백만원), 주당 주식배당, 순이익·EPS(연결 우선). 회사·사업연도당 1행, 배당 기록 없음은 `has_data=False`. `ts` = 결산일, `filed` = 접수일 | OpenDART `alotMatter` | 사업연도 2015~, `dart_fin`과 같은 종목(상장폐지 포함) | 매일 (실행당 8천 호출씩 백필 → 이후 새 사업보고서만) |
| `kr_dividend_schedule` (symbol=ALL) | KR **배당 기준일·배당락일**: `ts` = 기준일, `ex_date` = 배당락일(KRX 거래일로 계산: 기준일 이하 마지막 거래일의 전 거래일), `dps`(주당 현금배당금, 원), `kind`(결산·중간·분기), `stock_kind`(보통·우선), `divi_rate`(액면 대비 %), 주식배당률, `pay_date`. 공시일 없음 → 배당락 가격 보정(총수익)용 | KIS 예탁원정보 배당일정 (`HHKDB669102C0`) | 2010년 이후 KRX 상장 종목(상장폐지 포함), 기준일 2010~ (1년 앞 발표분까지) | 매일 (상장 종목은 7일마다 재조회) |
| `dart_periodic_filings` (symbol=ALL) | 2015~2023 정기보고서(사업·반기·분기) 목록, 정정 포함: `ts`(접수일), `rcept_no`, `kind`, `period`('YYYY.MM'), `correction`. 2024~는 `dart_disclosures`에 있다. `dividends.first_filed()`가 둘을 합쳐 최초 접수일을 준다 | OpenDART 공시 목록 (회사별) | `dart_fin`과 같은 종목 | 회사당 1회 |
| `dart_executives` (symbol=ALL) | **기업 연고표 — 임원 현황** 원문: `nm`(이름), `sexdstn`, `birth_ym`, `ofcps`(직위), `rgist_exctv_at`(사내·사외이사, 미등기), `fte_at`(상근 여부), `chrg_job`(담당 업무), **`main_career`(주요 경력·학력 텍스트, 줄바꿈 포함)**, `mxmm_shrholdr_relate`(최대주주와의 관계), `hffc_pd`(재직 기간), `tenure_end_on`(임기 만료일). 1행 = 보고서 1개의 임원 1명. 아래 공통 열 참고 | OpenDART `exctvSttus` (사업보고서) | 사업연도 2015~, `dart_fin`과 같은 종목(상장폐지 포함) 중 사업보고서가 확인된 회사·연도 | 매일 (`dividends` 예산의 남은 몫으로 백필 → 이후 새 사업보고서만) |
| `dart_investments` (symbol=ALL) | **기업 연고표 — 타법인 출자 현황** 원문: `inv_prm`(피출자 법인명), `frst_acqs_de`(최초 취득일), `invstmnt_purps`(출자 목적), 기초·증감·기말 수량·지분율(`*_qota_rt`)·장부가, 피출자사 최근 사업연도 총자산·순이익. 1행 = 출자 1건. 금액은 쉼표 문자열 그대로 | OpenDART `otrCprInvstmntSttus` | 같음 | 같음 |
| `dart_major_holders` (symbol=ALL) | **기업 연고표 — 최대주주 현황** 원문: `nm`(주주명), `relate`(관계: 최대주주 본인·특수관계인·계열회사 임원 등), `stock_knd`, 기초·기말 주식 수·지분율, `rm`. 1행 = 주주 1명 × 주식 종류 | OpenDART `hyslrSttus` | 같음 | 같음 |
| `dart_company` (symbol=ALL) | **기업개황**: **`adres`(본사 주소)**, `ceo_nm`, `induty_code`(업종), `est_dt`(설립일), `acc_mt`(결산월), `jurir_no`·`bizr_no`, `hm_url` 등 응답 그대로 + `status`. `ts` = 받은 날. **현재 값만 있다** (아래 한계 참고) | OpenDART `company` | `dart_fin`과 같은 종목 | 회사당 1회 + 90일마다 다시 받아 변경을 쌓음 (`dividends.profile_company_refresh_days`) |
| `dart_profile_calls` (symbol=ALL) | 연고표 진행 기록: (`api`, `corp_code`, `bsns_year`)당 `status`('000' 있음·'013' 없음), `n_rows`, `rcept_no`, `fetched`. 이어받기·'없음' 재확인용 | — | — | 매일 |
| `edgar_facts` (symbol=ALL) | 미국 **재무 사실(fact) 긴 표**: `field`(eps_diluted·eps_basic·net_income·revenue·operating_income·equity·shares_outstanding·shares_diluted_wavg·shares_basic_wavg), `tag`·`tag_rank`(같은 항목의 us-gaap 태그 우선순위), `unit`, `start`(기간 시작, 시점 항목은 NaT)·`end`, `val`, `accn`(접수번호), `fy`·`fp`, `form`(10-K·10-Q·10-K/A·8-K…), **`filed`(공시일)**, `frame`. `ts` = 기간 끝. 같은 기간 값이 다음 공시(비교 재무제표·정정)에 다시 나오면 공시마다 행이 따로 있다 | SEC EDGAR XBRL companyfacts (키 없음, **`SEC_USER_AGENT` 필요**) | us_1d 유니버스(S&P500+NASDAQ100 **현재 구성종목**), XBRL 2009~ (그 전 기간은 이후 공시의 비교 값만) | 매일 (SEC 일별 색인에서 10-K·10-Q(정정 포함)를 낸 회사 + 30일 지난 회사만 다시 받음) |
| `edgar_companies` (symbol=ALL) | 종목 → CIK(SEC `company_tickers.json`), 회사명, 마지막 companyfacts 수집 시각(`ts`), 사실 수, `fields_v`(받을 때의 항목 버전) | SEC | 같음 | 매일 |
| `edgar_class_shares` (symbol=ALL) | **클래스별 표지 주식수**: `cik`, `accn`, `form`, `filed`, `member`(`축=클래스`, 차원 없는 회사 전체 값은 빈 문자열, 표지 주식수가 없는 공시는 `∅`), `asof`(표지 기준일), `val`. `ts` = asof | SEC 공시별 `{접수번호}-xbrl.zip`의 표지(dei) | companyfacts에 차원 없는 주식수가 있는 10-K·10-Q가 50% 미만인 CIK(2026-11-17 기준 STZ·ERIE·BRK·V·CVNA·BKR) + `edgar.class_force_ciks`, 2009-06~ | 매일 (새 공시만, 실행당 1,500건 상한) |
| `us_splits` (symbol=ALL) | **US 분할 이력**: `ts` = 효력일(분할 반영 첫 거래일), `ratio`(새/옛 주식수: 4:1 → 4.0, 1:10 → 0.1), `source`(yahoo), `alpaca_date`(Alpaca announcements v2의 effective_date — 발표일과 맞는 경우(NVDA 2024)도 틀린 경우(GOOGL 2022: 2021-12-27, 실제 2022-02-01)도 있어 **참고용**, 2020~ Yahoo 분할 100건 중 31건만), `alpaca_ratio`(교차 확인: 2020~ 59건 일치) | Yahoo(yfinance `Ticker.splits`). 비교한 소스: Alpaca v1 분할은 2016 전이 비어 있음(2014 AAPL 7:1 없음), us_1d `adj_close/close`는 Yahoo `close`가 이미 분할 반영이라 분할이 안 보임, EDGAR는 분할을 구조화된 값으로 주지 않음 → 전 기간이 있는 Yahoo를 기본으로. 무료로 믿을 만한 발표일은 못 찾음 | us_1d 유니버스 + 벤치마크 ETF, 2010~ (2026-11-17: 177종목 257건) | 매일 (전 종목 다시 받음) |
| `us_dividends` (symbol=ALL) | 미국 **현금·주식배당**: `ts` = 배당락일, `rate`(주당 금액, 원본·분할 미반영), `record_date`, `payable_date`, `special`, `foreign`, `cusip`. ETF·펀드 포함 | Alpaca corporate actions v1 (키 필요) | **시장 전체**(상장폐지 포함), 배당락 2016~ | 매일 (최근 150일 + 발표된 90일 앞까지 재수집) |
| `us_dividend_announcements` (symbol=ALL) | 미국 배당 발표 기록: `ts` = Alpaca `declaration_date` (**실제 결의일이 아님**, 아래 한계 참고), `cash`, `ex_date`, `record_date`, `payable_date`, `sub_type` | Alpaca corporate action announcements v2 (키 필요) | 시장 전체(2020~21은 뮤추얼펀드 다수), 2020~ | 매일 (최근 30일 재수집) |
| `news` (symbol=ALL) | 미국 주식·코인 뉴스 (Benzinga): `id`, `ts`(기사 작성 시각), `updated_at`, `symbols`(쉼표로 이은 티커, 코인은 `BTCUSD`), `headline`, `summary`, `source`, `url`. 본문 없음 | Alpaca News API 무료 플랜 (키 필요) | 2015-01~ | 매일 |
| `naver_trend` (symbol=ALL) | **네이버 검색어 트렌드** (일별 검색량 상대값): `ts`(날짜 KST 0시 → UTC), `keyword`, `symbol`(종목코드, 추가 검색어는 ""), `ratio`, **`win_start`·`win_end`(요청 창)**, `group`(같은 요청 키워드 5개, `\|`로 연결), `collected_at`. 키 = (ts, keyword, win_start, win_end). **창이 다르면 값이 상수배 다르다** → 아래 '네이버' 절의 시점 규칙대로 쓴다 | NAVER API HUB 데이터랩 (키 필요) | kr_daily 유니버스 보통주 중 2016년 이후 상장 중이던 종목명(약 2,900개) + `naver.trend.extra_keywords`, 2016-01~ | 매일 (어제까지 창 1개 갱신, 백필은 하루 2천 요청씩 약 7일) |
| `naver_news_live` (symbol=ALL) | **네이버 뉴스 검색 앞으로 수집**: `ts`(발행 시각, 분 단위 → UTC), `title`·`summary`(태그 제거), `link`, `originallink`, `domain`(언론사 대신 원문 주소 호스트), `query`, `query_kind`(theme·stock), `symbol`, `collected_at`. 키 = (link, query) — 한 기사가 여러 질의에 걸리면 질의마다 1행. 본문 없음. **저작권 대상: data/ 밖으로 내보내지 않는다** | NAVER API HUB 뉴스 검색 (같은 키) | 2026-11-18~ (과거분 없음, 기간 지정 불가) | 하루 2~3회 (07:30·17:00·로그인) |
| `ecos` | 기준금리, 국고채 3·10년, 회사채 AA- 3년, CD 91일, 원/달러 | 한국은행 ECOS | 2005~ | 매일 |
| `legacy/*.parquet` | 기존 stock-analysis-site DB에서 가져온 것: 일봉(KR 3,259·US 5,487종목, 2016-08~ +시가총액), 실제 지수 구성종목, 재무제표(K-IFRS 349사·US GAAP 511사, 2021~), 공시(SEC 1995~, DART 2024-08~), 내부자 거래(SEC Form 4 2000~) | Docker Postgres (1회 가져오기) | 표 참고 | 수동 (`collect legacy`) |
| `scan_hits` (symbol=ALL) | 장 전 스캐너가 규칙에 맞았다고 표시한 종목 (시장·셋업·종가·20일 평균 거래대금) | 위 데이터에서 계산 | 2026-11-15~ (셋업 함수로 과거 재계산 가능) | 매일 08:00 |
| `regime_1d` | **국면 지표** (시장정보팀이 계산, 매일 전체 재계산). 지표 값이 아직 없는 날(200일선·변동성 백분위 워밍업, 휴장이 5일 넘게 이어진 날)은 값과 `*_trend`·`*_vol` 라벨이 비어 있다(None). `risk_off_available` = 그날 셀 수 있는 위험회피 신호 개수(정의 4개), `risk_off_signals` = 그중 켜진 개수 | 위 데이터에서 계산 | 2010-01~ (추세 라벨 SPX·KOSPI 2010-10~, 변동성 라벨 2011-01~, BTC 추세 2018-03~) | 매일 08:00 |

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
- **FRED 신용 스프레드**(ICE BofA)는 라이선스 문제로 최근 3년만 제공된다. 그 이전 구간의 국면 지표에서는 이 신호가 빠진다: `regime_1d`는 이 신호를 '없음(0)'으로 세지 않고 `risk_off_available`에서 뺀다(2023-11 이전 최대 3개). 셀 수 있는 신호가 하나도 없는 날은 `risk_off_signals`·`risk_regime`이 비어 있다. 국면 기준(켜진 신호 3개 이상 위험회피, 1개 이상 중립)은 개수 그대로라, 신호가 적게 셀 수 있는 구간은 위험회피가 덜 나온다 — 구간 간 비교 시 `risk_off_available`로 나눠 보거나 같은 신호만 골라 쓴다.
- **공시 시점**: DART는 접수일만 있으므로 일봉 전략에서는 다음 날부터 알 수 있는 정보로 쓴다.
- **KR 재무는 `dart_fin`을 쓴다** (legacy는 2026-08 1회 가져오기라 갱신되지 않는다).
  - 시점: `filed` 다음 날부터 알 수 있는 정보로 쓴다. 기존의 +45일 규칙은 실제 접수보다 이를 수 있다. 예: 삼성전자 2025 반기는 추정 8/12, 실제 접수 8/14.
  - 4분기: 국내에는 4분기 보고서가 없다. 4분기는 `FY − Q3의 _cum`으로 계산하고, 알 수 있는 시점은 사업보고서 접수일이다. 4분기 잠정실적은 공시 목록의 "영업(잠정)실적"이 더 빠르다.
  - 정정: DART는 최신(정정) 값을 준다. 정정된 보고서는 `filed`가 정정일이라 시점은 보수적이다. 다만 정정 전 원래 값은 남지 않는다.
  - 업종: 금융업은 계정이 달라 매출액·영업이익이 비는 경우가 있다 (영업수익 등).
  - **지배주주 순이익** (`net_income_controlling`, `nic_source`):
    - `net_income`은 연결 당기순이익 전체(비지배지분 몫 포함)다. 주요계정 API(`fnlttMultiAcnt`)에는 지배주주 항목이 없어(당기순이익 2행 모두 전체) 단일회사 전체 재무제표(`fnlttSinglAcntAll`)에서 따로 받는다. 예: SK 2024 반기 3개월 전체 4,541억 중 지배 1,810억.
    - `cfs`: 연결 재무제표의 "지배기업 소유주 귀속 당기순이익" (표준 계정 ID, 없으면 계정명. 포괄이익 지배분은 쓰지 않는다).
    - `cfs_signfix`: 지배 + 비지배 ≠ 전체인데 지배 값의 부호를 바꾸면 맞는 경우 (제출 오류 보정).
    - `cfs_no_nci`: 연결인데 지배·비지배 구분 행이 아예 없음 → 비지배지분이 없다고 보고 당기순이익 전체.
    - **`ofs`: 별도재무제표만 있는 회사는 지배주주 순이익 = 당기순이익으로 둔다.** 별도만 낸다는 건 연결할 종속회사가 없다는 뜻이라 비지배지분이 생기지 않는다. 엔진에서 엄격하게 쓰려면 `nic_source == 'ofs'`를 걸러 NaN으로 바꾸면 된다.
    - `cfs_missing`: 연결인데 지배주주 순이익을 못 찾았거나(포괄이익만 나눈 회사 등) 아직 백필 전 → NaN. 표본 62건(2015~2026 무작위) 중 2건(3%).
    - 시점·4분기·정정 규칙은 다른 `dart_fin` 열과 같다 (`filed`, 4분기 = FY − Q3 `_cum`). 연결 보고서는 `dart_fin`과 같은 접수번호로 받은 값만 붙이고, 정정으로 접수번호가 바뀌면 다시 받을 때까지 NaN.
    - `fnlttSinglAcntAll`의 2015 분기·반기는 비어 있는 경우가 많다 (삼성전자 2015 3분기 013). 2015 사업보고서는 있다.
    - 금융지주·은행(예: 105560, 055550)은 `fnlttMultiAcnt`에 없어 `dart_fin`에 행이 없고, 따라서 지배주주 순이익도 없다.
  - **EPS용 주식수** (수집 안 함, 확인만):
    - `krx_1d.shares`는 종목코드별 상장주식수라 보통주(코드 끝 0)·우선주(끝 5·7 등)가 따로 있다. 자기주식을 빼지 않은 값이다.
    - 자기주식: OpenDART `stockTotqySttus`(주식의 총수: 발행·자기주식·유통주식수, 보통·우선 구분)와 `tesstkAcqsDspsSttus`(자기주식 취득·처분 현황)가 정기보고서마다 있다. 다만 초기 연도는 비어 있다 (삼성전자 2015 사업보고서는 두 API 모두 '-'). 회사·보고서당 1회라 백필하면 약 12만 호출(하루 2회 × 8천 기준 약 8일). 필요하면 개발·점검 HQ가 정한다.
- **기업 연고표** (`dart_executives`·`dart_investments`·`dart_major_holders`·`dart_company`, 수집기 `collectors/dart_profile.py`, L004 요청):
  - 공통 열(보고서 3종): `ts` = 결산일(`stlm_dt`), `corp_code`, `symbol`, `bsns_year`(결산일의 연도: 2016.03 결산 → 2016), `reprt_code`='11011', `seq`(응답 안 순서), `rcept_no`, `filed`(받은 보고서 접수일 = 정정이면 정정일), **`first_filed`(최초 접수일, 받을 때 `first_filed()`로 붙임)**, `fetched`, 그리고 응답 필드 전부(**문자열 원문 그대로**, 숫자도 쉼표 문자열). 키 = (`corp_code`, `bsns_year`, `reprt_code`, `seq`). 파싱(학력·지역·지분율 숫자화)은 lab에서 한다.
  - **시점 규칙: 사업보고서 최초 접수일(`first_filed`) 다음 날부터 쓴다** (`dart_dividend`와 같은 규칙). `first_filed`가 비어 있으면 `dividends.first_filed()`(`kind='사업보고서'`, `period`=결산 'YYYY.MM')로 다시 붙인다. `filed`(정정일)를 쓰면 늦어지고, `ts`(결산일)를 쓰면 미래참조다(사업보고서는 결산 뒤 약 3개월에 나온다).
  - 값은 최신(정정) 보고서 값이다. 정정이 임원·출자 내용을 바꿨다면 최초 접수일 기준으로는 약한 미래참조가 남는다 (배당·재무와 같음).
  - 임원 현황은 사업보고서 기준일의 명단이다. 연중 선임·사임은 다음 사업보고서까지 안 보인다 (중간에 바뀐 것은 공시 목록의 "임원ㆍ주요주주특정증권등소유상황보고서"·"최대주주변경" 등으로 보완).
  - **`dart_company`(본사 주소 등)는 현재 값만 준다 — 과거 이력이 없다.** 2026-11 수집분을 과거 구간에 그대로 쓰면, 그 사이 본사를 옮긴 회사는 미래참조다. 과거 구간 소재지 연고에는 위험을 적고 쓰거나 다른 근거(사업보고서 본문 등)로 확인한다. 90일마다 다시 받으므로 수집 시작 뒤 변경은 `ts`(받은 날) 기준으로 쓸 수 있다.
  - 대상: 사업보고서가 있다고 확인된 (회사, 사업연도)만 묻는다 (`dart_dividend` 값 있음 ∪ `first_filed()`의 사업보고서). 2026-11-18 기준 26,282쌍(2015 2,013 ~ 2025 2,761) × 3종 + 기업개황 3,184 ≈ **8.2만 호출**.
  - 데이터가 없다는 답(013)은 사업보고서 마감·지연 제출을 넉넉히 본 다음 해 7월 이후에 받았으면 끝, 그 전이면 14일 뒤 다시 묻는다.
- **DART 호출 예산 배분** (`collect dividends`, `dividends.dart_max_calls` 실행당 8천 하나를 나눠 쓴다. 새 상한 없음):
  1) `dart_periodic_filings`(최초 접수일, 회사당 1회) → 2) `dart_dividend`(배당) → 3) 기업 연고표(남은 예산 전부: 기업개황 → 보고서 3종, 최근 사업연도부터 회사마다 3종을 이어서).
  앞 단계가 먼저 쓰고, 뒤 단계는 남은 만큼만 쓴다. 1)·2)는 2026-11-17에 백필이 끝나 지금은 새 보고서만 확인한다(실행당 수십 회) → 연고표가 거의 8천을 쓴다.
  DART 020(하루 한도 초과)을 받으면 그 단계에서 멈추고 뒤 단계도 쉬며, 다음 실행이 받은 것을 건너뛰고 이어받는다.
  같은 키를 `kr-public`(`dart_disclosures`·`dart_fin`·`dart_fin_owner` 실행당 8천)도 쓴다.
- **배당 데이터 시점** (검토: `docs/research/2026-11-16-dividend-data-sources.md`):
  - `dart_dividend`: DART가 최신(정정) 보고서를 주므로 `filed`가 정정일인 행이 많다 (결산일 +120일 넘는 행 22%, +1년 넘는 행 6%. 예: 현대차 2015 → 2022-02). **최초 접수일**(`collectors.dividends.first_filed()`, `kind='사업보고서'`, `period`=결산 'YYYY.MM') 다음 날부터 쓴다. 값은 정정 후 값이라 정정이 배당 수치를 바꿨다면 아주 약한 미래참조가 남는다. 사업보고서(3월)는 배당결정 공시(보통 2월, `dart_disclosures`의 "현금ㆍ현물배당결정")보다 늦다 → 그 점에선 보수적. 같은 문제가 `dart_fin`에도 있으니 재무 전략도 `first_filed()`를 쓰면 된다.
  - `us_dividends`: 공시일이 없다. 배당은 배당락 전에 발표되므로 `ts`(배당락일)부터 쓰면 미래참조는 없지만 늦다 (실제 발표는 보통 1~4주 전).
  - `us_dividend_announcements`: Alpaca의 `declaration_date`는 실제 결의일보다 늦다 (2022~ 대개 배당락 1~4일 전, 2020~21은 배당락 뒤도 많음). `ts` 다음 날부터 쓰면 미래참조는 없지만 **발표 이벤트 시점으로 쓰면 안 된다.** 실제 결의일이 필요하면 다른 소스가 필요하다 (리서치 문서 참고).
  - `kr_dividend_schedule`: 공시일이 없다. 배당락일 가격 보정에만 쓰고, 신호(배당수익률 등)에는 `dart_dividend` + 최초 접수일을 쓴다. 배당락일은 기준일 전에 정해져 있으므로 배당락일 당일 보정은 미래참조가 아니다. 마지막 수집 거래일 뒤 기준일은 `ex_date`가 비어 있다 (다음 갱신 때 채워짐).
  - `kr_adj_factor`는 분할·증자만 반영하고 배당은 반영하지 않는다 (삼성전자 수정계수는 2018-05 액면분할 외 변화 없음).
    대신 백테스트 엔진(`backtest/data.py` `load("kr_stock")`)이 `kr_dividend_schedule`의 배당락일에 배당소득세 15.4%(`config/settings.yaml` `kr_stock.dividend_tax_pct`)를 뗀 배당을 재투자해 **총수익**으로 계산한다.
    세율은 `dividend_tax` 인자(lab.yaml `dividend_tax`)로 바꾸고, `reinvest_dividends=False`면 예전처럼 가격 수익만 계산한다. report에 "KR 배당 재투자 반영" 여부가 찍힌다.
    `store.read("kr_adj_factor")`로 직접 수정주가를 만들면 여전히 배당이 빠진다.
- **US 재무 (`edgar_facts`)**:
  - 설정: SEC는 User-Agent에 연락처를 요구한다. 환경변수 또는 `.env`에 `SEC_USER_AGENT="이름 이메일"`을 사용자가 넣는다. 없으면 수집기는 건너뛰고 상태에 `skipped`를 남긴다. 코드·로그에는 이 값을 쓰지 않는다.
  - 시점: `filed` 다음 거래일부터 알 수 있다. 같은 (`field`, `start`, `end`)에 공시가 여러 개면
    - 최초 공시값(first): `filed`가 가장 이른 행. 정정 전 원래 숫자가 남아 있으므로 미래참조가 없다.
    - 최신값(last as-of): asof 이전 `filed` 중 가장 늦은 행. 10-K/A·재작성 값은 그 `filed`부터 반영된다.
    - 즉 정정(10-K/A, 10-Q/A)과 다음 해 비교 재무제표의 재작성 값을 지우지 않고 모두 남겨 두고, 엔진이 시점 규칙으로 고른다 (KR `dart_fin`은 최신 값만 남는 것과 다르다).
  - 기간: 10-Q에는 3개월 값과 연초 누계(YTD)가 같이 있다 (`start`로 구분). 4분기 보고서는 없으므로 4분기 = 10-K 연간 − 3분기 누계. 회계연도가 달력과 다른 회사가 많다 (AAPL 9월 말).
  - 태그: 같은 항목도 회사·연도마다 태그가 다르다 (매출: Revenues → RevenueFromContractWithCustomer…). 기간마다 `tag_rank`가 가장 작은 행을 쓴다. `net_income`의 1순위 `NetIncomeLoss`는 지배주주 귀속 순이익이다.
  - 주식수: `shares_outstanding`의 1순위 `dei:EntityCommonStockSharesOutstanding`은 표지 기준일(공시 직전) 값, `us-gaap:CommonStockSharesOutstanding`은 기간 말 값이다. 클래스가 여러 개인 회사(GOOG/GOOGL, BRK)는 클래스별로 나뉘거나 합계만 있을 수 있다. GOOG·GOOGL처럼 같은 CIK면 같은 사실이 두 종목에 다 들어간다.
  - 실제 JSON (2026-11-16 확인): fact 항목 = `start`(기간 항목만), `end`, `val`, `accn`, `fy`, `fp`, `form`, `filed`, `frame`(일부만). 정정 공시(10-K/A·10-Q/A)도 들어 있다 (AAPL 받는 항목 중 10-K/A 25행). 8-K 사실도 있다 (실적 발표 첨부 등).
  - 시작 시점: 유니버스 515개사의 첫 10-K·10-Q XBRL 공시일은 2009-04-15가 가장 이르고, 2009년 239개사·2010년 140개사·2011년 34개사이며 나머지는 상장·분사 시점(중앙값 2010-07). 첫 공시의 비교 재무제표 덕분에 기간 끝은 그보다 앞선 값도 있다 (가장 이른 기간 끝 1989-12-31, 드묾).
  - 적재 방식: **첫 백필은 일괄 파일, 갱신은 회사별 API.** 처음 받는 종목이 `edgar.bulk_min_new`(50) 이상이면 `companyfacts.zip`(전 기업, 매일 밤 재생성)을 데이터 루트 아래 임시 폴더에 받아 유니버스 회사 파일만 꺼내 같은 `parse_facts`로 적재하고 zip은 지운다. 이때 `edgar_companies.ts`는 zip 생성 시각(Last-Modified)이라, 그 뒤 공시는 일별 색인·30일 주기 규칙으로 회사별 API가 다시 받는다.
  - 일괄 파일 크기 (2026-11-16 측정): zip 1.43GB(다운로드 약 4분), 20,588개사, 해제 시 19.5GB. 유니버스 515개사 파일은 해제 1.95GB, JSON 읽기 19초. 백필 때 데이터 루트 디스크에 약 1.5GB가 잠시 필요하다.
  - 범위: 현재 구성종목만이라 **생존편향**이 있다. us-gaap만 받으므로 IFRS로 내는 외국 회사(20-F/40-F 중 IFRS)는 비어 있다 (ASML은 US GAAP라 있음). 8-K로만 나온 사실은 30일 주기 재수집 때 들어온다.
- **US 주식수·시가총액** (`src/autotrader/data/shares.py`, 엔진 `market_cap` 필드의 바탕):
  - 쓰는 법: `shares.daily_shares(symbols, start, end)` → us_1d 봉마다 `close`, `shares`(오늘 기준 = close와 같은 분할 기준), `market_cap`(USD), `cik`, `filed`, `asof`, `source`, `age_days`. 공시별 값은 `shares.share_events()`, 종목별 메타는 `shares.quality()`.
  - 시점: 공시일(`filed`) **다음 날**부터 쓴다. 기준일(`asof`)이 `us_shares.max_age_days`(400일)보다 오래되면 NaN.
  - 분할 기준 맞추기: us_1d 가격은 오늘 기준으로 분할 소급 반영, EDGAR 주식수는 공시 당시 값이다. 공시값에 그 뒤 분할 비율을 곱해 오늘 기준으로 바꾼다 → `close × shares` = 그날 실제 시가총액. 미래 분할 비율을 쓰지만 가격과 상쇄되므로 미래참조가 아니다. 표지 값은 `asof` 뒤, 재무제표 값은 `filed` 뒤 분할을 곱한다(재무제표는 공시 전 분할을 소급 반영하는 것이 규정). 공시 직후 분할을 표지에 미리 반영한 회사(ETN 2011)가 있어 공시일+15일 기준 값도 만들어 직전 값에 가까운 쪽을 쓴다.
  - **분사형 배수**: Yahoo는 분사(spin-off)도 소수 비율 '분할'로 적고 `close`를 그만큼 소급 보정한다(ABT 2013 2.084 = AbbVie, A 2014 1.398 = Keysight). 실제 주식수는 그대로이므로 이런 배수는 공시가 아니라 **봉 날짜 기준**으로 곱한다. 실제 분할인지 분사형인지는 분할 45일 전까지의 공시 주식수와 분할 뒤 첫 공시 주식수의 비율로 정하고(`classify_splits`), 자료가 없으면 비율 모양(p/q, q ≤ 4)으로 정한다.
  - 후보 우선순위(공시 하나 안에서): 클래스 합계 > dei 표지(`EntityCommonStockSharesOutstanding`) > 재무상태표(`CommonStockSharesOutstanding`) > 기본 가중평균 > 희석 가중평균. 가중평균은 분기 평균이라 기간 말 값과 1~2% 다를 수 있다.
  - 이상치 처리(직전 값·그날 가격만 쓰므로 미래참조 없음): ① 공시일 종가 × 주식수가 5천만~10조 달러 밖이면 10^3·10^6·10^9로 나눠 맞춘다(회사 XBRL 입력 오류: ON 2019 4.6×10^14주, YUM, GRMN, PKG, SRE, CB). ② 직전 채택값 3개 중앙값 대비 0.5~2배면 채택, 벗어나도 같은 공시의 다른 태그와 15% 안이거나 직전에 버린 값과 같은 새 수준이면 실제 변화로 채택. ③ 그래도 안 맞으면 1000배 단위 오류인지 보고 고치거나 버린다(`flag` = ok·unchecked·scale_fixed·level_change·dropped).
  - **옛 CIK**: 지주사 재편·1:1 주식 교환 합병으로 CIK가 바뀐 종목(XOM·DIS·GOOGL·GOOG·BLK·AVGO·CI·ICE·ETN·MDT·APA·STE·TPL·BG·APO·MRVL)은 `edgar.predecessors`의 옛 CIK 사실을 그 종목 이름으로 `edgar_facts`에 넣는다(`cik` 열은 옛 CIK). 새 CIK 첫 공시 전 공시분만 넣는다. 교환비가 1:1이 아닌 합병(LIN·DD·BKR·VTRS·AMCR 등)은 넣지 않아 합병 전 기간이 비어 있다. **이 추가로 해당 종목의 다른 US 재무 필드(매출 등)도 합병 전 기간 값이 생긴다.**
  - **다중 클래스 방침**: 시가총액은 **회사 단위**다. 클래스별로 나뉜 주식수는 상장 클래스 주식으로 환산해 합친다(`us_shares.class_ratio`; 적지 않은 클래스는 1). BRK-B = A주 × 1,500 + B주. GOOG·GOOGL·FOX·FOXA·NWS·NWSA처럼 한 회사 종목이 여럿이면 각 종목에 회사 전체 시총이 나온다(클래스 가격 차이만큼만 다름: 중앙값 GOOG/GOOGL 0.7%, NWS/NWSA 2.7%, FOX/FOXA 7.6%) → **순위를 매길 때는 `cik`로 묶어 한 종목만 고른다**(어느 클래스를 살지는 전략이 정함). 클래스별 시총이 필요하면 `edgar_class_shares`에서 직접 계산한다.
  - 한계: V는 B·C주 전환비가 시기마다 바뀌어 빼므로 시총이 약 5~20% 작다(2015-06-30 1,314억 달러 vs 실제 약 1,650억). 분할 직후~첫 공시 사이 클래스 비율이 바뀐 회사, 2011-06 이전 XBRL이 없는 소형사(첫 1~1.5년), 외국 회사(20-F)는 비어 있다. 현재 구성종목만이라 생존편향이 있다.
  - **verifier**: `uv run python -m autotrader.data.shares_check --out <폴더>` — 분할 전후 연속성, 주식수 급변, 단위 이상치(전 기간 중앙값의 10배↑/↓), 알려진 날짜 시총 상위 10 표본, 비상식 시총(8조 달러↑·1천만 달러↓), 결측률을 점검하고 종목별 표(`us_shares_quality.csv`)를 남긴다. 반환 코드 = 기준 미달 항목 수.
  - 점검 결과 (2026-11-17, 메인 data/ + 이 작업에서 받은 분할·옛 CIK·클래스 표본, 2010-01~2026-11, 비용 무관 데이터 점검):

    | 항목 (기준) | 전략부 L002 방식 (shares_outstanding 최신값 × 종가) | 새 방식 |
    |---|---|---|
    | 분할 전후 주식수 연속성 실패 (≤5%) | 66.7% | 4.1% (DD·DUK·TMUS 등 같은 날 합병·분사가 겹친 경우) |
    | 시총 상위 10 표본 6개 날짜 통과 | 0/6 (CNP·STX·PKG가 1위 등) | 6/6 |
    | 비상식 시총 날 수 | 27,599 | 0 |
    | 단위 이상치가 있는 종목 수 (0) | 81 | 2 (BE: 상장 직후 주식수, EXE: 파산 재편 전 옛 주식) |
    | 전체 결측률 (봉 기준) | 9.2% (오래된 값도 채움) | 5.2% (400일 넘은 값은 비움) |
    | 결측: XOM / META / GOOGL / DIS / AVGO / BRK-B / V | 98% / 100% / 36% / 55% / 50% / 0%(A주 수) / 0%(2010 값 고정) | 0% / 1.4% / 0% / 0% / 7% / 0.7% / 0% |
    | 최대 시총(10억 달러): ON / YUM / GRMN / PKG | 3,891,188 / 25,194,929 / 16,452 / 20,747 | 52 / 46 / 60 / 23 |
- **legacy 재무제표 (US)**: 회계기간 기준이라 공개일이 아니다. 분기 +45일, 연간 +90일 공개 지연 규칙을 적용해서 쓴다. 같은 기간 중복과 9개월 누적 행이 섞여 있다.
- **legacy US 일봉**: 분할 반영 가격이고 상장폐지 종목이 없다 (생존편향).
- **KR 분봉 유니버스**: 시총 상위 350개 + 실제 KOSPI200·KOSDAQ150 구성종목(2026-08 기준) 합집합, 420개.
- **KRX 지수 API**: 현재 키로는 401(미신청 서비스). 필요하면 KRX Open API에서 지수 서비스를 추가 신청한다.
- **뉴스 시점** (`news`): 기사는 `ts`(작성 시각, UTC) 이후에만 쓸 수 있다. 분봉 전략은 `ts`가 속한 봉의 다음 봉부터, 일봉 전략은 장 마감 뒤 기사면 다음 거래일부터 쓴다. headline·summary는 나중에 고쳐질 수 있고 저장된 건 수집 시점의 최신 값이라, `updated_at > ts`인 기사는 원래 문구와 다를 수 있다.
- **뉴스 범위**: Benzinga 한 곳이고 영문뿐이다. 국내 종목 뉴스는 없다. `symbols`는 Benzinga 태깅이라 빠지거나 넓게 붙은 경우가 있다.
- 데이터 스냅샷(lab 재현용)은 백테스트 엔진을 만들 때 함께 구현한다.

## 네이버 (NAVER API HUB) — `naver_trend`, `naver_news_live`

- 수집기: `src/autotrader/data/collectors/naver.py`. 키: `.env`의 `NAVER_CLIENT_ID`·`NAVER_CLIENT_SECRET` (네이버 클라우드 API Gateway 헤더 `X-NCP-APIGW-API-KEY-ID`·`X-NCP-APIGW-API-KEY`). 없으면 건너뛴다.
- **정기 수집 등록**: `scripts/collect_all.ps1`의 기본 소스 목록에 `naver-trend`, `naver-news`가 들어 있다. 작업 스케줄러 `autotrader-collect`(07:30·17:00·로그인)가 메인 폴더에서 이 스크립트를 부르므로 master에 병합되면 따로 등록할 것이 없다. 하나만 돌릴 때:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\collect_all.ps1 -Sources naver-trend,naver-news
  uv run python -m autotrader.data.cli collect naver-trend
  ```
- **호출 한도** (`config/data.yaml`의 `naver`): 실제 HTTP 요청(재시도 포함)마다 `data/state.sqlite`의 `api_calls` 표에 KST 날짜별로 센다. 하루·월 한도에 닿으면 받은 데까지 저장하고 다음 실행에 이어받는다.
  공식 한도: 트렌드 월 5만 건, 뉴스 검색은 문서마다 하루 2.5만 / 월 77.5만(작은 쪽 기준), 키당 50회/초.
- **수집 기록과 빠진 날 점검**: 실행마다 `data/state.sqlite`의 `naver_runs` 표에 1행을 남긴다 — `dataset`, `started_at`·`finished_at`(UTC), `kst_date`, `status`(ok = 끝까지 / limit = 한도에 닿아 일부만, 다음 실행에 이어받음 / error = 오류로 멈춤 / skip = 키 없음), `rows`(새로 저장한 행 수, 뉴스는 이미 있던 기사 제외), `calls`(그 실행의 호출 수), `error`.
  ```powershell
  uv run python -m autotrader.data.cli naver-gaps                         # 뉴스: 첫 수집일~오늘 중 문제 있는 날
  uv run python -m autotrader.data.cli naver-gaps --dataset naver_trend --since 2026-11-18 --all
  ```
  `problem`: `missing`(그날 실행 기록 없음), `failed`(실행했지만 ok·limit 없음), `empty`(성공했지만 새 행 0 — 뉴스는 거래일이면 의심). `trading_day`는 krx_1d로 판단(주말 False, krx_1d가 아직 없는 날은 빈 값). 거래일(또는 아직 모르는 평일)에 문제가 있으면 종료 코드 1. 코드에서는 `naver.gaps()`, `naver.runs()`.
  **앞으로 검증의 시작일** = `naver.runs("naver_news_live")`에서 status가 ok이고 rows > 0인 첫 `kst_date` 이후 첫 거래일. 그 뒤 `missing`·`failed`인 거래일은 그날 장중 기사가 비어 있을 수 있다(질의당 최신 1,100건까지는 다음 실행이 메우므로 `누락 가능` 로그가 없으면 대개 채워진다).
- **과금 멈춤**: NAVER API HUB는 "한시적 무료, 유료 전환 시 사전 공지"다. 401·402·403, 본문에 유료·과금·결제 등의 말이 있는 오류, 429 반복이면 `NaverStop`으로 즉시 멈추고 상태에 오류를 남기며 텔레그램(`logs/alerts.log`)으로 알린다. 이때는 개발·점검 HQ가 사용자에게 보고한다 (비용은 사용자 승인).

### 검색어 트렌드 시점 규칙 (lab이 지킬 것)
- `ratio`는 **같은 요청 안의 모든 키워드·날짜 중 최댓값 = 100**인 상대값이다. 그래서 전체 기간을 한 번에 받으면 미래 최댓값으로 과거를 정규화하는 미래참조가 된다.
- 받는 방식: 1월 1일·7월 1일에 시작하는 **1년 고정 창** (반씩 겹침, 2016-01-01부터). 끝난 창은 한 번 받고, 안 끝난 창 중 가장 먼저 시작한 창은 매일 `[창 시작, 어제]`로 다시 받는다(처음엔 창 전체, 그 뒤엔 끝 14일만 저장).
- 2026-11-18 샘플 확인: 같은 키워드를 다른 창·다른 그룹 구성으로 받아도 **값의 비가 상수**였다(겹친 181일·214일에서 비의 변동 0, 합동/단독 비 23.086 ± 0.0003). 즉 창이 바뀌면 상수배만 달라지고 과거 값이 다시 계산되지 않았다(이후에도 그런지는 매일 갱신 창으로 계속 확인 가능).
- **날짜 t의 값은 `win_end ≤ t`인 창의 행만 쓴다.** 매일 갱신 창 행은 `win_end` = 수집 전날이므로, `win_end` = D 인 행은 **D+1일 장 시작 전부터** 쓴다 (네이버 집계는 하루 단위, 그날 값은 다음 날 공개).
- 창을 이을 때: 같은 키워드의 겹치는 날로 비(scale)를 구해 앞 창 기준으로 곱한다(`median(새 창 / 앞 창)`). 이은 시계열의 수준은 첫 창(2016) 최댓값 기준 상수배이므로, **비율 신호(오늘 / 지난 N일 평균)는 안전**하고, 키워드끼리 절대 수준 비교는 같은 요청(`group`) 안에서만 의미가 있다.
- 한 요청에 키워드 5개를 함께 받아 그중 최댓값이 100이 된다. 검색량이 아주 큰 종목과 같은 그룹이면 작은 종목 값이 작아져(소수 5자리) 0이나 거친 값이 될 수 있다 — `ratio == 0` 행은 '검색량 0'이 아니라 '정밀도 부족'일 수 있다.
- 키워드는 지금 종목명이다. 사명 변경 전 이름은 받지 않는다 (필요하면 `extra_keywords`로 요청).

### 뉴스 앞으로 수집 규칙
- 질의: `naver.news.theme_queries`(관련주·테마주·특징주·수혜주·상한가) + `krx_1d` **최근 거래일**에 15% 이상 오른 종목명(상승률 큰 순 60개). `krx_1d`는 KRX가 다음 영업일에 공개하므로 17:00 실행 때 급등 종목은 **전 거래일** 것이다 (그날 종목은 theme 질의의 `상한가`·`특징주` 기사로 잡힌다).
- 최신순으로 100건씩 넘기다 지난번 마지막 기사 시각보다 오래된 기사가 나오면 멈춘다. API는 질의당 최신 1,100건까지만 주므로 그 안에서 못 따라잡으면 로그에 `누락 가능`이 남는다 (샘플: `특징주` 9시간에 100건, `테마주` 1주에 100건 → 하루 2회면 충분).
- 시점: `ts`는 네이버가 준 발행 시각(분 단위)이다. 기사는 **`ts` 다음 분봉부터**, 일봉 전략은 장 마감(15:30 KST) 뒤 기사면 다음 거래일부터 쓴다. 제목·요약은 받을 때의 값이다 (`collected_at`).
- 과거분은 없다 — 2026-11-18부터 쌓인다. 과거 검증(2012~)에는 빅카인즈 쪽을 쓴다.

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
