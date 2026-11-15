# AI 반도체 시황 점검과 앞으로의 변수 (2026-11-15)

> 목적: 최근 AI 반도체 급락의 원인을 정리하고, 앞으로 방향을 가를 변수와 확인 일정을 정리한다.
> 가격 예측·매수·매도 권유가 아니다. 시나리오는 "어떤 조건이면 어떤 흐름이 되는가"만 다룬다.

## 1. 현황 — 로컬 데이터 (`us_1d` adj_close, `kr_1d` 종가, 2026-11-13 종가 기준)

| 종목 | 1주 | 1개월 | 3개월 | 1년 | 52주 고점 대비 |
|---|---:|---:|---:|---:|---:|
| NVDA | -12.7% | -25.0% | -27.5% | -15.8% | -31.9% |
| AMD | +18.9% | -5.0% | +5.5% | +106.3% | -17.8% |
| AVGO | +18.2% | +0.8% | -11.4% | -1.6% | -27.7% |
| MU | +12.8% | -16.4% | -23.1% | +218.2% | -35.8% |
| ASML | +4.9% | -9.7% | -20.5% | +45.0% | -24.6% |
| AMAT | +11.4% | -13.8% | -30.7% | +61.5% | -48.7% |
| LRCX | +12.7% | -18.6% | -34.2% | +40.6% | -47.8% |
| KLAC | +10.0% | -12.7% | -26.0% | +27.7% | -49.5% |
| INTC | +6.0% | -37.4% | -42.4% | +57.5% | -57.7% |
| MRVL | +11.0% | -35.0% | -33.1% | +75.8% | -50.4% |
| ARM | +9.1% | -22.2% | -29.8% | +28.1% | -56.6% |
| 삼성전자 | +10.8% | -0.5% | -12.7% | +88.1% | -42.3% (고점 2026-06-18) |
| SK하이닉스 | +8.9% | -2.3% | -13.5% | +98.9% | -57.8% (고점 2026-06-22) |
| 한미반도체 | +13.0% | -5.1% | -14.4% | +28.8% | -55.5% |

읽는 법:
- 고점은 대부분 6월 전후, 3개월 기준으로는 거의 전 종목이 -10~-40%다. 이번 조정은 이미 몇 달째 이어지고 있다.
- 1주 수익률(11-06 → 11-13)은 11-05 폭락 다음 날을 기준점으로 잡기 때문에 대부분 반등으로 나온다. **NVDA만 계속 빠졌다**(CEO 사임). 업종 전체 문제였던 하락이 이번 주에는 NVDA 개별 문제로 나뉘었다.
- 1년 기준으로는 메모리(MU·하이닉스·삼성)가 여전히 크게 올라 있다. 고점 대비 낙폭도 그만큼 크다.
- 한계: US는 현재 구성종목 기준이라 생존편향이 있다(이번 비교에는 영향이 작다). 수수료·세금·슬리피지가 없는 단순 가격 수익률이다.

## 2. 무엇이 있었나 (웹 보도 종합)

1. **10월~: NVDA 6주 연속 하락** — 2012년 이후 최장 연속 하락이다. 시가총액 약 1.6~1.7조 달러가 사라졌다.
2. **10-29: 하이퍼스케일러 capex 상향 보류** — Google과 Microsoft가 capex 전망을 올리지 않았다. Microsoft는 연간 전망을 1,750억에서 1,700억 달러로 낮췄다. 부품 가격 상승과 납기 지연 탓이라고 설명했고, 2027년 상반기에는 다시 늘 것이라고 했다.
3. **11-04~05: 미 중간선거 민주당 압승 → 반도체 폭락** — SOX가 하루 약 -12%로, 2025년 4월 이후 최대 낙폭이었다. 같은 날 Qualcomm 실적도 부진했다. 선거 뒤 민주당 지도부(Sanders, AOC 등)가 "전국 데이터센터 모라토리엄"을 주장했고, 데이터센터를 다룬 주민투표 41건 중 40건이 규제 쪽으로 통과됐다.
4. **11-02~06: 한국 외국인 대량 매도** — 외국인은 유가증권시장에서 약 10.4조 원을 순매도했다. 그중 하이닉스가 5.4조, 삼성이 2.2조다. 외국인 보유율은 삼성 45.95%(2009년 이후 최저), 반도체 업종 46.6%(2013년 이후 최저)까지 내려갔다.
5. **11-12: Jensen Huang NVIDIA CEO 사임**(SEC 8-K로 확인) — CFO Colette Kress가 임시 CEO를 맡고, 후임은 내부와 외부 후보를 모두 검토한다. 사임 배경과 관련된 스캔들 보도는 매체마다 내용이 달라 확인되지 않았다. HSBC는 Hold로, Bernstein은 Market-Perform으로 투자의견을 낮췄다.

## 3. 앞으로를 가를 변수

### 수요 쪽 (가장 중요)
- **하이퍼스케일러 capex**: 2026년 가이던스 합계는 약 7,300억 달러다(Alphabet 1,950~2,050억, Microsoft 약 1,700억, Meta 1,350~1,450억, Amazon 약 2,200억). 2027년 공식 가이던스는 아직 아무도 내지 않았다. 컨센서스는 4사 합계 약 9,350억 달러(+28%), Goldman은 Oracle을 포함해 약 1.2조 달러로 본다. Alphabet은 2027년 capex가 "크게 늘어날 것"이라고 했다. → **숫자 자체는 아직 증가 쪽이다.** 시장은 증가 속도가 둔해지는지를 보고 있다.
- **정책·전력**: 연방 차원의 모라토리엄은 법안이 제안된 단계다. 실제 허가는 주 단위로 결정되고, 뉴욕은 이미 모라토리엄을 시행 중이며 텍사스는 행정명령으로 신규 개발을 멈춘 상태다. 새 의회는 2027년 1월에 출범하므로, 구체적인 법안과 표결은 그 뒤에야 나온다. 그동안은 불확실성 자체가 할인 요인으로 남는다.
- **수출 규제**: 선거 이후 칩 수출 정책에 대한 보도는 거의 없다. 확인되지 않은 변수로 남겨 둔다.

### 공급·가격 쪽 (메모리)
- 리서치 기관들은 2027년 HBM 가격이 크게 오른다고 본다. TrendForce는 평균판매가 +121%, UBS는 +79%를 제시했다. HBM4 단가는 기가비트당 약 4.5달러로, HBM3E(약 1.5달러)의 세 배 수준이다.
- 삼성, 하이닉스, Micron은 2027년 물량이 대부분 이미 계약됐다는 보도가 있다. 신규 생산능력은 2028~29년에 들어온다.
- 위험 요인: 증설이 과하면 공급 과잉이 올 수 있다. 시장 일부는 "메모리 가격 정점"을 이유로 이미 매도하고 있다. 판매량보다 가격이 이익을 좌우하는 구조라서, 가격 기대가 꺾이면 이익 추정치가 빠르게 내려간다.

### 개별 이벤트
- **NVDA 3분기(FY27) 실적: 11-17 장 마감 후.** 가이던스는 매출 약 1,080억 달러(±2%), 총이익률 74%(±0.5%p)이고, 컨센서스는 약 1,090억 달러다. 볼 것은 데이터센터 매출, Blackwell(GB300) 출하, Rubin 일정, 4분기 가이던스, 그리고 임시 CEO 체제에 대한 설명이다.
- CEO 후임 발표 시점과 인선.

## 4. 시나리오 (조건 → 흐름, 확률·목표가 없음)

| 시나리오 | 조건 | 예상되는 흐름 |
|---|---|---|
| 회복 | NVDA 실적·가이던스가 컨센서스를 넘고, 하이퍼스케일러가 2027년 capex 증가를 숫자로 확인하고, 주 단위 규제가 일부 지역에 그침 | 낙폭이 컸던 장비·메모리 쪽 반등 폭이 클 수 있음. 외국인 보유율이 바닥이라 수급이 돌아오면 한국 반도체 탄력이 큼 |
| 박스권 (기본) | 실적은 괜찮지만 정책 불확실성이 1월 의회 출범 전까지 이어짐 | 이벤트마다 등락하고 종목 간 차이가 커짐(이번 주 NVDA와 AMD·AVGO가 반대로 움직인 것처럼). 상관관계가 높아서 업종 전체가 함께 출렁이는 날이 많음 |
| 추가 하락 | 4분기 가이던스 실망, capex 하향, 연방 모라토리엄 법안 진전, HBM 가격 기대 꺾임 중 하나 이상 | 수요(capex)와 가격(HBM) 기대가 함께 꺾이면, 1년 상승 폭이 컸던 메모리 쪽 조정이 깊어질 수 있음 |

## 5. 우리 데이터로 이어서 볼 것

- `regime_1d`: `risk_regime`, `vix`, `hy_spread`, `kospi_rv_pctile`. 업종 문제인지 시장 전체 위험회피인지 구분한다.
- `dart_disclosures`: 삼성·하이닉스 단일판매·공급계약 공시. HBM 계약을 확인하는 신호가 된다.
- 외국인 수급 데이터는 아직 수집하지 않는다. 필요하면 KRX 투자자별 매매동향을 수집 대상에 추가할지 검토한다.
- 뉴스는 아직 수집하지 않는다(`docs/data.md` 한계 항목). 이번 정리는 웹 검색 결과에 의존했다.

## 6. 한계

- 웹 보도는 2차 출처가 많고, 수치가 매체마다 조금씩 다르다(SOX 낙폭 9~12% 등). 확인된 1차 출처는 NVIDIA 8-K와 회사 가이던스 정도다.
- NVDA 스캔들의 세부 내용과 Intel 급락 폭(-29%) 같은 일부 주장은 확인하지 못해 본문에서 뺐다.
- 로컬 가격은 11-13 종가까지다. 미래참조는 없다(관측된 과거 값만 사용).

## 출처
- [Reuters/Investing.com — Global AI stocks sink as Nvidia crisis deepens](https://www.investing.com/news/stock-market-news/global-ai-stocks-sink-as-nvidia-crisis-deepens-sector-selloff-5021776)
- [Bloomberg — Semiconductor Stocks Plunge After US Election](https://www.bloomberg.com/news/articles/2026-11-05/chips-selloff-worsens-as-fears-about-ai-trade-grip-stock-market)
- [CNBC — AI stocks hammered by midterm shock](https://www.cnbc.com/2026/11/06/ai-data-center-midterms-selloff.html)
- [CNBC — Nvidia stock drops for 6th week](https://www.cnbc.com/2026/11/13/nvidia-stock-jensen-affair-ai.html)
- [SEC — NVIDIA 8-K CEO transition](https://www.sec.gov/Archives/edgar/data/0001045810/000104581026000082/nvidiaannouncesceotransiti.htm)
- [Forex.com — Nvidia earnings preview](https://www.forex.com/en/news-and-analysis/nvidia-earnings-preview-nvda-s-first-test-of-the-post-jenson-huang-era/)
- [CNBC — Google and Microsoft put off raising capex forecasts](https://www.cnbc.com/2026/10/29/microsoft-google-ai-capex-estimates.html)
- [CNBC — Democrats call for data center moratorium](https://www.cnbc.com/2026/11/05/democrats-data-center-restrictions-midterms.html)
- [CNBC — Data center ballot measures](https://www.cnbc.com/2026/11/04/data-center-ballot-measures-2026-midterm.html)
- [Seoul Economic Daily — HBM prices to more than double in 2027](https://en.sedaily.com/finance/2026/10/01/hbm-prices-to-more-than-double-in-2027-on-supply-crunch)
- [TrendForce — HBM prices potentially up 50%+ in 2027](https://www.trendforce.com/news/2026/08/25/news-nvidias-reported-15-server-hike-boosts-samsung-sk-hynix-with-hbm-prices-potentially-up-50-in-2027/)
- [헤럴드경제 — 외국인 하이닉스 5.4조·삼성 2.2조 매도](https://biz.heraldcorp.com/article/10929440)
- [한국일보 — 삼전·닉스에서만 8.2조 던졌다](https://www.hankookilbo.com/news/article/A2026110614150004204)
