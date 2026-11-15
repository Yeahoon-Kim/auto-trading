# AI 반도체 밸류체인 종합 분석 (2026-11-15)

> 방식: llm-desk 설계(architecture.md 7·8절)를 수동으로 적용 — 애널리스트 4명(기술·펀더멘털·공시/뉴스·국면) → 강세/약세 토론 → 판단.
> llm-desk는 아직 코드로 구현되지 않아 이번 분석은 대화 세션에서 진행했고, `decision_log`에는 기록하지 않았다.
> 투자 권유가 아닌 리서치 정리다. 매매 판단은 사용자가 직접 한다.

## 0. 데이터와 한계

| 항목 | 내용 |
|---|---|
| 재무 | yfinance `income_stmt`·`quarterly_*`·`info` (2026-11-15 조회). 연간 4~5년, 분기 최근 5개 |
| 주가 | yfinance 수정주가, 1년 (마지막 거래일 2026-11-13) |
| 뉴스·수주 | 웹 검색 (2026-08~11월 기사). 2차 출처가 많아 **확인 수준을 표시**함 |
| 한계 | ① 한국 기업 yfinance 분기 실적은 2026-06월분까지만 있음 → 3분기는 회사 발표치(뉴스)로 보완 ② 한미반도체 yfinance 재무는 단위·값이 불안정해 정량 비교에서 뺌 ③ yfinance `forwardPE`는 한국 종목에서 비정상(2~3배)이라 쓰지 않고, 시가총액/연환산 영업이익을 직접 계산함 ④ **선정 편향**: 지금 잘나가는 기업을 골랐으므로 "핵심기업" 목록 자체에 생존편향이 있다 ⑤ 백테스트가 아니므로 비용·미래참조 점검은 해당 없음 |

---

## 1. 밸류체인과 핵심기업 선정

```
[설계 IP/EDA] → [가속기 설계] → [파운드리·첨단패키징] ← [메모리(HBM)]
  Arm/Synopsys    NVIDIA, AMD,     TSMC (CoWoS)         SK하이닉스, 삼성전자, Micron
                  Broadcom(ASIC)        ↑                      ↑
                                 [전공정 장비]           [후공정 장비]
                                 ASML, AMAT, Lam         한미반도체(TC본더)
                                         ↓
                          [수요] 하이퍼스케일러 capex (MS·구글·아마존·메타·오라클) + AI 랩(OpenAI·Anthropic)
```

| 단계 | 선정 기업 | 선정 이유 (병목·점유율) |
|---|---|---|
| 가속기 | **NVIDIA** | GPU+NVLink+CUDA. AI 가속기 매출의 대부분 |
| 가속기 | **AMD** | 2번째 GPU 공급자. OpenAI 6GW 계약 |
| 가속기(맞춤) | **Broadcom** | 구글 TPU·메타·OpenAI·Anthropic 맞춤 ASIC + 이더넷 스위치 |
| 파운드리 | **TSMC** | 최첨단 공정·CoWoS 사실상 독점 |
| 메모리 | **SK하이닉스** | HBM 1위, NVIDIA 주력 공급 |
| 메모리 | **삼성전자** | HBM4 공급 확대, 범용 D램 최대 생산자 |
| 메모리 | **Micron** | HBM 3강 중 하나, 미국 유일 |
| 장비 | **ASML** | EUV 독점 — 공급 증설의 최종 병목 |
| 장비 | **Applied Materials, Lam Research** | 증착·식각. 메모리·첨단패키징 투자에 직결 |
| 후공정 장비 | **한미반도체** | HBM TC본더 (SK하이닉스 주력) |

---

## 2. 과거 실적 (연간·분기)

### 2-1. 연간 매출 / 영업이익

| 기업 | 통화 | FY-2 | FY-1 | 최근 FY | 비고 |
|---|---|---|---|---|---|
| NVIDIA (FY 1월말) | $B | 60.9 / 33.0 | 130.5 / 81.5 | **215.9 / 130.4** (FY26) | 3년 매출 8배 |
| AMD | $B | 22.7 / 0.4 | 25.8 / 2.1 | **34.6 / 3.7** (2025) | 이익률 낮음 (11%) |
| Broadcom (FY 10월말) | $B | 35.8 / 16.5 | 51.6 / 15.0 | **63.9 / 26.1** (FY25) | VMware 인수 상각 후 회복 |
| TSMC | TWD B | 2,162 / 921 | 2,894 / 1,322 | **3,809 / 1,936** (2025) | |
| SK하이닉스 | 조원 | 32.8 / −7.7 | 66.2 / 23.5 | **97.1 / 47.2** (2025) | 2023 적자 → 2년 만에 역대 최대 |
| 삼성전자 | 조원 | 258.9 / 6.6 | 300.9 / 32.7 | **333.6 / 43.6** (2025) | |
| Micron (FY 8월말) | $B | 25.1 / 1.3 | 37.4 / 9.8 | **133.2 / 99.3** (FY26) | 1년 매출 3.6배 |
| ASML | €B | 27.6 / 9.0 | 28.3 / 9.0 | **32.7 / 11.3** (2025) | 2026 가이던스 €44~45B |
| AMAT (FY 10월말) | $B | 26.5 / 7.7 | 27.2 / 7.9 | **28.4 / 8.5** (FY25) | 3년 정체 후 2026 반등 |
| Lam (FY 6월말) | $B | 14.9 / 4.3 | 18.4 / 5.9 | **23.2 / 8.2** (FY26) | |

### 2-2. 최근 분기 흐름 (매출 / 영업이익)

| 기업 | 5분기 전 → 최근 | 최근 분기 OPM | 성장 판단 |
|---|---|---|---|
| NVIDIA | 46.7/28.4 → 57.0 → 68.1 → 81.6 → **96.2/63.7** (7월분기) | 66% | 분기당 +15~20% 꾸준. 다음 분기 가이던스 $108B |
| AMD | 9.2/1.3 → 10.3 → 10.3 → 11.5 → **13.4/2.8** (9월분기) | 21% | MI450 출하 시작으로 4분기 데이터센터 +50% 가이던스(2차 출처) |
| Broadcom | 16.0/6.1 → 18.0 → 19.3 → 22.2 → **29.6/16.1** (7월분기) | 54% | 직전 분기 대비 +33% 급가속 |
| TSMC | 990/501 → 1,046 → 1,134 → 1,270 → **1,494/877** (9월분기) | 59% | 6분기 연속 사상 최대 |
| SK하이닉스 | 22.2/9.2 → 24.4 → 32.8 → 52.6 → 79.3 → **104.6/83.3** (3Q26, 회사 발표) | **80%** | 4분기 만에 영업이익 9배 |
| 삼성전자 | 74.6/4.7 → 86.1 → 93.8 → 133.9 → 171.5 → **195.3/107.4** (3Q26, 회사 발표) | 55% | 메모리가 견인, 모바일·파운드리 일부 적자 |
| Micron | 11.3/3.7 → 13.6 → 23.9 → 41.5 → **54.2/43.8** (8월분기) | 81% | 다음 분기 가이던스 $61.5B, GM 86% |
| ASML | 7.5/2.5 → 9.7 → 8.8 → 9.3 → **11.7/5.2** (3Q26) | 44% | 사상 최대 |
| AMAT | 7.3/2.2 → 6.8 → 7.0 → 7.9 → **9.1/3.1** (7월분기) | 34% | 2026 들어 가속 |
| Lam | 5.3/1.8 → 5.3 → 5.8 → 6.7 → **8.2/3.3** (9월분기) | 40% | 가속 |

**요약**: 2026년 이익 증가의 중심이 **가속기(NVIDIA) → 메모리**로 이동했다. 메모리 3사는 영업이익률이 80% 안팎으로, 역사적으로 반도체 업종에서 본 적 없는 수준이다. 장비는 2026년 들어서야 가속이 시작됐다(후행).

---

## 3. 재무 안전성

| 기업 | 현금 | 총차입 | 순현금 | 유동비율 | 부채/자본(%) | FCF(TTM) | 판단 |
|---|---|---|---|---|---|---|---|
| NVIDIA | $62.5B | $38.9B | +23.6 | 4.6 | 17 | $41.8B | 매우 안전. **재고 4분기 만에 $15→31.6B (2.1배)** 는 감시 항목 |
| AMD | $15.8B | $9.2B | +6.6 | 2.7 | 13 | $7.4B | 안전. 재고 1분기에 $8.5→11.5B (MI450 선행 생산) |
| Broadcom | $24.0B | $59.4B | **−35.4** | 2.5 | 60 | $30.6B | 차입 가장 많음(VMware). FCF로 2년 내 상환 가능 수준 |
| TSMC | TWD 4,017B | 818B | +3,199 | 2.5 | 12 | TWD 1,203B | 매우 안전. capex $60~64B를 자체 현금으로 감당 |
| SK하이닉스 | 114.5조 | 15.8조 | +98.7조 | 26.9* | 5 | — | 2023 적자기 차입을 모두 정리. 2026 주주환원 90조+ 발표 |
| 삼성전자 | 214.8조 | 17.2조 | +197.6조 | 9.6* | 3 | 83.8조 | 최상위 |
| Micron | $43.4B | $5.2B | +38.2 | 3.3 | 4 | $29.1B | 안전. 단 capex $30.7B → 상반기만 $25B로 급증 |
| ASML | €12.7B | €2.0B | +10.7 | 1.5 | 9 | €10.4B | 안전 (선수금 구조라 유동비율 낮게 보임) |
| AMAT | $9.3B | $6.5B | +2.8 | 2.3 | 25 | $2.5B | 양호. FCF가 이익 대비 적음 (capex 2배 증가) |
| Lam | $7.4B | $3.7B | +3.7 | 2.5 | 27 | $4.5B | 양호 |

\* yfinance 원값. 한국 기업 유동비율은 분기 재무상태표 기준이라 값이 커 보일 수 있음.

**결론**: 10개사 모두 재무 위험은 낮다. Broadcom만 순차입. 위험은 재무가 아니라 **업황(가격·수요) 쪽**에 있다.

---

## 4. 계약·수주로 본 미래 전망

| 기업 | 수주·계약 소식 | 확인 수준 |
|---|---|---|
| NVIDIA | 3Q FY27 가이던스 $108B(±2%), GM 74%. FY28 매출 **약 +70%** 전망, 공급 병목 최소 FY28 말까지. Vera Rubin 8월 출하 시작. AWS 2027~28 GPU 200만 개 추가 배치 | 가이던스는 회사 발표. 수주잔고 $279B는 단일 출처 |
| NVIDIA (리스크) | **11월 젠슨 황 CEO 사임 보도**, 시총 $4T 아래로, 주간 −12%. 후임 미정 상태로 실적 발표 예정 | 다수 언론 보도. 세부는 확인 필요 |
| AMD | OpenAI 6GW 계약(1차 1GW MI450, 2026 하반기부터). OpenAI에 최대 1.6억 주 워런트(배치 마일스톤 연동). 4Q 데이터센터 +50% QoQ 가이던스 | 계약은 공식. 4Q 가이던스는 2차 출처 |
| Broadcom | Anthropic TPU 주문 $10B + 추가 $11B, Anthropic 2026 1GW·2027 3GW+ TPU. OpenAI 맞춤칩 2027부터 1GW+. AI 관련 수주잔고 $73B(18개월). **"2027 AI 칩 매출 $100B 이상 가시권"**(CEO) | CEO 발언. OpenAI 규모는 출처마다 다름 |
| TSMC | 3Q26 매출 $46.7B, HPC 68%. 2026 capex $60~64B 유지, 2026~28 capex는 이전 3년보다 크게 증가. N2 2027 2배+. CoWoS 2026말 월 13만 → 2028말 26만 장(언론 추정). 단 **"2027 계획은 신중하게"** | 회사 발표 + 언론 추정 구분 |
| SK하이닉스 | 3Q26 매출 104.6조, 영업이익 83.3조(OPM 80%). **2027 HBM 물량·가격 협상 완료**. 고객 14곳과 장기계약(LTA), 2027~28 생산능력의 50% 이상 확보. 4Q부터 HBM4가 주력. 2030년까지 공급 부족 전망 | 회사 발표(감사 전 수치) |
| 삼성전자 | 3Q26 매출 195.3조, 영업이익 107.4조. HBM4 판매 크게 확대(엔비디아 Rubin 연결은 2차 보도). 파운드리 2nm 신규 수주, Tesla AI6 $16.5B(2025 계약, 2033년까지) | 회사 발표 + 2차 보도 |
| Micron | FY26 4Q 매출 $54.2B(+379%). 다음 분기 $61.5B·GM 86%. **2027 생산의 75% 이상 이미 확정**, HBM 2027 상당 부분 매진·가격은 2026보다 크게 높음. 2027·28 수요>공급. 신규 팹은 2028 | 회사 발표 |
| ASML | 3Q26 매출 €11.7B 사상 최대. **2027·28 장비 공급이 수요를 못 따라감**, 고객이 2029~30 주문 문의. EUV 2026 약 65대 → 2027 +30% | 회사 발표. 수주액은 미공개 |
| 한미반도체 | SK하이닉스와 HBM4용 TC본더 계약(2026년 2건 이상). 단, 엔비디아 HBM4 인증 지연 시 발주 이연 우려 보도 | 공시 + 언론. 3Q26 실적은 이번 검색에서 확인 못 함 |
| 수요단 | 하이퍼스케일러 capex 2026 약 $730~785B → **2027 $1조 안팎**(Moody's·BofA·Evercore). 다만 2027엔 capex가 FCF를 넘을 수 있다는 분석(Reuters/LSEG) | 7월 기준 추정. 10월 실적 반영 전 |
| 메모리 가격 | TrendForce: 4Q26 범용 D램 계약가 +10~15% QoQ로 **상승 폭 둔화**, PC D램 +13~18%. 2027 HBM 가격 +70~140% 가능하나 일부 LTA에 가격 상한. 11월 초 현물가 혼조, Morgan Stanley는 4Q26 고점 가능성 제기(2차 출처) | TrendForce 원문 + 2차 출처 |

---

## 5. 애널리스트 4인 의견 (llm-desk 1단계)

### ① 기술(가격) 애널리스트
| 종목 | 1개월 | 3개월 | 1년 | 52주 고점 대비 | 200일선 대비 |
|---|---|---|---|---|---|
| SOXX(업종) | −13% | −20% | +56% | −32% | −7% |
| NVIDIA | −25% | −28% | −14% | −32% | −20% |
| AMD | −5% | +6% | +116% | −18% | +26% |
| Broadcom | +1% | −11% | +2% | −28% | −5% |
| TSMC | −6% | −7% | +41% | −18% | −1% |
| SK하이닉스 | −4% | −15% | +99% | **−58%** (6/22 고점) | −20% |
| 삼성전자 | −2% | −12% | +106% | −42% | −13% |
| Micron | −16% | −23% | +216% | −36% | +2% |
| ASML | −10% | −20% | +49% | −25% | −6% |
| AMAT | −14% | −31% | +65% | −49% | −17% |
| Lam | −19% | −34% | +53% | −48% | −20% |
| 한미반도체 | −5% | −15% | +37% | −56% | −28% |

의견: **업종 전체가 하락 추세.** 한국 메모리·장비는 6월 고점 이후 5개월째 조정이고, NVIDIA는 10월 고점 후 한 달 새 −25%로 가장 최근에 꺾였다. 200일선 위에 있는 건 AMD·Micron뿐. 추세 신호로는 "아직 바닥 확인 전".

### ② 펀더멘털 애널리스트
시가총액 ÷ 최근 분기 영업이익×4 (단순 연환산, 낮을수록 이익 대비 싸다):

| 기업 | 배수 | 해석 |
|---|---|---|
| SK하이닉스 | **약 2.6배** | 시장이 "지금 이익은 고점, 오래 못 간다"고 가격을 매김 |
| 삼성전자 | 약 3.1배 | 같음 |
| Micron | 약 5.0배 | 같음 (가이던스는 더 높음) |
| NVIDIA | 약 15배 | 다음 해 +70% 가이던스 감안 시 역사적 저점권 |
| TSMC | 약 18배 | |
| Lam / AMAT | 약 21배 / 24배 | |
| ASML / Broadcom | 약 26배 | |
| AMD | 약 78배 | 이익이 아직 작음 — 2027 MI450 성공에 거는 가격 |

의견: 이익 성장·재무 모두 최상. 메모리의 극단적 저배수는 "싸다"가 아니라 **"이익의 지속성을 안 믿는다"** 는 뜻 — 메모리 업황에서 PER이 가장 낮을 때가 고점이었던 과거 패턴(2018, 2022)과 같은 모양이다. 반대로 이번엔 장기계약(LTA)·가격 확정이 2027~28까지 있다는 점이 과거와 다르다.

### ③ 공시·뉴스 애널리스트
- 긍정: 메모리 3사 모두 **2027 물량·가격을 이미 확정**(SK하이닉스 협상 완료, Micron 75%+ 확정). ASML은 2027~28 장비 부족을 말함. Broadcom "2027 AI 칩 $100B+". NVIDIA FY28 +70%.
- 부정: NVIDIA CEO 사임(경영 공백). TSMC "2027 신중". 범용 D램 가격 상승률 둔화와 현물가 혼조. 하이퍼스케일러 capex가 FCF를 넘는 구간 진입 우려. 미국 선거 결과가 AI 투자 속도에 미칠 영향 논란.
- 의견: **실적·수주 뉴스는 여전히 강하고, 악재는 "가격·심리·경영" 쪽**이다. 수주가 깨진 소식은 아직 없다.

### ④ 국면 애널리스트
- 11/5 SOX −12.2%(하루), 나스닥 조정 진입. 11/13 물가 상회·유가·채권 약세가 겹침.
- 의견: 실적이 아니라 **할인율(금리·물가)과 위험선호**가 주가를 끌어내리는 국면. 이런 국면에서는 실적이 좋아도 배수가 줄어든다. 프로젝트의 `regime_1d`로 확인 필요(이번 분석에서는 미실행).

---

## 6. 강세 / 약세 토론 (llm-desk 2단계)

**강세 논거**
1. 수요가 계약으로 잠겨 있다: 메모리 2027~28 LTA, Broadcom·AMD의 GW 단위 계약, AWS 200만 GPU.
2. 공급이 물리적으로 막혀 있다: EUV 증설 +30%, CoWoS 2028까지 2배, 신규 메모리 팹은 2028 — 공급 과잉이 오려면 최소 2년.
3. 하이퍼스케일러 capex 2027 $1조 — 아직 증가 중.
4. 가격이 이미 많이 빠졌다: 메모리 연환산 영업이익의 2.6~5배, NVIDIA 15배.
5. 재무가 탄탄해 하락기에도 자사주 매입·배당(SK하이닉스 90조+) 여력이 있다.

**약세 논거 (반드시 작성)**
1. **메모리 이익률 80%는 지속 불가능한 숫자**다. 역사적으로 메모리 OPM 고점 = 주가 고점이었고, 주가는 이익보다 6~12개월 먼저 꺾인다. SK하이닉스 주가는 6월에 이미 고점을 찍었다.
2. 범용 D램 가격 상승률이 둔화하기 시작했다(4Q +10~15%로 감속). 이익의 "2차 미분"이 음수로 돌아서는 시점이 주가에 가장 나쁘다.
3. LTA에 **가격 상한**이 들어가기 시작 — 2027 가격 상승의 상단이 막힐 수 있다.
4. 하이퍼스케일러가 2027년 capex > FCF 구간에 들어가면 차입·주주 압박으로 증가율이 꺾일 수 있다. 장비·메모리는 capex 증가율에 반응하지, 절대액에 반응하지 않는다.
5. NVIDIA 경영 공백 — 밸류체인 전체의 수요 신호를 내는 회사의 불확실성.
6. NVIDIA 재고 4분기 만에 2.1배, AMD 재고 1분기 +35% — 출하 준비일 수도, 수요 둔화 신호일 수도 있다.

---

## 7. 판단 (llm-desk 3단계)

**업종 관점: 펀더멘털은 "강함 유지", 주가는 "조정 진행 중". 2027년까지 실적 가시성은 높지만, 시장은 이미 2028년 이후 둔화를 가격에 넣고 있다.**

밸류체인 위치별 관점:

| 구간 | 관점 | 근거 |
|---|---|---|
| 메모리 (SK하이닉스·삼성·Micron) | **실적 정점 근처, 주가는 사이클 논쟁 중** | 2027 물량·가격 확정은 강하지만, 범용 D램 가격 감속이 시작. 이익이 늘어도 배수가 더 줄 수 있음 |
| 가속기 (NVIDIA·Broadcom) | **펀더멘털 최강, 단기 이벤트 위험** | FY28 +70%, $100B 가시권. NVIDIA는 CEO 공백이라는 실적 외 위험 |
| AMD | **실행에 거는 구간** | 배수가 높아 MI450 램프 성과가 확인돼야 정당화 |
| 파운드리 (TSMC) | **가장 균형 잡힘** | 독점 + 순현금 + 적정 배수. 단 "2027 신중" 발언 |
| 장비 (ASML·AMAT·Lam·한미) | **후행 수혜, 단 고베타** | 2027~28 장비 부족은 긍정. 그러나 3개월 −20~34%로 가장 크게 흔들림. 한미는 HBM4 인증 일정에 민감 |

시나리오 (2027년 기준, 확률은 주관적 추정):

| 시나리오 | 조건 | 확률 | 결과 |
|---|---|---|---|
| 강세 | capex $1조+ 확인, HBM4 순조, 범용 D램 가격 보합 이상 | 35% | 메모리 이익 2027 추가 증가, 배수 재평가 |
| 기본 | capex 증가율 둔화(+20~30%), 범용 D램 2027 하반기 하락 전환, HBM 유지 | 45% | 가속기·TSMC는 이익 성장 지속, 메모리는 이익 정체 → 주가 횡보 |
| 약세 | AI 투자 회수 우려로 capex 동결, 재고 조정 | 20% | 메모리·장비 이익 급감 (과거 사이클처럼 −50%+) |

**앞으로 확인할 지표 (판단 갱신 조건)**
1. NVIDIA 3Q FY27 실적(11월 하순): $108B 달성 여부, 재고 증가 설명, 후임 CEO
2. 하이퍼스케일러 2027 capex 가이던스 (4Q 실적, 2027-01~02월)
3. TrendForce 1Q27 계약가 전망 — 범용 D램 상승률이 0% 아래로 가는지
4. SK하이닉스·Micron 4Q 실적에서 HBM4 비중과 2028 LTA 진척
5. ASML 4Q 수주액 공개 여부 (장비 사이클의 선행지표)
6. 프로젝트 `regime_1d` 국면 지표 — 위험회피 국면이 끝나는지

---

## 8. 출처

실적·가이던스
- [SK hynix 3Q26 (Webull 게재 보도자료)](https://www.webull.com/news/15822865136734208), [Sammyfans — 14개 LTA, 2029까지](https://www.sammyfans.com/2026/10/28/sk-hynix-q3-margin-hits-record-80-secures-14-ltas-with-memory-supply-booked-to-2029/amp/), [Guru3D](https://www.guru3d.com/story/sk-hynix-posts-record-quarter-and-expects-memory-shortage-through-2030/)
- [Samsung 3Q26 Results](https://news.samsung.com/global/samsung-electronics-announces-third-quarter-2026-results), [Light Reading](https://lightreading.com/finance/memory-boom-carries-samsung-to-another-bumper-quarter), [Magica](https://magica.com/news/samsung-q3-2026-profit-ai-memory-device-losses)
- [Pulse2 — NVIDIA Q2 FY27, Q3 가이던스 $108B](https://pulse2.com/nvidia-operating-income-jumps-124-to-63-7-billion-as-revenue-doubles-and-q3-outlook-hits-108-billion/), [Jon Peddie](https://www.jonpeddie.com/news/nvidia-financial-results-for-q226-fy27/)
- [Micron 가이던스 $61.5B (AI Weekly)](https://aiweekly.co/alerts/micron-q4-revenue-jumps-379-to-542b-on-ai-memory-demand-guides-fiscal-2027-q1), [24/7 Wall St.](https://247wallst.com/investing/2026/10/01/micron-blew-past-estimates-and-says-it-cant-see-when-the-memory-shortage-ends/)
- [TSMC 3Q26 (CRN Asia)](https://www.crnasia.com/news/2026/components-and-peripherals/tsmc-third-quarter-revenue-tops-guidance-as-ceo-says-ai-dema), [The Next Web — 2027 신중](https://thenextweb.com/news/tsmc-q3-2026-record-profit-prudent-2027), [TrendForce — CoWoS 2028 2배](https://www.trendforce.com/news/2026/09/14/news-tsmc-reportedly-targets-22-2nm-16-3nm-capacity-boost-by-mid-2027-cowos-to-double-by-2028/)
- [ASML 3Q26 (investingLive)](https://investinglive.com/stocks/asml-says-2027-and-2028-supply-will-fall-short-of-demand-as-q3-sales-hit-a-record/), [Unite.AI](https://unite.ai/asml-warns-2027-2028-chip-tool-output-will-fall-short-of-ai-demand)
- [AMD 3Q26 (sec-api 분석, 미검증)](https://sec-api.io/insights/amd-q3-2026-earnings-sec-filing-analysis-data-center-revenue-openai-and-meta-warrants-guidance), [The Stack — Helios](https://www.thestack.technology/amd-helios-open-rack-ai/)

수주·계약
- [DCD — Broadcom Anthropic 주문](https://www.datacenterdynamics.com/en/news/broadcom-bulges-with-anthropic-ai-orders/), [SiliconANGLE](https://siliconangle.com/2026/03/04/broadcoms-ai-chip-business-jumps-106-cruises-another-solid-earnings-revenue-beat/), [LetsDataScience](https://letsdatascience.com/news/broadcom-secures-major-custom-ai-deals-43729fca)
- [딜사이트 — 한미반도체 HBM4 TC본더](https://dealsite.co.kr/articles/150315), [딜사이트](https://dealsite.co.kr/articles/150642)

수요·가격·국면
- [DCD — Moody's capex 2027 $1조 근접](https://datacenterdynamics.com/en/news/moodys-hyperscaler-capex-forecasts-marked-up-by-85bn-to-close-in-on-1trn-by-2027/), [Khaleej Times — FCF 압박](https://www.khaleejtimes.com/business/tech/ai-investment-boom-puts-big-techs-free-cash-flow-under-pressure)
- [TrendForce 11/11](https://www.trendforce.com/presscenter/news/20261111-13303.html), [TrendForce 9/30](https://www.trendforce.com/presscenter/news/20260930-13258.html), [FourWeekMBA — 메모리 고점?](https://fourweekmba.com/ai-has-memory-peaked-spot-prices-slip-as-micron-rallies-24/)
- [Yahoo Finance — NVIDIA CEO 사임, 반도체 매도](https://finance.yahoo.com/markets/article/nvidia-stock-slips-as-investors-digest-ceo-exit-rest-of-chip-sector-falls-144818245.html), [CNBC 11/6](https://www.cnbc.com/2026/11/06/tech-stocks-sk-hynix-samsung-softbank.html), [Invezz 11/13](https://invezz.com/news/2026/11/13/why-are-micron-sk-hynix-and-sandisk-stocks-falling-on-friday/)
