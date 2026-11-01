# 포트폴리오 층 — 팩터 위험모델 · 제약 최적화 · 충격 비용 · 성과 귀속 (v0.9.0)

- 날짜: 2026-11-01
- 범위: 시스템 트레이딩 11단계(유니버스→데이터→알파 발굴→알파 검증→ML 예측→**위험모델**→**최적화**→**거래비용**→
  백테스트→집행→**성과 귀속·모니터링**) 중 네 레포 어디에도 없는 칸을 코어에 만든다. 전략은 여기 두지 않는다 —
  위험모델의 스타일 팩터는 "무엇이 위험을 설명하나"의 측정 도구지 알파 주장이 아니다.
- 사용자 지시: 11단계 그림을 주며 "이게 중요한 레포에서 작업을 시작해 … 질문없이 바로 구상해서 들어가봐"(2026-11-01).

## 1. 왜 — 2026-11-01 감사

| 단계 | 지금 있는 것 | 빈 곳 |
|---|---|---|
| 4 알파 검증 | `backtest.crosssectional.rank_ic`(NW-t), `stats.sharpe`(DSR·PSR), `stats.cv`(purge·embargo), `runtime.oos` | IC 감쇠·ICIR·회전율(portfolio-research `research/ic.py` 에만 있고 미국용) |
| 6 위험모델 | 없음(portfolio-research `covariance.py` 의 Ledoit-Wolf 표본 공분산뿐) | 노출·팩터 공분산·특이위험 |
| 7 최적화 | `rank_tilt_backtest`·`staggered_tranche_backtest` 는 순위 동일가중 | 위험·섹터·베타·회전율·ADV 제약이 있는 최적화 |
| 8 비용 | `costs.tax`(일자별 세율), `costs.model`(수수료·고정 슬리피지), `backtest.orderbook`(10호가 sweep) | 주문 크기 의존 충격(%ADV) |
| 11 귀속·모니터링 | `risk.killswitch`, `runtime.nightly` | 팩터 기여 분해, 알파 감쇠 추적 |

swing-it 이 통과시킨 유일한 알파(PEAD)는 **확산형·기관 엣지**(`swing-it/docs/alpha-research.md` §4.1)다 — 개별 거래가 얇고
다수 종목 분산으로 북 Sharpe 를 만든다. 확산형 엣지의 북 품질은 신호보다 **구성(위험·비용)** 에서 갈린다. 그런데 지금은
그 구성을 잴 도구가 없어 "동일가중 순위 틸트"가 유일한 선택지다.

## 2. 원칙(기존 코어 계약 그대로)

- 기존 함수 출력은 바꾸지 않는다. 전부 **새 모듈·새 함수**다.
- 통계·진단 함수는 숫자만 낸다(`pass`/`fail`/`verdict` 키 금지 — 기존 테스트가 막는다).
- 순수 함수(numpy/pandas) 우선. DB 를 읽는 것은 `data/` 의 로더 하나(`--extra db`). 최적화만 새 선택 의존성
  `portfolio = ["cvxpy>=1.5"]`(기본 솔버 Clarabel). 없으면 `import` 시점이 아니라 호출 시점에 안내 예외.
- 패널 규약은 `backtest.crosssectional` 과 같다: `code × date` numpy 배열, `dates` 는 열 라벨. 날짜 t 의 값은 **t 장 마감까지
  알 수 있는 정보만** 쓴다(호출부 책임이 아니라 로더 책임 — §3.5).

## 3. 모듈

### 3.1 `risk/factor_model.py` — 횡단면 팩터 위험모델 (numpy)

구조: r_t = X_{t-1} f_t + u_t (노출은 전일 마감 기준). 공분산 Σ = X F Xᵀ + diag(D).

- `standardize(raw, cap, mask, *, clip=3.0) -> ndarray[N]`: 결측 제외 → 중앙값·MAD 로 ±clip 윈저 → **시총가중 평균 0**,
  동일가중 표준편차 1. 결측은 0(평균)으로 채우지 않고 NaN 으로 둔다(호출부가 마스크로 뺀다).
- `industry_dummies(labels: ndarray[N] of str) -> (ndarray[N, G], list[str])`.
- `cross_section_regression(r, X_style, X_ind, w) -> (f, u)`: 한 날짜. 시장(전 종목 1) + 업종 더미 + 스타일.
  시장과 업종 더미의 공선성은 **업종 수익의 시총가중 합 = 0** 제약으로 푼다(Barra 방식, 제약 WLS 를 변수 치환으로 정확히).
  가중 w = √시총. 결측 행 제외. 반환 f 는 [market, industries..., styles...] 순서.
- `factor_returns(R, X_style, ind_labels, cap, mask) -> FactorReturns(f: ndarray[T, K], u: ndarray[N, T], names)`: 날짜 루프.
- `ewma_cov(F, *, half_life, nw_lags=0) -> ndarray[K, K]`: 지수가중 + Newey-West 보정(Bartlett). 일 단위.
- `specific_var(U, *, half_life, shrink=0.3, size=None, n_groups=10) -> ndarray[N]`: 잔차 EWMA 분산을
  시총 분위 그룹 평균 쪽으로 베이지안 수축(가중 `shrink`). 관측 수가 `half_life` 미만인 종목은 그룹 평균.
- `RiskModel`(dataclass, frozen): `codes, names, X(N×K), F(K×K), D(N)`, 메서드 `cov()`, `var(w)`,
  `decompose(w) -> {factor_var, specific_var, by_factor: dict}`(by_factor 는 x_kᵀ F 행 기여의 합 = factor_var).
- `build_risk_model(..., asof_index, *, cov_half_life=90, spec_half_life=90, nw_lags=2) -> RiskModel`:
  asof_index 까지의 f·u 만 쓴다(미래 누출 테스트 필수).
- `bias_statistic(pred_sd, realized) -> dict(mean, q05, q95, n)`: 예측 표준편차로 나눈 실현 수익의 표준편차(롤링 창별).
  위험모델 검증의 표준 지표 — 1 이면 예측이 맞고 >1 이면 과소예측. 숫자만.

### 3.2 `costs/impact.py` — 크기 의존 비용

- `sqrt_impact_bp(notional, adv_value, daily_vol, *, y=1.0) -> ndarray`: 제곱근 법칙 `Y · σ_daily · √(Q/ADV)` (bp).
  기본 y=1.0 은 문헌 중앙값(Tóth et al. 2011, Y≈O(1)). **보정 전 자리표시값이고 그렇게 문서화** — 소비 레포는 1·2배로 돌린다.
- `trade_cost_bp(side, notional, adv_value, daily_vol, market, trade_date, *, half_spread_bp, y=1.0) -> ndarray`:
  수수료(`costs.model` 정가) + 매도 거래세(`costs.tax` 일자별) + 반스프레드 + 충격. 할인·면제 없음(정가 원칙).
- `participation(notional, adv_value) -> ndarray`: %ADV. 최적화 제약과 보고에 같이 쓴다.
- 단위: `adv_value` 는 **원**. `daily_bars.trade_value`(백만원)를 받는 쪽이 ×1e6 하고, 함수는 ADV 가 1e6 미만이면
  `ValueError`(백만원 단위를 그대로 넣은 사고 2회 — quant-airflow §3).

### 3.3 `portfolio/optimize.py` — 롱온리 제약 평균-분산 (cvxpy)

max_w  αᵀw − (λ/2)·wᵀΣw − Σ_i c_i|Δw_i| − Σ_i k_i|Δw_i|^{3/2}

- Σ 는 팩터 형태 그대로 쓴다(y = Xᵀw 보조변수, N×N 을 만들지 않는다 — 2,500 종목에서 필수).
- 제약: `sum(w) = budget`(기본 1, 현금 허용 시 ≤), `0 ≤ w ≤ max_weight`, 업종 노출 `|Σ_{i∈g} w_i − b_g| ≤ sector_band`
  (b = 벤치 비중, 기본 시총가중 유니버스), 베타(시장 노출) 범위, 스타일 노출 범위(선택), 회전율 `Σ|Δw| ≤ max_turnover`,
  **참여율** `|Δw_i|·AUM ≤ max_participation·ADV_i`, 매수 금지 마스크(거래정지·관리종목·상한가 — 호출부가 준다).
- c_i = (수수료+세율+반스프레드)/1e4, k_i = Y·σ_i·√(AUM/ADV_i)/1e4 — §3.2 와 같은 식(1.5 제곱 = 비용 bp × 거래량).
- `optimize(alpha, model, w_prev, *, aum, adv_value, daily_vol, cost_params, constraints) -> OptResult`
  (`weights, expected_alpha, expected_risk, est_cost, turnover, exposures, status, solver`). 실패(infeasible)는 예외가
  아니라 `status` 로 돌려주고 `weights = w_prev`(거래 안 함) — 백테스트가 조용히 빈 북을 쥐지 않게 status 를 기록한다.
- 기준 구현 확인: 제약이 없을 때 해석해 `w* = (1/λ)Σ⁻¹α` 와 1e-6 이내로 같아야 한다.

### 3.4 `risk/attribution.py` + `stats/ic.py` — 귀속·모니터링

- `factor_attribution(w, X, f, u) -> DataFrame[date, market, <industry>…, <style>…, specific, total]`:
  날짜별 포트 수익 = Σ_k (wᵀX)_k f_k + wᵀu. **total 은 실제 포트 수익과 1e-12 안에서 같다**(항등식 테스트).
- `ic_decay(signal, close, dates, horizons=(1,5,10,20,40,60)) -> DataFrame[h, ic_mean, ic_sd, icir, nw_t, n]`:
  `rank_ic` 를 지평별로 부르는 얇은 층(기존 함수 재사용, 출력 불변).
- `rolling_ic(signal, close, dates, *, horizon, window=12) -> DataFrame`: 모델 드리프트 모니터링용 이동 ICIR.
- `signal_turnover(signal, *, top_q=0.2) -> Series`: 상위 분위 구성의 기간 간 교체율.

### 3.5 `data/risk_inputs.py` — DB 로더 (`--extra db`)

kr_quant 에서 한 번에 읽어 패널로: `daily_bars_adjusted`(종가·거래대금 → 수익률·ADV), `shares_outstanding_history`
(knowledge_date as-of — 시총 = **원 종가**(`daily_bars_raw`) × 그때 알던 주식수, 정정 G21-1), `wics_membership`(주간
스냅샷 → 다음 스냅샷 전까지 앞으로 채움, 2019~ — 그 전은 `None` 반환하고 호출부가 업종 없이 돈다), 상장폐지 포함
(`delisted_stocks`; 폐지 종목 누락은 `AssertionError` — swing-it `read_prices` 와 같은 가드).
2026-09-14~ NXT·애프터마켓 혼입(종가·거래량)은 `data.daily` 의 정규장 기준가 경로를 쓴다.

스타일 원천(위험 설명용, 표준 세트): `size`(log 시총), `beta`(252일 대 KOSPI·반감 63), `momentum`(252−21일),
`resid_vol`(63일 잔차 변동성), `liquidity`(log 20일 ADV/시총), `value`(EP = TTM 순이익 / 시총, `earnings` 의 보수적
as-of — max(avail_date, periodic rcept_dt) 다음 거래일, swing-it `eps_asof` 규약). 정의는 로더의 상수 하나에 모은다.

## 4. 완료 조건 (먼저 정한 합격 바 — 이 줄은 결과를 본 뒤 고치지 않는다)

1. 합성 데이터: 알려진 X·f·D 로 만든 수익에서 `factor_returns` 가 f 를 상관 > 0.99 로, `ewma_cov` 가 F 를 원소 상대오차
   중앙값 < 10% 로 복원. 미래 누출 테스트(asof 뒤 값을 망가뜨려도 `build_risk_model` 출력 불변).
2. `optimize`: 무제약 해석해와 1e-6, 모든 제약이 활성일 때 위반 ≤ 1e-6, 2,500 종목·K≈35 에서 한 번 풀기 < 5초(simnode).
3. `factor_attribution` 항등식 1e-12. `ic_decay` 의 h=40 행이 기존 `rank_ic` 와 바이트 동일.
4. 실데이터(kr_quant, 2019-07~2023-12, 월말 asof, 21거래일 앞): 바이어스 통계의 평균이 [0.8, 1.2] 안 —
   대상 (a) 무작위 롱온리 50종목 100개, (b) 시총가중 유니버스, (c) 스타일 6개 각각의 팩터 모방 포트.
   밖이면 실패로 기록하고 원인을 고친다(바를 넓히지 않는다). 2024~ 는 swing-it 봉인 소모 구간이라 쓰지 않는다.
5. 기존 테스트·골든 전부 그대로(numba·폴백 두 잡), ruff 통과. 릴리스(태그·PyPI)는 하지 않는다 — 사용자 지시 때.

## 5. 범위 밖 (다음 단계)

- **5 ML 수익 예측** — 위험모델 잔차 수익을 라벨로 쓰는 walk-forward(이 층이 먼저 있어야 라벨이 깨끗하다).
- **10 집행 VWAP/TWAP** — 키움 시장가(`trde_tp`) 동작이 실서버로 확인되지 않았다(전역 원칙: 브로커 가설은 스모크 먼저).
- **연구 적용(swing-it)** — PEAD 를 α 로 넣어 기존 동일가중 순위 틸트 북 대 최적화 북을 2017~2023 walk-forward 로
  비교(비용 1·2배). swing-it `GUARDRAILS.md` 순서대로 사전등록 후 실행, 깨끗한 확인은 전진 모의매매뿐.

## 6. 결과 — 완료 조건 4 (실데이터 바이어스, 2026-11-01)

재현: `uv run --extra db python scripts/validate_risk_model.py`. 실제 asof 는 2019-07-31~2023-10-31 월말 52회다
(21거래일 앞이 2023-12-28 안에 들어오는 마지막 월말이 10월). 바이어스 = z(실현/예측 표준편차)의 표준편차이고,
올바른 모델에서도 52표본의 표준오차는 약 0.10 이다.

| 대상 | v1(유동성 = log 20일 ADV/시총) | v2(유동성 = CNE5 회전율 1·3·12개월) |
|---|---|---|
| 무작위 롱온리 50종목 ×100 (평균 [q05, q95]) | 0.973 [0.912, 1.046] | 0.972 [0.910, 1.046] |
| 시총가중 유니버스 | 0.993 | 0.991 |
| 규모 | 0.981 | 0.996 |
| 베타 | 0.988 | 0.994 |
| 모멘텀 | 0.927 | 0.907 |
| 잔차변동성 | 0.870 | 0.821 |
| 유동성 | **0.749 — 실패** | 0.835 |
| 가치(EP) | 0.902 | 0.908 |

**v1 은 유동성에서 바를 못 넘었다.** 원인 진단(같은 창, 결과를 본 뒤):
1. 자기상관 가설은 기각 — 유동성 팩터 수익의 21일 분산비 0.94(√ 0.97)로는 0.75 가 안 나온다.
2. 팩터·특이 부분을 따로 재면 각각 1.00·1.03 으로 맞는데 합이 0.75 다 → 보유 중 노출이 움직인다.
3. 고정 순수 팩터 포트의 21일 평균 자기 노출: 규모 0.997 · 베타 0.957 · 모멘텀 0.916 · 가치 0.935 ·
   유동성 0.895 · 잔차변동성 0.876. 20일 ADV 는 거래량 급증이 빠르게 빠지며 노출이 평균회귀한다.

**고친 것 하나**: 유동성 정의를 문헌 표준(Barra CNE5 LIQUIDTY, 회전율 STOM·STOQ·STOA 0.35/0.35/0.30)으로 바꿨다.
결과를 보기 전에 정의를 문헌에서 그대로 가져왔고 가중치를 탐색하지 않았다. 그래도 **같은 데이터로 본 뒤의 수정
1회**라는 사실은 남긴다. v2 는 8개 대상이 모두 [0.8, 1.2] 안이다.

읽는 법·남은 한계:
- 스타일 포트가 전부 1 아래(0.82~1.00)다. 노출 이동만큼 위험을 조금씩 과대예측한다는 뜻이고, 최적화에 쓰면
  보수적인 쪽(위험을 덜 잡는 쪽이 아니다)으로 틀린다. 잔차변동성 0.821 은 바 경계에 가깝다.
- 2024~ 는 봉인 소모 구간이라 독립 표본이 아직 없다. 다음 확인은 2026-09-13 이전의 전진 구간이 아니라
  운영 중 매달 같은 통계를 쌓는 것(§3.4 모니터링)이다.
