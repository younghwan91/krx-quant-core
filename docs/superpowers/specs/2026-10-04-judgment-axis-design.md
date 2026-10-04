# 판정 축 — 우위 판정 공용 함수 · 야간 실행 신뢰성 · 핀 드리프트 점검 (v0.6.0)

- 날짜: 2026-10-04
- 범위: krx-quant-core 에 "우위가 있는가"를 세 소비 레포가 같은 식으로 재는 함수와, simnode 야간 실행이
  조용히 실패하지 않게 하는 운영 도구를 더한다. 실행 계층(OMS 교체)·주식선물은 이번 범위 밖.
- 사용자 지시: "트레이딩 레포들의 관계를 분석하고 코어가 핵심이 되는 축을 정리·계획 → 자율 실행"(2026-10-04).

## 1. 왜 — 2026-10-04 감사

| 사실 | 근거 |
|---|---|
| 실주문이 도는 레포가 없다. scalp-it 은 09-16 부터 dry-run, daytrade-it 은 `--paper`+`live_STOP` | `scripts/pair_detect_daily.sh`, `scripts/news_signal_daemon.sh` |
| scalp-it 101번: 현물 일중은 **비용 41.5bp > 어떤 우위** 로 측정 종결. 95(오라클 천장)·98(체크포인트 선택편향)·101(일중/야간 드리프트, 매치드 널) 측정 코드가 전부 scalp-it 스크립트에만 있다 | `scripts/ceiling_95/ceiling.py`, `scripts/selection_98/selection_bias.py`, `docs/research/manju/101` |
| daytrade-it `kqc nightly` job 이 09-23 부터 매일 rc=127 — cron PATH 에 `uv` 가 없다. 아무도 몰랐다 | `~/.kqc/nightly/*/daytrade-it.json` |
| 핀이 셋으로 갈라졌다: scalp-it 0.5.1 · daytrade-it 0.5.0(워크트리 0.4.1) · swing-it 0.4.1 | 각 pyproject |

코어의 역할은 "같은 시장 규칙·같은 비용"에서 한 단계 올라가 **"같은 판정 기준"** 이 되는 것이다. 허위 우위로 실주문을 켜는
사고를 세 레포가 같은 함수로 막는다.

## 2. 호스트 관계(설계 전제)

trader(08:55~15:40 수집·주문 데몬) 와 simnode(16:00~ 전진 판정·`kqc nightly`·DB primary) 는 같은 경로의 같은 레포를
07:50 pull-all 로 맞춘다. 이 스펙의 모든 측정 함수는 순수 함수(numpy/pandas)라 어디서든 테스트되고, **실행 진입점만**
simnode 가드를 받는다(기존 `runtime.host` 원칙).

## 3. 엔진 C — 판정 함수 (`stats`, `backtest`)

### 3.1 `stats/selection.py` — 체크포인트·후보 선택편향 (scalp-it 98번 이식)
- `expected_max_of_m(values, m) -> float`: n 개 후보에서 비복원 m 개를 뽑았을 때 최댓값의 기댓값, 순서통계량 정확식.
  m=1 → 평균, m≥n → 최댓값.
- `argmax_first(rows, key)`: strict `>`·먼저 만난 쪽 승 — `pick()` 과 같은 동점 처리.
- `selection_bias_report(rows, *, key="val_daymean_bp", pooled_key="val_mean_bp", n_key="val_n", min_n=30) -> SelectionBiasReport`
  : 후보 필터(`n ≥ min_n`, key 가 NaN 아님) → 풀 통계(평균·중앙·sd·최대, 풀링 평균, gap 중앙·p90) → 두 규칙의 선택
  → `premium_vs_random`, `premium_vs_median`, `expected_max_curve` (m ∈ {1,2,3,5,10,20,n}), 선택된 gap 과 분위,
  거래 수 vs 일평균 Spearman 3종(`stats.metrics.spearman`, scipy 없음), 거래 수 사분위별 요약, `rule_delta`.
  판정은 하지 않는다 — 숫자만 돌려준다. 후보가 2 개 미만이면 `note` 만 채운다.
- 골든: 원본 `expected_max` 를 테스트 파일에 복사해 무작위 배열·m 전 범위에서 `==`. `argmax_first` 동점 사례.

### 3.2 `backtest/lob/ceiling.py` — 오라클 천장 (scalp-it 95번 이식)
- `oracle_long(bid, ask, horizon, *, cost) -> (taker, maker)`: t+1 진입, [t+2, t+1+H] 사후 최적 매도 − cost.
- `oracle_short(bid, ask, horizon, *, cost)`: 대칭.
- `through_fill_second(bid, fill_sec=10) -> int64[n]`: 지정가 bid[t+1] 이 뚫려 체결된 초(없으면 −1). `_jit.njit` 커널 + 파이썬 폴백.
- `window_max(x, h)`, `maker_pessimistic(bid, fill_idx, maxask, *, cost)`.
- `ceiling_table(bid, ask, *, horizons, cost, lo, hi, fill_sec) -> DataFrame` : 지평별 taker/maker/maker_pessimistic 의
  중앙·평균·n (09:05~15:10 창은 호출부 인자).
- `cost` 는 필수 인자(원본 상수 0.0023 을 박지 않는다 — `costs.round_trip_cost` 를 넘긴다).
- 골든: 원본 `oracle`·`oracle_short`·`fill_second`·`window_max`·`maker_pessimistic` 를 테스트에 복사, 합성 경로 3 시드에서
  `np.array_equal(equal_nan=True)`. numba/폴백 동일.

### 3.3 `backtest/drift.py` — 일중·야간 수익 분해 (101번 §1·§2, daytrade-it 10년 측정)
- `intraday_overnight(frame, *, open="open", close="close", prev_close="prev_close") -> DataFrame[intraday_bp, overnight_bp, close_to_close_bp]`.
  `prev_close` 가 없으면 `code` 별 `shift(1)` 로 만든다(첫 행 NaN).
- `drift_summary(frame, value, *, by=None, date="date") -> DataFrame[mean_bp, median_bp, n, n_months, neg_month_share, t_hac]`
  : 전체 또는 `by`(버킷 열) 별. 월 부호 비율은 "그 달의 평균이 음수인 달 / 전체 달". t 는 `newey_west_t(lag=5)` 의 일별 평균열.
- `rank_buckets(frame, value, *, by_date="date", n=4, labels=None)`: 날짜별 분위 버킷(전일 거래대금 등 **전일** 정보로 호출부가 만든 열에 적용).

### 3.4 `stats/matched_null.py` — 매치드 널 대조 (101번 A 설계, 95h 변동성 매칭 일반화)
- `matched_control(trades, pool, *, strata, n_per=1, seed=0) -> DataFrame`: 각 트레이드와 같은 strata 키 조합의 풀에서
  n_per 개를 무작위로 뽑아 `trade_id`·`control_idx` 를 돌려준다. 짝이 없는 strata 는 `unmatched` 로 보고.
- `matched_alpha(trades, controls, *, value, cluster="date", n_boot=2000, seed=0, ci=0.95) -> MatchedAlpha`
  : alpha = mean(trade value) − mean(control value), 날짜 클러스터 부트스트랩 CI(날짜를 복원 추출하고 그 날짜의 트레이드·대조를
  통째로 넣는다), `n_trades`, `n_controls`, `n_clusters`, 층별 alpha 표.
- 판정하지 않는다. "CI 하한 > 0" 은 사전등록 문서가 쓴다.

## 4. 운영 — `runtime.nightly`, `kqc`

### 4.1 nightly 실행 파일 해석
- `cmd[0]` 을 `shutil.which(cmd[0], path=<augmented PATH>)` 로 찾는다. augmented = 현재 PATH + `~/.local/bin` + `~/.cargo/bin`
  + `/usr/local/bin`. 자식에도 그 PATH 를 넘긴다(uv 가 그 안에서 python 을 찾아야 하므로).
- 못 찾으면 rc=127 + 로그에 **찾아본 경로** 를 적는다.

### 4.2 `kqc nightly status [--days N] [--out DIR]`
- `~/.kqc/nightly/<date>/<repo>.json` 을 최근 N 일(기본 7) 읽어 레포×job 표(날짜별 rc)를 출력한다.
- 가장 최근 날짜에 실패(rc≠0 또는 timeout)가 하나라도 있으면 exit 1, 연속 실패 일수를 같이 적는다.
  어느 호스트에서든 읽기만 하므로 호스트 가드 없음.
- `kqc nightly <repo>` 는 끝에 `kqc nightly: <repo> FAILED <job>(rc=…)` 를 stderr 에 한 줄 더 쓴다 — cron 로그에서 grep 되게.

### 4.3 `kqc pins [--root ~/git] [repo ...]`
- 각 레포의 `pyproject.toml` 에서 `krx-quant-core` 요구사항과 `uv.lock` 의 잠긴 버전을 읽어 표로 낸다. 기본 대상은
  `scalp-it daytrade-it swing-it` 중 존재하는 것. 기준 버전은 이 패키지의 `__version__`.
- 잠긴 버전이 기준보다 낮은 레포가 있으면 exit 1. 수정은 하지 않는다.

## 5. 오류 처리
- 측정 함수는 빈 입력·NaN 전부에서 예외 대신 NaN/빈 표를 돌려준다(원본과 같다). strata 열이 없으면 `KeyError` 에 열 이름.
- `kqc nightly status` 는 깨진 JSON 한 파일을 건너뛰고 경고한다.

## 6. 테스트
- 모듈별 단위 테스트 + 3.1·3.2 골든(원본 복사본 대조). numba 없는 경로는 `KRX_QUANT_CORE_DISABLE_NUMBA=1` 서브프로세스
  (기존 parity 테스트 패턴).
- 테스트는 simnode: `kqc run krx-quant-core --ref feat/judgment-axis -- uv run --extra dev --extra fast pytest -q`. CI 녹색.

## 7. 릴리스·소비자
- `0.6.0` 태그 → PyPI. README 에 §3·§4 추가.
- daytrade-it·scalp-it·swing-it: 핀을 `0.6.0` 으로 올리는 PR. daytrade-it 의 nightly rc=127 은 4.1 로 해소된다
  (nightly.toml 은 안 고쳐도 된다). 실주문 경로 변경 없음.
- 다음 단계(이번 범위 밖, 순서대로): scalp-it 100번이 3.3·3.4 를 쓰도록 안내 → OMS 교체(엔진 A 마무리) →
  101번 B-2 결과가 양수일 때만 kiwoom-client 선물 모듈·파생 비용 스케줄.
