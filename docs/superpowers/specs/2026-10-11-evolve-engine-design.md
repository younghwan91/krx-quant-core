# v0.8.0 — 진화 엔진 `research.evolve` (2026-10-11)

사용자 지시(2026-10-11): "스윙잇에서 계속 GA 방식으로 매매를 진화하고 있다. 체계적 포맷으로 가져올 게
있으면 만들고 초가속화에 신경 써", "GA 이든 뭐든 보고 판단해 SOTA 로 도와", "검증 에이전트는 붙이고".

## 근거 — swing-it 패턴 GA 실측 (simnode, 2026-10-11 17:40, 작업자 1개, 3세대 12 작업자 실행과 동시)

`research/experiments/pattern_ga_run.py`(feature-combined-strategy 워크트리) 한 런(폴드 1, 시드 0, 예산 5,000):

| 층 | ms/평가 | 비율 |
|---|---|---|
| CMAwM `ask`/`tell`(cmaes 0.13.1, scipy 없음) | 0.41 | 20% |
| `decode` + `gkey`(벡터 → dict → 튜플) | 0.26 | 13% |
| `evaluate` 파이썬 접착(`candidates`·`run`·`np.percentile`·`penalty`·`monthly`) | ~0.7 | 35% |
| memo·이력 등 | ~0.6 | |
| **합** | **2.0** | numba 커널 자체는 작은 몫 |

- CMAwM 비용의 정체: 표본마다 `B·diag(D)` 63×63 행렬곱을 다시 하고(재표본 평균 7회), scipy 가 없어
  `chi2_ppf` 가 표본·세대마다 100회 이분법 × `np.vectorize` 로 돈다.
- swing-it `ga_fast._jit` 은 `cache=False` — 후보 실행마다 작업자 12개가 각자 약 9초 JIT 한다.
- 후보 하나 = 폴드 4 × 시드 10 × 10,000 평가 + 재시작 풀 선별 40만 건 ≈ 150초(12 작업자).
- **병목은 속도만이 아니라 잡음이다.** 2세대에서 같은 후보(G1C)의 시드 집합 간 표준편차 ±1.03%p 가
  후보 간 차이와 같은 크기 → 시드 집합 3~5개로는 순위를 못 가린다. 가속은 시드 집합 수(검정력)로 바꿔 쓴다.

## 최적화기 판단

- **골격은 CMA-ES(BIPOP) 유지.** 63차원 혼합(연속·순서형·범주형), 잡음 있는 적합도 — 이 계열이 맞다.
- **범주형은 CatCMAwM(Hamano et al. 2025, arXiv 2504.07884)** 을 두 번째 백엔드로. 지금은 범주형 하나를
  수준 수만큼의 연속 로짓으로 펴서(연속 39 중 상당수) 평평한 방향을 늘린다. CatCMAwM 은 연속·정수·범주형을
  각자의 분포로 다룬다. cmaes 0.13.1(swing-it 고정 버전)에 이미 있다. 기존 백엔드와 같은 예산으로 비교한다
  (어느 쪽이 나은지는 **소비 레포의 사전등록 비교**로 정한다 — 코어는 둘 다 같은 인터페이스로 낸다).
- **세대 수준은 레이싱.** 후보마다 시드 집합을 고정 개수 돌리지 않고, 리더와 구분 안 되는 후보에만 시드
  집합을 더한다. 잡음 규칙은 swing-it 의 것(평균 차 > 두 표준편차 중 큰 값 × 2, ddof=1)을 그대로 쓴다.

## 모듈 `krx_quant_core.research.evolve`

전략은 여기 없다. 코어는 **탐색 공간·최적화기·세대 프로토콜·작업자 풀·일반 커널**을 내고, 표현형
(canonical)·평가(적합도)는 소비 레포가 콜백으로 준다.

### `space` — 탐색 공간

```python
space = Space([
    Real("rs_min", 0.0, 90.0, off="lo"),          # [0,1] 정규화, 끔 끝에서 off_zone 안이면 NaN(끔)
    Choice("entry", ("bottom", "horiz", "trend")), # 범주형
    Ordinal("theta", (0.06, 0.08, 0.10)),         # 순서형(정수 축)
], off_zone=0.02, digits=4)
space = space.override({"nom": {...}, "disc": {...}, "cont": {...}})   # swing-it PATTERN_GA_MENU 형식
```

- 선언 순서와 무관하게 **기존 배치(legacy layout)** = `[Real…][Choice 로짓…][Ordinal 단계…]`. swing-it 의
  `CONT·NOM·DISC` 벡터와 같은 배치다(같은 x → 같은 유전자).
- `decode(X) -> G`: `(n, dim)` → `(n, n_genes)` float64 유전자 행렬. Real = 실제 값(`round(·, digits)`,
  끔 = NaN), Choice = 로짓 argmax 수준 번호(동률 = 앞), Ordinal = `clip(round(x))` 수준 번호. **같은 값의
  중복 수준은 첫 번호로 접는다**(한 수준 덮어쓰기 복제 `(v, v)` 가 같은 키가 되게 — swing-it 검증 MUST-FIX 3).
- `keys(G) -> int64[n]`: 행 바이트 해시(NaN 정규화, −0 → 0). 같은 표현형 = 같은 키.
- `to_dict(row)` / `from_dict(g)`: 보고·기존 dict 코드와 왕복(`None` = 끔, 수준 값 원형).
- `random(rng, n)`: swing-it `random_x` 와 같은 난수 순서(균등 `rng.random((n, dim))` 뒤 순서형 열을
  `rng.integers` 로 덮음).
- `override(menu)`: `{"nom","disc","cont"}` 덮어쓰기, `"inf"` → ∞, `null` → 끔, 모르는 이름이면 `KeyError`,
  한 수준 순서형은 `(v, v)` 로 복제.

### `optim` — BIPOP 구동기 + 백엔드

```python
res = bipop(fitness, space, budget=10_000, seed=0, x0=x0, x_restart=pool, backend="cmawm",
            canonical=canon, sigma0=0.3, stall=True)
res.best_key, res.best_value, res.hist_keys, res.hist_values, res.restarts, res.uniq
```

- `fitness(G_unique) -> (values, counts)` — **배치 콜백**. 구동기가 세대마다 λ개를 한 번에 묻고, decode →
  `canonical(G)`(소비자, 벡터화) → 키 → memo 에 없는 행만 넘긴다. 값은 최대화.
- 재시작 규칙은 swing-it `cma_bipop_x0` 와 **같은 난수 순서**(Hansen 2009 BIPOP, `stall` 정체 재시작,
  첫 런 평균 = `x0`, 재시작 평균 = `x_restart` 풀에서 추출, 예산 도달 시 부분 세대는 tell 안 함).
  이력·재시작 기록(`regime`·`popsize`·`sigma`·`evals`·`gens`·`why`·`zero_frac`)도 같은 모양.
- 백엔드:
  - `"cmawm"` — `cmaes.CMAwM` 를 상속한 가속판 `FastCMAwM`. 세대마다 `B·diag(D)` 한 번, 이산 부호화
    배치, scipy 없을 때의 `chi2_ppf`·`norm_cdf` 를 **같은 연산 순서**의 스칼라 함수 + 값 memo 로 대체.
    **cmaes 0.13.x 라이브러리와 비트 단위로 같은 궤적**(시험이 대조). 다른 cmaes 버전이면 라이브러리
    `ask`/`tell` 로 폴백(경고).
  - `"catcmawm"` — `cmaes.CatCMAwM`. Real → `x_space`[0,1], Ordinal → `z_space`(수준 번호), Choice →
    `c_space`. 첫 평균·재시작 평균은 기존 배치 벡터의 Real·Ordinal 부분, 범주형은 균등.
  - `"cmawm_lib"` — 순정 라이브러리(대조용).
- cmaes 는 선택 의존성: extra `evo = ["cmaes==0.13.1"]`.

### `protocol` — 세대 포맷 (swing-it META_EVOLUTION 을 일반화)

- `EvoLedger(label, logs_dir)` — `stats.trials` 원장 위:
  `register(candidate, menu, seeds, rationale, **extra)` 가 **결과 전** 줄을, `result(candidate, seeds,
  metrics)` 가 결과 줄을 적는다. `pending()` = 결과 없는 등록(실패) — DSR N 에 그대로 센다(보수).
  `n_trials()` = 원장의 서로 다른 줄 수.
- `summarize(vals)` → 평균·표준편차(ddof=1)·n. `distinguish(a, b, k=2.0)` → `"a"`·`"b"`·`None`.
- `race(candidates, run, seed_sets, *, min_sets=3, max_sets=…, k=2.0)` — 라운드마다 살아 있는 후보에 시드
  집합 하나씩 더 돌린다. 현재 리더보다 잡음 규칙으로 **확실히 나쁜** 후보는 탈락. 살아 있는 후보가 하나거나
  `max_sets` 에 닿으면 끝. `run(candidate, seed_offset) -> float` 은 소비자가 준다(보통 `EvoPool` 위).
- `Seal(days, seal_date)` — 봉인 경계 행, `clip(arr)`(경계 이후 끊기), `check(exit_rows)`(넘으면
  `SealBreach`). swing-it 검증 MUST-FIX 1 의 일반형.
- `digest(obj)` — 정렬된 JSON 의 sha256(결정성 확인: 같은 후보 두 번 → 같은 digest).
- `provenance(modules=…)` — git HEAD·dirty·모듈 소스 해시·numpy/numba/cmaes 버전.

### `pool` — 작업자 풀

`EvoPool(workers, initializer, initargs)` — spawn, 자식 시작 전 `NUMBA_NUM_THREADS`·`OMP_NUM_THREADS`
등을 1 로, `map(fn, tasks)` 는 **과제 순서대로** 결과를 낸다(끝난 순서 집계 금지 — swing-it MUST-FIX 2).
`progress` 콜백. 후보 여럿 × 시드 집합 여럿을 한 풀에 넣어 JIT·캐시 로딩을 한 번만 한다.

### `kernels` — 일반 가속 커널 (`stats` 수준, numba 선택)

- `block_boot_lower(monthly, block, n_boot, seed, q)` — swing-it `ga_fast._block_boot_py` +
  `np.percentile`(linear) 과 **비트 동일**한 한 번 호출(정렬 + numpy `_lerp` 와 같은 보간식).
- `monthly_sums(x, month_of, t_lo)` — 연속 월 구간 합(`np.unique` + `bincount` 와 같은 덧셈 순서).
- `h32` 해시 난수(lowbias32 변형) 공개.

## 시험

- `space`: 배치·decode·끔·중복 수준 접기·키 안정성·덮어쓰기·dict 왕복, swing-it decode 의미 재현표.
- `optim`: `FastCMAwM` vs `cmaes.CMAwM` 비트 동일 궤적(혼합 시험 함수 200세대, scipy 없는 경로),
  BIPOP 구동기 vs swing-it `cma_bipop_x0` 사본 — 같은 시드 → 같은 이력, 배치 memo 정확성,
  CatCMAwM 수렴 스모크.
- `protocol`: 원장 결과 전/결과/실패 집계, 잡음 규칙(G2 표 재현), 레이싱, 봉인.
- `kernels`: `np.percentile` 비트 동일(크기·분위 무작위 1,000회), numba ↔ 파이썬 동일.
- **실데이터 대조(스크래치, simnode):** swing-it 엔진을 평가 콜백으로 넣고 `pattern_ga_run.ga_run_one` vs
  코어 구동기 — 같은 폴드·시드·예산에서 이력 값 비트 동일, 그리고 ms/평가.

## 범위 밖

- swing-it 평가 접착(`candidates`·`run`)의 전 JIT 화 — 전략 코드라 swing-it 에서 한다. 코어는 배치 평가
  계약과 예시를 README 에 두고, swing-it 세션에 채택 순서(① `_jit` `cache=True` ② 코어 `bipop` +
  `Space` ③ `kernels` ④ 평가 접착 njit 화)를 전달한다. 그 워크트리는 3세대 실행 중이라 직접 고치지 않는다.
- 다목적(COMOCatCMAwM), 대리모델(BO) — 평가가 ~0.1ms 대로 내려가면 대리모델 이득이 없다.

## 검증

각 단계(스펙·구현·실데이터 대조)마다 독립 검증 에이전트가 따라붙는다. 반영 전에는 완료를 보고하지 않는다.
