# v0.7.0 — 공용 데이터 레이어 + 가속 (2026-10-08)

사용자 지시(2026-10-08): "코어의 디펜던시가 트레이더 레포들을 잘 보필하는지 점검하고 최대한 많은 공통
비중을 챙겨오고, 캐싱과 가속화·메모리, numba". 세 소비 레포(scalp-it·daytrade-it·swing-it) 감사 결과
공통 1순위가 같았다 — **코어에 데이터 읽기 층이 없다**.

## 감사 요약(근거)

- scalp-it: 하루치 틱·호가를 `pd.read_sql` 로 통째 읽고 `dict(tuple(groupby))` 로 복사하는 로더가 9벌,
  날짜별 피클 캐시 로더 17벌(`~/of80` 17GB, float64·object 열 혼재). `select distinct ts::date from ticks`
  로 9.8GB 하이퍼테이블을 훑어 날짜 목록을 낸다(5곳).
- daytrade-it: 분봉을 매 실행 Postgres 에서 다시 읽는다(코드×일 단위 쿼리 포함), 전 거래일을 005930
  일봉으로 찾는다, `Decimal(str(x))` 로 분봉 OHLC 를 만든다.
- swing-it: `SELECT *` 로 `daily_bars_adjusted` 전체를 매번 읽고 같은 표를 5~6번 피벗, `code×date`
  vs `date×code` 방향 혼동으로 실제 버그 기록(`inst_flow_accel_gate.py:61`).
- 코어 자체 느린 곳: `lob.ceiling` 미래 창 O(n·h)(→ 커밋 2a34d66 에서 O(n)), `panels`
  (`pivot_table`·`transform(lambda)`), `crosssectional._trailing_adv`(날짜 루프), `paired_bootstrap`
  (n_boot 파이썬 루프), `matched_control`(트레이드마다 후보 배열 복사), `tick_size_int`(호출마다 Decimal).

이전 결정 "읽기 클라이언트는 레포별 독립 사본"(swing-it `storage.py`, 2026-09-12)은 **쓰기 통합**만 다룬
것이고 캐시·자료형 문제는 다루지 않았다. 이번 데이터 층은 **옵트인**이다 — 기존 어댑터를 깨지 않고,
소비 레포가 로더를 하나씩 갈아끼운다.

## `krx_quant_core.data`

| 모듈 | 내용 |
|---|---|
| `db` | `resolve_dsn()`(인자 → `KR_QUANT_DB` → `KQC_ENV_FILE`/`.env` 탐색), `connect()`(psycopg3 우선, 없으면 psycopg2), `fetch_frame()` — Postgres 는 `COPY … TO STDOUT (FORMAT csv)` + pandas C 파서(`read_sql` 대비 1.7×, 실측 7.6s→4.5s/일), 그 밖의 DB-API(sqlite 테스트)는 커서. |
| `store` | `ColumnStore` — `{root}/{dataset}/v{N}/{key}/{col}.npy` + `meta.json`. 열 단위로 쓰고 `mmap_mode="r"` 로 읽는다(16 워커 스윕이 같은 페이지 캐시를 공유, 필요한 열만 디스크에서 올라옴). 원자적 쓰기(임시 파일 → `os.replace`), 열 추가는 `flock`. 의존성 없음(pyarrow 불필요 — 소비 레포 둘에 없다). |
| `intraday` | `CodeDay`(종목별 CSR: `codes`·`ptr`·열 배열) + `load_ticks(day)`·`load_quotes(day, levels)`·`load_minute_bars(day)`. 시각은 `sec`(자정 기준 int32 초), 가격 int32, 수량 int64, 방향 int8, 강도 float64(DB 비트 동일 — float32 는 피처 골든을 깬다). `CodeDay.frame(code)` 가 `lob.build_second_grid` 가 받는 DataFrame(`ts` 복원)을 낸다. **닫힌 날(KST 오늘 이전)만** 캐시 — 오늘은 매번 DB, 행 0 인 날은 캐시 안 함, 백필 뒤엔 `refresh=True`. 정수열은 int64 로 파싱 후 범위 검사(C 파서에 int32 를 주면 조용히 wrap). |
| `daily` | `load_daily_bars(start, end, adjusted=)` 긴 프레임(`code` 범주형) + `daily_panel()` → `DailyPanel(codes, dates, 값 배열)` 방향 명시(`frame(field, orient=)`). 캐시는 **연도별**, 지문 `(count, max(date), Σround(close×100), Σvolume)` 정수 합(float 합은 PG 병렬 집계에서 흔들린다). 지문이 바뀐 해만 새 세대로 통째 교체. |
| (daily 안) | `trading_calendar()` — `daily_bars` 의 distinct date(0.76s)를 하루 한 번 디스크 캐시 → `TradingCalendar`. 직전 평일이 없으면(수집 실패) 캐시 안 하고 10분 뒤 재질의. |

## 가속·메모리(기존 API, 값 동일) — 실제 반영분

- `backtest.lob.ceiling`·`features.forward_labels`: 미래 창 극값 O(n·h) → 단조 덱 O(n)(`lob._window`).
- `market.ticks.tick_size_int` float 밴드 이분탐색, `tick_size_array(prices, etf=)`, `ETF_TICK_BANDS`,
  `lob.etf_tick_table()`.
- `backtest.ticktime.trailing_sums`·`forward_max_last` — scalp-it `tick_sanity` 루프의 numba 이식(비트 동일).
- `run_replay(cost_model=, liquidate_at_end=)`.

측정 후 **하지 않은 것**: `backtest.panels`(재작성 이득 ≤25%, 3,000×2,500 기준 각 ~1s — swing-it 의 해법은
반복 피벗을 `data.daily_panel` 한 번으로 바꾸는 것), `crosssectional._trailing_adv`(0.8s, nanmean 합 순서가
배열 메모리 배치에 따라 달라 비트 동일 재현 비용이 큼), `paired_bootstrap`(0.4s)·`matched_control`(0.12s)
— 이미 1초 미만.

## 추가 통계(소비 레포 수작업 대체)

- `stats.matched_null.cluster_bootstrap_diff_ci(a, ca, b, cb)` — 두 표본 평균차, 날짜 공동 재추출
  (daytrade-it `diff_ci` 와 같은 숫자; scalp-it `tick_sanity.bootstrap_diff_ci` 는 관측 i.i.d. 라 숫자가 다르다).

## 하지 않는 것

- 소비 레포 코드 변경(각 레포 세션 몫 — 채택 안내만).
- 수집기 쓰기 경로(COPY writer)·`TimeWindow` 실시간 상태 — 실매매 경로라 다음 단계에서 패리티 근거와 함께.
