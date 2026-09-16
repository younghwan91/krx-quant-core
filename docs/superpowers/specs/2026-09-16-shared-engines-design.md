# 공용 엔진 — trader 주문 관리 계층 · 이벤트 리플레이 · simnode 스윕/야간 실행 (v0.5.0)

- 날짜: 2026-09-16
- 범위: krx-quant-core(이 레포)에 공용 엔진 3개를 만들고 daytrade-it·scalp-it 이 쓰게 한다. swing-it 은 이번 범위 밖.
- 사용자 지시: "트레이딧·스캘핏이 코어의 최적화를 적극 활용", "simnode 가 시뮬레이션을 자주", "trader 매매용 더 좋은 클래스",
  "세계적 레포에서 배울 것 적용", "필요한 엔진은 코어에, 특히 공용 엔진", "워크트리 → 테스트 → 머지·푸시".

## 1. 왜 — 감사 결과(2026-09-16)

| 문제 | daytrade-it | scalp-it |
|---|---|---|
| 코어 활용 | 약 25%: 가드·킬스위치·체결·stats·lob 미사용 | 약 40%: lob·킬스위치·CV·OOS·kqc 미사용 |
| 복제 코드 | `execute_trade` 자체 가드, `live_STOP` 자체 킬, `_simulate` 자체 루프, bayes `SELL_COST` 하드코딩 | `COST=0.0023` 류 약 25곳, 자체 tick 함수 6곳, 자체 체결 시뮬 4개(rl_87·rl_94·tiger_86·of_sim) |
| 실매매 상태 | 09-15 데몬 이중 기동으로 이중 매수(지금은 flock 으로만 막음). 청산 경로는 가드를 우회 | 포지션을 `RiskGuard`·`OrderExecutor` 둘이 따로 추적. 메모리에만 있어 크래시하면 사라짐. 재시작 대사 없음 |
| 페이퍼 | `PaperBroker` 는 있지만 데몬이 안 씀. dry-run 은 가드 전에 return | dry-run 은 POST 만 생략 |
| 시뮬 빈도 | 스케줄 없음(bayes 브랜치 수동) | forward_gate 크론만. 스윕은 직렬 `itertools.product` |

## 2. 배울 것(외부 레포)과 적용 위치

| 출처 | 가져오는 원리 | 적용 |
|---|---|---|
| NautilusTrader | 전략 코드가 백테스트·실매매에서 **같다**. 차이는 어댑터(브로커·데이터)뿐. 시작 시 실계좌 대사 | `EngineCore`+`Strategy` 프로토콜, `OrderManager.reconcile()` |
| QuantConnect Lean | 브로커 모델·체결 모델·수수료 모델 분리 | `Broker` 프로토콜 / `PaperBroker(fill_basis)` / 코어 `costs` |
| hftbacktest | L2 대기열 위치 모델(v0.3 도입 완료) | `PaperBroker` 는 초 단위 스냅샷엔 `backtest.fills`, 격자 연구엔 기존 `lob.queue` |
| vectorbt / Optuna | 대량 파라미터 스윕·조기 가지치기, 병렬 | `research.run_sweep`(프로세스 풀·캐시), `research.optuna_search`(extra) |
| MLflow | 실행마다 코드·데이터·파라미터·지표 기록 | 기존 `start_run`+`TRIALS.jsonl` 재사용 — 스윕의 모든 config 가 DSR 의 N 에 들어간다 |

## 3. 엔진 A — 주문 관리 계층 (`krx_quant_core.execution`, trader)

기존 `OrderIntent`·`OrderResult`·`OrderGuard`·`KillSwitch` 는 **바꾸지 않는다**. 그 위에 새 모듈을 더한다.

### 3.1 `execution/events.py`
- `OrderStatus(StrEnum)`: `blocked`, `submitted`, `partial`, `filled`, `canceled`, `rejected`, `unknown`(최종 리뷰 추가 — 나갔는지 모름, 재시도 금지).
- `Fill(frozen)`: `ord_no: str`, `code: str`, `side: "buy"|"sell"`, `qty: int`, `price: int`, `ts: datetime`.
- `Holding(frozen)`: `code`, `qty: int`, `avg_price: float`.
- `OpenOrder(frozen)`: `ord_no`, `code`, `side`, `qty`, `remaining: int`, `price: int`.

### 3.2 `execution/broker.py` — `Broker` 프로토콜
```python
class Broker(Protocol):
    def submit(self, intent: OrderIntent) -> OrderResult: ...
    def cancel(self, ord_no: str, intent: OrderIntent) -> OrderResult: ...
    def open_orders(self) -> list[OpenOrder]: ...
    def holdings(self) -> dict[str, Holding]: ...
    def poll_fills(self) -> list[Fill]: ...   # 지난 호출 이후 새 체결만
```

### 3.3 `execution/paper.py` — `PaperBroker`
- 같은 프로토콜을 따르는 모의 브로커. `on_quote(code, ts, bid, ask, last=None)` 로 시장을 먹인다.
- 도착 즉시 반대 호가가 지정가 이내면 반대 호가에 전량 체결(테이커). 아니면 대기 → 이후 `on_quote` 마다
  `backtest.fills.limit_buy_filled/limit_sell_filled(fill_basis)` 로 판정해 지정가에 체결. 기본 `fill_basis="through"`(보수).
- `latency_sec`(기본 0): 주문 뒤 이 초가 지난 첫 시세부터 판정한다.
- 보유 부족 매도는 `blocked`(사유 `paper: 보유 부족`). 비용은 브로커가 빼지 않는다(장부 몫).
- `ord_no` 는 `P000001` 부터 순번.

### 3.4 `execution/kiwoom_broker.py` — `KiwoomBroker`
- `KiwoomBroker(api, *, dry_run=True)`. `api` 는 `kiwoom_client.KiwoomAPI` 호환 덕타입(`api.order.buy_order(**body)` 등).
  테스트는 가짜 api.
- 제출은 **재시도하지 않는다**(이중 주문 방지). 조회는 예외 시 1회 재시도.
- `dry_run=True` 면 POST 안 하고 `OrderResult(dry_run=True, submitted=False)`.
- `holdings()`=계좌평가잔고(kt00018 `evaluation_balance_detail`), `open_orders()`=미체결(ka10075 `unfilled_orders`),
  `poll_fills()`=체결(ka10076 `filled_orders`)을 주문번호·누적 체결 수량 기준으로 증분만 돌려준다. 부호·앞자리 0 은 `strip_sign`.
- 토큰 폐기(logout)는 절대 하지 않는다(`TOKEN_SHARING_WARNING`). 토큰 갱신은 kiwoom-client 가 한다.
- 응답 필드명은 kiwoom-client 가 정본. 확인 못 한 필드는 docstring 에 "미검증"으로 적는다.

### 3.5 `execution/book.py` — `PositionBook`
- `apply(fill)`: 매수는 평단 갱신, 매도는 실현손익(원, **비용 차감**: `costs.KoreanCostModel` 기본 설정, 체결일 기준 세율).
- `position(code) -> Holding | None`, `positions() -> dict`, `realized_krw: float`, `n_round_trips`, `trades` (청산 단위 원장 → `backtest.ledger` 호환 DataFrame `to_frame()`).
- `journal: Path | None`: 모든 fill 을 JSONL 로 append(`fsync`). `PositionBook.load(journal)` 로 재생해 복원한다. 날짜 파일은 호출부가 정한다.

### 3.6 `execution/oms.py` — `OrderManager`, `InstanceLock`, `ReconcileReport`
- `InstanceLock(path)`: `fcntl.flock(LOCK_EX|LOCK_NB)`. 이미 잡혀 있으면 `AlreadyRunning`. 같은 계좌를 도는 데몬 이중 기동 방지.
- `OrderManager(broker, *, guard: OrderGuard, book: PositionBook, kill: KillSwitch | None = None, clock=now_kst)`.
  - `buy(code, qty, price, *, ref_price)`: 순서 = 킬스위치(걸리면 `blocked`, 사유 `kill:<reason>`) → `guard.reason(...)`(기존 판정 순서·사유 문자열 그대로) → 제출 → 통과 시 `guard.record_order`.
  - `sell(code, qty, price, *, ref_price=None)`: **청산 경로.** 킬스위치·블록리스트·화이트리스트·횟수 상한으로 막지 않는다
    (청산을 막으면 실포지션이 무감시로 남는다 — 가드 N-1 과 같은 이유). 막는 것은 둘뿐: 수량 ≤ 장부 보유(장부 없으면 브로커 보유), 가격 > 0.
  - `cancel(ord_no, intent)`.
  - `sync() -> list[Fill]`: `broker.poll_fills()` → `book.apply` → 매도 체결이면 `kill.record_trade(realized, on=date)`.
  - `reconcile() -> ReconcileReport`: 장부 vs `broker.holdings()`. `only_book`, `only_broker`, `qty_mismatch`, `ok`. **고치지 않고 보고만 한다.**
  - 모든 결과는 `orders.jsonl`(선택 `order_log: Path`)에 `OrderResult.to_record()` + `status`.

### 3.7 `execution/engine.py` — 같은 전략 코드, 두 어댑터
- 이벤트: `Quote(ts, code, bid, ask, bid_qty=0, ask_qty=0)`, `Trade(ts, code, price, qty)`, `Bar(ts, code, open, high, low, close, volume)`.
- `Strategy` 프로토콜: `on_start(ctx)`, `on_event(ev, ctx)`, `on_end(ctx)` (on_start/on_end 는 선택 — 없으면 건너뜀).
- `StrategyContext`: `oms: OrderManager`, `now: datetime`, `last: dict[str, Quote|Trade|Bar]`.
- `EngineCore(strategy, oms)`: `feed(ev)` = (브로커가 `PaperBroker` 면 `on_quote` 로 시세 전달) → `oms.sync()` → `strategy.on_event`.
  실매매는 웹소켓 이벤트를 `KiwoomBroker` 위 `EngineCore.feed` 로, 리플레이는 아래 3.8 이 `PaperBroker` 위로 먹인다.

### 3.8 `backtest/replay.py` — `ReplayEngine`
- `merge_events(quotes=None, trades=None, bars=None) -> list[Event]`: DataFrame → 시각 정렬(같은 시각은 Quote→Trade→Bar, 안정 정렬).
- `run_replay(strategy, events, *, guard_config=OrderGuardConfig(max_qty=10, price_band_pct=0, max_orders_per_code=0, max_orders_total=0), kill_config=None, fill_basis="through", latency_sec=0) -> ReplayResult`.
- `ReplayResult`: `fills`(DataFrame), `orders`(list[dict]), `book`(PositionBook), `metrics`(`backtest.ledger` 지표 — 청산 원장 기반).
- 한 번의 `run_replay` 는 한 거래일 한 계좌. 여러 날은 호출부가 나눠 `run_sweep` 로 병렬.

## 4. 엔진 B — simnode 스윕·야간 실행 (`krx_quant_core.research`, `runtime.nightly`)

### 4.1 `research/sweep.py`
- `grid(**axes) -> list[dict]` (데카르트 곱, 키 순서 고정).
- `run_sweep(objective, configs, *, label, repo_root, data: DataSpec, n_jobs=None, cache=True, seed=0, allow_dirty=False) -> SweepResult`
  - 스윕 전체를 `start_run(label, {"sweep": n, "configs_fp": ...})` 한 번으로 감싼다(simnode·clean·OOS 게이트 그대로).
  - **모든 config** 를 `record_trial(label, cfg, logs_dir=<label dir>)` 로 원장에 적는다 → DSR 의 N 이 스윕 크기를 반영.
  - `objective(cfg) -> dict[str, float | list[float]]` 은 모듈 최상위 함수(피클 가능). `ProcessPoolExecutor(n_jobs)`;
    `n_jobs` 기본 `min(os.cpu_count() - 2, KQC_MAX_JOBS or 14)`. `n_jobs=1` 이면 프로세스 안 띄움(테스트·디버그).
  - 캐시: `~/.kqc/cache/<label>/<sha256(git_sha, data, config_fp)>.json`. 코드나 데이터가 바뀌면 키가 바뀐다.
  - 한 config 가 예외면 그 행에 `error` 를 남기고 나머지는 계속.
- `SweepResult`: `frame`(config 열 + 지표 열), `n_trials`, `best(metric, returns_key=None)` — `returns_key` 가 있으면
  그 수익 배열로 `deflated_sharpe(…, n_trials=n_trials)` 를 같이 돌려준다(판정은 안 함).

### 4.2 `research/optuna.py` (extra `opt = ["optuna>=4"]`)
- `optuna_search(objective, space, *, label, repo_root, data, n_trials, n_jobs=1, direction="maximize", seed=0) -> SweepResult`.
  `space(trial) -> dict` 로 config 를 뽑고, 모든 trial 을 원장에 기록. TPE 시드 고정. optuna 가 없으면 ImportError 에 설치 안내.

### 4.3 `runtime/nightly.py` + `kqc nightly`
- 레포의 `research/nightly.toml`:
  ```toml
  [[job]]
  name = "pair-sweep"
  cmd = ["uv", "run", "python", "scripts/pair_sweep.py", "--days", "20"]
  timeout_min = 60
  weekdays_only = true
  ```
- `kqc nightly <repo_root> [--only NAME] [--dry-run]`: simnode 전용. job 을 순서대로 `nice -n 10` 으로 실행, 타임아웃 시 kill.
  로그 `~/.kqc/nightly/<YYYY-MM-DD>/<repo>-<name>.log`, 요약 `~/.kqc/nightly/<YYYY-MM-DD>/<repo>.json`
  (`name, rc, secs, timed_out`). job 하나가 실패해도 다음 job 은 돈다. 종료코드 = 실패 수(최대 1).
- 소비 레포는 `deploy/crontab.simnode` 에 `kqc nightly` 한 줄을 둔다(장 마감 후, 기존 forward_gate 뒤).

## 5. 소비 레포 — 무엇을 바꾸나

원칙: **내일(2026-09-17) 07:50 pull-all 로 실매매에 반영된다.** 실주문 경로의 동작은 기본값에서 바뀌지 않아야 한다.
실주문 경로에서 코어로 바꾸는 건 **수치·사유 문자열이 같음을 테스트로 보인 것만**. 새 능력은 플래그나 섀도로 켠다.

### 5.1 daytrade-it
1. 코어 0.5.0 으로 올린다.
2. `live_STOP` 킬 확인 두 곳 → 코어 `KillSwitch(stop_file=..., sticky_stop_file=False)` 의 `stop_file_present`(비고정 = 현행 동작).
3. 뉴스 데몬 시작 시 코어 `InstanceLock(data/auto_trader/daemon.lock)` 추가(기존 flock 과 병행 — 이중 방어).
4. 장전 점검 `preopen_check` 에 **보고 전용** 대사(`OrderManager.reconcile`: 장부 jsonl 재생 vs `KiwoomBroker.holdings`) — 실패해도 exit code 에 반영하지 않고 경고만.
5. `PaperBroker` 기반 `--paper` 모드: `news_signal_poller.py --paper` 가 코어 `OrderManager(PaperBroker)` 로 같은 AutoTrader 판단을 흘린다(실주문 없음).
6. MCP `backtest._simulate` 의 자체 체결·비용 → 코어 `run_replay`(Bar 이벤트) + `costs`.
7. `research/nightly.toml` + `deploy/crontab.simnode`(신규) — 리플레이/스윕 job.
8. bayes 브랜치는 main 이 아니므로 건드리지 않는다.

### 5.2 scalp-it
1. 코어 0.5.0 으로 올린다.
2. 자체 tick 함수 → 코어 `market.ticks`(골든 대조 테스트로 동일 확인).
3. 연구 스크립트의 `COST = 0.0023` 류 → `costs.round_trip_cost(trade_date, market)`. 2026 날짜에서 값이 같음을 테스트. 날짜가 없는 곳은 그대로 두고 주석만.
4. `pair_sweep.py` 직렬 루프 → 코어 `run_sweep`(병렬·캐시·시행 수 기록).
5. 포지션 영속: `OrderExecutor` 의 체결을 코어 `PositionBook(journal=data/positions/<date>.jsonl)` 에 **추가 기록**(기존 상태기계는 그대로). 재시작 시 보고 전용 `reconcile` 로그.
6. `cli_pair_detect.py` 에 코어 `InstanceLock` 추가.
7. `research/nightly.toml` + `crontab.simnode` 에 `kqc nightly` 한 줄.
8. 실주문 체결 판정·킬 판정 교체는 **이번에 하지 않는다**(수치 동일성 증명 비용이 크고 내일 실매매) — 다음 단계로 적는다.

## 6. 오류 처리
- 브로커 제출 예외·`return_code` 없는 제출 결과 → `OrderResult(submitted=True, return_code=None, ...)`, 상태 `unknown`(최종 리뷰에서 `rejected` 에서 변경). 재시도 없음. 매수는 횟수에 센다.
- 장부 journal 쓰기 실패는 예외를 올린다(상태가 조용히 사라지면 안 된다).
- `reconcile` 은 절대 주문을 내지 않는다.
- 스윕 objective 예외는 행 단위로 격리. `start_run` 게이트 거부는 그대로 `RunRefused`.
- 야간 job 타임아웃은 프로세스 그룹 kill.

## 7. 테스트
- 코어: 모듈마다 단위 테스트. 가짜 kiwoom api(`order`/`account` 네임스페이스), `PaperBroker` 시나리오(즉시 체결·대기 체결·부분 없음·보유 부족), `PositionBook` 비용·재생 동일성,
  `OrderManager`(킬이 매수만 막고 매도는 통과, 가드 사유 문자열 그대로, `InstanceLock` 이중 획득 실패), 리플레이 결정론(같은 입력 → 같은 fills),
  `run_sweep`(n_jobs=1·2 결과 동일, 캐시 적중, 원장 N), `nightly`(타임아웃·실패 격리, 호스트 게이트는 monkeypatch).
- 테스트는 simnode 에서 돈다(`kqc run krx-quant-core --ref <branch> -- uv run pytest` 또는 simnode 워크트리). CI(GitHub) 도 녹색이어야 머지.
- 소비 레포: 기존 테스트 전부 + 동일성 테스트. `preopen_check` 가 FAIL 0.

## 8. 릴리스
- 코어 `0.5.0`: 기능 브랜치 3개 → `feat/shared-engines` 로 합쳐 PR → main 머지 → `v0.5.0` 태그(PyPI publish 워크플로).
- 소비 레포: 각자 워크트리 브랜치 → `krx-quant-core==0.5.0` 핀 → PR → CI 녹색 → main 머지·푸시.
