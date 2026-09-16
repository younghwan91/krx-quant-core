# 공용 엔진 v0.5.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** krx-quant-core 에 trader 주문 관리 계층·이벤트 리플레이·simnode 스윕/야간 실행 엔진을 만들고 daytrade-it·scalp-it 이 쓰게 한다.

**Architecture:** 기존 값 객체(`OrderIntent`/`OrderResult`)·가드·킬스위치·체결 판정·비용·시행 원장·`start_run` 위에 얹는다. 전략 코드는 `EngineCore` 하나로 실매매(`KiwoomBroker`)와 리플레이(`PaperBroker`)에서 같다.

**Tech Stack:** Python ≥3.11, numpy/pandas, 표준 라이브러리(`fcntl`, `concurrent.futures`, `tomllib`, `subprocess`), 선택 optuna.

**Spec:** `docs/superpowers/specs/2026-09-16-shared-engines-design.md`

## Global Constraints

- Python `>=3.11`, ruff `0.16.2`, line-length 100, lint `E,F,I,W,UP,B`. `ruff check src/ tests/` 통과.
- 기존 공개 API·사유 문자열·수치는 **바꾸지 않는다**(설계 원칙 1). 기존 테스트 전부 통과.
- 통계 함수는 PASS/FAIL 을 돌려주지 않는다(설계 원칙 2).
- 새 런타임 의존성 금지. optuna 는 extra `opt`.
- docstring·주석은 한국어, 기존 모듈 문체(왜를 적는다).
- 테스트·백테스트는 simnode 에서만: `ssh simnode-local` 으로 워크트리 브랜치를 체크아웃해 `uv run --extra dev pytest -q`.
- 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- 소비 레포: 실주문 경로 기본 동작 불변. 2026-09-17 07:50 pull-all 로 실매매 반영.

## File Structure

| 파일 | 책임 |
|---|---|
| `src/krx_quant_core/execution/events.py` | `OrderStatus`, `Fill`, `Holding`, `OpenOrder` 값 객체 |
| `src/krx_quant_core/execution/broker.py` | `Broker` 프로토콜 |
| `src/krx_quant_core/execution/book.py` | `PositionBook` — 체결 → 보유·실현손익(비용 차감)·journal |
| `src/krx_quant_core/execution/paper.py` | `PaperBroker` |
| `src/krx_quant_core/execution/kiwoom_broker.py` | `KiwoomBroker`(kiwoom-client 덕타입 어댑터) |
| `src/krx_quant_core/execution/oms.py` | `OrderManager`, `InstanceLock`, `AlreadyRunning`, `ReconcileReport` |
| `src/krx_quant_core/execution/engine.py` | `Quote`/`Trade`/`Bar` 이벤트, `Strategy`, `StrategyContext`, `EngineCore` |
| `src/krx_quant_core/backtest/replay.py` | `merge_events`, `run_replay`, `ReplayResult` |
| `src/krx_quant_core/research/__init__.py`, `sweep.py`, `optuna.py` | `grid`, `run_sweep`, `SweepResult`, `optuna_search` |
| `src/krx_quant_core/runtime/nightly.py` | `load_jobs`, `run_nightly`, `JobResult` |
| `src/krx_quant_core/runtime/cli.py` | `kqc nightly` 서브커맨드 추가 |
| `tests/execution/test_{events,book,paper,kiwoom_broker,oms,engine}.py`, `tests/backtest/test_replay.py`, `tests/research/test_sweep.py`, `tests/runtime/test_nightly.py` | 단위 테스트 |

---

### Task 1: 값 객체 · Broker 프로토콜 · PositionBook

**Files:** Create `execution/events.py`, `execution/broker.py`, `execution/book.py`; Test `tests/execution/test_book.py`

**Interfaces — Produces:**
```python
class OrderStatus(StrEnum): BLOCKED="blocked"; SUBMITTED="submitted"; PARTIAL="partial"; FILLED="filled"; CANCELED="canceled"; REJECTED="rejected"
@dataclass(frozen=True) class Fill: ord_no: str; code: str; side: Literal["buy","sell"]; qty: int; price: int; ts: datetime
@dataclass(frozen=True) class Holding: code: str; qty: int; avg_price: float
@dataclass(frozen=True) class OpenOrder: ord_no: str; code: str; side: str; qty: int; remaining: int; price: int
class Broker(Protocol): submit(intent)->OrderResult; cancel(ord_no, intent)->OrderResult; open_orders()->list[OpenOrder]; holdings()->dict[str, Holding]; poll_fills()->list[Fill]
class PositionBook:
    def __init__(self, *, journal: Path | str | None = None, market_of: Callable[[str], str] = lambda c: "KOSPI", cost_model: KoreanCostModel | None = None) -> None
    def apply(self, fill: Fill) -> float           # 이 체결로 실현된 순손익(원). 매수는 0.0
    def position(self, code: str) -> Holding | None
    def positions(self) -> dict[str, Holding]
    realized_krw: float; n_round_trips: int
    def to_frame(self) -> pd.DataFrame             # 청산 원장: code, entry_ts, exit_ts, qty, entry_price, exit_price, pnl
    @classmethod
    def load(cls, journal: Path | str, **kw) -> PositionBook   # journal 재생(재기록 안 함)
```
- 매수 fill: `avg = (avg*q + price*qty)/(q+qty)`. 매도 fill: 보유 초과면 `ValueError`. 순손익 = `(price-avg)*qty - 매수측 commission - 매도측 commission - 매도세`; 비용은 `cost_model.cost_of_trade(Decimal(price), Decimal(qty), side, market, fill.ts.date())` 의 `commission + tax`, 매수 측은 평단 기준 notional 로 계산. `cost_model` 기본은 **슬리피지 0** 설정(`CostModelConfig` 에서 slippage 0 — 체결가가 이미 실제 가격).
- journal: `{"ord_no","code","side","qty","price","ts"}` 한 줄 append + flush + `os.fsync`.

**Tests:** 매수 두 번 평단; 매도 부분 청산 실현손익이 비용 차감과 일치(2026-09-16 KOSDAQ 세율 사용); 보유 초과 매도 ValueError; journal 재생 → 같은 positions·realized; to_frame 행 수 = 매도 fill 수.

- [ ] 실패 테스트 작성 → 실행(FAIL) → 구현 → 실행(PASS) → 커밋 `feat(execution): 체결 값 객체·Broker 프로토콜·PositionBook`

### Task 2: PaperBroker

**Files:** Create `execution/paper.py`; Test `tests/execution/test_paper.py`
**Consumes:** Task 1 types, `backtest.fills.limit_buy_filled/limit_sell_filled`, `OrderIntent`, `OrderResult`.
**Produces:**
```python
class PaperBroker:
    def __init__(self, *, fill_basis: FillBasis = "through", latency_sec: float = 0.0, holdings: dict[str, Holding] | None = None) -> None
    def on_quote(self, code: str, ts: datetime, bid: float, ask: float, last: float | None = None) -> list[Fill]
    # + Broker 프로토콜 전부
```
- `submit`: 매도 수량 > (보유 − 이미 걸린 매도 잔량) 이면 `OrderResult(submitted=False, blocked=True, blocked_reason="paper: 보유 부족")`. 아니면 `ord_no=f"P{n:06d}"`, `return_code=0`, `submitted=True`, `dry_run=False`, 대기열에 등록(도착시각 = 마지막으로 본 그 종목 시세 ts + latency; 시세 전이면 첫 시세부터).
- `on_quote`: 도착시각 이후 시세에서 — 매수: `ask <= limit` 이면 ask 가격(정수)에 전량 체결(테이커). 아니면 `last` 가 있고 `limit_buy_filled(limit, last, basis)` 면 limit 에 전량. 매도 대칭(`bid >= limit` → bid, `limit_sell_filled`). 체결은 보유 갱신 + `poll_fills` 버퍼.
- `cancel`: 대기 중이면 제거 `return_code=0`, 없으면 `return_code=-1, return_msg="paper: 주문 없음"`.
**Tests:** 마켓어블 매수 즉시 ask 체결; 대기 매수가 through 에서 last==limit 이면 미체결, last<limit 체결; touch 는 last==limit 체결; latency 2초면 1초 뒤 시세로는 체결 안 됨; 보유 부족 매도 차단; cancel 후 체결 없음; poll_fills 증분.

- [ ] TDD 사이클 → 커밋 `feat(execution): PaperBroker — 같은 Broker 프로토콜의 모의 체결`

### Task 3: KiwoomBroker

**Files:** Create `execution/kiwoom_broker.py`; Test `tests/execution/test_kiwoom_broker.py`
**Consumes:** Task 1, `OrderIntent.body()/cancel_body()`, `kiwoom_spec.strip_sign`.
**Produces:** `class KiwoomBroker: __init__(self, api: Any, *, dry_run: bool = True)` + Broker 프로토콜.
- 제출: `api.order.buy_order(**intent.body())` / `sell_order`. 응답 dict 의 `return_code`(int), `return_msg`, `ord_no`. 예외 → `OrderResult(submitted=True, return_code=None, return_msg=f"{type(e).__name__}: {e}")`. **재시도 없음.**
- 취소: `api.order.cancel_order(**intent.cancel_body(ord_no))` — 필드는 kiwoom-client `domestic/order.py` 의 `cancel_order` 시그니처로 확인해 맞춘다(원주문번호 키·취소수량 키 이름).
- 조회는 `_read(fn, **kw)`: 예외 시 1회 재시도.
  - `holdings()`: `api.account.evaluation_balance_detail(qry_tp="1", dmst_stex_tp="KRX")` → 목록 필드(kiwoom-client/kiwoom_spec 정본 확인, 보통 `acnt_evlt_remn_indv_tot`)의 `stk_cd`(앞 `A` 제거·`strip_sign`), `rmnd_qty`, `pur_pric`.
  - `open_orders()`: `api.account.unfilled_orders(...)` → `oso` 목록: `ord_no`, `stk_cd`, `io_tp_nm`/`trde_tp`(매수/매도 판별), `ord_qty`, `oso_qty`, `ord_pric`.
  - `poll_fills()`: `api.account.filled_orders(...)` → `cntr` 목록의 주문번호별 누적 `cntr_qty` 와 `cntr_pric`; 지난번 누적보다 늘어난 만큼만 `Fill` 로 낸다.
  - 필드명이 kiwoom-client 에서 확인되지 않으면 모듈 docstring "미검증 필드" 목록에 적고, 파서는 `_first(row, *keys)` 로 후보 키를 차례로 본다.
- `dry_run=True`: POST 없이 `OrderResult(intent, dry_run=True, submitted=False)`; 조회 메서드는 정상 동작(읽기).
- `logout`/토큰 폐기 메서드는 만들지 않는다.
**Tests:** 가짜 api(`SimpleNamespace(order=..., account=...)`, 호출 기록): dry-run 이면 order 호출 0; 성공 응답 → ok, ord_no; 예외 → 재시도 없이 1회 호출; holdings 파싱(`A005930`, `"+000010"`); poll_fills 두 번 호출 시 증분만; 조회 1회 실패 후 성공.

- [ ] TDD 사이클 → 커밋 `feat(execution): KiwoomBroker — kiwoom-client 어댑터(제출 무재시도·증분 체결)`

### Task 4: OrderManager · InstanceLock · ReconcileReport

**Files:** Create `execution/oms.py`; Test `tests/execution/test_oms.py`
**Consumes:** Tasks 1–2 (테스트는 PaperBroker 사용), `OrderGuard.reason(intent, ref_price)`, `OrderGuard.record_order(code)`, `KillSwitch.check(on) -> str|None`, `KillSwitch.record_trade(pnl_krw, on)`.
**Produces:**
```python
class AlreadyRunning(RuntimeError)
class InstanceLock:  # context manager; __init__(path), acquire(), release()
@dataclass(frozen=True) class ReconcileReport: only_book: dict[str,int]; only_broker: dict[str,int]; qty_mismatch: dict[str, tuple[int,int]]  # (book, broker)
    @property ok -> bool
@dataclass class ManagedResult: result: OrderResult; status: OrderStatus
class OrderManager:
    def __init__(self, broker: Broker, *, guard: OrderGuard, book: PositionBook, kill: KillSwitch | None = None, clock: Callable[[], datetime] = now_kst, order_log: Path | str | None = None) -> None
    def buy(self, code: str, qty: int, price: int, *, ref_price: float | None) -> ManagedResult
    def sell(self, code: str, qty: int, price: int) -> ManagedResult
    def cancel(self, ord_no: str, intent: OrderIntent) -> ManagedResult
    def sync(self) -> list[Fill]
    def reconcile(self) -> ReconcileReport
```
- `buy`: `kill.check(clock().date())` 사유 있으면 BLOCKED `blocked_reason=f"kill:{r}"`; `guard.reason` 사유 있으면 BLOCKED(사유 그대로); 제출; `result.ok and not dry_run` 이거나 dry_run 통과면 `guard.record_order(code)`. 상태: blocked→BLOCKED, ok→SUBMITTED, 아니면 REJECTED.
- `sell`: 보유 = `book.position(code)` 가 있으면 그 qty, 없으면 `broker.holdings()` 의 qty; `qty > 보유` → BLOCKED `"청산 수량이 보유 초과: {qty} > {held}"`; `price <= 0` → BLOCKED `"지정가가 0 이하: {price}"`. 킬·가드로는 막지 않는다.
- `sync`: fills 를 `book.apply`, 매도 fill 이면 `kill.record_trade(pnl, on=fill.ts.date())`.
- order_log: 결과마다 `{**result.to_record(ts=clock()), "status": status}` JSONL append.
**Tests:** 킬 발동 후 buy BLOCKED·sell 은 SUBMITTED; 가드 사유 문자열이 `OrderGuard.reason` 과 동일; dry-run·실패 시 record_order 카운트 규칙; sell 보유 초과 BLOCKED; sync 가 book·kill 갱신(연속손절 트리거); reconcile 세 가지 차이; InstanceLock 같은 경로 두 번째 acquire → AlreadyRunning(별도 프로세스 없이 두 개의 fd 로 검증 — flock 은 fd 단위).

- [ ] TDD 사이클 → 커밋 `feat(execution): OrderManager — 매수는 킬·가드, 청산은 보유만 본다 · InstanceLock · 대사 보고`

### Task 5: EngineCore · run_replay

**Files:** Create `execution/engine.py`, `backtest/replay.py`; Test `tests/execution/test_engine.py`, `tests/backtest/test_replay.py`
**Consumes:** Tasks 1,2,4.
**Produces:**
```python
@dataclass(frozen=True) class Quote: ts: datetime; code: str; bid: float; ask: float; bid_qty: float = 0; ask_qty: float = 0
@dataclass(frozen=True) class Trade: ts: datetime; code: str; price: float; qty: float
@dataclass(frozen=True) class Bar: ts: datetime; code: str; open: float; high: float; low: float; close: float; volume: float
Event = Quote | Trade | Bar
class Strategy(Protocol): def on_event(self, ev: Event, ctx: StrategyContext) -> None  # on_start/on_end 선택
@dataclass class StrategyContext: oms: OrderManager; now: datetime | None; last: dict[str, Event]
class EngineCore:
    def __init__(self, strategy, oms: OrderManager) -> None
    def start(self) -> None; def feed(self, ev: Event) -> list[Fill]; def end(self) -> None
    ctx: StrategyContext
def merge_events(*, quotes: pd.DataFrame | None = None, trades: pd.DataFrame | None = None, bars: pd.DataFrame | None = None) -> list[Event]
@dataclass class ReplayResult: fills: pd.DataFrame; orders: list[dict]; book: PositionBook; trades: pd.DataFrame
def run_replay(strategy, events: Iterable[Event], *, guard_config: OrderGuardConfig | None = None, kill_config: KillSwitchConfig | None = None, fill_basis: FillBasis = "through", latency_sec: float = 0.0, market_of: Callable[[str], str] = lambda c: "KOSPI") -> ReplayResult
```
- `EngineCore.feed`: `ctx.now = ev.ts`; `ctx.last[ev.code] = ev`; PaperBroker 면 Quote→`on_quote(code, ts, bid, ask)`, Trade→직전 Quote 의 bid/ask 와 `last=price` 로 `on_quote`, Bar→`on_quote(code, ts, bid=low, ask=high, last=close)` 가 아니라 **보수적으로** `bid=close, ask=close, last=close`; 그 뒤 `oms.sync()`; 그 뒤 `strategy.on_event`. 반환은 sync 된 fills.
- `merge_events`: DataFrame 열 이름 = 데이터클래스 필드명. 정렬 키 `(ts, kind_order{Quote:0,Trade:1,Bar:2}, 입력 순서)`.
- `run_replay` 기본 guard_config = `OrderGuardConfig(max_qty=10, price_band_pct=0.0, max_orders_per_code=0, max_orders_total=0)`; OrderManager 의 clock 은 `lambda: ctx.now or datetime.min(KST)` 로 이벤트 시각을 따른다. 끝나면 `strategy.on_end`, 미청산은 그대로 둔다(book 에 남음).
**Tests:** 매수 후 다음 Quote 에서 마켓어블 매도하는 장난감 전략 → fills 2, trades 1행, pnl = 비용 차감 값; 같은 입력 두 번 → fills DataFrame equal; merge_events 동시각 순서; EngineCore 가 KiwoomBroker 가짜 api 위에서도 같은 전략 호출(on_quote 없이 sync 만).

- [ ] TDD 사이클 → 커밋 `feat(engine): EngineCore·run_replay — 실매매·리플레이 같은 전략 코드`

### Task 6: research.run_sweep · optuna_search

**Files:** Create `research/__init__.py`, `research/sweep.py`, `research/optuna.py`; Modify `pyproject.toml`(extra `opt = ["optuna>=4.0"]`); Test `tests/research/test_sweep.py`
**Consumes:** `runtime.start_run`, `runtime.DataSpec`, `stats.trials.record_trial/config_fingerprint/count_trials`, `runtime.gitstate.git_head`, `stats.sharpe.deflated_sharpe_from_sample`.
**Produces:**
```python
def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]
@dataclass class SweepResult: frame: pd.DataFrame; n_trials: int; label: str
    def best(self, metric: str, *, returns_key: str | None = None, maximize: bool = True) -> dict[str, Any]  # {"config":..., metric:..., "dsr": {...}|None}
def run_sweep(objective: Callable[[dict], dict], configs: Sequence[dict], *, label: str, repo_root: Path | str, data: DataSpec, n_jobs: int | None = None, cache: bool = True, seed: int = 0, allow_dirty: bool = False, trials_dir: Path | str | None = None) -> SweepResult
def optuna_search(objective, space: Callable[[Any], dict], *, label, repo_root, data, n_trials: int, n_jobs: int = 1, direction: str = "maximize", metric: str = "score", seed: int = 0, allow_dirty: bool = False) -> SweepResult
```
- `run_sweep`: `with start_run(label, {"sweep_n": len(configs), "configs_fp": config_fingerprint({"c": configs}), "seed": seed}, repo_root=..., data=..., seed=seed, allow_dirty=..., trials_dir=...) as run:` 안에서 각 config `record_trial(label, cfg, logs_dir=<trials_dir or repo_root/runs_root>)` — `start_run` 이 쓰는 원장 폴더와 **같은 곳**(gitstate/runs 코드에서 경로 계산을 재사용). 실행: `n_jobs==1` 이면 직렬, 아니면 `ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn")가 아니라 기본)`; 결과 행 `{**cfg, **metrics}`; 예외 시 `{**cfg, "error": repr(e)}`. 캐시 디렉터리 `Path(os.environ.get("KQC_CACHE", "~/.kqc/cache")).expanduser()/label`, 키 `sha256(json.dumps([git_head, asdict(data), config_fingerprint(cfg)]))`. 리스트 지표(수익 배열)는 캐시 JSON 에 그대로 저장. `run.log_result({"n": len, "errors": k, "n_trials": count})`. `n_jobs` 기본 `max(1, min((os.cpu_count() or 2) - 2, int(os.environ.get("KQC_MAX_JOBS", 14))))`.
- `optuna_search`: `import optuna` 실패 시 `ImportError('pip install "krx-quant-core[opt]"')`. `TPESampler(seed=seed)`; 각 trial `cfg = space(trial)`; `record_trial`; `objective(cfg)[metric]` 반환; 한 번의 `start_run` 으로 감싼다.
- 테스트는 simnode 호스트 게이트를 `monkeypatch.setattr("socket.gethostname", lambda: "simnode")` (기존 `tests/runtime/test_runs.py` 의 `_host`·`repo` 픽스처 방식을 그대로 복사) 로 통과시킨다.
**Tests:** grid 곱·순서; n_jobs=1 과 2 frame 동일(정렬 후); 두 번째 실행 캐시 적중(objective 호출 카운터 파일로 확인); TRIALS 원장 N == 서로 다른 config 수; objective 예외 행 격리; best 가 returns_key 로 dsr dict 포함; optuna 미설치면 skip(`pytest.importorskip`).

- [ ] TDD 사이클 → 커밋 `feat(research): run_sweep·optuna_search — 병렬·캐시·모든 config 시행 원장 기록`

### Task 7: runtime.nightly · `kqc nightly`

**Files:** Create `runtime/nightly.py`; Modify `runtime/cli.py`; Test `tests/runtime/test_nightly.py`
**Consumes:** `runtime.host.require_backtest_host`, `market.session.now_kst`.
**Produces:**
```python
@dataclass(frozen=True) class Job: name: str; cmd: list[str]; timeout_min: float = 60; weekdays_only: bool = True
@dataclass(frozen=True) class JobResult: name: str; rc: int | None; secs: float; timed_out: bool; skipped: str | None = None
def load_jobs(repo_root: Path | str) -> list[Job]          # research/nightly.toml, tomllib
def run_nightly(repo_root: Path | str, *, only: str | None = None, dry_run: bool = False, out_root: Path | str | None = None, today: date | None = None) -> list[JobResult]
```
- `run_nightly`: `require_backtest_host()`; `out = out_root or ~/.kqc/nightly`; 날짜 폴더; 주말이고 weekdays_only 면 `skipped="weekend"`. 실행 `subprocess.Popen(["nice","-n","10",*cmd], cwd=repo_root, stdout=log, stderr=STDOUT, start_new_session=True)`; `wait(timeout)`; 타임아웃이면 `os.killpg(proc.pid, SIGKILL)`. dry_run 이면 실행 안 하고 `skipped="dry-run"`. 요약 `<out>/<date>/<repo_name>.json` 에 리스트 저장.
- CLI: `kqc nightly REPO_ROOT [--only NAME] [--dry-run]` → 실패(rc≠0 또는 timed_out) 하나라도 있으면 1.
**Tests:** toml 파싱; 성공·실패(`false`)·타임아웃(`sleep 5`, timeout_min=0.01) 세 job → 서로 격리, 요약 JSON; 주말 skip; only 필터; 호스트 아니면 RunRefused/RuntimeError; CLI 종료코드.

- [ ] TDD 사이클 → 커밋 `feat(runtime): kqc nightly — research/nightly.toml job 을 simnode 에서 격리 실행`

### Task 8: 내보내기 · 문서 · 릴리스 v0.5.0

**Files:** Modify `execution/__init__.py`, `backtest/__init__.py`, `README.md`, `pyproject.toml`(version `0.5.0`), `uv.lock`
- [ ] `execution/__init__.py` 에 새 이름 export(`Broker, Fill, Holding, OpenOrder, OrderStatus, PositionBook, PaperBroker, KiwoomBroker, OrderManager, InstanceLock, AlreadyRunning, ReconcileReport, EngineCore, Strategy, StrategyContext, Quote, Trade, Bar`). `backtest/__init__.py` 에 `run_replay, merge_events, ReplayResult` — 순환 import 가 생기면 `backtest.replay` 는 export 하지 않고 모듈 경로로만 쓴다.
- [ ] README 모듈 트리·"공용 엔진 (v0.5)" 절: 실매매/리플레이 같은 전략 예시, run_sweep 예시, kqc nightly 예시, 외부 레포에서 배운 표.
- [ ] simnode 에서 전체 `pytest -q`, `ruff check src/ tests/` → PASS.
- [ ] PR → GitHub CI 녹색 → main 머지 → `git tag v0.5.0 && git push origin v0.5.0` → PyPI publish 워크플로 성공 확인(`pip index versions krx-quant-core`).

### Task 9: daytrade-it 적용

워크트리 `~/git/daytrade-it-engines`(branch `feat/core-engines`, origin/main 기준). 스펙 5.1 항목 1–7.
- [ ] `krx-quant-core==0.5.0` 핀, `uv lock`.
- [ ] `live_STOP` 확인 두 곳(`interfaces/mcp/tools.py::_kill_switch_active`, `application/trading/auto_trader.py::_kill_switch_on`) → 코어 `KillSwitch(KillSwitchConfig(stop_file=path, sticky_stop_file=False)).stop_file_present()`. 기존 테스트로 동작 동일 확인.
- [ ] `scripts/news_signal_poller.py` 시작부에 `InstanceLock(data/auto_trader/daemon.lock)`; `AlreadyRunning` 이면 로그 후 exit 0(크론 재시작 루프가 멈추지 않게 기존 flock 경로와 같은 종료코드 확인).
- [ ] `--paper` 모드: `OrderManager(PaperBroker)` 로 AutoTrader 판단 흘리기(실주문 경로 기본값 불변). 단위 테스트.
- [ ] `preopen_check` 에 보고 전용 reconcile(경고만, exit code 불변).
- [ ] MCP `backtest._simulate` → 코어 `run_replay`(Bar) — 기존 MCP 테스트가 기대하는 응답 키 유지.
- [ ] `research/nightly.toml`(MCP 백테스트 스모크 job) + `deploy/crontab.simnode` 신규 `kqc nightly` 줄.
- [ ] simnode 에서 전체 테스트, `scripts/preopen_check.sh` 스모크 → PR → CI 녹색 → 머지·푸시.

### Task 10: scalp-it 적용

워크트리 `~/git/scalp-it-engines`(branch `feat/core-engines`). 스펙 5.2 항목 1–7.
- [ ] `krx-quant-core==0.5.0` 핀, lock.
- [ ] 자체 tick 함수(`realtime/tiger_shadow.py:_tick`, `scripts/flow_83/features.py:_tick_np`, `realtime/etf_strength.py:etf_tick`, `scripts/etf_strength_81.py`, `orderflow_80/of_build.py|of_strength.py|of_scratch.py`) → 코어 `market.ticks` — 교체 전후 1..2,000,000원 전 구간 대조 테스트. ETF 호가단위가 주식과 다르면 교체하지 않고 사유 주석.
- [ ] `COST = 0.0023` 류 → `round_trip_cost(date, Market.KOSDAQ|KOSPI)`; 2026 날짜 값 동일 assert. 날짜 문맥 없는 곳은 유지.
- [ ] `scripts/pair_sweep.py` → `run_sweep`(objective 최상위 함수화).
- [ ] `OrderExecutor` 체결 시 `PositionBook(journal=data/positions/<date>.jsonl).apply` **추가 기록**; 시작 시 reconcile 로그(보고 전용).
- [ ] `cli_pair_detect.py` 에 `InstanceLock(data/pair_detect.lock)`.
- [ ] `research/nightly.toml`(pair_sweep) + `deploy/crontab.simnode` 에 `kqc nightly` 줄(17:30, forward_gate 뒤).
- [ ] simnode 전체 테스트, preopen_check → PR → CI 녹색 → 머지·푸시.
