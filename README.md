# krx-quant-core

**한국 주식(KOSPI·KOSDAQ) 퀀트 공통 코어.** 호가단위·가격제한폭·세션 규칙, 일자별
증권거래세 비용모델, 키움 주문 가드, DART 중대공시 분류, 킬스위치, 체결 시뮬레이션,
Deflated Sharpe·purged CV 검증 통계를 한 패키지로 묶었다.

전략은 여기 없다. 전략을 가진 레포들이 **같은 시장 규칙과 같은 비용 숫자**를 쓰게 하는
것이 이 패키지의 존재 이유다.

| 소비자 | 매매 스타일 | 실행 | 이 패키지에서 쓰는 것 |
|---|---|---|---|
| [scalp-it](https://github.com/younghwan91) (비공개) | 스캘핑 | 자동 실주문 | 호가·상한가·주문 가드·호가 스윕·체결 규칙·호스트 가드 |
| [daytrade-it](https://github.com/younghwan91) (비공개) | 데이트레이딩 | 자동 실주문 | 비용모델·공시 리스크 게이트·DART DB·코드 정규화·호스트 가드 |
| [swing-it](https://github.com/younghwan91/swing-it) | 스윙 | 리서치 + 수동 체결 | DSR·purged CV·부트스트랩·취약성 진단·횡단면 시뮬 |

## 왜 따로 뺐나

세 레포가 같은 규칙을 각자 들고 있었고, 실제로 어긋나 있었다.

- `normalize_code` 가 두 레포에 **복붙**돼 있었다. 한쪽 옛 구현은 `0155E0` 을 `001550`
  (완전히 다른 회사)으로 바꾸던 버그를 안고 있었다.
- 거래비용이 레포마다 `0.0023`·`0.0034`·`0.0064`·`50bp` 로 흩어져 있었고, 거래세가
  2023→2024→2025→2026 에 네 번 바뀌었는데 일자별로 적용하는 코드가 **어디에도 없었다**.
- 체결 판정(`through`/`touch`)이 같은 레포 안에서도 알 수 없는 값을 받으면 한 곳은
  through, 다른 곳은 touch 로 **반대로** 읽었다.
- 하한가 계산은 없었다. 대칭식으로 짜면 틀린다 — 실측 대조 결과는 아래.

## 설치

```bash
pip install krx-quant-core==0.5.0
# 초 격자 호가 리플레이(backtest.lob)를 numba 로 가속하려면 extra 로: "krx-quant-core[fast]==0.5.0"
# optuna 스윕(research.optuna_search)까지 쓰려면: "krx-quant-core[fast,opt]==0.5.0"

# PyPI 릴리스 전(또는 태그 고정 개발 중)에는 git 태그로:
pip install "krx-quant-core @ git+https://github.com/younghwan91/krx-quant-core@v0.5.0"
```

Python ≥ 3.11. 의존성은 `kiwoom-client`(호가단위 표의 정본), `numpy`, `pandas` 뿐이다.
선택 extra `fast` 는 `numba` 를 더한다 — 없으면 같은 커널을 파이썬으로 돌려 같은 숫자를 낸다.

## 모듈

```
krx_quant_core/
├── market/     종목코드·Market, 호가단위, 상/하한가, KST 세션, 거래일 달력
├── costs/      일자별 거래세 스케줄, KoreanCostModel(Decimal), round_trip_cost(float)
├── execution/  주문 관리 계층 — OrderManager·InstanceLock(oms.py), PositionBook(book.py),
│               Broker 프로토콜·PaperBroker·KiwoomBroker, EngineCore(같은 전략, 실매매/리플레이),
│               키움 REST 주문 스펙, OrderIntent/OrderResult, OrderGuard(순수 가드)
├── risk/       DART 중대공시 분류·RiskGate, DartDisclosureDB, KillSwitch
├── backtest/   호가 스윕 VWAP·왕복비용, 지정가 체결 규칙, 트레이드 원장 지표, 횡단면 시뮬,
│               replay.py(run_replay — EngineCore+PaperBroker 로 과거 이벤트 리플레이)
│   └── lob/    틱·호가 → 초 격자 특징·경로, 배치=실시간 공용 커널, 에피소드 시뮬, 무작위 대조군, 지정가 대기열 모델
├── research/   run_sweep(격자·병렬·캐시), optuna_search(TPE, extra `opt`) — 둘 다 모든
│               config 를 시행 원장에 적는다
├── stats/      Deflated/Probabilistic Sharpe, t-haircut, purged walk-forward, 부트스트랩, 취약성
└── runtime/    호스트 가드, 실행 기록 start_run, OOS 하드 잠금, kqc CLI(simnode 원격 실행 · kqc nightly)
```

```python
from datetime import date
from krx_quant_core.market import Market, limit_up_price, limit_down_price, shift_ticks
from krx_quant_core.costs import round_trip_cost, tax_rate

limit_up_price(24_250)          # 31500
limit_down_price(239_000)       # 167500.0
shift_ticks(49_950, 2)          # Decimal('50100') — 밴드 경계를 넘어도 유효 호가
tax_rate(date(2025, 6, 2), Market.KOSPI).total   # Decimal('0.0015')
round_trip_cost(date(2026, 9, 14), Market.KOSDAQ, slippage_one_way=0.0015)  # 0.0053
```

### 초 격자 호가 리플레이 (`backtest.lob`, v0.2)

scalp-it 80·81·82번에 흩어져 있던 틱·호가 리플레이를 옮겼다. 원본 `build_code`·
`_strength_exit_labels`·`random_control` 사본과 합성 틱·호가로 대조해 **비트 단위 동일**
(골든 테스트), numba 경로와 파이썬 폴백도 동일, 배치와 실시간 증분도 동일하다.

```python
from datetime import date
from krx_quant_core.backtest.lob import build_second_grid, simulate_exits, random_entry_control

feat, path = build_second_grid(ticks, quotes)     # 09:00~15:20 초 격자, 09:05~15:10 특징 행
r = simulate_exits(feat.sec, path["bid"], path["ask"], path["strength"],
                   strength_drop=3.0, max_hold=300,          # 82번 청산 규칙
                   trade_date=date(2026, 9, 8), market="KOSDAQ")  # 비용 = round_trip_cost
r.net, r.hold, r.reason                          # 순수익·보유초·청산사유(EXIT_*)
```

실시간은 `SecondFeatureStream().update(...)` 를 초마다 부른다 — 배치와 같은 `step` 커널이다.
체결 가정은 원본 그대로 낙관적이다(1호가 전량 체결, 잔량·대기열 무시).

### 지정가 대기열 모델 (`backtest.lob.queue`, v0.3)

`touch`/`through` 는 대기열 위치를 모를 때의 두 극단이다. 10단계 호가 스냅샷과 가격별 체결량으로
**내 앞 잔량**을 추적한다(hftbacktest L2 모델 방식, MIT): `risk_averse`(취소는 전부 내 뒤),
`prob_power`·`prob_log`(취소를 앞·뒤에 확률 배분). 도착 즉시 반대 호가 스윕, 부분 체결, 정수 초 지연.

```python
from krx_quant_core.backtest.lob import build_book_grid, build_level_trades, simulate_limit_orders

book, trades = build_book_grid(quotes), build_level_trades(ticks)
r = simulate_limit_orders(+1, book.bid_px[secs, 0], 10, secs, book, trades,
                          max_wait=60, latency=1, queue_model="risk_averse")
r.filled_qty, r.avg_price, r.status      # STATUS_FILLED / PARTIAL / CANCELED / NOT_PLACED
```

실데이터 점검(2026-09-10, 체결 많은 20종목, 30초마다 매수1호가 합류 10주, 최대 60초 대기):

| 모델 | 체결률 | 체결 60초 뒤 마크아웃 |
|---|---|---|
| touch | 87.3% | **+3.2bp** |
| prob_log | 71.0% | −4.2bp |
| risk_averse | 70.4% | −4.6bp |
| through | 61.4% | −9.4bp |

touch 가정은 체결률만 부풀리는 게 아니라 **역선택을 지운다**(마크아웃 부호가 뒤집힌다).
데이터가 1초 절삭이라 초 미만 순서·지연은 모델링하지 않는다 — 가정 전체는 모듈 docstring.


## 백테스트 실행 기반 (`runtime`, v0.4)

백테스트는 **simnode 에서만** 돈다. 숫자마다 어디서·어떤 코드로·어떤 데이터로·몇 번째 시도로 나왔는지
기계가 남긴다.

```python
from krx_quant_core.runtime import DataSpec, start_run

with start_run("scalp84-flow", config, repo_root=ROOT,
               data=DataSpec("2026-08-24", "2026-09-07", "train"), seed=84) as run:
    ...
    run.log_result({"mean_bp": -3.1, "n": 812, "dsr_trials": run.n_trials})
```

- 게이트: simnode 아님 · 추적 파일 미커밋 · 잠긴 OOS 구간 → `RunRefused`.
- 기록: `research/runs/<label>/RUNS.jsonl`(정본, git) + `TRIALS.jsonl`(DSR 의 N 자동) + Postgres
  `kqc_runs` 색인(`[db]` extra, 실패해도 실행 계속).
- OOS: `kqc oos define` → `kqc prereg lock` → `start_run(..., final=True)` 는 label 당 **한 번**.

```bash
kqc run scalp-it -- uv run python scripts/x.py   # trader 에서: 푸시된 sha 를 simnode worktree 에서 실행
kqc runs ls scalp84-flow --repo-root ~/git/scalp-it
```

## 공용 엔진 (v0.5)

daytrade-it·scalp-it 감사 결과(2026-09-16) 둘 다 코어를 25~40%만 쓰고 있었다 — 가드·킬스위치는
있는데 체결·주문관리는 각자 복제, 페이퍼 모드는 데몬이 안 씀, 재시작 대사가 없었다. v0.5 는
그 위에 얹는 세 겹이다: **주문 관리 계층**(`execution`), **같은 전략이 실매매/리플레이를 도는 엔진**
(`execution.engine` + `backtest.replay`), **simnode 스윕·야간 실행**(`research` + `runtime.nightly`).

| 배워온 곳 | 원리 | 적용 |
|---|---|---|
| NautilusTrader | 전략 코드는 백테스트·실매매에서 **같다** — 다른 건 브로커/데이터 어댑터뿐. 시작 시 실계좌 대사 | `EngineCore`+`Strategy` 프로토콜, `OrderManager.reconcile()` |
| QuantConnect Lean | 브로커 모델·체결 모델·수수료 모델을 분리 | `Broker` 프로토콜 / `PaperBroker(fill_basis=...)` / 기존 `costs` |
| hftbacktest (MIT) | L2 호가 대기열 위치 모델 | `PaperBroker` 는 초 단위 스냅샷엔 `backtest.fills`, 격자 연구엔 `backtest.lob.queue` |
| vectorbt / Optuna | 대량 파라미터 스윕·병렬·조기 가지치기 | `research.run_sweep`(프로세스 풀·캐시), `research.optuna_search`(extra `opt`) |
| MLflow | 실행마다 코드·데이터·파라미터·지표를 기록 | 기존 `runtime.start_run` + `TRIALS.jsonl` 재사용 — 스윕의 모든 config 가 DSR 의 N 에 들어간다 |

### 1. 같은 전략, 실매매와 리플레이

`Strategy`(`on_start`/`on_event`/`on_end`)는 `StrategyContext.oms`(`OrderManager`)만 보고 어디서
체결되는지 모른다. 브로커만 바뀐다 — 웹소켓 이벤트는 `KiwoomBroker` 위 `EngineCore` 로, 과거
이벤트는 `PaperBroker` 위로 흘린다(`backtest.replay.run_replay` 가 그 배선을 대신 해 준다).

```python
from krx_quant_core.execution import (
    EngineCore, KiwoomBroker, OrderGuard, OrderGuardConfig, OrderManager, PositionBook, Quote,
)

class MyStrategy:
    def on_event(self, ev, ctx) -> None:
        if isinstance(ev, Quote) and ctx.oms.book.position(ev.code) is None:
            ctx.oms.buy(ev.code, 1, int(ev.ask), ref_price=ev.ask)

# 리플레이 — run_replay 는 backtest 최상위가 아니라 backtest.replay 에서 임포트한다
# (execution.paper ↔ backtest.replay 상호 의존이라 backtest/__init__ 이 이걸 다시 내보내면 순환 임포트가 난다).
from krx_quant_core.backtest.replay import merge_events, run_replay

events = merge_events(bars=daily_bars_df)   # ts, code, open/high/low/close, volume
result = run_replay(MyStrategy(), events)
result.fills, result.metrics, result.book.realized_krw

# 실매매 — 같은 전략, KiwoomBroker 위
broker = KiwoomBroker(api, dry_run=True)   # dry_run=False 는 실주문
guard = OrderGuard(OrderGuardConfig(max_qty=10, price_band_pct=0.05))
oms = OrderManager(broker, guard=guard, book=PositionBook(journal=Path("data/positions/2026-09-17.jsonl")))
engine = EngineCore(MyStrategy(), oms)
engine.feed(Quote(now_kst(), "005930", 70_000, 70_100))   # 웹소켓 이벤트마다 호출
```

### 2. simnode 파라미터 스윕

```python
from krx_quant_core.research import grid, run_sweep
from krx_quant_core.runtime import DataSpec

def objective(cfg: dict) -> dict:
    ...  # 모듈 최상위 함수 — ProcessPoolExecutor 로 자식 프로세스에 피클된다
    return {"sharpe": ..., "mean_bp": ...}

configs = grid(threshold=[0.5, 0.6, 0.7], hold_sec=[60, 120])
res = run_sweep(objective, configs, label="scalp84-sweep", repo_root=ROOT,
                data=DataSpec("2026-08-24", "2026-09-07", "train"))
res.frame, res.n_trials, res.best("sharpe")   # 모든 config 가 TRIALS.jsonl 에 적힌다
```

### 3. `kqc nightly` — simnode 야간 실행

레포에 `research/nightly.toml` 을 두면 `kqc nightly` 가 순서대로(job 하나가 실패해도 다음은 돈다)
`nice -n 10` 으로 실행하고, 로그와 요약(`name, rc, secs, timed_out`)을 `~/.kqc/nightly/<날짜>/` 에 남긴다.

```toml
# research/nightly.toml
[[job]]
name = "pair-sweep"
cmd = ["uv", "run", "python", "scripts/pair_sweep.py", "--days", "20"]
timeout_min = 60
weekdays_only = true
```

```bash
kqc nightly ~/git/scalp-it                       # 크론 한 줄(장 마감 후, simnode 전용)
kqc nightly ~/git/scalp-it --only pair-sweep --dry-run
```

### 안전 규칙 — 지키는 것과 아직 못 미더운 것

- **매수는 킬·가드가 막지만 매도(청산)는 막지 않는다.** `OrderManager.sell` 이 막는 건 1주 미만·
  가용 보유(보유 − 걸린 매도 잔량)보다 많이 파는 것·0 이하 지정가, 셋뿐이다. 킬이 걸린 날일수록
  들고 있는 포지션은 빠져나가야 한다 — 청산까지 막으면 실포지션이 감시 없이 남는다.
- **주문 제출은 재시도하지 않는다.** `KiwoomBroker.submit` 실패는 그대로 `rejected` 로 끝난다 —
  이중 주문 쪽이 더 위험하다. 조회(미체결·잔고·체결)만 예외 시 1회 재시도한다.
- **`poll_fills`(`ka10076` 체결 조회)는 미검증이다.** 필드명이 kiwoom-client 에 예시가 없어
  브리프 추정값을 쓴다 — **모의계좌로 실호출 확인 전에는 실주문 데몬에 쓰지 말 것.**
  `KiwoomBroker.holdings()`(`kt00018`)·`open_orders()`(`ka10075`)는 검증됐다.
- **`InstanceLock`** 은 같은 계좌를 도는 데몬이 두 번 뜨는 사고(2026-09-15 이중 매수 원인)를
  `flock(LOCK_EX|LOCK_NB)` 로 막는다. 이미 잡혀 있으면 `AlreadyRunning` — 데몬은 이걸 정상 종료로
  다뤄야 한다(재시작 루프가 계속 두 번째 인스턴스를 죽이면 안 된다).
- **`OrderManager.reconcile()` 은 절대 주문을 내지 않는다.** 장부 vs `broker.holdings()` 차이를
  보고만 한다 — 어느 쪽이 맞는지는 사람이 판단한다.

## 설계 원칙

1. **이식은 수치 동일.** 소비 레포가 실매매일에 갈아탈 수 있어야 한다. 포트마다 원본
   코드와 무작위 입력으로 대조했다 — 주문 가드 16,000건 사유 문자열 동일, 킬스위치
   9,000 스텝 동일, 실제 `dart.db` 공시 26,551건 분류 동일, 통계·횡단면 시뮬 40 시드
   NaN 포함 동일.
2. **판정하지 않는다.** 통계 함수는 숫자를 리포트할 뿐 `PASS`/`FAIL` 을 돌려주지 않는다
   (테스트가 강제). 합격선은 사전등록 문서에 있어야 한다.
3. **모르는 건 모른다고 적는다.** 시장가 `trde_tp` 는 `"03"`/`"3"` 둘 중 무엇인지 실호출로
   확인된 적이 없어 `TRDE_TP_MARKET_VERIFIED = False` 로 둔다. 2025 이전 거래세 단계는
   2차 출처만 있다고 docstring 에 적었다.
4. **토큰은 공유 자원이다.** 두 실매매 프로세스가 같은 앱키를 쓴다. 연결 해제 시 토큰을
   폐기하면 다른 프로세스가 죽는다(`kiwoom_spec.TOKEN_SHARING_WARNING`).

## 가격제한폭 — 실측 대조

`daily_bars`(2023-02 이후)에서 고가·저가가 기준가 ±29~31% 인 행에 후보 규칙을 대조했다.

| 규칙 | 규칙끼리 갈리는 행에서 적중 |
|---|---|
| 상한가 = ⌊기준가×1.3 을 **그 가격대** 틱⌋ | **1,211** |
| 상한가 = 기준가 + ⌊기준가×0.3 을 기준가 틱⌋ | 165 |
| 하한가 = ⌈기준가×0.7 을 그 가격대 틱⌉ (대칭식) | 25 |
| 하한가 = 기준가 − ⌊기준가×0.3 을 **기준가** 틱⌋ | **105** |

즉 상한가와 하한가는 **대칭이 아니다**.

## 거래세 스케줄 (매도 시)

| 시행일 | KOSPI 거래세 + 농특세 | KOSDAQ |
|---|---|---|
| 2026-01-01 | 0.05% + 0.15% = 0.20% | 0.20% |
| 2025-01-01 | 0.00% + 0.15% = 0.15% | 0.15% |
| 2024-01-01 | 0.03% + 0.15% = 0.18% | 0.18% |
| 2023-01-01 | 0.05% + 0.15% = 0.20% | 0.20% |
| 2021-01-01 | 0.08% + 0.15% = 0.23% | 0.23% |
| 2019-06-03 | 0.10% + 0.15% = 0.25% | 0.25% |

2026 개정은 시행령 개정 보도로 확인했고, 그 이전 단계는 2차 출처다. 법령 원문으로 확인되면
이 표를 갱신한다.

## 개발

```bash
uv sync --extra dev
uv sync --extra dev --extra fast   # numba 경로까지
uv run pytest -q        # 632 tests
uv run ruff check src tests
```

태그 `v*` 를 푸시하면 `publish.yml` 이 PyPI 로 올린다(trusted publishing 설정 후).

## 로드맵

- **v0.5** 로 주문 관리 계층(`execution.oms`)·`KiwoomBroker`/`PaperBroker`·`EngineCore`+
  `backtest.replay`·`research.run_sweep`/`optuna_search`·`kqc nightly` 가 들어갔다(위 "공용
  엔진" 참고). 남은 것: `poll_fills`(`ka10076`) 모의계좌 실호출 확인, scalp-it `RiskGuard` 킬
  판정을 `KillSwitch` 로 위임(대조 테스트는 이미 있음), daytrade-it/scalp-it 실주문 경로를
  코어 엔진으로 갈아타는 건 수치 동일성 증명이 끝난 부분부터 단계적으로.
- ETF 호가단위(kiwoom-client 정본 추가 대기). 대기열 모델(`backtest.lob.queue`)은 실주문
  체결로 계속 보정할 것.
- PyPI 릴리스.

## 라이선스

Apache-2.0
