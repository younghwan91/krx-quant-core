# krx-quant-core Reference Guide

New here? Start with the [README](../README.en.md).

<!-- toc -->
- [Design Principles](#design-principles)
- [Price Limits — Empirical Verification](#price-limits--empirical-verification)
- [Transaction Tax Schedule (on Sell)](#transaction-tax-schedule-on-sell)
- [Run Ledger & OOS Lock (`runtime`)](#run-ledger--oos-lock-runtime)
- [Same Strategy, Live & Replay (`execution` + `backtest.replay`)](#same-strategy-live--replay-execution--backtestreplay)
- [Fill Assumptions (`backtest.lob`)](#fill-assumptions-backtestlob)
- [Edge Evaluation Axes (`backtest` + `stats`)](#edge-evaluation-axes-backtest--stats)
- [Data Layer (`data`) — Read Once, Columnar `mmap`](#data-layer-data--read-once-columnar-mmap)
- [Search: Sweeps & Evolution (`research`)](#search-sweeps--evolution-research)
- [Development](#development)
- [Roadmap](#roadmap)
<!-- /toc -->

## Design Principles

1. **Numerical parity on migration.** Consumers must be able to switch over on a live trading day. Every port was cross-checked against original code on random and real data (verified at v0.1.0) —
   16,000 order-guard reason strings identical, 9,000 kill-switch steps identical, 26,551 real `dart.db` disclosures classified identically, and stats/cross-sectional sims identical across 40 seeds including NaNs. Numba kernels and Python fallbacks pass the same golden tests.
2. **No pass/fail judgments.** Statistical functions return numbers only; tests forbid returning `PASS`/`FAIL`. Thresholds belong in each repo's pre-registration docs.
3. **Mark unknowns explicitly.** Because market-order `trde_tp` (`"03"` vs `"3"`) has never been verified in a live call, `TRDE_TP_MARKET_VERIFIED = False`. Pre-2025 tax steps are documented as secondary-source only in docstrings.
4. **Tokens are shared resources.** Multiple live daemons share the same app key. Revoking a token on disconnect kills the other process (`kiwoom_spec.TOKEN_SHARING_WARNING`).

## Price Limits — Empirical Verification

Tested candidate rules against `daily_bars` (since 2023-02) on rows where high/low fell within ±29–31% of the base price.
Measured when writing v0.1.0 in 2026-09; the reproduction script is not in this repo.

| Rule | Hits where rules disagree |
|---|---|
| Limit-up = ⌊base × 1.3 floored to **target-band** tick⌋ | **1,211** |
| Limit-up = base + ⌊base × 0.3 floored to base-band tick⌋ | 165 |
| Limit-down = ⌈base × 0.7 ceiled to target-band tick⌉ (symmetric) | 25 |
| Limit-down = base − ⌊base × 0.3 floored to **base-band** tick⌋ | **105** |

## Transaction Tax Schedule (on Sell)

| Effective Date | KOSPI (Tax + Rural Special Tax) | KOSDAQ |
|---|---|---|
| 2026-01-01 | 0.05% + 0.15% = 0.20% | 0.20% |
| 2025-01-01 | 0.00% + 0.15% = 0.15% | 0.15% |
| 2024-01-01 | 0.03% + 0.15% = 0.18% | 0.18% |
| 2023-01-01 | 0.05% + 0.15% = 0.20% | 0.20% |
| 2021-01-01 | 0.08% + 0.15% = 0.23% | 0.23% |
| 2019-06-03 | 0.10% + 0.15% = 0.25% | 0.25% |

The 2026 revision was verified from decree amendment reports; earlier steps come from secondary sources. This table will be updated once verified against primary statutory texts.

## Run Ledger & OOS Lock (`runtime`)

Backtests run only on a dedicated host (`simnode`). Every number records where, which code, which data, and which attempt produced it.

```python
from krx_quant_core.runtime import DataSpec, start_run

with start_run("scalp84-flow", config, repo_root=ROOT,
               data=DataSpec("2026-08-24", "2026-09-07", "train"), seed=84) as run:
    ...
    run.log_result({"mean_bp": -3.1, "n": 812, "dsr_trials": run.n_trials})
```

- Gates: non-`simnode` host, dirty tracked files, or locked OOS window → `RunRefused`.
- Records: `research/runs/<label>/RUNS.jsonl` (canonical, in git) + `TRIALS.jsonl` (automatic `N` for DSR) + Postgres `kqc_runs` index (`[db]`, runs continue even if DB indexing fails).
- OOS: `kqc oos define` → `kqc prereg lock` → `start_run(..., final=True)` runs **once** per label. Editing the pre-registration doc breaks the hash and refuses final evaluation; even if final evaluation crashes mid-run, it cannot be re-run. Reverting requires a human deleting it in git, leaving a commit trail.

```bash
kqc run scalp-it -- uv run python scripts/x.py   # Run pushed SHA in a simnode worktree
kqc runs ls scalp84-flow --repo-root ~/git/scalp-it
kqc nightly ~/git/scalp-it                       # Run jobs in research/nightly.toml sequentially (post-close cron)
kqc nightly status --days 7                      # Recent repo × job status table; exits 1 on recent failure
kqc pins                                         # Table of core versions pinned by consumers; exits 1 if any lag
```

## Same Strategy, Live & Replay (`execution` + `backtest.replay`)

`Strategy` (`on_start`/`on_event`/`on_end`) only sees `StrategyContext.oms` and does not know where orders execute. Only the broker swaps —
WebSocket events flow into `EngineCore` over `KiwoomBroker`, while historical events flow over `PaperBroker`
(following NautilusTrader's "same strategy in backtest and live" and Lean's separation of broker, fill, and fee models).

```python
from krx_quant_core.execution import (
    EngineCore, KiwoomBroker, OrderGuard, OrderGuardConfig, OrderManager, PositionBook, Quote,
)

class MyStrategy:
    def on_event(self, ev, ctx) -> None:
        if isinstance(ev, Quote) and ctx.oms.book.position(ev.code) is None:
            ctx.oms.buy(ev.code, 1, int(ev.ask), ref_price=ev.ask)

# Replay — import run_replay from backtest.replay, not top-level backtest
# (execution.paper <-> backtest.replay are mutually dependent; re-exporting in backtest/__init__ would cause a circular import).
from krx_quant_core.backtest.replay import merge_events, run_replay

events = merge_events(bars=daily_bars_df)   # ts, code, open/high/low/close, volume
result = run_replay(MyStrategy(), events)
result.fills, result.trades, result.book.realized_krw   # trades = closed-trade ledger (one row per sell fill)

# Live — same strategy over KiwoomBroker
broker = KiwoomBroker(api, dry_run=True)   # dry_run=False submits live orders
# fills_verified=False (default): ka10076 fill lookup is unverified, so poll_fills/prime return [] (warns once).
# Only enable KiwoomBroker(api, dry_run=False, fills_verified=True) after verifying fields on a paper account.
guard = OrderGuard(OrderGuardConfig(max_qty=10, price_band_pct=0.05))
oms = OrderManager(broker, guard=guard, book=PositionBook(journal=Path("data/positions/2026-09-17.jsonl")),
                   order_log=Path("logs/orders.jsonl"))
engine = EngineCore(MyStrategy(), oms)
engine.feed(Quote(now_kst(), "005930", 70_000, 70_100))   # Call on every WebSocket quote event
# Sync fills on a timer — with fills_verified=True, polling on every feed() would exceed REST rate limits.
#   e.g. every 2s via loop.call_later / thread timer: oms.sync()
```

> Naming note: `execution.Trade` is a market tick trade event; `backtest.Trade` is one row in the closed-trade ledger.
> When importing both in one file, alias one (e.g. `from krx_quant_core.execution import Trade as TradeTick`).

### Safety Rules — What Is Enforced vs. Still Unverified

- **Kill switch and guard block buys, never sells (exits).** `OrderManager.sell` only blocks `< 1` share, selling more than available holdings (`held - open sell qty`), or limit price `<= 0`. Especially on a kill-switch day, open positions must be able to exit — blocking exits leaves live positions unmonitored.
- **Order submissions are never retried.** Timeouts, exceptions, and responses without `return_code` become `OrderStatus.UNKNOWN` (`"unknown"`) — **unknown whether it went through; do not retry.** Check via open orders and holdings. Unknown buys count toward the daily attempt limit (otherwise retries could breach the cap). Only `rc != 0` broker rejections become `rejected`. Queries (open orders, holdings, fills) retry once on exception.
- **`poll_fills` (`ka10076`) is unverified.** Because `kiwoom-client` has no example of its field names, brief-estimated names are used. `KiwoomBroker(fills_verified=False)` is therefore the default, making `poll_fills` and `prime` return `[]` without querying — **do not enable before testing against a paper account.** `OrderManager.sync` logs query exceptions as `poll_fills_failed` and returns `[]` so the engine does not crash on every event.
- **`holdings()` (`kt00018`) and `open_orders()` (`ka10075`) are not fully verified either.** In particular, `ka10075`'s buy/sell indicator fields (`io_tp_nm`, `sell_tp_nm`, `sell_tp`) are unverified in live calls. Unparseable rows are skipped (`skipped_rows`), and if rows exist but all are skipped, a warning is logged — a silent empty list would zero out reserved sell quantities and trigger duplicate exits.
- **Shared account: only apply fills from own orders.** Because `scalp-it` and `daytrade-it` share an account, fill queries include each other's fills. `OrderManager` only applies fills matching its own order numbers (`own_orders`, zero-padded normalized via `normalize_ord_no`) to the book and kill switch, routing others to `foreign_fills` and `foreign_fill` logs (`own_orders_only=False` only when an account is dedicated). Fills from `UNKNOWN` orders lacking an order number also land here for human review.
- **In-memory state is lost on restart (follow-up item).** `own_orders`, partial-exit cumulative PnL, and kill-switch state do not survive restarts. Fills from pre-restart orders land in `foreign_fills` — use `reconcile()` to cross-check.
- **`InstanceLock`** prevents two daemons on the same account (previously a cause of double buys) using `flock(LOCK_EX|LOCK_NB)`. If already held, it raises `AlreadyRunning` — daemons should treat this as a clean exit so restart loops do not keep killing the second instance.
- **`OrderManager.reconcile()` never submits orders.** It only reports discrepancies between the local book and `broker.holdings()` — a human decides which is right.
- **Consumer-repo convention switches.** Defaults follow all principles above; daemon-specific legacy conventions are opt-in:
  `preorder_holdings_check` (check holdings right before buying, fail-closed on error), `guard_sells` (apply guard/counts to exits),
  `count_rejected` (count at guard pass time), `loss_eps` (`1e-9` treating breakeven as a win).

## Fill Assumptions (`backtest.lob`)

Ticks and quotes are aggregated onto a 1-second grid for features and paths, where batch and live call the **exact same `step` kernel** (golden tests verify original vs. batch, batch vs. live, and Numba vs. Python fallback with `np.array_equal(equal_nan=True)`). Intraday restarts recover by replaying that day's grid through `SecondFeatureStreamV2.from_history(...)`.

```python
from datetime import date
from krx_quant_core.backtest.lob import build_second_grid, simulate_exits

feat, path = build_second_grid(ticks, quotes)     # 09:00–15:20 1s grid, 09:05–15:10 feature rows
r = simulate_exits(feat.sec, path["bid"], path["ask"], path["strength"],
                   strength_drop=3.0, max_hold=300,
                   trade_date=date(2026, 9, 8), market="KOSDAQ")  # cost = round_trip_cost
r.net, r.hold, r.reason                          # net return, hold seconds, exit reason (EXIT_*)
```

<details>
<summary>Building live per-second inputs (<code>backtest.lob.features_v2</code>) — identical rules to batch <code>aggregate_seconds_v2</code></summary>

Quote row `x` (`INPUT_COLUMNS_V2`) passed to `SecondFeatureStreamV2().update_array(...)` when second `s` closes:

```python
x = [
    last_tick.best_bid, last_tick.best_ask,   # Last tick values in that second; keep NaN if missing
    last_tick.strength,                       # (all three NaN if no ticks in that second)
    q.bidqty1, q.askqty1,                     # q = last "valid" snapshot in that second (bid1>0 and ask1>0), else NaN
    q.bidqty1 + q.bidqty2 + q.bidqty3,        #   × q.bid1 (skip NaN levels in sum) — bid_depth3
    ...,                                      #   ask_depth3 computed the same way
    r.bidqty1, r.askqty1,                     # r = last snapshot in that second (valid or not), keep NaN
]
# If a valid snapshot exists, bid/ask are overwritten with that snapshot's bid1/ask1.
row = stream.update_array(x, prices, volumes, sides)   # Trades in that second, ordered by (ts, seq)
```

</details>

`touch` and `through` are the two extremes when queue position is unknown. `backtest.lob.queue` tracks **quantity ahead of you** using 10-level order-book snapshots and per-price trade volume (following the hftbacktest L2 model): `risk_averse` (all cancels assumed behind you), `prob_power`, and `prob_log`.

```python
from krx_quant_core.backtest.lob import build_book_grid, build_level_trades, simulate_limit_orders

book, trades = build_book_grid(quotes), build_level_trades(ticks)
r = simulate_limit_orders(+1, book.bid_px[secs, 0], 10, secs, book, trades,
                          max_wait=60, latency=1, queue_model="risk_averse")
r.filled_qty, r.avg_price, r.status      # STATUS_FILLED / PARTIAL / CANCELED / NOT_PLACED
```

In a real-data check (one day 2026-09-10, 20 actively traded stocks, joining bid-1 with 10 shares every 30s, max 60s wait), `touch` showed 87% fill rate and +3.2bp 60s post-fill markout, `risk_averse` showed 70% and −4.6bp, and `through` showed 61% and −9.4bp. The `touch` assumption doesn't just inflate fill rates — it **erases adverse selection**.
Because data is truncated to 1-second resolution, sub-second ordering and latency are not modeled — see module docstrings for full assumptions.

## Edge Evaluation Axes (`backtest` + `stats`)

All three repos use the same pure functions to prevent turning on live orders over spurious edge. None return pass/fail judgments — thresholds belong in each repo's pre-registration.

```python
from datetime import date
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.backtest import drift_summary, intraday_overnight, rank_buckets
from krx_quant_core.backtest.lob import ceiling_table
from krx_quant_core.stats import matched_alpha, matched_control, selection_bias_report

# 1. Oracle ceiling — does even perfect hindsight exit timing beat costs?
cost = round_trip_cost(date(2026, 10, 2), "KOSDAQ", slippage_one_way=0.0)   # tax + fees; spread comes from quotes
t = ceiling_table([(path["bid"], path["ask"]) for path in paths], cost=cost, horizons=(10, 60, 300, 1800))
t[t.kind == "taker"][["horizon", "median_bp", "frac_pos"]]

# 2. Selection bias — how many bp does picking the validation max add with zero skill?
rep = selection_bias_report(val_log_rows)      # val_n, val_daymean_bp, val_mean_bp columns
rep.premium_vs_random, rep.expected_max_curve

# 3. Intraday vs. overnight — is intraday drift structurally negative in this universe?
d = intraday_overnight(daily)                              # open/close/prev_close -> intraday_bp, overnight_bp
d["bucket"] = rank_buckets(d, "turnover_prev", n=4)        # prior-day info only (same-day leaks look-ahead)
drift_summary(d, "intraday_bp", by="bucket")               # mean_bp, neg_month_share, t_hac

# 4. Matched null — do post-signal returns beat random entries in the same stratum (date × min × vol × turnover)?
ctl = matched_control(trades, pool, strata=["date", "minute", "vol_q", "turnover_q"], n_per=5)
a = matched_alpha(trades, pool, ctl, value="net_bp", cluster="date")   # date-clustered bootstrap
a.alpha, a.ci_low, a.ci_high, a.by_stratum
```

`cost` is a required argument — cost constants are never hardcoded.

## Data Layer (`data`) — Read Once, Columnar `mmap`

Reads Postgres (`KR_QUANT_DB`) via `COPY … TO STDOUT`, stores each column as `.npy`, and opens via `mmap` thereafter. Only requested columns touch disk, and 16 workers reading the same day share one page cache. Cache directory `$KQC_DATA` (default `~/.kqc/data`). Opt-in, leaving existing adapters intact. Only closed days (before KST today) are cached; today always queries DB.

```python
from datetime import date
from krx_quant_core import data as D

tk = D.load_ticks(date(2026, 10, 7))                 # Per-ticker CSR: tk.codes, tk.ptr, tk.cols["price"] (int32)
qs = D.load_quotes(date(2026, 10, 7), levels=3)      # Cache stores all 10 levels; reads only 3
mb = D.load_minute_bars(date(2026, 10, 7))           # sec = minute start second
bars = D.load_daily_bars("2016-01-01", "2026-10-08")  # Yearly cache + fingerprint — only changed years re-fetch
panel = D.daily_panel(bars, ["close", "trade_value"], dtype="float32")
panel.frame("close", orient="date_x_code")           # orient is required — prevents missing-.T bugs
cal = D.trading_calendar()                           # daily_bars dates, DB queried once per KST day
```

Measured on `simnode` (single day 2026-10-07, cold → warm): 1.9M ticks 5.4s → 1ms, 900K quotes 9.0s → 2ms, 10-year 5.8M daily bars 16s → 1.1s.
`frame()` upcasts integer columns to `int64` — when multiplying raw arrays (`get()`), call `.astype(np.int64)` first (`int32 × int32` overflows).

### KRX Data Pitfalls (unreleased on `main` — opt-in, default outputs unchanged)

Since 2026-09-14, KRX after-market trading (16:00–20:00) has been mixed into Kiwoom daily bars, minute bars, and investor-flow closes — `daily_bars` high, low, close, and volume are no longer regular-session values. Next-day price limits are based on the **regular-session** close. Some minute bars are also retroactively adjusted for ex-rights and splits.

```python
mb = D.load_minute_bars(date(2026, 10, 8), session="regular")    # 09:00–15:32 bars only
bars = D.load_daily_bars("2026-09-01", "2026-10-12", session="regular")  # Re-aggregate 09-14+ from regular minute bars
bp = D.base_price(date(2026, 10, 12), ["005930"])                 # base 262,000 (daily bar close 263,000)
limit_up_price(bp.base[0])                                       # 340,500 — daily bar close would give 341,500
```

`base_price` selects in priority order: same-day flow identity → prior-day regular closing minute bar → flow close → `daily_bars_raw` → `daily_bars` (overridable via `sources=`), and adjusts via `corporate_actions` on ex-dates. The `src` column records which source was used.

## Search: Sweeps & Evolution (`research`)

Every config is logged to `TRIALS.jsonl` and counted in DSR's `N` — nothing can be run and omitted from the trial count.

```python
from krx_quant_core.research import grid, run_sweep
from krx_quant_core.runtime import DataSpec

def objective(cfg: dict) -> dict:
    ...  # Top-level module function — pickled to worker processes via ProcessPoolExecutor
    return {"sharpe": ..., "mean_bp": ...}

configs = grid(threshold=[0.5, 0.6, 0.7], hold_sec=[60, 120])
res = run_sweep(objective, configs, label="scalp84-sweep", repo_root=ROOT,
                data=DataSpec("2026-08-24", "2026-09-07", "train"))
res.frame, res.n_trials, res.best("sharpe")
```

`research.evolve` provides the compute engine and protocol for generational evolution (per-generation candidate menu → real search per fold → sealed window evaluated once at the end). Strategy phenotype and fitness are consumer callbacks. `pip install "krx-quant-core[evo]"`.

| Module | Contents |
|---|---|
| `space` | `Real`, `Choice`, `Ordinal` search space, `decode`, collapsing equal-value levels, `override(menu)` |
| `optim` | `bipop(evaluate, space, budget=, seed=, …)` — BIPOP restarts, batch-evaluating only un-memoized phenotypes per generation. Backends `cmawm`, `catcmawm` (categoricals as distributions) |
| `protocol` | `EvoLedger` (registers before results → incomplete runs still count in DSR `N`), `race` (eliminates candidates via paired-difference SE over identical seed sets), `Seal`, `SealBreach`, `provenance` |
| `pool` | `EvoPool` — task-order aggregation |
| `kernels` | Monthly sum → moving block bootstrap → percentile in one pass, `block_boot`, `h32` |

```python
from krx_quant_core.research.evolve import Space, Real, Choice, Ordinal, bipop

space = Space([Real("rs_min", 0.0, 90.0, off="lo"), Choice("entry", ("bottom", "horiz", "trend")),
               Ordinal("theta", (0.06, 0.08, 0.10)),
               Ordinal("exit_safe", (0, 1), optional=True)]).override(menu)

def evaluate(G):                       # Only un-memoized phenotypes, in first-seen order
    return fits, n_trades              # (objective [maximized], entry count)

res = bipop(evaluate, space, budget=10_000, seed=0, x0=x0, x_restart=pool, canonical=canon)
res.best_payload, res.best_value, res.uniq, res.restarts, res.top(5)
```

`race` exploits the fact that all candidates run on the same seed set, pruning via paired-difference SE (shrinking as `√n`). `k`, `min_sets`, and seed order are written to the ledger before round 1, and winner's curse among survivors is avoided by evaluating only once on the sealed window.

## Development

```bash
uv sync --extra dev                      # + --extra fast(numba) --extra opt(optuna) --extra db(psycopg) --extra evo(cmaes)
uv run pytest -q                         # ~1 minute
uv run ruff check src tests
KRX_QUANT_CORE_DISABLE_NUMBA=1 uv run --extra dev --extra fast pytest -q   # Force Python fallback even with numba installed
```

CI runs Python 3.11–3.13 × `[dev,fast]` and `[dev]` without Numba separately — both paths must pass the same golden tests.
Pushing a `v*` tag triggers `publish.yml` to publish to PyPI. Consumer repos do not change until they bump their pin (checked via `kqc pins`).

Design specs live in `docs/superpowers/specs/` (rationale per module), and implementation plans in `docs/superpowers/plans/`.

## Roadmap

- Verify `poll_fills` (`ka10076`) and `open_orders` buy/sell discriminator fields via live paper-account calls; keep `fills_verified=False` until then.
- Persist `OrderManager` state across restarts (`own_orders`, partial-exit cumulative PnL, kill-switch state) and resolve the `execution.Trade` / `backtest.Trade` naming collision.
- Continue calibrating `backtest.lob.queue` against live order fills; add ETF tick-size table once canonicalized in `kiwoom-client`.
- Migrate consumer repos' live execution and data paths onto core incrementally as numerical parity is proven.
