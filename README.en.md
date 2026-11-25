<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="krx-quant-core" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

# krx-quant-core

**Korean stock tick sizes, price limits, and transaction taxes, computed correctly for each trade date.**

A shared core where KOSPI/KOSDAQ quant systems align on market rules in one place. Upper and lower price limits are asymmetric, transaction taxes vary by effective date, and out-of-sample windows open only once. No strategies live here.

[![CI](https://github.com/younghwan91/krx-quant-core/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/krx-quant-core/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/krx-quant-core.svg)](https://pypi.org/project/krx-quant-core/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

[Quick start](#quick-start) · [Reference Guide](docs/guide.en.md) · [한국어](README.md) · [Issues](https://github.com/younghwan91/krx-quant-core/issues)

</div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero-dark.png">
  <img alt="KRX upper and lower price limits are asymmetric — tick-size truncation across 1,000–1,000,000 KRW base prices skewing ±30%, and securities transaction tax steps by effective date" src="docs/images/hero-light.png" width="100%">
</picture>

<p align="center"><sub>Asymmetric upper/lower price limit skew by base-price band and securities transaction tax steps by effective date</sub></p>

- **Upper and lower limits are asymmetric.** From a 154,200 KRW base price, limit-up is 200,000 KRW (+29.70%) and limit-down is 108,000 KRW (−29.96%). Across 130 `daily_bars` rows where rules diverge, the symmetric formula matched 25 rows while the asymmetric formula matched 105 ([empirical check](docs/guide.en.md#price-limits--empirical-verification)).
- **Transaction taxes depend on the date.** Six steps from 0.25% in 2019 to 0.15% in 2025 and 0.20% in 2026. Cost functions take the trade date and apply that day's rate.
- **Out-of-sample windows open once.** The run ledger locks OOS ranges in git, and final evaluation can be started only once per label. Retries are refused.

## Quick start

```bash
pip install krx-quant-core==0.9.0            # extras: [fast] numba · [opt] optuna · [db] psycopg · [evo] cmaes
```

```python
from datetime import date
from krx_quant_core.market import Market, limit_up_price, limit_down_price, shift_ticks
from krx_quant_core.costs import round_trip_cost, tax_rate

limit_up_price(24_250)          # 31500
limit_down_price(239_000)       # 167500.0
shift_ticks(49_950, 2)          # Decimal('50100') — valid tick even across band boundaries
tax_rate(date(2025, 6, 2), Market.KOSPI).total   # Decimal('0.0015')
round_trip_cost(date(2026, 9, 14), Market.KOSDAQ, slippage_one_way=0.0015)  # 0.0053
```

Requires Python 3.11 or later. Required dependencies are only `kiwoom-client` (canonical tick-size table), `numpy`, and `pandas`. Without `[fast]`, the same kernels run in pure Python and produce identical numbers.

```bash
# If you need unreleased features, pin a commit on main:
pip install "krx-quant-core @ git+https://github.com/younghwan91/krx-quant-core@<commit-sha>"
```

## Why It Matters

These are the pitfalls that quietly skew Korean backtests and order code. Each one is a single function call here.

| Pitfall | Hand-rolled result | This package |
|---|---|---|
| One constant for transaction tax | The sell-side rate was 0.18% in 2024, 0.15% in 2025, and 0.20% in 2026. Running 2026 trades at the 2025 rate understates every sell by 0.05%p | `tax_rate(date, market)` picks that day's rate from the effective-date table |
| Lower limit computed symmetrically | From a 239,000 KRW base, the symmetric formula (base × 0.7 rounded up to that band's tick) gives 167,300 KRW, but the real limit-down is 167,500 KRW. Of 130 empirical rows where the rules diverge, it matched only 25 | `limit_down_price` truncates the width with the base price's tick (matched 105) |
| Moving n ticks as "price + n × current tick" | Two ticks above 49,950 KRW is not 50,050 KRW: at 50,000 KRW and up the tick is 100 KRW, so the order is rejected | `shift_ticks(49_950, 2)` crosses the band boundary and returns 50,100 |
| Copying the tick table into each repo | One boundary off by one means rejected orders or wrong limit-up calls | The canonical table lives in [kiwoom-client](https://github.com/younghwan91/kiwoom-client), which places real orders; this package only derives from it |
| Treating tickers as numbers | Stripping the new alphanumeric code `0155E0` to digits yields a different company | `normalize_code("0155E0")` keeps `0155E0`; `normalize_code("5930")` returns `005930` |

**Scope and sources:**

- The tick table is the one after the 2023-01-25 tick-size reform (unified across KOSPI, KOSDAQ, KONEX). Low KOSDAQ bands differed before that date, so it does not fit pre-2023-01-25 data. Sources are KRX reform notices and broker notices (`kiwoom_client.tick_size` docstring).
- Price limits follow the ±30% rule in force since 2015-06-15. On days when the base price is not the previous close (ex-rights, new listings), the caller must pass the base price.
- The 2026-01-01 tax increase was confirmed from reporting on the Ministry of Economy and Finance decree amendment. Earlier steps come from secondary sources and have not yet been checked against the statute text (law.go.kr). The KONEX rate is unverified and modeled the same as KOSDAQ ([tax table](docs/guide.en.md#transaction-tax-schedule-on-sell)).

## What's Inside

| Module | Purpose |
|---|---|
| `market/` | Ticker normalization & `Market`, tick sizes (scalar, vectorized, ETF), price limits, KST sessions, trading calendar |
| `costs/` | Effective-date tax schedule, `KoreanCostModel` (`Decimal`), `round_trip_cost` (`float`), `impact` (square-root impact) |
| `execution/` | `OrderManager`, `InstanceLock`, `PositionBook`, `Broker` (`PaperBroker`, `KiwoomBroker`), `EngineCore`, `OrderGuard` |
| `risk/` | DART material disclosure classification & `RiskGate`, `DartDisclosureDB`, `KillSwitch`, `factor_model`, `attribution` |
| `backtest/` | Order-book sweep VWAP & round-trip cost, limit-order fill rules, trade-ledger metrics, cross-sectional sim, `replay`, `drift`, `ticktime` |
| `backtest/lob/` | Ticks & quotes → 1s grid features & paths (shared batch/live kernel), episode sim, random control, limit-order queue model, `ceiling` |
| `data/` | Single `COPY` read from shared DB into columnar `.npy` + `mmap`, daily/minute/tick/quote cache, regular-session base-price recovery, `risk_inputs.load_risk_inputs` |
| `stats/` | Deflated/Probabilistic Sharpe, t-haircut, purged walk-forward, bootstrap, fragility, selection bias, stratified matched control, `ic` |
| `portfolio/` | `optimize` — constrained mean-variance optimization over the factor risk model (requires `[portfolio]` extra) |
| `research/` | `run_sweep` (grid, parallel, cached), `optuna_search`, `evolve` (BIPOP CMAwM, generation protocol, paired racing, sealing) |
| `runtime/` | Host guard, `start_run` ledger, OOS hard lock, `kqc` CLI (remote execution, `nightly`, `pins`) |

Each strategy repository pins an exact `krx-quant-core==X.Y.Z` release.

| Consumer | Style | Execution | Used from this package |
|---|---|---|---|
| [scalp-it](https://github.com/younghwan91/scalp-it) | Scalping | Automated live | Ticks, price limits, order guard, book sweep, fill rules, host guard |
| [daytrade-it](https://github.com/younghwan91/daytrade-it) | Day trading | Automated live | Cost model, disclosure risk gate, DART DB, code normalization, host guard |
| [swing-it](https://github.com/younghwan91/swing-it) | Swing | Research + manual | DSR, purged CV, bootstrap, fragility diagnostics, cross-sectional sim, evolution engine, optimization |

## Why Extract It

Three repositories each maintained their own copies of the same rules, and they had drifted apart.

- **Divergent ticker normalization**: `normalize_code` was copy-pasted across two repos; one old implementation turned `0155E0` into `001550` (a completely different company).
- **Scattered cost constants**: Transaction costs varied across repos (`0.0023`, `0.0034`, `0.0064`, `50bp`), and no code applied the 2023–2026 tax steps by date.
- **Inverted fill assumptions**: When given an unknown value, one place treated it as `through` while another in the same repo treated it as `touch`.
- **Missing lower limits**: There was no lower-limit calculation; symmetric formulas get it wrong.

Four design principles govern the core: **numerical parity** on migration against original code, **no pass/fail judgments** from statistical functions, **explicitly flagging unverified broker fields** (`False` defaults), and **never revoking shared app-key tokens** on disconnect.

## Works Well With

| Repo | What it does |
|---|---|
| [kiwoom-client](https://github.com/younghwan91/kiwoom-client) | Python client for the Kiwoom REST API — 182 endpoints, real-time WebSocket, MCP server; holds the canonical tick table this package uses |
| [krx-news-client](https://github.com/younghwan91/krx-news-client) | Korean stock news and DART disclosures in one schema |
| [kiwoom-order-guard](https://github.com/younghwan91/kiwoom-order-guard) | Claude Code hook that checks orders an AI agent sends through Kiwoom MCP/plugins before they go out |
| [fin-checkup](https://github.com/younghwan91/fin-checkup) | Traffic-light view of 17 financial metrics for Korean listed companies, each with its formula and thresholds; measurements only, not investment advice |

## Docs

| Doc | Contents |
|---|---|
| [Reference Guide](docs/guide.en.md) | Price limits check, tax schedule, run ledger & OOS lock, live safety rules, LOB fill assumptions, edge evaluation axes, data layer, search & evolution, development, roadmap |
| [Design Specs](docs/superpowers/specs/) | Rationale and design decisions per module |
| [Implementation Plans](docs/superpowers/plans/) | Version-by-version implementation plans |
| [Release Process Diagram](docs/diagrams/release-process.html) | Interactive release and pin-sync workflow |

## Contributing & Support

- Report bugs, ask questions, or propose features via [GitHub Issues](https://github.com/younghwan91/krx-quant-core/issues).
- See the [Development section in the Reference Guide](docs/guide.en.md#development) for local setup and test commands.
- **Author**: **Younghwan Chae (채영환)** · [GitHub @younghwan91](https://github.com/younghwan91) · [LinkedIn](https://www.linkedin.com/in/younghwan-chae/)

## License

[Apache-2.0](LICENSE)
