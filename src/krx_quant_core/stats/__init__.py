"""검증 통계 — Deflated Sharpe·성과지표·워크포워드/purged CV·취약성·시행 원장.

전부 순수 함수(numpy/pandas 만)라 DB·브로커·시계를 모른다. 세 소비 레포가 같은 수익열에서
같은 숫자를 내게 하는 게 이 패키지의 존재 이유다.
"""

from .cv import (
    Fold,
    FoldMask,
    entry_mask,
    fold_consistency,
    fold_slices,
    oos_fixed,
    purge_embargo,
    rdist,
    resolve_exit_dates,
    rolling_folds,
    slice_by_entry,
    walk_forward,
)
from .fragility import (
    fragility_report,
    max_loss_streak,
    median_trade,
    monster_share,
    tail_removal,
    win_conditional,
)
from .metrics import (
    PPY,
    ann_sharpe,
    cagr,
    max_drawdown,
    newey_west_t,
    paired_bootstrap,
    paired_date_bootstrap,
    quantile_summary,
    regime_buckets,
    spearman,
    summarize_periods,
)
from .sharpe import (
    bootstrap_mean_ci,
    deflated_sharpe,
    deflated_sharpe_from_sample,
    expected_max_sharpe_h0,
    probabilistic_sharpe,
    t_haircut,
)
from .trials import config_fingerprint, count_trials, ledger_path, read_trials, record_trial

__all__ = [
    "PPY",
    "Fold",
    "FoldMask",
    "ann_sharpe",
    "bootstrap_mean_ci",
    "cagr",
    "config_fingerprint",
    "count_trials",
    "deflated_sharpe",
    "deflated_sharpe_from_sample",
    "entry_mask",
    "expected_max_sharpe_h0",
    "fold_consistency",
    "fold_slices",
    "fragility_report",
    "ledger_path",
    "max_drawdown",
    "max_loss_streak",
    "median_trade",
    "monster_share",
    "newey_west_t",
    "oos_fixed",
    "paired_bootstrap",
    "paired_date_bootstrap",
    "probabilistic_sharpe",
    "purge_embargo",
    "quantile_summary",
    "rdist",
    "read_trials",
    "record_trial",
    "regime_buckets",
    "resolve_exit_dates",
    "rolling_folds",
    "slice_by_entry",
    "spearman",
    "summarize_periods",
    "t_haircut",
    "tail_removal",
    "walk_forward",
    "win_conditional",
]
