"""stats.metrics — swing-it ``tests/test_engine_metrics.py`` 이식 + scalp-it 페어드 부트스트랩.

각 지표를 알려진 답과 경계(빈 배열·전부 NaN·원소 1개·변동 0)로 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.stats.metrics import (
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

_MONTHS = [f"20{y:02d}-{m:02d}" for y in range(18, 24) for m in range(1, 13)]  # 72 months


def _series(mean: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, vol, len(_MONTHS)), index=_MONTHS, name="ret")


# --- ann_sharpe ----------------------------------------------------------------

def test_ann_sharpe_known_value():
    r = np.array([0.01, -0.01, 0.02, 0.00, 0.03])
    expected = r.mean() / r.std() * np.sqrt(12)
    assert abs(ann_sharpe(r) - expected) < 1e-12


def test_ann_sharpe_zero_vol_is_nan():
    assert np.isnan(ann_sharpe(np.full(12, 0.01)))


def test_ann_sharpe_empty_is_nan():
    assert np.isnan(ann_sharpe(np.array([])))


def test_ann_sharpe_single_element_is_nan():
    assert np.isnan(ann_sharpe(np.array([0.05])))


def test_ann_sharpe_drops_nan_and_honors_ppy():
    r = np.array([0.01, np.nan, -0.01, 0.02])
    assert abs(ann_sharpe(r, ppy=252) - ann_sharpe(np.array([0.01, -0.01, 0.02]), ppy=252)) < 1e-12


# --- cagr -----------------------------------------------------------------------

def test_cagr_known_value():
    r = np.full(12, 0.01)  # +1%/mo, 12 months
    assert abs(cagr(r) - (1.01 ** 12 - 1)) < 1e-9


def test_cagr_empty_is_nan():
    assert np.isnan(cagr(np.array([])))


def test_cagr_all_nan_is_nan():
    assert np.isnan(cagr(np.array([np.nan, np.nan])))


def test_cagr_single_element():
    # one +10% period annualizes to (1.1)^(12/1) - 1
    assert abs(cagr(np.array([0.10])) - (1.10 ** 12 - 1.0)) < 1e-9


# --- max_drawdown ---------------------------------------------------------------

def test_max_drawdown_known():
    # +10%, -50%, +10%: peak 1.1 then trough 0.55 -> dd = 0.55/1.1 - 1 = -0.5
    assert abs(max_drawdown(np.array([0.1, -0.5, 0.1])) - (-0.5)) < 1e-12


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown(np.array([0.01, 0.02, 0.03])) == 0.0


def test_max_drawdown_empty_is_nan():
    assert np.isnan(max_drawdown(np.array([])))


# --- newey_west_t ---------------------------------------------------------------

def test_newey_west_t_zero_lag_matches_plain_t():
    x = np.array([0.02, 0.01, -0.01, 0.03, 0.00, 0.02])
    mu, t = newey_west_t(x, lag=0)
    se = x.std(ddof=0) / np.sqrt(len(x))  # var = (d@d)/n is the population var
    assert abs(mu - x.mean()) < 1e-12
    assert abs(t - x.mean() / se) < 1e-9


def test_newey_west_t_insufficient_returns_nan():
    mu, t = newey_west_t(np.array([0.01, 0.02]), lag=5)
    assert np.isnan(mu) and np.isnan(t)


def test_newey_west_t_drops_nan():
    x = np.array([0.02, np.nan, 0.01, -0.01, 0.03])
    m1 = newey_west_t(x, lag=1)
    m2 = newey_west_t(np.array([0.02, 0.01, -0.01, 0.03]), lag=1)
    assert m1 == m2



# --- summarize_periods ----------------------------------------------------------

def test_summarize_empty_returns_zero_n():
    s = summarize_periods(pd.DataFrame(columns=["net", "turnover"]), horizon=20)
    assert s["n"] == 0
    assert np.isnan(s["sharpe"])


def test_summarize_known_fields():
    periods = pd.DataFrame({"net": [0.02, -0.01, 0.03, 0.00], "turnover": [0.5, 0.5, 0.5, 0.5]})
    s = summarize_periods(periods, horizon=20)
    assert s["n"] == 4
    assert abs(s["mean_net"] - 0.01) < 1e-12
    assert abs(s["hit_rate"] - 0.5) < 1e-12
    assert abs(s["avg_turnover"] - 0.5) < 1e-12


def test_summarize_payoff_ratio():
    periods = pd.DataFrame({"net": [0.04, -0.02], "turnover": [1.0, 1.0]})
    s = summarize_periods(periods, horizon=20)
    assert abs(s["payoff_ratio"] - (0.04 / 0.02)) < 1e-12
    assert abs(s["best"] - 0.04) < 1e-12
    assert abs(s["worst"] - (-0.02)) < 1e-12



# --- spearman -------------------------------------------------------------------

def test_spearman_monotonic():
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([10, 20, 30, 40])
    assert spearman(a, b) == 1.0
    assert spearman(a, -b) == -1.0


def test_spearman_single_element_is_nan():
    assert np.isnan(spearman(pd.Series([1.0]), pd.Series([2.0])))



# --- quantile_summary -----------------------------------------------------------

def _merged(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"score": rng.normal(0, 1, n), "fwd_ret": rng.normal(0, 0.05, n)})


def test_quantile_summary_buckets():
    out = quantile_summary(_merged(50), 5)
    assert list(out.columns) == ["quantile", "n", "mean_fwd", "hit_rate"]
    assert len(out) == 5
    assert out["n"].sum() == 50


def test_quantile_summary_too_few_returns_empty():
    out = quantile_summary(_merged(3), 5)
    assert out.empty
    assert list(out.columns) == ["quantile", "n", "mean_fwd", "hit_rate"]



# --- paired_bootstrap -----------------------------------------------------------

def test_paired_bootstrap_detects_clear_winner():
    strong = _series(0.015, 0.03, seed=1)
    weak = _series(0.000, 0.03, seed=2)
    res = paired_bootstrap(strong, weak, n_boot=500, seed=0)
    assert res["d_sharpe_ci"][0] > 0
    assert res["prob_a_better_sharpe"] > 0.9


def test_paired_bootstrap_ties_include_zero():
    a = _series(0.005, 0.03, seed=3)
    b = _series(0.005, 0.03, seed=4)
    lo, hi = paired_bootstrap(a, b, n_boot=500, seed=0)["d_sharpe_ci"]
    assert lo < 0 < hi


def test_paired_bootstrap_too_short_returns_nan():
    a = pd.Series([0.01, 0.02, 0.03])
    res = paired_bootstrap(a, a, block=6, n_boot=100, seed=0)
    assert np.isnan(res["d_sharpe_ci"][0]) and res["n"] == 3


# --- regime_buckets -------------------------------------------------------------

def test_regime_buckets_sign_count():
    r = pd.Series([0.02] * 18 + [-0.01] * 18 + [0.03] * 18 + [0.01] * 18, index=_MONTHS)
    regs = regime_buckets(r, n=4)
    assert len(regs) == 4
    assert sum(x["positive"] for x in regs) == 3


def test_regime_buckets_too_few_returns_empty():
    assert regime_buckets(pd.Series([0.01, 0.02]), n=4) == []


def test_max_drawdown_counts_first_period_loss():
    """초기자본 1.0 을 peak 로 세우지 않으면 첫 구간 손실이 0 으로 읽힌다(swing-it 실측 반례)."""
    assert abs(max_drawdown(np.array([-0.20, 0.05, 0.05])) - (-0.20)) < 1e-12


# --- paired_date_bootstrap (scalp-it validate/compare.py 이식) -------------------

def _portfolio(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(values), freq="D"),
            "n_positions": [15] * len(values),
            "mean_excess": values,
        }
    )


def test_paired_date_bootstrap_detects_a_real_difference():
    base = np.random.default_rng(0).normal(scale=0.01, size=200)
    out = paired_date_bootstrap(
        _portfolio(list(base + 0.02)), _portfolio(list(base)), n_boot=1000
    )
    assert np.isclose(out["mean_diff"], 0.02, atol=0.005)
    assert out["ci_low"] > 0
    assert out["p_greater"] > 0.95
    assert out["n_dates"] == 200


def test_paired_date_bootstrap_finds_nothing_when_series_are_identical():
    values = list(np.random.default_rng(1).normal(scale=0.01, size=200))
    out = paired_date_bootstrap(_portfolio(values), _portfolio(values), n_boot=1000)
    assert out["mean_diff"] == 0.0
    assert out["ci_low"] == out["ci_high"] == 0.0


def test_paired_date_bootstrap_pairs_on_date_not_position():
    a = _portfolio([0.1, 0.2, 0.3])
    b = _portfolio([0.1, 0.2, 0.3]).iloc[::-1].reset_index(drop=True)
    assert paired_date_bootstrap(a, b, n_boot=200)["mean_diff"] == 0.0


def test_paired_date_bootstrap_rejects_disjoint_dates():
    a = _portfolio([0.1, 0.2])
    b = _portfolio([0.1, 0.2])
    b["date"] = b["date"] + pd.Timedelta(days=365)
    with pytest.raises(ValueError, match="공통 날짜"):
        paired_date_bootstrap(a, b, n_boot=100)


def test_paired_date_bootstrap_is_seed_deterministic():
    rng = np.random.default_rng(7)
    a, b = _portfolio(list(rng.normal(size=50))), _portfolio(list(rng.normal(size=50)))
    assert paired_date_bootstrap(a, b, seed=3) == paired_date_bootstrap(a, b, seed=3)
