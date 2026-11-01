"""stats.ic — rank_ic 재사용 동일성(완료 조건 3)·ICIR·이동 IC·회전율."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.backtest.crosssectional import rank_ic
from krx_quant_core.stats import ic_decay, rolling_ic, signal_turnover

_VERDICT_KEYS = {"pass", "fail", "passed", "failed", "ok", "verdict",
                 "go", "nogo", "deploy", "accept", "reject"}


def _panels(n_days=400, n_codes=60, drift=0.002, seed=0, noise=0.0):
    """tests/backtest/test_backtest_crosssectional.py 의 합성 패널 + 신호 잡음 선택."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()
    C = np.empty((n_codes, n_days))
    sig = np.empty((n_codes, n_days))
    for i in range(n_codes):
        rank = i / (n_codes - 1)
        rets = drift * (rank - 0.5) + rng.normal(0, 0.01, n_days)
        C[i] = 1000 * np.cumprod(1 + rets)
        sig[i] = rank
    if noise:
        sig = sig + rng.normal(0, noise, sig.shape)
    V = np.full((n_codes, n_days), 1e5)
    V *= rng.uniform(0.5, 1.5, V.shape)
    return C, V, sig, dates


def test_h40_row_identical_to_rank_ic():
    C, V, sig, dates = _panels(noise=0.3)
    kw = dict(adv_floor=60000.0, start_index=40)
    dec = ic_decay(sig, C, dates, trade_value=V, **kw)
    ref = rank_ic(C, V, sig, dates, horizon=40, **kw)
    row = dec.loc[dec["h"] == 40].iloc[0]
    assert row["ic_mean"] == ref["ic_mean"]
    assert row["nw_t"] == ref["ic_nw_t"]
    assert row["n"] == ref["n_days"]
    assert np.float64(row["ic_mean"]).tobytes() == np.float64(ref["ic_mean"]).tobytes()
    assert np.float64(row["nw_t"]).tobytes() == np.float64(ref["ic_nw_t"]).tobytes()


def test_every_horizon_matches_rank_ic_and_icir_is_mean_over_sd():
    C, V, sig, dates = _panels(noise=0.3)
    dec = ic_decay(sig, C, dates, adv_floor=0, start_index=40)
    assert list(dec.columns) == ["h", "ic_mean", "ic_sd", "icir", "nw_t", "n"]
    assert dec["h"].tolist() == [1, 5, 10, 20, 40, 60]
    for _, row in dec.iterrows():
        ref = rank_ic(C, np.zeros_like(C), sig, dates, horizon=int(row["h"]),
                      adv_floor=0, start_index=40, return_series=True)
        assert row["ic_mean"] == ref["ic_mean"] and row["nw_t"] == ref["ic_nw_t"]
        assert row["ic_sd"] == float(np.std(ref["ic"], ddof=1))
        assert row["icir"] == row["ic_mean"] / row["ic_sd"]
    assert (dec["ic_mean"] > 0).all()


def test_rank_ic_default_output_unchanged():
    C, V, sig, dates = _panels()
    base = rank_ic(C, V, sig, dates, horizon=20, adv_floor=0, start_index=40)
    ext = rank_ic(C, V, sig, dates, horizon=20, adv_floor=0, start_index=40, return_series=True)
    assert set(base) == {"ic_mean", "ic_nw_t", "n_days", "frac_positive", "regimes"}
    assert {k: ext[k] for k in base} == base
    assert len(ext["ic"]) == base["n_days"] == len(ext["ic_dates"])
    assert float(np.mean(ext["ic"])) == base["ic_mean"]


def test_adv_floor_without_trade_value_raises():
    C, _, sig, dates = _panels(n_days=200)
    with pytest.raises(ValueError):
        ic_decay(sig, C, dates, adv_floor=5000.0)
    with pytest.raises(TypeError):
        ic_decay(sig, C, dates, horizon=5)


def test_thin_universe_gives_nan_not_error():
    C, _, sig, dates = _panels(n_days=200, n_codes=10)
    dec = ic_decay(sig, C, dates, horizons=(5,), start_index=40)
    assert dec["n"].iloc[0] == 0 and math.isnan(dec["ic_mean"].iloc[0])
    assert math.isnan(dec["ic_sd"].iloc[0]) and math.isnan(dec["icir"].iloc[0])


def test_rolling_ic_non_overlapping_samples():
    C, _, sig, dates = _panels(noise=0.3)
    h, win = 20, 6
    out = rolling_ic(sig, C, dates, horizon=h, window=win, start_index=40)
    assert list(out.columns) == ["date", "ic", "ic_mean", "ic_sd", "icir", "n"]
    pos = [dates.index(d) for d in out["date"]]
    assert all(b - a >= h for a, b in zip(pos[:-1], pos[1:], strict=True))
    ref = rank_ic(C, np.zeros_like(C), sig, dates, horizon=h, adv_floor=0,
                  start_index=40, return_series=True)
    lookup = dict(zip(ref["ic_dates"], ref["ic"], strict=True))
    assert all(out["ic"].iloc[j] == lookup[d] for j, d in enumerate(out["date"]))
    assert out["ic_mean"].iloc[: win - 1].isna().all()
    j = win + 2
    seg = out["ic"].iloc[j - win + 1: j + 1].to_numpy()
    assert out["ic_mean"].iloc[j] == pytest.approx(seg.mean(), abs=1e-15)
    assert out["ic_sd"].iloc[j] == pytest.approx(seg.std(ddof=1), abs=1e-15)
    assert out["icir"].iloc[j] == out["ic_mean"].iloc[j] / out["ic_sd"].iloc[j]
    assert out["n"].iloc[j] == win


def test_turnover_constant_signal_is_zero():
    _, _, sig, dates = _panels(n_days=50)
    tov = signal_turnover(sig, top_q=0.2, dates=dates)
    assert tov.name == "turnover" and tov.index.tolist() == dates
    assert math.isnan(tov.iloc[0])
    assert (tov.iloc[1:] == 0.0).all()


def test_turnover_reshuffled_signal_is_high():
    rng = np.random.default_rng(1)
    N, T = 500, 40
    S = np.column_stack([rng.permutation(N).astype(float) for _ in range(T)])
    tov = signal_turnover(S, top_q=0.2)
    # 무작위 재배치의 기대 교체율 = 1 − top_q = 0.8
    assert tov.iloc[1:].mean() == pytest.approx(0.8, abs=0.05)
    assert tov.iloc[1:].min() > 0.6


def test_turnover_known_swap_and_nan_handling():
    S = np.array([[5.0, 5.0, np.nan], [4.0, 1.0, np.nan], [3.0, 4.0, np.nan],
                  [2.0, 3.0, np.nan], [1.0, 2.0, np.nan]])
    tov = signal_turnover(S, top_q=0.4)               # k = 2
    assert tov.iloc[1] == 0.5                          # {0,1} → {0,2}
    assert math.isnan(tov.iloc[2])                     # 빈 구성


def test_outputs_have_no_verdict_keys():
    C, _, sig, dates = _panels(n_days=200)
    frames = [ic_decay(sig, C, dates, horizons=(5,), start_index=40),
              rolling_ic(sig, C, dates, horizon=5, start_index=40)]
    for df in frames:
        assert not {c.lower() for c in df.columns} & _VERDICT_KEYS
        assert not any(df[c].dtype == bool for c in df.columns)
