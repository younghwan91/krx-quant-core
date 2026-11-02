"""risk.attribution — 귀속 항등식(완료 조건 3)·단일 팩터 검산·결측 규약."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.risk.attribution import factor_attribution


def _random(N=80, T=60, K=7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, N, K))
    f = rng.normal(0, 0.01, size=(T, K))
    u = rng.normal(0, 0.02, size=(N, T))
    w = rng.dirichlet(np.ones(N), size=T).T          # [N, T], 열 합 1
    w[rng.random((N, T)) < 0.3] = 0.0                # 일부는 안 쥠
    names = [f"k{k}" for k in range(K)]
    dates = pd.bdate_range("2021-01-04", periods=T).strftime("%Y-%m-%d").tolist()
    return w, X, f, u, names, dates


@pytest.mark.parametrize("seed", range(5))
def test_total_equals_actual_portfolio_return_1e12(seed):
    w, X, f, u, names, dates = _random(seed=seed)
    out = factor_attribution(w, X, f, u, names=names, dates=dates)
    r = np.einsum("tnk,tk->nt", X, f) + u            # r_t = X_{t-1} f_t + u_t
    actual = (w * r).sum(axis=0)
    assert list(out.columns) == ["date", *names, "specific", "total"]
    assert out["date"].tolist() == dates
    assert np.max(np.abs(out["total"].to_numpy() - actual)) <= 1e-12
    parts = out[[*names, "specific"]].to_numpy().sum(axis=1)
    np.testing.assert_array_equal(parts, out["total"].to_numpy())


def test_sequence_of_per_date_exposures_matches_stacked():
    w, X, f, u, names, dates = _random(seed=7)
    a = factor_attribution(w, X, f, u, names=names, dates=dates)
    b = factor_attribution(w, [X[t] for t in range(X.shape[0])], f, u, names=names, dates=dates)
    pd.testing.assert_frame_equal(a, b)


def test_single_factor_sanity():
    # 시장 하나, 노출 전부 1, 특이 0 → 시장 기여 = 투자비중 × f, 특이 0.
    N, T = 10, 4
    X = np.ones((T, N, 1))
    f = np.array([[0.01], [-0.02], [0.0], [0.03]])
    u = np.zeros((N, T))
    w = np.full((N, T), 0.05)                        # 50% 투자
    out = factor_attribution(w, X, f, u, names=["market"], dates=list(range(T)))
    np.testing.assert_allclose(out["market"], 0.5 * f[:, 0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(out["specific"], np.zeros(T))
    np.testing.assert_allclose(out["total"], 0.5 * f[:, 0], rtol=0, atol=1e-15)


def test_specific_only_when_factor_returns_zero():
    w, X, f, u, names, dates = _random(seed=3)
    out = factor_attribution(w, X, np.zeros_like(f), u, names=names, dates=dates)
    assert (out[names].to_numpy() == 0).all()
    np.testing.assert_allclose(out["total"], (w * u).sum(axis=0), rtol=0, atol=1e-15)


def test_missing_on_unheld_names_ignored_on_held_propagates():
    w, X, f, u, names, dates = _random(seed=1)
    i0 = int(np.flatnonzero(w[:, 0] == 0)[0])
    X[0, i0, :] = np.nan
    u[i0, 0] = np.nan
    w2 = w.copy()
    w2[i0, 1] = np.nan                               # NaN 비중 = 안 쥠
    out = factor_attribution(w2, X, f, u, names=names, dates=dates)
    assert np.isfinite(out.loc[0, "total"])

    i1 = int(np.flatnonzero(w[:, 2] > 0)[0])
    u[i1, 2] = np.nan
    out = factor_attribution(w, X, f, u, names=names, dates=dates)
    assert np.isnan(out.loc[2, "specific"]) and np.isnan(out.loc[2, "total"])
    assert np.isfinite(out.loc[3, "total"])


def test_shape_and_name_validation():
    w, X, f, u, names, dates = _random(N=5, T=3, K=2)
    with pytest.raises(ValueError):
        factor_attribution(w, X[:, :, :1], f, u, names=names, dates=dates)
    with pytest.raises(ValueError):
        factor_attribution(w, X, f, u, names=["a"], dates=dates)
    with pytest.raises(ValueError):
        factor_attribution(w, X, f, u, names=["total", "b"], dates=dates)
    with pytest.raises(ValueError):
        factor_attribution(w, X, f, u, names=names, dates=dates[:2])
