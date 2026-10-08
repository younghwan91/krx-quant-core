"""backtest.lob._window — 단조 덱 미래 창 극값이 무식한 nanmax/nanmin 과 같은지(nan·짧은 배열·긴 창)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from krx_quant_core.backtest.lob._window import forward_extreme


def _brute(x, lo, h, largest):
    n = len(x)
    out = np.full(n, np.nan)
    f = np.nanmax if largest else np.nanmin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in range(n):
            w = x[t + lo : t + lo + h]
            if len(w):
                out[t] = f(w)
    return out


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("lo", [0, 1, 2])
@pytest.mark.parametrize("h", [1, 2, 3, 7, 50, 400])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_bruteforce(largest, lo, h, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 300))
    x = np.round(rng.normal(0, 3, n))  # 동점이 많게
    x[rng.random(n) < 0.3] = np.nan
    if seed == 2:
        x[: n // 2] = np.nan  # 긴 nan 구간
    got = forward_extreme(x, lo, h, largest=largest)
    np.testing.assert_array_equal(got, _brute(x, lo, h, largest))


def test_edges():
    assert forward_extreme(np.empty(0), 2, 5, largest=True).shape == (0,)
    assert np.isnan(forward_extreme(np.ones(3), 0, 0, largest=True)).all()
    with pytest.raises(ValueError):
        forward_extreme(np.ones(3), -1, 2, largest=True)
