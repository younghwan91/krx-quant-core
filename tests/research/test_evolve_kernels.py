"""research.evolve.kernels — swing-it ga_fast 원본·numpy 와 비트 동일."""

from __future__ import annotations

import numpy as np
import pytest

from krx_quant_core.research.evolve import kernels as K

M32 = 0xFFFFFFFF


# --- swing-it research/signals/ga_fast.py 원본 사본(순수 파이썬) ---
def _h32_ref(x):
    x = x & M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x5BD1E995) & M32
    x ^= x >> 16
    return x


def _block_boot_ref(monthly, block, n_boot, seed):
    M = monthly.shape[0]
    out = np.empty(n_boot)
    nstart = M - block + 1
    nb = (M + block - 1) // block
    hs = _h32_ref(seed * 0x2545F491 + 0x1B873593)
    for b in range(n_boot):
        hb = _h32_ref(hs ^ _h32_ref(b + 0x27D4EB2F))
        s = 0.0
        cnt = 0
        for k in range(nb):
            u = _h32_ref(hb ^ _h32_ref(k * 3 + 1))
            st = (u * nstart) >> 32
            for j in range(block):
                if cnt < M:
                    s += monthly[st + j]
                    cnt += 1
        out[b] = s / M * 12.0
    return out


def test_h32_matches_reference():
    for x in [0, 1, 7, 2**31 - 1, 2**32 - 1, 123456789, 0x2545F491 * 3 + 0x1B873593]:
        assert int(K.h32(x)) == _h32_ref(x)


@pytest.mark.parametrize("seed", [0, 7_919, 20261018])
def test_block_boot_bit_identical(seed):
    rng = np.random.default_rng(seed)
    for m in (6, 13, 48, 77):
        mo = rng.normal(0.002, 0.03, m)
        ref = _block_boot_ref(mo, 3, 200, seed)
        got = K.block_boot(mo, 3, 200, seed)
        assert np.array_equal(ref, got)


def test_percentile_linear_bit_identical_random():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        n = int(rng.integers(1, 700))
        x = rng.normal(size=n) * 10 ** rng.uniform(-4, 2)
        if rng.random() < 0.2:
            x = np.round(x, 2)                         # 동률
        q = float(rng.choice([2.5, 5.0, 50.0, 97.5, 0.0, 100.0, rng.uniform(0, 100)]))
        assert K.percentile_linear(x, q) == np.percentile(x, q)


def test_percentile_nan_and_empty():
    assert np.isnan(K.percentile_linear(np.array([1.0, np.nan, 2.0]), 2.5))
    with pytest.raises(ValueError):
        K.percentile_linear(np.zeros(0), 2.5)


def test_block_boot_lower_equals_numpy_composition():
    rng = np.random.default_rng(3)
    for m in (6, 24, 96):
        mo = rng.normal(0.0, 0.05, m)
        want = float(np.percentile(_block_boot_ref(mo, 3, 500, 0), 2.5))
        assert K.block_boot_lower(mo, 3, 500, 0) == want


def test_monthly_sums_matches_unique_bincount():
    rng = np.random.default_rng(5)
    month_of = np.repeat(np.arange(40), rng.integers(15, 23, 40))
    x = rng.normal(0, 0.01, 300)
    t_lo = 57
    mo = month_of[t_lo:t_lo + len(x)]
    u, inv = np.unique(mo, return_inverse=True)
    want = np.bincount(inv, weights=x)
    got, ids = K.monthly_sums(x, month_of, t_lo)
    assert np.array_equal(got, want)
    assert np.array_equal(ids, u)


def test_monthly_sums_rejects_unsorted():
    with pytest.raises(ValueError):
        K.monthly_sums(np.ones(3), np.array([0, 2, 1]), 0)


def test_monthly_block_lower_short_and_full():
    month_of = np.repeat(np.arange(30), 20)
    x = np.random.default_rng(0).normal(0, 0.01, 100)          # 5개월
    assert K.monthly_block_lower(x, month_of, 0, 3, 500, 0) == -1.0
    x = np.random.default_rng(0).normal(0, 0.01, 400)
    mo, _ = K.monthly_sums(x, month_of, 0)
    want = float(np.percentile(_block_boot_ref(mo, 3, 500, 0), 2.5))
    assert K.monthly_block_lower(x, month_of, 0, 3, 500, 0) == want


@pytest.mark.skipif(not K.HAVE_NUMBA, reason="numba 없음")
def test_numba_and_python_paths_identical():
    mo = np.random.default_rng(9).normal(0, 0.04, 50)
    py = K.block_boot.py_func(mo, 3, 300, 11)
    assert np.array_equal(py, K.block_boot(mo, 3, 300, 11))
    assert K._percentile.py_func(py, 2.5) == K._percentile(py, 2.5)


def test_nb_aliases_callable_from_njit():
    """소비자 njit 커널 안에서 부를 수 있다(numba 가 있으면 실제 njit 안에서)."""
    rng = np.random.default_rng(4)
    month_of = np.repeat(np.arange(30), 20)
    x = rng.normal(0, 0.01, 400)

    def user(x, month_of):
        sums = np.empty(x.shape[0])
        ids = np.empty(x.shape[0], np.int64)
        k = K.monthly_sums_nb(x, month_of, 0, sums, ids)
        return K.block_boot_lower_nb(sums[:k], 3, 500, 0, 2.5), K.percentile_nb(sums[:k], 50.0)

    if K.HAVE_NUMBA:
        import numba
        user = numba.njit(user)
    lo, med = user(x, month_of)
    mo, _ = K.monthly_sums(x, month_of, 0)
    assert lo == K.block_boot_lower(mo, 3, 500, 0) and med == np.percentile(mo, 50.0)
