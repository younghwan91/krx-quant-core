"""backtest.ticktime — scalp-it scripts/tick_sanity.py 원본 복사본과 비트 단위 동일."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from krx_quant_core.backtest.ticktime import forward_max_last, trailing_sums

# ---- 원본 복사본(scalp-it tick_sanity.py) ------------------------------------------------


def _orig_trailing(ts_sec, side, volume, strength, window):
    n = ts_sec.shape[0]
    mean_strength = np.full(n, np.nan)
    buy_ratio = np.full(n, np.nan)
    count = np.zeros(n, dtype=np.int64)
    buy_vol = np.where(side > 0, volume, 0).astype(np.float64)
    s_sum = v_sum = bv_sum = 0.0
    left = 0
    for i in range(n):
        s_sum += float(strength[i])
        v_sum += float(volume[i])
        bv_sum += float(buy_vol[i])
        lo = ts_sec[i] - window
        while ts_sec[left] <= lo:
            s_sum -= float(strength[left])
            v_sum -= float(volume[left])
            bv_sum -= float(buy_vol[left])
            left += 1
        c = i - left + 1
        count[i] = c
        if c > 0:
            mean_strength[i] = s_sum / c
        if v_sum > 0:
            buy_ratio[i] = bv_sum / v_sum
    return mean_strength, buy_ratio, count


def _orig_forward(ts_sec, price, horizon):
    n = ts_sec.shape[0]
    mfe = np.full(n, np.nan)
    close = np.full(n, np.nan)
    dq: deque[int] = deque()
    r = 0
    for i in range(n):
        hi = ts_sec[i] + horizon
        while r < n and ts_sec[r] <= hi:
            if r > i:
                while dq and price[dq[-1]] <= price[r]:
                    dq.pop()
                dq.append(r)
            r += 1
        while dq and dq[0] <= i:
            dq.popleft()
        entry = float(price[i])
        last_idx = r - 1
        if entry > 0 and last_idx > i:
            close[i] = price[last_idx] / entry - 1.0
            if dq:
                mfe[i] = price[dq[0]] / entry - 1.0
    return mfe, close


def _day(seed, n=4000):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(32400, 55200, n))  # 같은 초 다수
    price = (10000 + np.cumsum(rng.integers(-1, 2, n)) * 10).astype(np.int32)
    vol = rng.integers(1, 3000, n).astype(np.int64)
    side = rng.choice(np.array([-1, 1], np.int8), n)
    strength = rng.uniform(30, 250, n)
    return ts.astype(np.int32), price, vol, side, strength


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("window", [1, 10, 60, 600])
def test_trailing_matches_original(seed, window):
    ts, price, vol, side, stg = _day(seed)
    ms, br, cnt = _orig_trailing(ts, side, vol, stg, window)
    buy = np.where(side > 0, vol, 0).astype(np.float64)
    (s, v, bv), c = trailing_sums(ts, stg, vol, buy, window=window)
    np.testing.assert_array_equal(c, cnt)
    np.testing.assert_array_equal(np.where(c > 0, s / c, np.nan), ms)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_array_equal(np.where(v > 0, bv / v, np.nan), br)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("horizon", [1, 5, 30, 300])
def test_forward_matches_original(seed, horizon):
    ts, price, *_ = _day(seed)
    mfe, close = _orig_forward(ts, price, horizon)
    fmax, last = forward_max_last(ts, price, horizon)
    entry = price.astype(np.float64)
    ok = (entry > 0) & (last >= 0)
    got_close = np.full(len(ts), np.nan)
    got_close[ok] = price[last[ok]] / entry[ok] - 1.0
    got_mfe = np.full(len(ts), np.nan)
    m = ok & np.isfinite(fmax)
    got_mfe[m] = fmax[m] / entry[m] - 1.0
    np.testing.assert_array_equal(got_close, close)
    np.testing.assert_array_equal(got_mfe, mfe)


def test_validation():
    with pytest.raises(ValueError):
        trailing_sums([3, 2, 1], [1, 1, 1], window=5)
    with pytest.raises(ValueError):
        forward_max_last([1, 2], [1.0], 5)
    f, last = forward_max_last(np.array([1.0]), np.array([5.0]), 10)
    assert np.isnan(f[0]) and last[0] == -1
