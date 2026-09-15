"""backtest.lob — scalp-it 원본(80·82번)과 숫자가 같은지, 배치·실시간·numba·폴백이 같은지."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest
from _lob_golden import (
    HS,
    build_code,
    random_control,
    strength_exit_labels,
    synthetic_day,
)

from krx_quant_core.backtest.lob import (
    FEATURE_COLUMNS,
    INPUT_COLUMNS,
    PATH_KEYS,
    STEP_COLUMNS,
    SecondFeatureStream,
    aggregate_seconds,
    build_second_grid,
    compute_features,
    random_entry_control,
    simulate_exits,
)
from krx_quant_core.costs import round_trip_cost

COST_2026 = round_trip_cost(date(2026, 9, 8), "KOSPI")

# pandas rolling.mean(보정합) 대 누적합 차 — ofi10 만 ulp 수준 차이를 허용한다.
_ULP_COLS = {"ofi10"}


@pytest.fixture(scope="module", params=[0, 1, 2])
def day(request):
    ticks, quotes = synthetic_day(request.param)
    return ticks, quotes


def test_cost_matches_original_constant():
    assert COST_2026 == pytest.approx(0.0023, abs=1e-15)


def test_build_second_grid_matches_original(day):
    ticks, quotes = day
    ref_df, ref_path = build_code(ticks, quotes)
    df, path = build_second_grid(ticks, quotes, label_horizons=HS, label_cost=0.0023)

    assert len(ref_df) > 1000  # 합성 데이터가 필터를 충분히 통과해야 비교가 의미 있다
    assert list(df.columns) == list(ref_df.columns)
    assert list(df.columns[: len(FEATURE_COLUMNS)]) == list(FEATURE_COLUMNS)
    pd.testing.assert_index_equal(df.index, ref_df.index)
    for c in df.columns:
        assert df[c].dtype == ref_df[c].dtype, c
        got, exp = df[c].to_numpy(), ref_df[c].to_numpy()
        if c in _ULP_COLS:
            np.testing.assert_allclose(got, exp, rtol=1e-6, equal_nan=True, err_msg=c)
        else:
            np.testing.assert_array_equal(got, exp, err_msg=c)
    assert set(path) == set(PATH_KEYS) == set(ref_path)
    for k in PATH_KEYS:
        np.testing.assert_array_equal(path[k], ref_path[k], err_msg=k)


def test_empty_quotes_matches_original():
    ticks, quotes = synthetic_day(7)
    ref_df, ref_path = build_code(ticks, quotes.iloc[:0])
    df, path = build_second_grid(ticks, quotes.iloc[:0])
    np.testing.assert_array_equal(df.index, ref_df.index)
    for c in FEATURE_COLUMNS:
        np.testing.assert_allclose(df[c], ref_df[c], rtol=1e-6, equal_nan=True, err_msg=c)
    for k in PATH_KEYS:
        np.testing.assert_array_equal(path[k], ref_path[k], err_msg=k)


def test_stream_equals_batch_bit_for_bit(day):
    ticks, quotes = day
    X, _ = aggregate_seconds(ticks, quotes)
    batch = compute_features(X)
    stream = SecondFeatureStream()
    rows = np.empty_like(batch)
    for t in range(X.shape[0]):
        rows[t] = stream.update_array(X[t])
    assert stream.seconds == X.shape[0]
    np.testing.assert_array_equal(rows, batch)
    # dict API 도 같은 값
    s2 = SecondFeatureStream()
    for t in range(700):
        d = s2.update(**dict(zip(INPUT_COLUMNS, X[t].tolist(), strict=True)))
    np.testing.assert_array_equal(np.array([d[c] for c in STEP_COLUMNS]), batch[699])


def test_labels_require_cost():
    ticks, quotes = synthetic_day(0, n_ticks=300, n_quotes=300)
    with pytest.raises(ValueError, match="label_cost"):
        build_second_grid(ticks, quotes, label_horizons=(5,))


def test_missing_columns_raise():
    ticks, quotes = synthetic_day(0, n_ticks=300, n_quotes=300)
    with pytest.raises(ValueError, match="best_bid"):
        build_second_grid(ticks.drop(columns="best_bid"), quotes)


# --- 시뮬레이터 ---------------------------------------------------------------


@pytest.fixture(scope="module")
def grid():
    ticks, quotes = synthetic_day(3)
    df, path = build_second_grid(ticks, quotes)
    return df, {k: v.astype(np.float64) for k, v in path.items()}


def test_default_sim_matches_82_labels(grid):
    df, p = grid
    secs = df.sec.to_numpy(np.int64)
    ref_net, ref_hold = strength_exit_labels(
        secs, p["bid"], p["ask"], p["strength"], 3.0, 300, 0.0023
    )
    r = simulate_exits(secs, p["bid"], p["ask"], p["strength"], strength_drop=3.0, max_hold=300,
                       trade_date=date(2026, 9, 8), market="KOSDAQ")  # fmt: skip
    np.testing.assert_array_equal(r.net.astype(np.float32), ref_net)
    np.testing.assert_array_equal(r.hold, ref_hold)
    assert np.isfinite(r.net).sum() > 1000
    assert set(np.unique(r.reason)) <= {0, 1, 2}


def test_latency_and_boundaries():
    L = 20
    bid = np.full(L, 100.0)
    ask = np.full(L, 101.0)
    ask[5] = 110.0  # 지연 2초면 이 가격에 산다
    stg = np.full(L, np.nan)
    r = simulate_exits([3, 18, 19, -5], bid, ask, stg, max_hold=4, latency=2, cost=0.0)
    assert r.entry_price[0] == 110.0
    assert r.hold[0] == 4 and r.reason[0] == 2
    assert r.net[0] == pytest.approx(100 / 110 - 1)
    # e = 20 >= L-1, e = 21, e < 0 → 진입 불가
    assert np.isnan(r.net[1:]).all() and (r.reason[1:] == 0).all() and (r.hold[1:] == 0).all()
    # e = 18 = L-2 → 가능, 격자 끝(L-1)에서 시간청산 1초
    r2 = simulate_exits([17], bid, ask, stg, max_hold=10, latency=1, cost=0.0)
    assert r2.hold[0] == 1 and r2.reason[0] == 2


def test_cost_is_subtracted_once():
    bid = np.array([100.0, 100.0, 102.0, 102.0])
    ask = np.array([100.0, 100.0, 103.0, 103.0])
    r0 = simulate_exits([0], bid, ask, max_hold=1, cost=0.0)
    r1 = simulate_exits([0], bid, ask, max_hold=1, cost=0.0023)
    assert r0.net[0] - r1.net[0] == pytest.approx(0.0023, abs=1e-15)
    with pytest.raises(ValueError, match="trade_date"):
        simulate_exits([0], bid, ask, max_hold=1)


def test_stop_take_and_order():
    # 진입 1,000원(틱 1원). 익절 3틱 = 1,003 을 hi 가 넘어야 체결.
    L = 10
    bid = np.full(L, 999.0)
    ask = np.full(L, 1000.0)
    hi = np.full(L, np.nan)
    hi[3] = 1003.0  # 같으면 안 된다(넘어야)
    hi[4] = 1004.0
    r = simulate_exits([0], bid, ask, hi=hi, take_ticks=3, max_hold=8, cost=0.0)
    assert r.reason[0] == 4 and r.exit_price[0] == 1003.0 and r.hold[0] == 3
    # 같은 초에 익절과 손절이 둘 다면 익절이 먼저(원본 of_scratch 순서)
    bid2 = bid.copy()
    bid2[4] = 990.0
    r = simulate_exits([0], bid2, ask, hi=hi, take_ticks=3, stop_ticks=2, max_hold=8, cost=0.0)
    assert r.reason[0] == 4
    # 손절: bid ≤ 1000 − 2틱
    bid3 = bid.copy()
    bid3[2] = 998.0
    r = simulate_exits([0], bid3, ask, stop_ticks=2, max_hold=8, cost=0.0)
    assert r.reason[0] == 3 and r.exit_price[0] == 998.0 and r.hold[0] == 1
    with pytest.raises(ValueError, match="hi"):
        simulate_exits([0], bid, ask, take_ticks=1, max_hold=3, cost=0.0)


def test_stop_tick_uses_band_of_entry_price():
    # 진입 2,000원 → 틱 5원(kiwoom 정본). 손절 1틱 = 1,995.
    bid = np.array([1999.0, 1999.0, 1996.0, 1995.0])
    ask = np.array([2000.0, 2000.0, 2000.0, 2000.0])
    r = simulate_exits([0], bid, ask, stop_ticks=1, max_hold=3, cost=0.0)
    assert r.reason[0] == 3 and r.exit_price[0] == 1995.0


# --- 대조군 -------------------------------------------------------------------


@dataclass
class _Ep:
    secs: np.ndarray
    buy: np.ndarray
    sell: np.ndarray


def test_random_control_matches_82():
    rng = np.random.default_rng(5)
    eps = {}
    for d in ("2026-09-08", "2026-09-09"):
        for c in ("005930", "000660"):
            secs = np.sort(rng.choice(np.arange(300, 22000), 400, replace=False))
            buy = 1000 + rng.normal(0, 3, 400)
            eps[(d, c)] = _Ep(secs, buy, buy - 1)
    eps[("2026-09-10", "tiny")] = _Ep(
        np.array([5, 6]), np.array([10.0, 10.0]), np.array([9.0, 9.0])
    )
    keys = [k for k in eps for _ in range(30)]
    holds = rng.integers(1, 400, len(keys))
    holds[-5:] = 10_000  # 후보 없음 → nan, 난수 안 뽑음
    trades = pd.DataFrame(
        {"day": [k[0] for k in keys], "code": [k[1] for k in keys], "hold": holds}
    )
    ref = random_control(trades, eps, 0.0023).ret_ctrl.to_numpy()
    got = random_entry_control(keys, holds, eps, cost=0.0023, rng=82)
    np.testing.assert_array_equal(got, ref)
    assert np.isnan(got[-5:]).all()
