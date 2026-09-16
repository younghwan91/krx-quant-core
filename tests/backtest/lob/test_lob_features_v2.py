"""backtest.lob v2(정본 37피처 중 종목 단위 35열) — scalp-it 83·84번 원본과,
배치·실시간·numba·폴백끼리 비트 단위로 같은지.
횡단면 순위 두 열은 편의 함수(:func:`cross_section_ranks_v2`)만 원본과 대조한다.

비교는 전부 ``np.array_equal(..., equal_nan=True)`` + 자료형 일치다. 허용 오차 없음 — 정책이
학습한 숫자 그대로를 실매매에서 받아야 하기 때문이다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from _lob_golden_v2 import FEATURES_JSON, features_by_code, synthetic_market

from krx_quant_core.backtest import lob
from krx_quant_core.backtest.lob import (
    FEATURE_SET_V2,
    FEATURE_SET_V2_EXCLUDED,
    INPUT_COLUMNS_V2,
    STEP_COLUMNS_V2,
    SecondFeatureStreamV2,
    aggregate_seconds_v2,
    compute_features_v2,
    cross_section_ranks_v2,
)

SEEDS = (0, 1, 2)


def _codes(ticks):
    return sorted(ticks.code.unique())


_I_CUMVAL = STEP_COLUMNS_V2.index("cumval")
_I_LAST = STEP_COLUMNS_V2.index("last")


def _batch(ticks, quotes):
    """종목별 배치 → (종목, 초, 열)."""
    codes = _codes(ticks)
    mats = []
    for c in codes:
        X, tr = aggregate_seconds_v2(ticks[ticks.code == c], quotes[quotes.code == c])
        mats.append(compute_features_v2(X, *tr))
    return codes, np.stack(mats)


@pytest.fixture(scope="module", params=SEEDS)
def market(request):
    ticks, quotes, prevclose = synthetic_market(request.param)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = features_by_code(ticks, quotes, prevclose)
    return request.param, ticks, quotes, prevclose, ref


def test_feature_set_is_canonical():
    assert FEATURE_SET_V2_EXCLUDED == ("cum_rank", "dayret_rank")
    assert list(FEATURE_SET_V2) == [f for f in FEATURES_JSON if f not in FEATURE_SET_V2_EXCLUDED]
    assert len(FEATURE_SET_V2) == 35 and len(FEATURES_JSON) == 37
    assert STEP_COLUMNS_V2 == (*FEATURE_SET_V2, "cumval", "last")
    assert not set(FEATURE_SET_V2_EXCLUDED) & set(STEP_COLUMNS_V2)


def test_batch_matches_original(market):
    seed, ticks, quotes, prevclose, ref = market
    codes, F = _batch(ticks, quotes)
    assert codes == sorted(ref)
    assert F.dtype == np.float64
    for ci, c in enumerate(codes):
        for j, name in enumerate(FEATURE_SET_V2):
            exp = ref[c][name]
            got = F[ci, :, j]
            assert got.dtype == exp.dtype == np.float64, name
            assert np.array_equal(got, exp, equal_nan=True), f"seed {seed} {c} {name}"
            # 원본 학습 캐시(flow84 pkl)는 float32 로 저장했다 — 같은 변환이면 같은 값
            e32, g32 = exp.astype(np.float32), got.astype(np.float32)
            assert g32.dtype == e32.dtype
            assert np.array_equal(g32, e32, equal_nan=True), f"seed {seed} {c} {name} f32"
    # 편의 함수: 원본 build_day 순위(동점 포함)
    pc = [prevclose.get(c, np.nan) for c in codes]
    cum_rank, dayret_rank = cross_section_ranks_v2(F[:, :, _I_CUMVAL], F[:, :, _I_LAST], pc)
    for ci, c in enumerate(codes):
        assert np.array_equal(cum_rank[ci], ref[c]["cum_rank"]), c
        assert np.array_equal(dayret_rank[ci], ref[c]["dayret_rank"]), c
    # 실시간 한 초씩(1차원) 매겨도 같다
    for t in (0, 500, 1499, 9000, 22799):
        r1, r2 = cross_section_ranks_v2(F[:, t, _I_CUMVAL], F[:, t, _I_LAST], pc)
        assert np.array_equal(r1, cum_rank[:, t]) and np.array_equal(r2, dayret_rank[:, t])


def test_synthetic_covers_edge_cases(market):
    _, ticks, quotes, _, ref = market
    c = _codes(ticks)[0]
    X, (ptr, _, _, _) = aggregate_seconds_v2(ticks[ticks.code == c], quotes[quotes.code == c])
    no_trade = np.diff(ptr) == 0
    assert no_trade.sum() > 1000  # 체결 없는 초
    assert np.isnan(ref[c]["spread_bp"][:100]).all()  # 빈 호가창 초
    assert (ref[c]["burst_secs10"] > 0).any() and (ref[c]["same_size_val10"] > 0).any()
    assert (ref[c]["sweep_ticks1"] > 0).any()
    assert np.isnan(ref[c]["micro_dev"]).any() and np.isfinite(ref[c]["micro_dev"]).sum() > 10000
    # 순위 동점(누적대금 0·전일 종가 없음)이 실제로 생긴다
    for k in sorted(ref)[2:]:  # 1500초부터 체결하는 두 종목 — 누적대금 0·dayret nan 동점
        assert (ref[k]["log_cumval"][:1000] == 0).all()


def test_stream_equals_batch_bit_for_bit(market):
    _, ticks, quotes, _, _ = market
    codes, F = _batch(ticks, quotes)
    streams = [SecondFeatureStreamV2() for _ in codes]
    inputs = []
    for c in codes:
        X, (ptr, price, vol, side) = aggregate_seconds_v2(
            ticks[ticks.code == c], quotes[quotes.code == c]
        )
        inputs.append((X, ptr, price, vol, side))
    n = F.shape[1]
    rows = np.empty_like(F)
    for t in range(n):
        for ci, (X, ptr, price, vol, side) in enumerate(inputs):
            a, b = ptr[t], ptr[t + 1]
            r = streams[ci].update_array(X[t], price[a:b], vol[a:b], side[a:b])
            assert r.dtype == np.float64
            assert r.shape == (len(STEP_COLUMNS_V2),)
            rows[ci, t] = r
    assert all(s.seconds == n for s in streams)
    assert rows.dtype == F.dtype
    for j, name in enumerate(STEP_COLUMNS_V2):
        assert np.array_equal(rows[:, :, j], F[:, :, j], equal_nan=True), name


def test_stream_returns_fresh_array_each_update():
    ticks, quotes, _ = synthetic_market(0, codes=("000010",), n_ticks=1500, n_quotes=1500)
    X, (ptr, price, vol, side) = aggregate_seconds_v2(ticks, quotes)
    s = SecondFeatureStreamV2()
    got = []
    for t in range(400):
        a, b = ptr[t], ptr[t + 1]
        got.append(s.update_array(X[t], price[a:b], vol[a:b], side[a:b]))
    ref = compute_features_v2(X[:400], ptr[:401], price, vol, side)
    # 앞서 받은 행이 뒤 호출에 덮이지 않았다
    assert np.array_equal(np.stack(got), ref, equal_nan=True)
    assert got[0][0] == 0.0 and got[399][0] == 399.0
    assert not np.shares_memory(got[0], got[1])
    # dict API
    s2 = SecondFeatureStreamV2()
    for t in range(300):
        a, b = ptr[t], ptr[t + 1]
        d = s2.update(
            price=price[a:b], volume=vol[a:b], side=side[a:b],
            **dict(zip(INPUT_COLUMNS_V2, X[t].tolist(), strict=True)),
        )  # fmt: skip
    assert list(d) == list(STEP_COLUMNS_V2)
    assert np.array_equal(np.array(list(d.values())), ref[299], equal_nan=True)


def test_stream_memory_is_bounded():
    """종목 수천 개를 띄우므로 인스턴스 메모리가 하루 내내 작게 고정돼야 한다."""
    ticks, quotes, _ = synthetic_market(3, codes=("000010",), n_ticks=20000, n_quotes=5000)
    X, (ptr, price, vol, side) = aggregate_seconds_v2(ticks, quotes)
    ref = compute_features_v2(X, ptr, price, vol, side)
    s = SecondFeatureStreamV2()
    sizes = []
    for t in range(X.shape[0]):
        a, b = ptr[t], ptr[t + 1]
        r = s.update_array(X[t], price[a:b], vol[a:b], side[a:b])
        if t % 997 == 0 or t == X.shape[0] - 1:
            assert np.array_equal(r, ref[t], equal_nan=True), t
        if t >= 600:
            sizes.append(s.nbytes)
    win10 = int((ptr[10:] - ptr[:-10]).max())  # 10초 창 최대 체결 수
    assert len(price) > 20000 and win10 < 256
    assert len(s._price) == 256 and len(s._ptr) == 11  # 늘지 않았다
    assert max(sizes) == min(sizes) < 100_000


def test_stream_buffer_grows_for_bursts_but_stays_windowed():
    # 한 초에 1,000건 → 버퍼는 창 크기만큼만 는다. 값은 배치와 같다.
    n = 40
    X = np.full((n, len(INPUT_COLUMNS_V2)), np.nan)
    X[:, 0], X[:, 1] = 1999.0, 2000.0
    X[:, 7], X[:, 8] = 10.0, 20.0
    counts = np.where(np.arange(n) % 7 == 3, 1000, 3)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    rng = np.random.default_rng(0)
    price = 1990.0 + rng.integers(0, 20, ptr[-1])
    vol = rng.choice([10.0, 50.0, 7.0], ptr[-1])
    side = rng.choice([-1.0, 0.0, 1.0], ptr[-1])
    ref = compute_features_v2(X, ptr, price, vol, side)
    s = SecondFeatureStreamV2()
    rows = []
    for t in range(n):
        rows.append(s.update_array(X[t], price[ptr[t]:ptr[t + 1]], vol[ptr[t]:ptr[t + 1]],
                                   side[ptr[t]:ptr[t + 1]]))  # fmt: skip
    assert np.array_equal(np.stack(rows), ref, equal_nan=True)
    assert len(s._price) <= 4096


def test_from_history_resumes_bit_for_bit():
    ticks, quotes, _ = synthetic_market(1, codes=("000010",), n_ticks=3000, n_quotes=3000)
    X, (ptr, price, vol, side) = aggregate_seconds_v2(ticks, quotes)
    ref = compute_features_v2(X, ptr, price, vol, side)
    cut = 9137  # 장중 재시작 시각
    s = SecondFeatureStreamV2.from_history(X[:cut], ptr[: cut + 1], price, vol, side)
    assert s.seconds == cut
    rows = [
        s.update_array(X[t], price[ptr[t] : ptr[t + 1]], vol[ptr[t] : ptr[t + 1]],
                       side[ptr[t] : ptr[t + 1]])
        for t in range(cut, cut + 700)
    ]  # fmt: skip
    assert np.array_equal(np.stack(rows), ref[cut : cut + 700], equal_nan=True)


def test_micro_dev_zero_depth_is_nan_like_original():
    ticks, quotes, prevclose = synthetic_market(0, codes=("000010",), n_ticks=3000, n_quotes=3000)
    qsec = (quotes.ts.dt.hour * 3600 + quotes.ts.dt.minute * 60 + quotes.ts.dt.second).to_numpy()
    qsec = qsec - 9 * 3600
    # 그 초에 스냅샷이 하나뿐인 행을 골라 1호가 잔량 양쪽을 0 으로
    uniq, cnt = np.unique(qsec, return_counts=True)
    target = int(uniq[(cnt == 1) & (uniq > 5000)][0])
    k = int(np.nonzero(qsec == target)[0][0])
    quotes = quotes.copy()
    quotes.loc[quotes.index[k], ["bidqty1", "askqty1"]] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = features_by_code(ticks, quotes, prevclose)["000010"]
    X, tr = aggregate_seconds_v2(ticks, quotes)
    assert X[target, 7] == 0.0 and X[target, 8] == 0.0
    F = compute_features_v2(X, *tr)
    j = STEP_COLUMNS_V2.index("micro_dev")
    assert np.isnan(ref["micro_dev"][target]) and np.isnan(F[target, j])
    assert np.array_equal(F[:, j], ref["micro_dev"], equal_nan=True)
    s = SecondFeatureStreamV2.from_history(X[:target], tr[0][: target + 1], *tr[1:])
    a, b = tr[0][target], tr[0][target + 1]
    r = s.update_array(X[target], tr[1][a:b], tr[2][a:b], tr[3][a:b])
    assert np.isnan(r[j])


def test_input_validation():
    with pytest.raises(ValueError, match="inputs"):
        compute_features_v2(np.zeros((5, 3)), np.zeros(6, np.int64), np.zeros(0), np.zeros(0),
                            np.zeros(0))  # fmt: skip
    with pytest.raises(ValueError, match="ptr"):
        compute_features_v2(np.zeros((5, len(INPUT_COLUMNS_V2))), np.zeros(5, np.int64),
                            np.zeros(0), np.zeros(0), np.zeros(0))  # fmt: skip
    s = SecondFeatureStreamV2()
    with pytest.raises(ValueError, match="length"):
        s.update_array(np.zeros(len(INPUT_COLUMNS_V2)), [1.0, 2.0], [1.0], [1.0])


_SCRIPT = r"""
import sys, time
import numpy as np
sys.path.insert(0, sys.argv[2])
from _lob_golden_v2 import synthetic_market
from krx_quant_core.backtest.lob import (
    HAVE_NUMBA, SecondFeatureStreamV2, aggregate_seconds_v2, compute_features_v2,
)
ticks, quotes, _ = synthetic_market(11, codes=("000010",))
X, (ptr, price, vol, side) = aggregate_seconds_v2(ticks, quotes)
F = compute_features_v2(X, ptr, price, vol, side)
s = SecondFeatureStreamV2()
S = np.empty_like(F)
t0 = time.perf_counter()
for t in range(X.shape[0]):
    a, b = ptr[t], ptr[t + 1]
    S[t] = s.update_array(X[t], price[a:b], vol[a:b], side[a:b])
dt = time.perf_counter() - t0
np.savez(sys.argv[1], have=HAVE_NUMBA, F=F, S=S, us=dt / X.shape[0] * 1e6)
"""


def _run(out: Path, *, disable: bool) -> dict[str, np.ndarray]:
    env = dict(os.environ)
    env.pop("KRX_QUANT_CORE_DISABLE_NUMBA", None)
    if disable:
        env["KRX_QUANT_CORE_DISABLE_NUMBA"] = "1"
    here = str(Path(__file__).parent)
    subprocess.run([sys.executable, "-c", _SCRIPT, str(out), here], check=True, env=env)
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.skipif(not lob.HAVE_NUMBA, reason="numba not installed ([fast] extra)")
def test_numba_and_fallback_identical_v2(tmp_path):
    fast = _run(tmp_path / "fast.npz", disable=False)
    slow = _run(tmp_path / "slow.npz", disable=True)
    assert bool(fast["have"]) and not bool(slow["have"])
    assert np.array_equal(fast["F"], slow["F"], equal_nan=True)
    assert np.array_equal(fast["S"], slow["S"], equal_nan=True)
    assert np.array_equal(fast["S"], fast["F"], equal_nan=True)
    print(f"stream µs/update numba={float(fast['us']):.1f} fallback={float(slow['us']):.1f}")
