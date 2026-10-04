"""backtest.lob.ceiling — scalp-it scripts/ceiling_95/ceiling.py 원본과 비트 단위 동일 + 폴백."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from krx_quant_core.backtest import lob
from krx_quant_core.backtest.lob import ceiling as C

COST = 0.0023
FILL_SEC = 10

# ---- 원본 복사본(상수 COST·FILL_SEC 그대로) --------------------------------------------------


def _orig_oracle(bid, ask, h):
    n = len(bid)
    best_bid = np.full(n, np.nan, np.float64)
    best_ask = np.full(n, np.nan, np.float64)
    for j in range(2, h + 2):
        sb = np.full(n, np.nan)
        sa = np.full(n, np.nan)
        sb[: n - j] = bid[j:]
        sa[: n - j] = ask[j:]
        best_bid = np.fmax(best_bid, sb)
        best_ask = np.fmax(best_ask, sa)
    entry_taker = np.full(n, np.nan)
    entry_maker = np.full(n, np.nan)
    entry_taker[: n - 1] = ask[1:]
    entry_maker[: n - 1] = bid[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        tk = best_bid / entry_taker - 1 - COST
        mk = best_ask / entry_maker - 1 - COST
    bad = ~(entry_taker > 0) | ~(entry_maker > 0) | ~(best_bid > 0) | ~(best_ask > 0)
    tk[bad] = np.nan
    mk[bad] = np.nan
    return tk, mk


def _orig_oracle_short(bid, ask, h):
    n = len(bid)
    lo_ask = np.full(n, np.nan, np.float64)
    lo_bid = np.full(n, np.nan, np.float64)
    for j in range(2, h + 2):
        sa = np.full(n, np.nan)
        sb = np.full(n, np.nan)
        sa[: n - j] = ask[j:]
        sb[: n - j] = bid[j:]
        lo_ask = np.fmin(lo_ask, sa)
        lo_bid = np.fmin(lo_bid, sb)
    e_taker = np.full(n, np.nan)
    e_maker = np.full(n, np.nan)
    e_taker[: n - 1] = bid[1:]
    e_maker[: n - 1] = ask[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        tk = e_taker / lo_ask - 1 - COST
        mk = e_maker / lo_bid - 1 - COST
    bad = ~(e_taker > 0) | ~(e_maker > 0) | ~(lo_ask > 0) | ~(lo_bid > 0)
    tk[bad] = np.nan
    mk[bad] = np.nan
    return tk, mk


def _orig_fill_second(bid):
    n = len(bid)
    e = np.full(n, -1, np.int64)
    for t in range(n - FILL_SEC - 3):
        P = bid[t + 1]
        if not (P > 0):
            continue
        for s in range(t + 2, t + 2 + FILL_SEC):
            if bid[s] > 0 and bid[s] < P:
                e[t] = s
                break
    return e


def _orig_window_max(x, h):
    n = len(x)
    out = np.full(n, np.nan)
    if n > h:
        from numpy.lib.stride_tricks import sliding_window_view

        out[: n - h + 1] = np.nanmax(sliding_window_view(x, h), axis=1)
    return out


def _orig_maker_pessimistic(bid, e, maxask):
    n = len(bid)
    out = np.full(n, np.nan)
    ok = e >= 0
    idx = np.nonzero(ok)[0]
    nxt = e[idx] + 1
    good = nxt < n
    idx, nxt = idx[good], nxt[good]
    P = bid[idx + 1]
    best = maxask[nxt]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = best / P - 1 - COST
    r[~(P > 0) | ~(best > 0)] = np.nan
    out[idx] = r
    return out


def synthetic_path(seed: int, n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """빈 호가(NaN·0)·틱 점프·정체 구간이 섞인 초 격자 bid/ask."""
    rng = np.random.default_rng(seed)
    tick = 10.0
    mid = 5000 + np.cumsum(rng.choice([-tick, 0, 0, 0, tick], size=n))
    spread = rng.choice([tick, tick, 2 * tick], size=n)
    bid = mid - spread / 2
    ask = mid + spread / 2
    bid = np.round(bid / tick) * tick
    ask = np.round(ask / tick) * tick
    gap = rng.random(n) < 0.03
    bid[gap] = np.nan
    ask[gap] = np.nan
    zero = rng.random(n) < 0.01
    bid[zero] = 0.0
    return bid.astype(np.float64), ask.astype(np.float64)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("h", [10, 60, 300])
def test_oracles_match_original(seed, h):
    bid, ask = synthetic_path(seed)
    tk, mk = C.oracle_long(bid, ask, h, cost=COST)
    otk, omk = _orig_oracle(bid, ask, h)
    assert np.array_equal(tk, otk, equal_nan=True)
    assert np.array_equal(mk, omk, equal_nan=True)
    stk, smk = C.oracle_short(bid, ask, h, cost=COST)
    ostk, osmk = _orig_oracle_short(bid, ask, h)
    assert np.array_equal(stk, ostk, equal_nan=True)
    assert np.array_equal(smk, osmk, equal_nan=True)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fill_second_window_max_pessimistic_match_original(seed):
    bid, ask = synthetic_path(seed)
    e = C.through_fill_second(bid, FILL_SEC)
    oe = _orig_fill_second(bid)
    assert e.dtype == np.int64 and np.array_equal(e, oe)
    assert (e >= 0).any()  # 뚫림 체결이 실제로 생기는 자료다
    for h in (10, 300):
        wm = C.window_max(ask, h)
        owm = _orig_window_max(ask, h)
        assert np.array_equal(wm, owm, equal_nan=True)
        mp = C.maker_pessimistic(bid, e, wm, cost=COST)
        omp = _orig_maker_pessimistic(bid, oe, owm)
        assert np.array_equal(mp, omp, equal_nan=True)


def test_ceiling_table_shape_and_consistency():
    paths = [synthetic_path(s) for s in (1, 2)]
    t = C.ceiling_table(paths, cost=COST, horizons=(10, 60), lo=100, hi=2800)
    assert set(t["kind"]) == {"taker", "maker", "maker_pessimistic", "short_taker", "short_maker"}
    assert len(t) == 10
    tk = t[(t.horizon == 10) & (t.kind == "taker")].iloc[0]
    mk = t[(t.horizon == 10) & (t.kind == "maker")].iloc[0]
    assert tk.n == mk.n > 0
    # 메이커(스프레드를 먹는 쪽)는 테이커보다 천장이 높고, 지평이 길수록 천장은 오른다.
    assert mk.median_bp > tk.median_bp
    tk60 = t[(t.horizon == 60) & (t.kind == "taker")].iloc[0]
    assert tk60.median_bp >= tk.median_bp
    mp = t[(t.horizon == 10) & (t.kind == "maker_pessimistic")].iloc[0]
    assert 0 < mp.fill_rate < 1
    # 직접 계산한 값과 같은가 (lo/hi·동일 초 집합 규칙)
    bid, ask = paths[0]
    bid2, ask2 = paths[1]
    vals = []
    for b, a in paths:
        tk_, mk_ = C.oracle_long(b, a, 10, cost=COST)
        ok = np.isfinite(tk_[100:2800]) & np.isfinite(mk_[100:2800])
        vals.append(tk_[100:2800][ok])
    assert tk.median_bp == float(np.median(np.concatenate(vals)) * 1e4)


def test_cost_is_explicit_and_shifts_result():
    bid, ask = synthetic_path(5)
    a, _ = C.oracle_long(bid, ask, 30, cost=0.0)
    b, _ = C.oracle_long(bid, ask, 30, cost=0.0041)
    ok = np.isfinite(a)
    assert np.allclose(a[ok] - b[ok], 0.0041)


def test_empty_paths_gives_nan_rows():
    t = C.ceiling_table([], cost=COST, horizons=(10,))
    assert len(t) == 5 and (t.n == 0).all() and t.median_bp.isna().all()


_SCRIPT = r"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from test_lob_ceiling_golden import synthetic_path
from krx_quant_core.backtest.lob import HAVE_NUMBA, through_fill_second
bid, _ = synthetic_path(9, n=4000)
np.savez(sys.argv[1], have=HAVE_NUMBA, e=through_fill_second(bid, 10))
"""


@pytest.mark.skipif(not lob.HAVE_NUMBA, reason="numba 가 없으면 폴백만 돈다")
def test_through_fill_numba_and_python_fallback_agree(tmp_path):
    here = Path(__file__).parent
    outs = []
    for disable in ("", "1"):
        out = tmp_path / f"r{disable or 0}.npz"
        env = {**os.environ, "KRX_QUANT_CORE_DISABLE_NUMBA": disable}
        subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(out), str(here)], check=True, env=env
        )
        outs.append(np.load(out))
    assert bool(outs[0]["have"]) is True and bool(outs[1]["have"]) is False
    assert np.array_equal(outs[0]["e"], outs[1]["e"])
