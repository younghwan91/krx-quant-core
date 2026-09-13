"""stats.sharpe — swing-it ``tests/test_gate_deflation.py`` 이식 + 알려진 값 고정.

원본 테스트는 ``gate_report(R, n_trials=N)["deflation"]`` 를 거쳤다. gate_report 는 swing-it
에 남으므로 여기서는 같은 계산 경로인 ``deflated_sharpe_from_sample`` 로 검증한다.

  (1) DSR: deflated Sharpe < raw Sharpe 이고, 시도 config 수 N 이 커지면 더 깎인다.
  (2) 반환 구조에 bool 판정 키가 **재귀적으로** 없다(리포터 불변).
  (3) 퇴화 표본은 예외 없이 NaN.
"""

from __future__ import annotations

import math
from math import e
from statistics import NormalDist

import numpy as np
import pytest

from krx_quant_core.stats.sharpe import (
    bootstrap_mean_ci,
    deflated_sharpe,
    deflated_sharpe_from_sample,
    expected_max_sharpe_h0,
    probabilistic_sharpe,
    t_haircut,
)

_VERDICT_KEYS = {"pass", "fail", "passed", "failed", "ok", "verdict",
                 "go", "nogo", "deploy", "accept", "reject"}


def _assert_no_boolean_verdict(obj) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                assert k.lower() not in _VERDICT_KEYS, f"판정 키 발견: {k}"
            assert not isinstance(v, bool), f"bool 판정 값 발견: {k}={v}"
            _assert_no_boolean_verdict(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_boolean_verdict(v)


def _edge_R(seed: int = 0, n: int = 400) -> np.ndarray:
    """양(+)의 엣지가 있는 R-멀티플 표본: 손절 −1R 절단 + 오른꼬리."""
    rng = np.random.default_rng(seed)
    ret = np.where(rng.random(n) < 0.40, -0.10, rng.exponential(0.08, n) + 0.04)
    return ret / 0.10


# --- (1) DSR < raw Sharpe, N 증가에 단조 깎임 --------------------------------------

def test_deflated_sharpe_below_raw_and_decreasing_in_trials():
    R = _edge_R(seed=1)
    raw_sharpe = R.mean() / R.std(ddof=1)
    assert raw_sharpe > 0

    defl = [deflated_sharpe_from_sample(R, n) for n in (1, 10, 100, 1000)]
    for d in defl:
        assert abs(d["observed_sharpe"] - raw_sharpe) < 1e-9
    for d in defl[1:]:
        assert d["deflated_sharpe"] < d["observed_sharpe"]

    sr0 = [d["expected_max_sharpe_h0"] for d in defl]
    dfl = [d["deflated_sharpe"] for d in defl]
    pgt = [d["prob_deflated_sharpe"] for d in defl]
    hair = [d["t_haircut"]["haircut_multiple"] for d in defl]
    assert sr0 == sorted(sr0)
    assert dfl == sorted(dfl, reverse=True)
    assert pgt == sorted(pgt, reverse=True)
    assert hair == sorted(hair)
    assert sr0[0] == 0.0  # N=1 은 뽑을 게 하나 → 벤치마크 0


def test_prob_sharpe_gt0_is_probability():
    d = deflated_sharpe_from_sample(_edge_R(seed=2), 50)
    assert 0.0 <= d["prob_sharpe_gt0"] <= 1.0
    assert 0.0 <= d["prob_deflated_sharpe"] <= 1.0
    assert d["prob_deflated_sharpe"] <= d["prob_sharpe_gt0"] + 1e-12


def test_helper_matches_sample_block():
    R = _edge_R(seed=3)
    mu = R.mean()
    sr = mu / R.std(ddof=1)
    sdp = R.std()
    skew = ((R - mu) ** 3).mean() / sdp ** 3
    kurt = ((R - mu) ** 4).mean() / sdp ** 4
    assert deflated_sharpe(sr, 25, len(R), skew, kurt) == deflated_sharpe_from_sample(R, 25)


def test_sample_block_drops_nonfinite():
    R = _edge_R(seed=6)
    dirty = np.concatenate([R, [np.nan, np.inf]])
    assert deflated_sharpe_from_sample(dirty, 5) == deflated_sharpe_from_sample(R, 5)


# --- (2) 리포터 불변 ----------------------------------------------------------------

def test_deflation_has_no_verdict_key():
    _assert_no_boolean_verdict(deflated_sharpe_from_sample(_edge_R(seed=5), 100))


# --- (3) 퇴화 방어 -------------------------------------------------------------------

def test_degenerate_sample_is_nan_safe():
    d = deflated_sharpe_from_sample(np.array([0.5]), 10)
    assert np.isnan(d["observed_sharpe"])
    assert np.isnan(d["prob_sharpe_gt0"])
    assert d["t_haircut"]["haircut_multiple"] > 1.0


def test_zero_dispersion_is_nan_safe():
    d = deflated_sharpe_from_sample(np.full(20, 0.5), 10)
    assert np.isnan(d["observed_sharpe"])


def test_near_zero_dispersion_is_degenerate():
    """0.3 같은 상수열의 반올림 잔차(~1e-17)도 퇴화로 본다 — swing-it 원본의 1e15 폭주 수정."""
    d = deflated_sharpe_from_sample(np.full(20, 0.3), 10)
    assert math.isnan(d["observed_sharpe"])


# --- 알려진 값 고정 -----------------------------------------------------------------

_N = NormalDist()


def test_expected_max_sharpe_closed_form():
    g = 0.5772156649015329
    want = 0.1 * ((1 - g) * _N.inv_cdf(1 - 1 / 10) + g * _N.inv_cdf(1 - 1 / (10 * e)))
    assert expected_max_sharpe_h0(10, 0.1) == pytest.approx(want, abs=0)
    assert expected_max_sharpe_h0(1, 0.1) == 0.0
    assert expected_max_sharpe_h0(10, float("nan")) == 0.0


def test_probabilistic_sharpe_normal_case():
    # skew=0, kurt=3 → 분모 = 1 + SR²/2
    sr, n = 0.1, 101
    z = sr * np.sqrt(n - 1) / np.sqrt(1 + 0.5 * sr ** 2)
    assert probabilistic_sharpe(sr, 0.0, n, 0.0, 3.0) == pytest.approx(_N.cdf(z), abs=1e-15)
    assert np.isnan(probabilistic_sharpe(sr, 0.0, 1, 0.0, 3.0))
    # 분모 ≤ 0 (강한 양의 왜도 × 큰 SR) → NaN
    assert np.isnan(probabilistic_sharpe(2.0, 0.0, 50, 5.0, 3.0))


def test_t_haircut_values():
    h1 = t_haircut(1)
    assert h1["haircut_multiple"] == pytest.approx(1.0)
    assert h1["base_hurdle_t"] == pytest.approx(1.959963984540054)
    h20 = t_haircut(20)
    assert 3.0 < h20["adjusted_hurdle_t"] < 3.1  # Harvey-Liu t>3.0 권고 수준
    assert t_haircut(0)["haircut_multiple"] == pytest.approx(1.0)  # N<1 은 1 로 클램프


def test_deflated_sharpe_default_sr_std():
    d = deflated_sharpe(0.2, 10, 101, 0.0, 3.0)
    assert d["expected_max_sharpe_h0"] == expected_max_sharpe_h0(10, 1 / np.sqrt(100))
    d2 = deflated_sharpe(0.2, 10, 101, 0.0, 3.0, sr_std=0.05)
    assert d2["expected_max_sharpe_h0"] == expected_max_sharpe_h0(10, 0.05)


# --- bootstrap_mean_ci ----------------------------------------------------------------

def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    R = _edge_R(seed=8)
    lo, hi = bootstrap_mean_ci(R, n_boot=1000, seed=0, ci=0.95)
    assert lo < R.mean() < hi
    assert (lo, hi) == bootstrap_mean_ci(R, n_boot=1000, seed=0, ci=0.95)
    lo90, hi90 = bootstrap_mean_ci(R, n_boot=1000, seed=0, ci=0.90)
    assert lo <= lo90 and hi90 <= hi


def test_bootstrap_ci_small_sample_is_nan():
    lo, hi = bootstrap_mean_ci(np.array([1.0, np.nan]), n_boot=10, seed=0, ci=0.95)
    assert np.isnan(lo) and np.isnan(hi)
