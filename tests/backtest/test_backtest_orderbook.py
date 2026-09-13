"""backtest.orderbook — scalp-it ``tests/test_pair_detector.py`` 호가 두께 부분 +
``tests/test_liquidity_sizing.py`` 순수함수 부분 이식.
"""

from __future__ import annotations

import math

import pytest

from krx_quant_core.backtest.orderbook import liquidity_size_multiplier, roundtrip_bp, sweep_vwap


def _book(px, qty, n=10, step=0):
    """``n`` 단계 호가 — 각 단계 ``qty`` 주. ``step`` 만큼 가격이 멀어진다."""
    return [(px + i * step, qty) for i in range(n)]


def test_roundtrip_bp_thick_book_is_cheap():
    """두꺼운 호가: 1000만원이 1호가에서 다 체결 → 충격은 스프레드뿐."""
    bids = _book(9_990.0, 10_000, step=-10)
    asks = _book(10_000.0, 10_000, step=10)
    # 스프레드 10원 / mid 9995 = 10.0bp
    assert roundtrip_bp(bids, asks, 10_000_000) == pytest.approx(10.0, abs=0.1)


def test_roundtrip_bp_thin_book_is_expensive():
    bids = _book(9_990.0, 200, step=-10)
    asks = _book(10_000.0, 200, step=10)
    assert roundtrip_bp(bids, asks, 10_000_000) > 50.0


def test_roundtrip_bp_insufficient_depth_is_inf():
    bids = _book(1_000.0, 10, step=-1)  # 10단계 × 1만원 = 10만원뿐
    asks = _book(1_001.0, 10, step=1)
    assert math.isinf(roundtrip_bp(bids, asks, 10_000_000))


def test_roundtrip_bp_missing_book_is_none():
    assert roundtrip_bp([], _book(100.0, 10), 1000) is None
    assert roundtrip_bp([(0.0, 10)], [(0.0, 10)], 1000) is None


def test_sweep_vwap_walks_levels_in_order():
    # 100원×10주=1000원, 110원×10주=1100원 → 1500원이면 10주 + 500/110 주
    levels = [(100.0, 10), (110.0, 10)]
    shares = 10 + 500 / 110
    assert sweep_vwap(levels, 1500.0) == pytest.approx(1500.0 / shares, rel=1e-15)
    assert sweep_vwap(levels, 1000.0) == pytest.approx(100.0)


def test_sweep_vwap_skips_empty_slots_and_rejects_nonpositive():
    assert sweep_vwap([(0.0, 5), (100.0, 0), (100.0, 10)], 500.0) == pytest.approx(100.0)
    assert sweep_vwap([(100.0, 10)], 0.0) is None
    assert sweep_vwap([(100.0, 10)], 1001.0) is None  # 호가 잔량 부족


@pytest.mark.parametrize("bp,expected", [
    (0.0, 1.0),
    (5.0, 1.0),
    (10.0, 1.0),  # 경계값은 아직 두꺼운 쪽
    (10.1, 0.2),
    (19.9, 0.2),
    (math.inf, 0.2),
])
def test_thin_book_gets_a_fraction(bp, expected):
    assert liquidity_size_multiplier(bp, thin_bp=10.0, thin_multiplier=0.2) == expected


def test_unknown_depth_is_treated_as_thin():
    assert liquidity_size_multiplier(None, thin_bp=10.0, thin_multiplier=0.2) == 0.2


def test_off_when_threshold_is_zero():
    assert liquidity_size_multiplier(50.0, thin_bp=0.0, thin_multiplier=0.2) == 1.0
    assert liquidity_size_multiplier(None, thin_bp=0.0, thin_multiplier=0.2) == 1.0
