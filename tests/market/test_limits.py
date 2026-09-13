"""상·하한가 — scalp-it 상한가 식과 한 치도 달라지면 안 된다."""

from __future__ import annotations

import math

import pytest

from krx_quant_core.market.limits import (
    LIMIT_RATE,
    limit_down_price,
    limit_up_price,
    near_limit_up,
)
from krx_quant_core.market.ticks import is_tick_valid


def _scalp_tick(price: float) -> int:
    """scalp-it ticksize.tick_size 의 원래 표(0 이하 → 1)."""
    if price <= 0:
        return 1
    for bound, tick in ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
                        (200_000, 100), (500_000, 500)):
        if price < bound:
            return tick
    return 1_000


def _scalp_limit_up(prev_close: float) -> float:
    """scalp-it realtime/pair_detector.py::limit_up_price 원문 그대로."""
    raw = float(prev_close) * 1.3
    tick = _scalp_tick(raw)
    return math.floor(raw / tick + 1e-9) * tick


# scalp-it 에서 실제로 뽑은 값(2026-09-14, pair_detector.limit_up_price).
_GOLDEN = {
    100: 130, 539: 700, 1_000: 1_300, 1_538: 1_999, 1_539: 2_000, 2_000: 2_600,
    3_846: 4_995, 6_230: 8_090, 15_760: 20_450, 19_999: 25_950, 20_000: 26_000,
    38_461: 49_950, 49_999: 64_900, 50_000: 65_000, 153_846: 199_900,
    199_999: 259_500, 384_615: 499_500, 1_234_567: 1_604_000, 10_000: 13_000,
}


@pytest.mark.parametrize("prev,expected", sorted(_GOLDEN.items()))
def test_limit_up_matches_scalp_golden(prev, expected):
    assert limit_up_price(prev) == expected
    assert limit_up_price(float(prev)) == expected


def test_limit_up_identical_to_scalp_formula_on_dense_sweep():
    prices = list(range(1, 3_000)) + list(range(3_000, 600_000, 7))
    prices += [p + 0.5 for p in range(1_500, 2_000)] + [384_615.38, 38_461.54]
    for p in prices:
        assert limit_up_price(p) == _scalp_limit_up(p), p


def test_limit_rate():
    assert LIMIT_RATE == 0.30


@pytest.mark.parametrize(
    "prev,expected",
    [
        (10_000, 7_000),
        (539, 378),        # 377.3 → 올림
        (71_500, 50_100),  # 50,050 → 5만원대 틱 100 으로 올림
        (2_000, 1_400),
        (7_150, 5_010),    # 5,005 는 5천원대 틱 10 의 배수가 아님 → 올림
        (1, 1),            # 0.7 → 1원
        (100_000, 70_000),
        # 실측 대조(daily_bars): 대칭 규칙과 갈리는 가격들
        (239_000, 167_500),
        (24_250, 17_000),
        (21_900, 15_350),
        (26_700, 18_700),
    ],
)
def test_limit_down_matches_krx(prev, expected):
    assert limit_down_price(prev) == expected


def test_limit_down_is_valid_tick_and_within_30pct():
    for p in list(range(1, 3_000)) + list(range(3_000, 800_000, 13)):
        if not is_tick_valid(p):
            continue
        down = limit_down_price(p)
        assert is_tick_valid(down), p
        assert down >= p * 0.7 - 1e-6, p
        up = limit_up_price(p)
        assert is_tick_valid(up), p
        assert up <= p * 1.3 + 1e-6, p


def _in_trigger_zone(*, price, prev_close, ticks=3):
    """scalp-it morning_report.in_trigger_zone 원문."""
    if prev_close <= 0 or price <= 0:
        return False
    limit = _scalp_limit_up(prev_close)
    if price >= limit:
        return False
    return price >= limit - ticks * _scalp_tick(limit)


@pytest.mark.parametrize("price", [12_960, 12_969, 12_970, 12_990, 12_999, 13_000, 13_010])
def test_near_limit_up_matches_morning_report_distance(price):
    prev = 10_000
    limit = limit_up_price(prev)
    expected = _in_trigger_zone(price=price, prev_close=prev) or price >= limit
    assert near_limit_up(price, prev, 3) is expected


def test_near_limit_up_ticks_and_guards():
    assert near_limit_up(12_970, 10_000, 3)
    assert not near_limit_up(12_970, 10_000, 2)
    assert not near_limit_up(0, 10_000)
    assert not near_limit_up(12_990, 0)
