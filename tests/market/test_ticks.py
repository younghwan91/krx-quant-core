"""호가단위 파생 연산 — 표의 정본은 kiwoom_client, 여기선 파생·호환 동작을 고정한다."""

from __future__ import annotations

from decimal import Decimal

import pytest

from krx_quant_core.market.ticks import (
    KRX_LOT_SIZE,
    is_tick_valid,
    round_to_tick,
    round_to_tick_up,
    shift_ticks,
    tick_size,
    tick_size_int,
    ticks_in,
    ticks_in_float,
)

# ---- scalp-it tests/test_ticksize.py 이식 -----------------------------------


@pytest.mark.parametrize(
    "price,expected",
    [
        (1_000, 1),
        (2_000, 5),
        (5_000, 10),
        (20_000, 50),
        (50_000, 100),
        (200_000, 500),
        (500_000, 1_000),
        (1_000_000, 1_000),
    ],
)
def test_tick_size_matches_krx_table(price, expected):
    assert tick_size_int(price) == expected
    assert tick_size(price) == Decimal(expected)


def test_tick_size_int_returns_plain_int_not_decimal():
    result = tick_size_int(70_000)
    assert type(result) is int
    # float 가격과 바로 섞어도 죽지 않아야 한다(scalp-it 실시간 경로가 이렇게 쓴다).
    assert 70_000.0 + 3 * result == 70_300.0


@pytest.mark.parametrize("price", [0, -1, -100])
def test_tick_size_int_nonpositive_falls_back_to_one_tick(price):
    assert tick_size_int(price) == 1


def test_decimal_tick_size_still_rejects_nonpositive():
    with pytest.raises(ValueError):
        tick_size(0)


def test_ticks_in_float_returns_plain_float_unrounded():
    result = ticks_in_float(2.2, 539)
    assert type(result) is float
    assert result == pytest.approx(2.2)


def test_ticks_in_float_nonpositive_price_uses_one_tick():
    assert ticks_in_float(-3, 0) == 3.0


# ---- scalp-it tests/test_pair_detector.py::test_tick_size_boundaries 이식 ---


@pytest.mark.parametrize(
    "price,expected",
    [
        (1, 1), (1_999, 1), (2_000, 5), (4_999, 5), (5_000, 10), (19_999, 10),
        (20_000, 50), (49_999, 50), (50_000, 100), (199_999, 100), (200_000, 500),
        (499_999, 500), (500_000, 1_000), (1_000_000, 1_000),
    ],
)
def test_tick_size_boundaries(price, expected):
    assert tick_size_int(price) == expected


def test_tick_size_int_matches_old_scalp_table_on_float_prices():
    """위임 전 scalp-it 표(limitup_timing.py 원본)와 float 가격에서도 같아야 한다."""

    def old(price: float) -> int:
        if price < 2_000:
            return 1
        if price < 5_000:
            return 5
        if price < 20_000:
            return 10
        if price < 50_000:
            return 50
        if price < 200_000:
            return 100
        if price < 500_000:
            return 500
        return 1_000

    for px in [0.5, 1, 539, 1_999.999, 2_000.0, 4_999.5, 5_000.0, 6_230, 19_999.99,
               20_000, 49_999.9, 50_000, 199_999.5, 200_000, 499_999.99, 500_000,
               1_234_567.8]:
        assert tick_size_int(px) == old(px), px


# ---- Decimal 파생 연산 ------------------------------------------------------


def test_reexports_are_kiwoom_client_objects():
    import importlib

    # kiwoom_client 패키지가 같은 이름의 함수를 노출해 서브모듈 속성을 가린다.
    kt = importlib.import_module("kiwoom_client.tick_size")

    assert tick_size is kt.tick_size
    assert round_to_tick is kt.round_to_tick
    assert ticks_in is kt.ticks_in


def test_lot_size():
    assert KRX_LOT_SIZE == 1


@pytest.mark.parametrize(
    "price,down,up",
    [
        ("70050", "70000", "70100"),
        ("70000", "70000", "70000"),
        ("1999.5", "1999", "2000"),
        ("4996", "4995", "5000"),
        ("539", "539", "539"),
        ("50123.4", "50100", "50200"),
    ],
)
def test_round_down_and_up(price, down, up):
    assert round_to_tick(Decimal(price)) == Decimal(down)
    assert round_to_tick_up(Decimal(price)) == Decimal(up)
    assert is_tick_valid(round_to_tick_up(Decimal(price)))


@pytest.mark.parametrize(
    "price,valid",
    [(1, True), (1999, True), (2000, True), (2003, False), (2005, True), (4995, True),
     (5005, False), (0, False), (-5, False), ("1.5", False), (500_500, False)],
)
def test_is_tick_valid(price, valid):
    assert is_tick_valid(price) is valid


@pytest.mark.parametrize(
    "price,n,expected",
    [
        (2000, -1, 1999),
        (1999, 1, 2000),
        (2000, 1, 2005),
        (5000, -1, 4995),
        (4995, 2, 5010),
        (20000, -2, 19980),
        (50000, -1, 49950),
        (200000, -1, 199900),
        (500000, -1, 499500),
        (499500, 1, 500000),
        (70000, 3, 70300),
        (70000, 0, 70000),
        (2, -1, 1),
    ],
)
def test_shift_ticks_across_bands(price, n, expected):
    result = shift_ticks(price, n)
    assert result == Decimal(expected)
    assert isinstance(result, Decimal)
    assert is_tick_valid(result)


def test_shift_ticks_roundtrip_is_identity_over_band_edges():
    for p in [1_990, 1_999, 2_000, 4_990, 5_000, 19_990, 20_000, 49_950, 50_000,
              199_900, 200_000, 499_500, 500_000]:
        for n in range(1, 25):
            assert shift_ticks(shift_ticks(p, n), -n) == p, (p, n)


def test_shift_ticks_rejects_invalid_price_and_underflow():
    with pytest.raises(ValueError):
        shift_ticks(2003, 1)
    with pytest.raises(ValueError):
        shift_ticks(1, -1)
