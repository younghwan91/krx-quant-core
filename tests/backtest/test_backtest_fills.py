"""backtest.fills — scalp-it 회계(``RiskGuard._advance``)·dry-run 실행기
(``_is_buy_filled``·``_check_sell_fill``)의 체결 판정과 같은 결과를 내는지.
"""

from __future__ import annotations

import pytest

from krx_quant_core.backtest.fills import EPS, limit_buy_filled, limit_sell_filled


@pytest.mark.parametrize("trade,through,touch", [
    (9_990.0, True, True),     # 관통
    (10_000.0, False, True),   # 닿기만 — touch 만 체결
    (10_010.0, False, False),  # 위에서 놂
])
def test_buy_fill_by_basis(trade, through, touch):
    assert limit_buy_filled(10_000.0, trade, "through") is through
    assert limit_buy_filled(10_000.0, trade, "touch") is touch


@pytest.mark.parametrize("trade,through,touch", [
    (10_010.0, True, True),
    (10_000.0, False, True),
    (9_990.0, False, False),
])
def test_sell_fill_by_basis(trade, through, touch):
    assert limit_sell_filled(10_000.0, trade, "through") is through
    assert limit_sell_filled(10_000.0, trade, "touch") is touch


def test_default_basis_is_through():
    assert limit_buy_filled(100.0, 100.0) is False
    assert limit_sell_filled(100.0, 100.0) is False


def test_eps_tolerance_matches_scalp_it():
    assert EPS == 1e-9
    lim = 100.0
    # through 는 EPS 보다 확실히 관통해야, touch 는 EPS 안쪽이면 닿은 것으로 본다
    assert limit_buy_filled(lim, lim - 5e-10, "through") is False
    assert limit_buy_filled(lim, lim + 5e-10, "touch") is True
    assert limit_sell_filled(lim, lim + 5e-10, "through") is False
    assert limit_sell_filled(lim, lim - 5e-10, "touch") is True


def test_through_implies_touch():
    for trade in (98.0, 99.999, 100.0, 100.001, 102.0):
        if limit_buy_filled(100.0, trade, "through"):
            assert limit_buy_filled(100.0, trade, "touch")
        if limit_sell_filled(100.0, trade, "through"):
            assert limit_sell_filled(100.0, trade, "touch")


def test_unknown_basis_raises():
    """원본 두 곳이 모르는 값을 서로 반대로(through/touch) 읽었다 — 여기선 막는다."""
    with pytest.raises(ValueError):
        limit_buy_filled(100.0, 99.0, "optimistic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        limit_sell_filled(100.0, 101.0, "")  # type: ignore[arg-type]
