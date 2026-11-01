"""크기 의존 비용(costs/impact.py) — 제곱근 법칙·정가 수수료·일자별 매도세·단위 가드."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from krx_quant_core.costs import (
    DEFAULT_COMMISSION_RATE,
    participation,
    sell_tax_rate,
    sqrt_impact_bp,
    trade_cost_bp,
)

ADV = np.array([5e9, 2e10, 3e8])
VOL = np.array([0.02, 0.015, 0.035])


def test_sqrt_impact_formula():
    q = np.array([1e8, 5e8, 3e6])
    got = sqrt_impact_bp(q, ADV, VOL, y=1.0)
    np.testing.assert_allclose(got, VOL * np.sqrt(q / ADV) * 1e4, rtol=1e-15)
    # Y 배수는 선형, 0 이면 0.
    np.testing.assert_allclose(sqrt_impact_bp(q, ADV, VOL, y=2.0), 2 * got, rtol=1e-15)
    assert np.all(sqrt_impact_bp(q, ADV, VOL, y=0.0) == 0.0)


def test_sqrt_impact_monotone_and_symmetric():
    q = np.linspace(0.0, 2e9, 50)
    imp = sqrt_impact_bp(q, 5e9, 0.02)
    assert imp[0] == 0.0
    assert np.all(np.diff(imp) > 0)
    # 부호(매도 음수 대금)는 보지 않는다.
    np.testing.assert_array_equal(sqrt_impact_bp(-q, 5e9, 0.02), imp)
    # 변동성·ADV 방향: 변동성↑ → 충격↑, ADV↑ → 충격↓.
    assert sqrt_impact_bp(1e8, 5e9, 0.03) > sqrt_impact_bp(1e8, 5e9, 0.02)
    assert sqrt_impact_bp(1e8, 1e10, 0.02) < sqrt_impact_bp(1e8, 5e9, 0.02)


@pytest.mark.parametrize("s", [4.0, 0.25, 9.0, 0.01])
def test_sqrt_impact_homogeneity(s):
    q = np.array([1e8, 5e8, 3e6])
    base = sqrt_impact_bp(q, ADV, VOL)
    # bp 는 √s 배, 원화 비용(bp·Q)은 s^{3/2} 배.
    np.testing.assert_allclose(sqrt_impact_bp(s * q, ADV, VOL), np.sqrt(s) * base, rtol=1e-13)
    # (Q, ADV) 를 같이 s 배 하면 불변(Q/ADV 의 함수).
    np.testing.assert_allclose(sqrt_impact_bp(s * q, s * ADV, VOL), base, rtol=1e-13)
    won = sqrt_impact_bp(s * q, ADV, VOL) / 1e4 * (s * q)
    np.testing.assert_allclose(won, s**1.5 * base / 1e4 * q, rtol=1e-13)


def test_participation():
    np.testing.assert_allclose(participation([1e8, -2e9], [1e9, 4e10]), [0.1, 0.05])


@pytest.mark.parametrize(
    "fn",
    [
        lambda adv: participation(1e8, adv),
        lambda adv: sqrt_impact_bp(1e8, adv, 0.02),
        lambda adv: trade_cost_bp(
            "BUY", 1e8, adv, 0.02, "KOSPI", date(2026, 3, 2), half_spread_bp=5
        ),
    ],
)
def test_adv_unit_guard(fn):
    # daily_bars.trade_value(백만원) 5,000 을 그대로 넣은 사고 — 원으로는 50억.
    with pytest.raises(ValueError, match="원 단위"):
        fn(np.array([5e9, 5000.0]))
    with pytest.raises(ValueError, match="원 단위"):
        fn(0.0)
    fn(np.array([5e9, 1e6]))  # 하한 그 자체는 통과


def test_nan_adv_passes_through_as_nan():
    out = sqrt_impact_bp([1e8, 1e8], [np.nan, 5e9], 0.02)
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_daily_vol_unit_guard():
    with pytest.raises(ValueError, match="분수"):
        sqrt_impact_bp(1e8, 5e9, 2.0)  # 퍼센트로 넣은 실수
    with pytest.raises(ValueError):
        sqrt_impact_bp(1e8, 5e9, -0.01)
    with pytest.raises(ValueError):
        sqrt_impact_bp(1e8, 5e9, 0.02, y=-1.0)


@pytest.mark.parametrize(
    ("on", "market"),
    [(date(2024, 6, 3), "KOSDAQ"), (date(2025, 3, 4), "KOSPI"), (date(2020, 1, 2), "코스닥")],
)
def test_trade_cost_components(on, market):
    q = np.array([1e8, 5e8, 3e6])
    hs = np.array([3.0, 5.0, 20.0])
    comm = float(DEFAULT_COMMISSION_RATE) * 1e4
    imp = sqrt_impact_bp(q, ADV, VOL)
    buy = trade_cost_bp("BUY", q, ADV, VOL, market, on, half_spread_bp=hs)
    sell = trade_cost_bp("sell", q, ADV, VOL, market, on, half_spread_bp=hs)
    np.testing.assert_allclose(buy, comm + hs + imp, rtol=1e-14)
    np.testing.assert_allclose(sell, comm + sell_tax_rate(on, market) * 1e4 + hs + imp, rtol=1e-14)


def test_trade_cost_per_name_side_and_market():
    # 시장별 세율 조회 경로(종목마다 다른 market)와 종목별 side 를 같이 본다.
    for d in (date(2019, 1, 2), date(2020, 6, 1), date(2024, 6, 3), date(2026, 3, 2)):
        sides = np.array(["BUY", "SELL", "SELL"])
        markets = np.array(["KOSPI", "KOSDAQ", "KOSPI"])
        got = trade_cost_bp(sides, 0.0, ADV, VOL, markets, d, half_spread_bp=0.0)
        comm = float(DEFAULT_COMMISSION_RATE) * 1e4
        tax = {mk: sell_tax_rate(d, mk) * 1e4 for mk in ("KOSPI", "KOSDAQ")}
        want = [comm, comm + tax["KOSDAQ"], comm + tax["KOSPI"]]
        np.testing.assert_allclose(got, want, rtol=1e-14)


def test_trade_cost_rejects_bad_inputs():
    with pytest.raises(ValueError, match="BUY"):
        trade_cost_bp("HOLD", 1e8, 5e9, 0.02, "KOSPI", date(2026, 3, 2), half_spread_bp=5)
    with pytest.raises(ValueError, match="half_spread"):
        trade_cost_bp("BUY", 1e8, 5e9, 0.02, "KOSPI", date(2026, 3, 2), half_spread_bp=-1)
