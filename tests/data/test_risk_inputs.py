"""data.risk_inputs 의 시점 규칙(DB 없이 — 순수 보조 함수만)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.data import risk_inputs as ri

DATES = pd.bdate_range("2021-01-04", periods=10)
CODES = np.array(["000001", "000002"])


def test_shares_usable_from_next_trading_day():
    sh = pd.DataFrame(
        {
            "code": ["000001"],
            "date": pd.to_datetime(["2020-12-31"]),
            "shares_outstanding": [100.0],
            "knowledge_date": pd.to_datetime(["2021-01-06"]),  # 3번째 거래일에 알게 됨
        }
    )
    out = ri._asof_shares(sh, CODES, DATES)
    assert np.isnan(out[0, :3]).all()  # 그날까지는 모른다
    assert (out[0, 3:] == 100.0).all()  # 다음 거래일부터
    assert np.isnan(out[1]).all()


def test_shares_late_correction_of_old_date_does_not_override_newer():
    sh = pd.DataFrame(
        {
            "code": ["000001"] * 3,
            "date": pd.to_datetime(["2020-06-30", "2020-12-31", "2020-06-30"]),
            "shares_outstanding": [100.0, 200.0, 150.0],
            "knowledge_date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-07"]),
        }
    )
    out = ri._asof_shares(sh, CODES, DATES)
    assert out[0, 1] == 100.0
    assert (out[0, 2:] == 200.0).all()  # 01-07 의 옛 기준일 정정(150)은 무시


def test_ttm_rules():
    st = {(2020, 4): (100.0, 80.0)}
    assert ri._ttm(st) == 100.0
    st[(2021, 1)] = (30.0, 20.0)
    assert ri._ttm(st) == 110.0  # 연간 100 + (30 − 20)
    st[(2021, 2)] = (40.0, 25.0)
    assert ri._ttm(st) == 125.0
    assert ri._ttm({(2021, 2): (40.0, 25.0), (2020, 4): (100.0, 1.0)}) == 100.0  # Q1 빠짐 → 연간
    assert np.isnan(ri._ttm({(2021, 1): (30.0, 20.0)}))  # 직전 연간 없음
    assert ri._ttm({**st, (2021, 4): (160.0, 100.0)}) == 160.0


def test_ttm_panel_uses_next_trading_day():
    earn = pd.DataFrame(
        {
            "code": ["000001", "000001"],
            "period": ["2020Q4", "2021Q1"],
            "known": pd.to_datetime(["2021-01-04", "2021-01-08"]),
            "netinc": [100.0, 30.0],
            "netinc_prior": [80.0, 20.0],
        }
    )
    out = ri._ttm_panel(earn, CODES, DATES)
    assert np.isnan(out[0, 0])
    assert (out[0, 1:5] == 100.0).all()
    assert (out[0, 5:] == 110.0).all()


def test_sector_panel_forward_fills_weekly_snapshot():
    wics = pd.DataFrame(
        {
            "code": ["000001", "000001", "000002"],
            "date": pd.to_datetime(["2021-01-04", "2021-01-11", "2021-01-11"]),
            "sector_cd": ["G10", "G20", "G45"],
        }
    )
    out = ri._sector_panel(wics, CODES, DATES)
    assert list(out[0, :5]) == ["G10"] * 5
    assert list(out[0, 5:]) == ["G20"] * 5
    assert all(v is None for v in out[1, :5])  # 첫 스냅샷에 없다 → WICS 밖
    assert list(out[1, 5:]) == ["G45"] * 5


def test_beta_recovers_known_beta_and_no_lookahead():
    rng = np.random.default_rng(0)
    t_len = ri.BETA_WINDOW + 40
    mkt = rng.normal(0, 0.01, t_len)
    betas = np.array([0.5, 1.0, 1.5])
    ret = betas[:, None] * mkt[None, :] + rng.normal(0, 0.002, (3, t_len))
    b, rv = ri._beta_resid_vol(ret, mkt)
    assert np.allclose(b[:, -1], betas, atol=0.05)
    assert np.allclose(rv[:, -1], 0.002, rtol=0.25)
    t = ri.BETA_WINDOW + 10
    ret2 = ret.copy()
    ret2[:, t + 1 :] = 9.0  # t 뒤를 망가뜨려도 t 까지의 값은 그대로
    b2, rv2 = ri._beta_resid_vol(ret2, mkt)
    assert np.array_equal(b[:, : t + 1], b2[:, : t + 1], equal_nan=True)
    assert np.array_equal(rv[:, : t + 1], rv2[:, : t + 1], equal_nan=True)


def test_refuses_aftermarket_window():
    with pytest.raises(ValueError, match="애프터마켓"):
        ri.load_risk_inputs(date(2026, 1, 1), date(2026, 10, 1), conn=object())
