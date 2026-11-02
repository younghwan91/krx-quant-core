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
    out, ref = ri._asof_shares(sh, CODES, DATES)
    assert (ref[0, 3:] == ri._days(np.datetime64("2020-12-31"))).all()  # 그 값의 기준일
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
    out, _ = ri._asof_shares(sh, CODES, DATES)
    assert out[0, 1] == 100.0
    assert (out[0, 2:] == 200.0).all()  # 01-07 의 옛 기준일 정정(150)은 무시


def test_original_filing_row_counts_from_first_public_then_correction_overrides():
    # first_public 입력(원본 + 본표)을 _asof_shares 가 그대로 받는다 — 로직은 안 바뀐다
    sh = pd.DataFrame(
        {
            "code": ["000001", "000001"],
            "date": pd.to_datetime(["2020-12-31", "2020-12-31"]),
            "shares_outstanding": [100.0, 110.0],
            "knowledge_date": pd.to_datetime(["2021-01-05", "2021-01-08"]),  # 원본, 정정
        }
    )
    out, ref = ri._asof_shares(sh, CODES, DATES)
    assert np.isnan(out[0, 1])  # 01-05 까지는 모른다
    assert (out[0, 2:5] == 100.0).all()  # 원본 값을 첫 공시 다음 거래일부터
    assert (out[0, 5:] == 110.0).all()  # 정정 값은 정정일 다음 거래일부터
    assert (ref[0, 2:] == ri._days(np.datetime64("2020-12-31"))).all()


def test_shares_sql_default_is_the_old_query():
    # 기본값은 옛 입력 그대로 — 수치 동일성 계약(CLAUDE.md). 옆 표 없는 DB 에서도 돈다.
    sql = ri._shares_sql(date(2023, 12, 28), "knowledge_date")
    assert sql == (
        "SELECT code, date, shares_outstanding, knowledge_date "
        "FROM shares_outstanding_history "
        "WHERE knowledge_date <= '2023-12-28'"
    )


def test_shares_sql_first_public_reads_null_stamp_rows_and_side_table():
    sql = ri._shares_sql(date(2023, 12, 28), "first_public")
    assert "coalesce(knowledge_date, date) <= '2023-12-28'" in sql  # 키움 NULL 행도 그날 안 값으로
    assert "UNION ALL" in sql and "shares_original_filing" in sql
    assert "first_public_date <= '2023-12-28'" in sql


def test_unknown_shares_basis_is_refused_before_touching_the_db():
    with pytest.raises(ValueError, match="shares_basis"):
        ri.load_risk_inputs("2021-01-04", "2021-02-01", shares_basis="latest", conn=object())
    with pytest.raises(ValueError, match="shares_basis"):
        ri._shares_sql(date(2023, 12, 28), "latest")


def _merger(rt, day=4):
    return pd.DataFrame({"code": ["000001"], "date": [DATES[day]], "upd_rt": [rt]})


def test_big_merger_blanks_cap_until_a_row_dated_after_the_event_is_public():
    # 017670 꼴: 사건 전 기준일 주식수를 쓰는 칸만 비운다.
    # 사건 뒤 기준일 행(8번째 거래일 공개)부터 돌아온다.
    ref = np.full((2, 10), ri._days(DATES[0]), dtype=np.int64)
    ref[0, 7:] = ri._days(DATES[5])  # 기준일 > 사건 날
    mask = ri._merger_gap_mask(ref, _merger(-82.75), CODES, DATES)
    assert mask[0].tolist() == [False] * 4 + [True] * 3 + [False] * 3
    assert not mask[1].any()  # 다른 종목은 그대로


@pytest.mark.parametrize("rt", [-16.0, -0.67, 19.99])
def test_small_merger_keeps_cap(rt):
    # 실측 merger 중앙값 −0.67% — 이 크기까지 지우면 시총이 괜히 빈다
    ref = np.full((2, 10), ri._days(DATES[0]), dtype=np.int64)
    assert not ri._merger_gap_mask(ref, _merger(rt), CODES, DATES).any()


@pytest.mark.parametrize("rt", [-82.75, -50.0, 47.17, 24.76])
def test_big_merger_either_direction(rt):
    # 인적분할 재상장은 위로 뛴다(207940 2025-11 +47.17%)
    ref = np.full((2, 10), ri._days(DATES[0]), dtype=np.int64)
    assert ri._merger_gap_mask(ref, _merger(rt), CODES, DATES)[0, 4:].all()


def test_merger_gap_ignores_codes_outside_panel_and_rows_already_after_event():
    ref = np.full((2, 10), ri._days(DATES[9]), dtype=np.int64)  # 기준일이 이미 사건 뒤
    assert not ri._merger_gap_mask(ref, _merger(-82.75), CODES, DATES).any()
    outside = pd.DataFrame({"code": ["999999"], "date": [DATES[4]], "upd_rt": [-82.75]})
    assert not ri._merger_gap_mask(ref, outside, CODES, DATES).any()
    assert not ri._merger_gap_mask(ref, outside.iloc[:0], CODES, DATES).any()


def test_roll_events_applies_only_events_after_reference_day():
    # 5:1 분할(upd_rt −80%)이 5번째 거래일에. 기준일이 그 전이면 ×5, 뒤면 그대로.
    ev = pd.DataFrame({"code": ["000001"], "date": [DATES[4]], "upd_rt": [-80.0]})
    shares = np.full((2, 10), 1_000.0)
    before = np.full((2, 10), ri._days(DATES[1]), dtype=np.int64)
    out = ri._roll_events(shares, before, ev, CODES, DATES)
    assert (out[0, :4] == 1_000.0).all()
    assert np.allclose(out[0, 4:], 5_000.0)
    assert (out[1] == 1_000.0).all()  # 사건 없는 종목
    after = np.full((2, 10), ri._days(DATES[6]), dtype=np.int64)
    assert np.allclose(ri._roll_events(shares, after, ev, CODES, DATES)[0, 6:], 1_000.0)
    # 기준일이 패널보다 앞이어도 그 사이 사건을 센다(사건 날짜로 고른다).
    old = np.full((2, 10), ri._days(np.datetime64("2020-12-31")), dtype=np.int64)
    assert np.allclose(ri._roll_events(shares, old, ev, CODES, DATES)[0, 4:], 5_000.0)


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
