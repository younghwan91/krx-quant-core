"""backtest.drift — 일중·야간 분해와 버킷 요약."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.backtest import drift as D


def _frame(seed=0, n_codes=6, n_days=130):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-02", periods=n_days)
    rows = []
    for i in range(n_codes):
        px = 10_000.0
        for d in days:
            pc = px
            o = pc * (1 + rng.normal(0.0005, 0.01))       # 야간 +5bp
            c = o * (1 + rng.normal(-0.0020 * (i + 1) / n_codes, 0.01))  # 일중 음수, i 클수록
            rows.append({"code": f"{i:06d}", "date": d, "open": o, "close": c,
                         "turnover_prev": float(i)})
            px = c
    return pd.DataFrame(rows)


def test_intraday_overnight_definitions_and_prev_close_shift():
    f = pd.DataFrame({
        "code": ["A", "A", "B"], "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-05"]),
        "open": [100.0, 110.0, 0.0], "close": [105.0, 99.0, 50.0],
    })
    out = D.intraday_overnight(f)
    a = out[out.code == "A"].sort_values("date")
    assert np.isnan(a.iloc[0].prev_close) and np.isnan(a.iloc[0].overnight_bp)
    assert a.iloc[0].intraday_bp == pytest.approx(500.0)
    assert a.iloc[1].prev_close == 105.0
    assert a.iloc[1].overnight_bp == pytest.approx((110 / 105 - 1) * 1e4)
    assert a.iloc[1].intraday_bp == pytest.approx((99 / 110 - 1) * 1e4)
    assert a.iloc[1].close_to_close_bp == pytest.approx((99 / 105 - 1) * 1e4)
    b = out[out.code == "B"].iloc[0]
    assert np.isnan(b.intraday_bp)  # 시가 0
    assert "intraday_bp" not in f.columns  # 원본 불변


def test_intraday_overnight_uses_given_prev_close():
    f = pd.DataFrame({"code": ["A"], "date": ["2026-01-05"], "open": [100.0], "close": [101.0],
                      "prev_close": [98.0]})
    out = D.intraday_overnight(f)
    assert out.overnight_bp.iloc[0] == pytest.approx((100 / 98 - 1) * 1e4)


def test_drift_summary_overall_and_by_bucket():
    f = D.intraday_overnight(_frame())
    s = D.drift_summary(f, "intraday_bp")
    assert list(s.index) == ["all"]
    r = s.loc["all"]
    assert r.n == f.intraday_bp.notna().sum()
    assert r.n_months == f.date.dt.to_period("M").nunique() and 0 <= r.neg_month_share <= 1
    assert r.mean_bp < 0 and r.t_hac < 0
    o = D.drift_summary(f, "overnight_bp").loc["all"]
    assert o.mean_bp > r.mean_bp

    f["bucket"] = D.rank_buckets(f, "turnover_prev", n=3)
    b = D.drift_summary(f, "intraday_bp", by="bucket")
    assert list(b.index) == [0.0, 1.0, 2.0]
    assert b.loc[0.0].mean_bp < b.loc[2.0].mean_bp  # 상위 버킷(큰 turnover)이 더 음수
    assert b.n.sum() == r.n


def test_drift_summary_empty_group():
    f = pd.DataFrame({"date": pd.to_datetime([]), "x": []})
    s = D.drift_summary(f, "x")
    assert s.loc["all"].n == 0 and np.isnan(s.loc["all"].mean_bp)


def test_rank_buckets_per_date_and_labels():
    f = pd.DataFrame({
        "date": ["d1"] * 4 + ["d2"] * 2,
        "v": [10, 40, 30, 20, 5, np.nan],
    })
    b = D.rank_buckets(f, "v", n=2)
    assert b.tolist()[:4] == [1.0, 0.0, 0.0, 1.0]  # 큰 값이 0
    assert b.iloc[4] == 0.0 and np.isnan(b.iloc[5])
    lab = D.rank_buckets(f, "v", n=2, labels=["top", "bottom"])
    assert lab.iloc[1] == "top" and lab.iloc[0] == "bottom"
    with pytest.raises(ValueError):
        D.rank_buckets(f, "v", n=2, labels=["x"])
