"""stats.matched_null — 층 매칭 대조와 클러스터 부트스트랩."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.stats import matched_null as M


def _data(seed=0, n_days=40, per_day=50, edge=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day_shock = rng.normal(0, 5)  # 날짜 공통 충격 → 클러스터
        for _ in range(per_day):
            vol = int(rng.integers(0, 3))
            r = day_shock + vol * 3 + rng.normal(0, 10)
            rows.append({"date": d, "vol_bucket": vol, "minute": int(rng.integers(0, 3)), "ret": r})
    pool = pd.DataFrame(rows)
    # 신호는 고변동 버킷에서만 켜진다 — 매칭 없이 비교하면 vol 효과가 alpha 로 둔갑한다
    sig = pool[pool.vol_bucket == 2].sample(frac=0.3, random_state=seed).copy()
    sig["ret"] = sig["ret"] + edge
    return sig, pool


def test_matched_control_respects_strata_and_reports_unmatched():
    sig, pool = _data()
    sig.loc[sig.index[0], "minute"] = 99  # 짝 없는 층
    c = M.matched_control(sig, pool, strata=["date", "vol_bucket", "minute"], n_per=2, seed=1)
    assert set(c.columns) == {"trade_idx", "control_idx", "date", "vol_bucket", "minute"}
    j = c.merge(pool, left_on="control_idx", right_index=True, suffixes=("", "_p"))
    assert (j.date == j.date_p).all() and (j.vol_bucket == j.vol_bucket_p).all()
    assert (j.minute == j.minute_p).all()
    assert (c.control_idx != c.trade_idx).all()  # 자기 자신 제외
    # 짝 없는 층(minute=99)은 반드시 unmatched. 자기 자신만 있는 층도 unmatched 가 된다.
    assert sig.index[0] in c.attrs["unmatched"]
    assert set(c.attrs["unmatched"]).isdisjoint(c.trade_idx)
    assert len(c.attrs["unmatched"]) + c.trade_idx.nunique() == len(sig)
    assert c.groupby("trade_idx").size().eq(2).all()
    with pytest.raises(KeyError, match="nope"):
        M.matched_control(sig, pool, strata=["nope"])


def test_matching_removes_regime_effect_and_detects_true_edge():
    sig, pool = _data(edge=0.0)
    naive = sig.ret.mean() - pool.ret.mean()
    assert naive > 2  # 매칭 없이 보면 변동성 효과가 alpha 처럼 보인다
    c = M.matched_control(sig, pool, strata=["date", "vol_bucket"], n_per=3, seed=2)
    a = M.matched_alpha(sig, pool, c, value="ret", cluster="date", n_boot=500, seed=3)
    assert abs(a.alpha) < 2.5
    assert a.ci_low < 0 < a.ci_high
    assert a.n_trades == len(sig) and a.n_controls == 3 * len(sig)
    assert a.n_clusters == sig.date.nunique() and a.n_unmatched == 0
    assert set(a.by_stratum.columns) == {"date", "vol_bucket", "alpha", "n_trades", "n_controls"}

    sig2, pool2 = _data(edge=8.0)
    c2 = M.matched_control(sig2, pool2, strata=["date", "vol_bucket"], n_per=3, seed=2)
    a2 = M.matched_alpha(sig2, pool2, c2, value="ret", cluster="date", n_boot=500, seed=3)
    assert 5 < a2.alpha < 11 and a2.ci_low > 0 and a2.p_greater > 0.99
    assert a2.trade_mean - a2.control_mean == pytest.approx(a2.alpha)


def test_cluster_bootstrap_ci_wider_than_iid_when_clustered():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(30), 40)
    x = rng.normal(0, 1, 30)[clusters] + rng.normal(0, 0.1, 1200)
    lo, hi, means = M.cluster_bootstrap_ci(x, clusters, n_boot=1000, seed=1)
    iid_lo, iid_hi, _ = M.cluster_bootstrap_ci(x, np.arange(1200), n_boot=1000, seed=1)
    assert (hi - lo) > 3 * (iid_hi - iid_lo)
    assert len(means) == 1000
    assert M.cluster_bootstrap_ci(x[:5], np.zeros(5), n_boot=10)[0] != M.cluster_bootstrap_ci(
        x[:5], np.zeros(5), n_boot=10
    )[0]  # 클러스터 1개 → NaN


def test_matched_alpha_empty_controls():
    sig, pool = _data()
    c = M.matched_control(sig.assign(date=-1), pool, strata=["date"])
    a = M.matched_alpha(sig, pool, c, value="ret")
    assert np.isnan(a.alpha) and a.n_trades == 0 and a.n_unmatched == len(sig)
