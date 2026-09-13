"""stats.fragility — swing-it ``tests/test_diagnostics_rdist.py`` 의 fragility 부분 이식.

손으로 만든 배열에서 괴물 의존도·최장 연패·꼬리제거·승리조건부가 정확한지 본다.
"""

from __future__ import annotations

import numpy as np

from krx_quant_core.stats.fragility import (
    fragility_report,
    max_loss_streak,
    median_trade,
    monster_share,
    tail_removal,
    win_conditional,
)


def test_monster_share_hand_built():
    # 총합=10, 상위 2건=[5,3]=8 → 0.8
    R = np.array([5.0, 3.0, 1.0, 1.0, -1.0, 1.0])
    assert abs(monster_share(R, k=2) - 0.8) < 1e-12


def test_monster_share_zero_total_is_nan():
    assert np.isnan(monster_share(np.array([1.0, -1.0]), k=1))
    assert np.isnan(monster_share(np.array([]), k=1))


def test_max_loss_streak_hand_built():
    # 손실(≤0) 연속: [win, loss, loss, loss, win, loss] → 최장 3
    assert max_loss_streak(np.array([1.0, -1.0, -0.5, -1.0, 2.0, -1.0])) == 3


def test_max_loss_streak_orders_by_entry():
    R = np.array([-1.0, 2.0, -1.0])
    entry = np.array(["2023-01-01", "2023-03-01", "2023-02-01"])
    assert max_loss_streak(R, entry=entry) == 2


def test_max_loss_streak_no_losses_and_threshold():
    assert max_loss_streak(np.array([1.0, 2.0])) == 0
    # loss_thr=0.5 면 0.3 도 손실로 센다
    assert max_loss_streak(np.array([0.3, 0.2, 1.0]), loss_thr=0.5) == 2
    # 0 은 손실(≤0) 쪽 — 무승부도 연패를 끊지 않는다
    assert max_loss_streak(np.array([-1.0, 0.0, -1.0])) == 3


def test_tail_removal_flips_expectancy():
    tr = tail_removal(np.array([-1.0, -1.0, -1.0, 10.0]), k=1)
    assert abs(tr["expectancy_full"] - 1.75) < 1e-12
    assert abs(tr["expectancy_ex"] - (-1.0)) < 1e-12
    assert abs(tr["cum_ex"] - (-3.0)) < 1e-12


def test_tail_removal_k_exceeds_n_is_nan_safe():
    tr = tail_removal(np.array([1.0, 2.0]), k=5)
    assert np.isnan(tr["expectancy_ex"]) and tr["cum_ex"] == 0.0


def test_win_conditional_fields():
    wc = win_conditional(np.array([-1.0, -1.0, 2.0, 4.0]))
    assert abs(wc["win_rate"] - 0.5) < 1e-12
    assert abs(wc["median_win"] - 3.0) < 1e-12
    assert abs(wc["mean_win"] - 3.0) < 1e-12
    assert abs(wc["max_win"] - 4.0) < 1e-12
    assert abs(wc["median_loss"] - (-1.0)) < 1e-12


def test_median_trade_and_empty():
    assert median_trade(np.array([-1.0, 0.5, 9.0])) == 0.5
    assert np.isnan(median_trade(np.array([np.nan])))


def test_fragility_report_shape():
    fr = fragility_report(np.array([-1.0, -1.0, 2.0, 4.0, -1.0, 6.0, np.nan]))
    assert set(fr) == {"n", "monster_share", "max_loss_streak", "tail_removal",
                       "median_trade", "win_conditional"}
    assert fr["n"] == 6
