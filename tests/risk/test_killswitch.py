"""킬스위치 — scalp-it tests/test_pair_risk_guard.py·test_profit_giveback.py 킬 케이스 거울.

원본은 가상 포지션을 손절·익절시켜 실현손익을 만들었다. 여기선 그 결과(원 손익)만
직접 넣는다: 손절 1회 = 트리거 틱 99.5 → −0.5% × 10만 = −500원.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from krx_quant_core.risk import (
    KILL_CONSECUTIVE,
    KILL_DAILY_LOSS,
    KILL_GIVEBACK,
    KILL_MANUAL,
    KillSwitch,
    KillSwitchConfig,
)

D = date(2026, 8, 31)
LOSS = -500.0
WIN = 600.0


def _ks(**kw) -> KillSwitch:
    return KillSwitch(KillSwitchConfig(**kw))


def test_reason_strings_match_scalp_it():
    assert (KILL_DAILY_LOSS, KILL_CONSECUTIVE, KILL_MANUAL, KILL_GIVEBACK) == (
        "daily_loss", "consecutive_losses", "manual_stop", "profit_giveback")


def test_everything_off_by_default():
    ks = KillSwitch()
    for _ in range(20):
        ks.record_trade(-1_000_000, D)
    assert ks.check(D) is None and not ks.killed


# --------------------------------------------------------------- 일일 손실한도

def test_daily_loss_limit_triggers_on_third_loss():
    ks = _ks(daily_loss_limit_krw=1400.0)
    assert ks.record_trade(LOSS, D) is None
    assert ks.record_trade(LOSS, D) is None
    assert ks.record_trade(LOSS, D) == KILL_DAILY_LOSS
    assert ks.killed and ks.kill_reason == KILL_DAILY_LOSS
    assert ks.check(D) == KILL_DAILY_LOSS


def test_daily_loss_limit_boundary_is_inclusive():
    ks = _ks(daily_loss_limit_krw=1000.0)
    ks.record_trade(LOSS, D)
    assert ks.record_trade(LOSS, D) == KILL_DAILY_LOSS


def test_pct_limit_takes_precedence():
    cfg = KillSwitchConfig(daily_loss_limit_krw=50_000, daily_loss_limit_pct=0.001,
                           daily_capital=1_000_000)
    assert cfg.loss_limit_krw() == 1000.0
    assert KillSwitchConfig(daily_loss_limit_krw=-700).loss_limit_krw() == 700.0
    assert KillSwitchConfig(daily_loss_limit_krw=None).loss_limit_krw() == 0.0
    ks = KillSwitch(cfg)
    ks.record_trade(LOSS, D)
    assert ks.record_trade(LOSS, D) == KILL_DAILY_LOSS


def test_kill_is_sticky_even_after_recovery():
    ks = _ks(daily_loss_limit_krw=500.0)
    assert ks.record_trade(LOSS, D) == KILL_DAILY_LOSS
    ks.record_trade(10_000, D)
    assert ks.check(D) == KILL_DAILY_LOSS


# --------------------------------------------------------------- 연속 손절

def test_consecutive_losses_trigger_killswitch():
    ks = _ks(max_consecutive_losses=2)
    assert ks.record_trade(LOSS, D) is None
    assert ks.record_trade(LOSS, D) == KILL_CONSECUTIVE


def test_win_resets_consecutive_counter():
    ks = _ks(max_consecutive_losses=3)
    ks.record_trade(LOSS, D)
    ks.record_trade(LOSS, D)
    assert ks.consecutive_losses == 2
    ks.record_trade(WIN, D)
    assert ks.consecutive_losses == 0 and not ks.killed
    assert (ks.wins, ks.losses) == (1, 2)


def test_breakeven_counts_as_win():
    ks = _ks(max_consecutive_losses=1)
    assert ks.record_trade(0.0, D) is None
    assert ks.consecutive_losses == 0


def test_is_loss_override_mirrors_scalp_it_return_based_outcome():
    ks = _ks(max_consecutive_losses=1)
    # 원 손익은 음수지만 호출부가(수익률 기준) 손실 아님으로 판정.
    assert ks.record_trade(-0.5, D, is_loss=False) is None
    assert ks.record_trade(0.0, D, is_loss=True) == KILL_CONSECUTIVE


def test_consecutive_checked_before_daily_loss():
    ks = _ks(max_consecutive_losses=2, daily_loss_limit_krw=1000.0)
    ks.record_trade(LOSS, D)
    assert ks.record_trade(LOSS, D) == KILL_CONSECUTIVE


def test_partial_realized_does_not_touch_streak():
    ks = _ks(max_consecutive_losses=2, daily_loss_limit_krw=1000.0)
    ks.record_trade(LOSS, D)
    assert ks.record_realized(300.0, D) is None
    assert ks.consecutive_losses == 1, "1차 분할 익절이 연속손절을 리셋하면 안 된다"
    assert ks.realized_krw == -200.0
    assert ks.record_realized(-800.0, D) == KILL_DAILY_LOSS


# --------------------------------------------------------------- 일일 리셋

def test_daily_reset_clears_counters_and_kill():
    ks = _ks(max_consecutive_losses=2)
    ks.record_trade(LOSS, D)
    ks.record_trade(LOSS, D)
    assert ks.killed
    assert ks.check(D + timedelta(days=1)) is None
    assert not ks.killed and ks.kill_reason == ""
    assert ks.consecutive_losses == 0 and ks.realized_krw == 0.0
    assert ks.peak_realized_krw == 0.0 and (ks.wins, ks.losses) == (0, 0)


def test_first_observation_is_not_a_reset():
    ks = _ks()
    ks.force_stop()                      # 날짜 관측 전 수동 킬
    assert ks.check(D) == KILL_MANUAL    # 첫 관측은 기록만 — 원본 _observe_date


def test_accepts_datetime_as_same_day():
    ks = _ks(max_consecutive_losses=1)
    ks.record_trade(LOSS, datetime(2026, 8, 31, 10, 0))
    assert ks.check(datetime(2026, 8, 31, 15, 0)) == KILL_CONSECUTIVE
    assert ks.check(D) == KILL_CONSECUTIVE
    assert ks.check(datetime(2026, 9, 1, 9, 0)) is None


# --------------------------------------------------------------- 수동 정지

def test_manual_stop_file_triggers_kill(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("stop", encoding="utf-8")
    ks = _ks(stop_file=str(stop))
    assert ks.check(D) == KILL_MANUAL


def test_no_stop_file_no_kill(tmp_path):
    ks = _ks(stop_file=tmp_path / "absent")
    assert ks.check(D) is None


def test_sticky_stop_file_survives_removal(tmp_path):
    stop = tmp_path / "pair_STOP"
    stop.touch()
    ks = _ks(stop_file=stop)
    assert ks.check(D) == KILL_MANUAL
    stop.unlink()
    assert ks.check(D) == KILL_MANUAL
    assert ks.check(D + timedelta(days=1)) is None


def test_non_sticky_stop_file_like_daytrade_live_stop(tmp_path):
    stop = tmp_path / "live_STOP"
    stop.touch()
    ks = _ks(stop_file=stop, sticky_stop_file=False, max_consecutive_losses=1)
    assert ks.check(D) == KILL_MANUAL and not ks.killed
    stop.unlink()
    assert ks.check(D) is None
    stop.touch()
    # 파일이 있는 동안엔 손실 트리거를 보지 않지만 막고는 있다.
    assert ks.record_trade(LOSS, D) == KILL_MANUAL and not ks.killed
    stop.unlink()
    assert ks.check(D) == KILL_CONSECUTIVE


def test_stop_file_beats_consecutive(tmp_path):
    stop = tmp_path / "STOP"
    ks = _ks(stop_file=stop, max_consecutive_losses=1)
    stop.touch()
    assert ks.record_trade(LOSS, D) == KILL_MANUAL


def test_force_stop_overrides_existing_reason():
    ks = _ks(max_consecutive_losses=1)
    ks.record_trade(LOSS, D)
    ks.force_stop()
    assert ks.kill_reason == KILL_MANUAL and ks.check(D) == KILL_MANUAL


# --------------------------------------------------------------- 이익 반납(천이오 B-3)

def _gb(**kw) -> KillSwitch:
    return _ks(profit_giveback_ratio=0.5, profit_giveback_min_krw=10_000, **kw)


def test_half_giveback_kills_the_day():
    ks = _gb()
    ks.record_trade(100_000, D)
    assert ks.record_trade(-50_000, D) == KILL_GIVEBACK


def test_shallow_giveback_does_not_kill():
    ks = _gb()
    ks.record_trade(100_000, D)
    assert ks.record_trade(-40_000, D) is None


def test_small_peak_is_ignored():
    ks = _gb()
    ks.record_trade(5_000, D)
    assert ks.record_trade(-5_000, D) is None


def test_giveback_off_by_default():
    ks = _ks()
    ks.record_trade(100_000, D)
    assert ks.record_trade(-100_000, D) is None


def test_giveback_checked_before_stop_file_and_consecutive(tmp_path):
    stop = tmp_path / "STOP"
    ks = _gb(stop_file=stop, max_consecutive_losses=1)
    ks.record_trade(100_000, D)
    stop.touch()
    assert ks.record_trade(-60_000, D) == KILL_GIVEBACK


def test_peak_tracks_intraday_high():
    ks = _gb()
    ks.record_trade(30_000, D)
    ks.record_trade(50_000, D)
    ks.record_trade(-20_000, D)
    assert ks.peak_realized_krw == 80_000
    assert ks.record_trade(-19_999, D) is None       # 40,001 > 40,000
    assert ks.record_trade(-1, D) == KILL_GIVEBACK   # 정확히 절반
