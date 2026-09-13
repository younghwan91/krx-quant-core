"""일일 킬스위치 — 손실한도·연속손절·이익반납·수동정지. 시계 없는 순수 상태기계.

scalp-it ``realtime/risk_guard.py`` ``RiskGuard`` 에서 **킬 판정만** 뽑아냈다. 가상
포지션·체결 가정·국면 배수 같은 전략 쪽 회계는 가져오지 않는다. 겹치는 개념은
의미를 원본과 **정확히** 맞췄다 — scalp-it 이 나중에 이쪽으로 위임해도 킬 시점이
한 틱도 달라지지 않게:

* **판정 순서**(:meth:`KillSwitch.check` 한 번에 하나만 발동):
  이익반납 → 수동정지 파일 → 연속손절 → 일일손실. 이미 킬이면 다시 보지 않는다.
* **킬은 그날 내내 유지된다.** 정지 파일을 지워도, 손실을 만회해도 안 풀린다.
  날짜가 바뀌어야(:meth:`check`·:meth:`record_trade` 에 다른 날짜가 들어와야) 리셋된다.
* **첫 관측일은 리셋이 아니다** — 원본 ``_observe_date`` 와 같다.
* 손실한도 ``loss_limit_krw()``: ``daily_loss_limit_pct`` 가 참이면 ``|자본×pct|``,
  아니면 ``|daily_loss_limit_krw|``. 0 이면 끔.
* 비교는 원본의 ``_EPS = 1e-9`` 여유를 그대로 둔다(``<= 한도 + eps``).
* 이익반납: 정점(``peak_realized_krw``)이 ``profit_giveback_min_krw`` 이상이고 0 초과일 때
  ``realized <= peak × ratio`` 면 킬. 정점은 판정 때마다 갱신된다.
* 수동 킬 :meth:`force_stop` 은 원본처럼 이미 킬이어도 사유를 ``manual_stop`` 으로 덮는다.

daytrade-it ``data/live_STOP`` 관례와의 차이
===========================================

daytrade-it 은 파일이 **있는 동안만** 주문을 거부하고, 지우면 즉시 재개된다(비고정).
scalp-it ``data/pair_STOP`` 은 한 번 보면 그날 끝(고정)이다. 둘 다 실운영 규약이라
하나로 뭉개지 않고 :attr:`KillSwitchConfig.sticky_stop_file` 로 고른다. 기본은 scalp-it
쪽(고정) — 킬스위치라는 이름에 맞는 보수적 쪽이다. ``stop_file`` 기본값은 ``None``
(끔)이다: 상대경로 기본값은 실행 디렉터리에 따라 조용히 다른 파일을 보게 되므로
각 레포가 명시한다(scalp-it ``"data/pair_STOP"``, daytrade-it ``"data/live_STOP"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

__all__ = [
    "KILL_CONSECUTIVE",
    "KILL_DAILY_LOSS",
    "KILL_GIVEBACK",
    "KILL_MANUAL",
    "KillSwitch",
    "KillSwitchConfig",
]

# 킬 사유 문자열 — scalp-it risk_guard 와 같은 값(로그·집계가 이 문자열로 거른다).
KILL_DAILY_LOSS = "daily_loss"
KILL_CONSECUTIVE = "consecutive_losses"
KILL_MANUAL = "manual_stop"
#: 정점 이익 대비 반납 하드스톱(scalp-it 천이오 B-3 "50% rule").
KILL_GIVEBACK = "profit_giveback"

_EPS = 1e-9


@dataclass(frozen=True)
class KillSwitchConfig:
    """킬스위치 파라미터. 모든 트리거는 0/None 이면 꺼진다."""

    daily_loss_limit_krw: float | None = 0.0   # 일일 손실한도(원). 0/None = 끔
    daily_loss_limit_pct: float | None = None  # 있으면 자본 대비 %로 한도 산출(우선)
    daily_capital: float = 1_000_000.0         # % 한도 기준 자본(원)
    max_consecutive_losses: int = 0            # 연속 손절 상한. 0 = 끔
    profit_giveback_ratio: float = 0.0         # 정점 대비 이 비율 이하로 떨어지면 킬. 0 = 끔
    profit_giveback_min_krw: float = 0.0       # 정점이 이만큼은 돼야 반납 규칙을 본다
    stop_file: str | Path | None = None        # 존재 시 수동 킬. None = 끔
    sticky_stop_file: bool = True              # True=scalp-it(그날 고정), False=daytrade-it

    def loss_limit_krw(self) -> float:
        """실효 일일 손실한도(원). pct 가 있으면 자본×pct, 없으면 절대액."""
        if self.daily_loss_limit_pct:
            return abs(self.daily_capital * self.daily_loss_limit_pct)
        return abs(self.daily_loss_limit_krw or 0.0)


def _as_date(on: date) -> date:
    # datetime 은 date 의 서브클래스라 그대로 비교하면 같은 날도 다르다고 나온다.
    return on.date() if isinstance(on, datetime) else on


class KillSwitch:
    """하루 단위 킬스위치. 시계를 모른다 — 날짜는 호출부가 넘긴다."""

    def __init__(self, config: KillSwitchConfig | None = None) -> None:
        self.config = config or KillSwitchConfig()
        self.day: date | None = None
        self.realized_krw: float = 0.0
        self.peak_realized_krw: float = 0.0
        self.consecutive_losses: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.killed: bool = False
        self.kill_reason: str = ""

    # ----- 일일 리셋 ----------------------------------------------------------

    def observe(self, on: date) -> None:
        """날짜를 관측한다. 처음이면 기록만, 바뀌었으면 카운터·킬을 리셋."""
        d = _as_date(on)
        if self.day is None:
            self.day = d
            return
        if d != self.day:
            self.day = d
            self.realized_krw = 0.0
            self.peak_realized_krw = 0.0
            self.consecutive_losses = 0
            self.wins = 0
            self.losses = 0
            self.killed = False
            self.kill_reason = ""

    # ----- 기록 --------------------------------------------------------------

    def record_trade(self, pnl_krw: float, on: date, *, is_loss: bool | None = None) -> str | None:
        """청산 1건을 기록하고 킬 상태를 갱신한다. 반환은 :meth:`check` 와 같다.

        승패는 기본적으로 ``pnl_krw < -eps`` 로 가른다. scalp-it 은 원 손익이 아니라
        **수익률**(분할 매도가 있었으면 합계 원 손익)로 가르므로, 그 판정을 그대로
        옮기려면 ``is_loss`` 를 명시해 넘긴다.
        """
        self.observe(on)
        self.realized_krw += pnl_krw
        loss = (pnl_krw < -_EPS) if is_loss is None else bool(is_loss)
        if loss:
            self.losses += 1
            self.consecutive_losses += 1
        else:
            self.wins += 1
            self.consecutive_losses = 0
        return self._refresh()

    def record_realized(self, pnl_krw: float, on: date) -> str | None:
        """승패 없이 실현손익만 더한다 — 분할 매도 1차분(scalp-it ``_maybe_partial``).

        포지션이 아직 안 끝났으니 연속손절 카운터는 건드리지 않는다. 1차 익절로
        카운터가 리셋되면 연속손절 킬이 무력화된다.
        """
        self.observe(on)
        self.realized_krw += pnl_krw
        return self._refresh()

    def force_stop(self) -> None:
        """수동 킬(파일 없이 즉시). 이미 킬이어도 사유를 manual_stop 으로 덮는다."""
        self._trigger(KILL_MANUAL)

    # ----- 판정 --------------------------------------------------------------

    def check(self, on: date) -> str | None:
        """``on`` 날짜 기준 킬 사유, 아니면 None. 신규 진입 직전에 부른다."""
        self.observe(on)
        return self._refresh()

    def stop_file_present(self) -> bool:
        sf = self.config.stop_file
        return bool(sf) and Path(sf).exists()

    def _refresh(self) -> str | None:
        cfg = self.config
        self.peak_realized_krw = max(self.peak_realized_krw, self.realized_krw)
        if self.killed:
            return self.kill_reason
        if (cfg.profit_giveback_ratio > 0
                and self.peak_realized_krw >= cfg.profit_giveback_min_krw
                and self.peak_realized_krw > 0
                and self.realized_krw <= self.peak_realized_krw * cfg.profit_giveback_ratio + _EPS):
            self._trigger(KILL_GIVEBACK)
            return self.kill_reason
        if self.stop_file_present():
            if cfg.sticky_stop_file:
                self._trigger(KILL_MANUAL)
                return self.kill_reason
            # 비고정(daytrade-it live_STOP): 파일이 있는 동안만 막는다. 나머지 트리거는
            # 원본 elif 순서대로 이번 판정에선 보지 않고, 파일이 사라진 뒤 판정에서 본다.
            return KILL_MANUAL
        if (cfg.max_consecutive_losses > 0
                and self.consecutive_losses >= cfg.max_consecutive_losses):
            self._trigger(KILL_CONSECUTIVE)
        elif (cfg.loss_limit_krw() > 0
              and self.realized_krw <= -cfg.loss_limit_krw() + _EPS):
            self._trigger(KILL_DAILY_LOSS)
        return self.kill_reason if self.killed else None

    def _trigger(self, reason: str) -> None:
        self.killed = True
        self.kill_reason = reason
