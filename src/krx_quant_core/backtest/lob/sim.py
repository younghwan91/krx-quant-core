"""초 격자 경로 위 롱 에피소드 시뮬레이터 — 시장가 진입, 규칙 청산.

scalp-it 82번 ``dataset._strength_exit_labels`` 와 81번 ``exit_at/trade``, 80-2번
``of_scratch.sim`` 의 청산 판정을 한 커널로 합쳤다. 기본값(지연 1초, 체결강도 꺾임 + 최대
보유, 손절·익절 없음)은 82번 라벨과 **같은 숫자**다.

체결 가정(원본 그대로, 낙관적이다):

- 진입: 신호 초 ``t`` 의 ``t + latency`` 초 **매도1호가**(전방채움 값) 전량 체결.
- 청산가: 판정 초의 **매수1호가**. 익절만 지정가라 그 초 체결 최고가가 목표가를
  **넘어야**(``hi > tp``) 목표가에 체결.
- 판정 순서(한 초 안): 익절 → 손절 → 체결강도 꺾임. 어느 것도 없으면 ``entry + max_hold`` 초.
- 호가 잔량·충격은 보지 않는다 — 소액 가정. 필요하면 :mod:`..orderbook` 의 스윕으로 보정.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from krx_quant_core.costs.model import round_trip_cost

from ._jit import njit
from .ticks import TickTable, stock_tick_table, tick_of

__all__ = [
    "EXIT_NONE",
    "EXIT_STOP",
    "EXIT_STRENGTH",
    "EXIT_TAKE",
    "EXIT_TIME",
    "SimResult",
    "simulate_exits",
]

#: 청산 사유 코드(``SimResult.reason``).
EXIT_NONE = 0  #: 진입 불가(격자 끝·호가 없음) 또는 청산가 없음 — ``net`` 은 nan.
EXIT_STRENGTH = 1  #: 진입 뒤 체결강도 최고치 − ``strength_drop`` 이하.
EXIT_TIME = 2  #: 최대 보유 시간(또는 격자 끝).
EXIT_STOP = 3  #: 매수1호가 ≤ 진입가 − ``stop_ticks`` 틱.
EXIT_TAKE = 4  #: 체결 최고가 > 진입가 + ``take_ticks`` 틱 → 목표가 체결.


@dataclass(frozen=True)
class SimResult:
    """신호마다 한 칸. 진입 불가면 ``net``·가격은 nan, ``hold`` 0, ``reason`` ``EXIT_NONE``."""

    net: NDArray[np.float64]
    hold: NDArray[np.int32]
    reason: NDArray[np.int8]
    entry_price: NDArray[np.float64]
    exit_price: NDArray[np.float64]


@njit
def _simulate(
    secs: NDArray[np.int64],
    bid: NDArray[np.float64],
    ask: NDArray[np.float64],
    stg: NDArray[np.float64],
    hi: NDArray[np.float64],
    latency: int,
    sdrop: float,
    max_hold: int,
    stop_ticks: float,
    take_ticks: float,
    cost: float,
    bounds: NDArray[np.float64],
    ticks: NDArray[np.float64],
    top: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int8],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    n = secs.shape[0]
    net = np.full(n, np.nan)
    hold = np.zeros(n, np.int32)
    reason = np.zeros(n, np.int8)
    epx = np.full(n, np.nan)
    xpx = np.full(n, np.nan)
    L = bid.shape[0]
    use_stg = sdrop == sdrop
    for i in range(n):
        e = secs[i] + latency
        if e < 0 or e >= L - 1:
            continue
        a = ask[e]
        if not (a > 0):
            continue
        tk = tick_of(a, bounds, ticks, top)
        tp = a + take_ticks * tk
        stop = a - stop_ticks * tk
        peak = stg[e]
        end = min(e + max_hold, L - 1)
        px = bid[end]
        xs = end
        why = EXIT_TIME
        for s in range(e + 1, end + 1):
            if take_ticks > 0:
                h = hi[s]
                if h == h and h > tp:
                    px = tp
                    xs = s
                    why = EXIT_TAKE
                    break
            if stop_ticks > 0 and bid[s] <= stop:
                px = bid[s]
                xs = s
                why = EXIT_STOP
                break
            if use_stg:
                v = stg[s]
                if v == v:
                    if not (peak == peak) or v > peak:
                        peak = v
                    if v <= peak - sdrop:
                        px = bid[s]
                        xs = s
                        why = EXIT_STRENGTH
                        break
        epx[i] = a
        if px > 0:
            net[i] = px / a - 1.0 - cost
            hold[i] = xs - e
            reason[i] = why
            xpx[i] = px
    return net, hold, reason, epx, xpx


def _f64(a: Any, name: str, n: int) -> NDArray[np.float64]:
    arr = np.ascontiguousarray(a, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be 1-D with the same length as bid ({n}), got {arr.shape}")
    return arr


def simulate_exits(
    entry_secs: Any,
    bid: Any,
    ask: Any,
    strength: Any = None,
    *,
    max_hold: int,
    latency: int = 1,
    strength_drop: float | None = None,
    stop_ticks: float = 0,
    take_ticks: float = 0,
    hi: Any = None,
    cost: float | None = None,
    trade_date: date | None = None,
    market: Any = None,
    tick_table: TickTable | None = None,
) -> SimResult:
    """신호 초마다 에피소드 하나(서로 겹쳐도 각자 독립 — 포지션 제한·쿨다운은 호출부 몫).

    Args:
        entry_secs: 신호 초(격자 인덱스, int).
        bid, ask, strength: 격자 경로(:func:`~.grid.build_second_grid` 의 ``path``).
        max_hold: 진입 뒤 최대 보유 초.
        latency: 신호 → 진입 지연 초(원본 1).
        strength_drop: 체결강도 꺾임 폭. ``None`` 이면 규칙 끔.
        stop_ticks, take_ticks: 진입가 호가단위 기준 손절·익절 틱 수. 0 이면 끔.
        hi: 초당 체결 최고가 경로 — ``take_ticks`` 를 쓰면 필수.
        cost: 왕복 비용률. 안 주면 ``trade_date``·``market`` 으로
            :func:`krx_quant_core.costs.round_trip_cost` 를 쓴다(ETF 처럼 세금이 다르면 직접 준다).
        tick_table: 손절·익절 틱 크기 표. 기본 주식(kiwoom-client 정본).

    Returns:
        :class:`SimResult`. ``net = exit/entry − 1 − cost``, ``hold`` 는 진입부터 청산까지 초.
    """
    if cost is None:
        if trade_date is None or market is None:
            raise ValueError("pass cost, or trade_date and market for round_trip_cost")
        cost = round_trip_cost(trade_date, market)
    if max_hold < 1 or latency < 0:
        raise ValueError("max_hold must be >= 1 and latency >= 0")
    b = np.ascontiguousarray(bid, dtype=np.float64)
    if b.ndim != 1:
        raise ValueError("bid must be 1-D")
    n = b.shape[0]
    a = _f64(ask, "ask", n)
    use_stg = strength_drop is not None
    if use_stg and strength is None:
        raise ValueError("strength path is required when strength_drop is given")
    s = _f64(strength, "strength", n) if strength is not None else np.full(n, np.nan)
    if take_ticks > 0 and hi is None:
        raise ValueError("hi path is required when take_ticks > 0")
    h = _f64(hi, "hi", n) if hi is not None else np.full(n, np.nan)
    secs = np.ascontiguousarray(entry_secs, dtype=np.int64).reshape(-1)
    tt = tick_table or stock_tick_table()
    net, hold, reason, epx, xpx = _simulate(
        secs, b, a, s, h, int(latency),
        float(strength_drop) if use_stg else np.nan,
        int(max_hold), float(stop_ticks), float(take_ticks), float(cost),
        tt.bounds, tt.ticks, tt.top,
    )  # fmt: skip
    return SimResult(net=net, hold=hold, reason=reason, entry_price=epx, exit_price=xpx)
