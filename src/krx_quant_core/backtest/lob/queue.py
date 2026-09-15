"""지정가 주문 체결 시뮬레이터 — 호가 대기열 위치(queue position) 모델.

:mod:`..fills` 의 ``touch``/``through`` 는 대기열 위치를 모른다는 전제의 양 극단이다
(맨 앞 / 맨 뒤). 여기서는 10단계 호가 스냅샷과 가격별 체결량으로 **내 앞 잔량**을 추적한다.
대기열 모델의 뼈대는 hftbacktest(nkaz001, MIT)의 L2 모델을 따랐다:

- ``risk_averse`` — 내 앞 잔량은 **체결로만** 줄어든다. 잔량 감소 중 체결로 설명 안 되는
  부분(취소)은 전부 내 뒤에서 났다고 본다. 가장 보수적인 대기열 모델.
- ``prob_power``/``prob_log`` — 취소를 내 앞·뒤에 확률로 나눈다. 뒤에 몫이 갈 확률
  ``p = f(back) / (f(back) + f(front))``, ``f(x) = x**power`` 또는 ``log(1+x)``.
  hftbacktest ``ProbQueueModel`` 의 추정식 ``front − (1−p)·chg + min(back − p·chg, 0)``.
- ``touch`` — 앞 잔량 0(맨 앞). 내 가격 체결량만큼 체결. :func:`..fills.limit_buy_filled` 의 touch.
- ``through`` — 앞 잔량 무한(맨 뒤). 가격이 관통하거나 반대 호가가 내 가격까지 와야 체결.

**데이터 한계와 그에 따른 가정**(키움 수집: 시각 1초 절삭, 호가 초당 스냅샷 1개):

1. 초 안 순서를 모른다. 스냅샷은 그 초 **끝** 상태로 본다.
2. 주문은 ``place_sec + latency`` 초(= ``e``)에 들어간다. 도착 시 대기열은 **``e`` 초 끝
   스냅샷**의 내 가격 잔량 전부를 내 앞으로 친다(``e`` 초 안에 나보다 늦게 온 주문까지 앞에
   세우는 보수 쪽 근사). 체결 판정은 ``e+1`` 초부터.
3. 도착 시 반대 호가가 내 가격 이내면 그 스냅샷 잔량만큼 **즉시 체결**(테이커), 남은 수량은
   내 가격에 새 최우선 호가로 선다(앞 잔량 0).
4. 매 초 판정 순서: 관통(체결가가 내 가격을 넘거나 반대 1호가가 내 가격 이내 → 남은 전량
   내 가격에 체결) → 내 가격 체결량으로 앞 잔량 소진·부분 체결 → 스냅샷 잔량 변화로 앞 잔량 갱신.
5. 내 가격이 보이는 10단계 밖(더 깊음)이면 앞 잔량을 모른다 — 보이게 되는 첫 초에 그 잔량
   전부를 앞으로 친다. 그 전에는 관통으로만 체결.
6. 지연은 정수 초다. 초 미만 지연은 이 데이터로 모델링할 수 없다 — 1초 지연이 최선의 근사.
7. 내 주문이 시장에 주는 영향(내 잔량이 호가에 더해져 남의 행동이 바뀌는 것)은 없다고 본다.

체결량이 크면 부분 체결이 된다. 비용(수수료·세금)은 가격만 돌려주고 호출부가 뺀다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._jit import njit
from .book import BookGrid, LevelTrades

__all__ = [
    "QUEUE_MODELS",
    "STATUS_CANCELED",
    "STATUS_FILLED",
    "STATUS_NOT_PLACED",
    "STATUS_PARTIAL",
    "LimitFillResult",
    "QueueModel",
    "simulate_limit_orders",
]

QueueModel = Literal["touch", "through", "risk_averse", "prob_power", "prob_log"]
QUEUE_MODELS: tuple[str, ...] = ("touch", "through", "risk_averse", "prob_power", "prob_log")
_M_TOUCH, _M_THROUGH, _M_RA, _M_POW, _M_LOG = range(5)

STATUS_NOT_PLACED = 0  #: 도착 초가 격자 밖이거나 호가 없음.
STATUS_FILLED = 1  #: 전량 체결.
STATUS_PARTIAL = 2  #: 대기 시간 끝에 일부만 체결, 나머지 취소.
STATUS_CANCELED = 3  #: 대기 시간 끝까지 한 주도 체결 안 됨.


@dataclass(frozen=True)
class LimitFillResult:
    """주문마다 한 칸. 체결 없으면 ``avg_price`` nan, ``first_fill_sec`` −1."""

    filled_qty: NDArray[np.float64]
    avg_price: NDArray[np.float64]
    taker_qty: NDArray[np.float64]  #: 도착 즉시 반대 호가를 치고 체결된 수량(수수료 구분용).
    first_fill_sec: NDArray[np.int64]
    done_sec: NDArray[np.int64]  #: 전량 체결 초 또는 취소 초.
    status: NDArray[np.int8]


@njit
def _qty_at(px: NDArray[np.float64], qty: NDArray[np.float64], price: float) -> float:
    for k in range(px.shape[0]):
        if px[k] == price:
            return qty[k]
    return np.nan


@njit
def _trade_vol(
    ptr: NDArray[np.int64], tpx: NDArray[np.float64], vol: NDArray[np.float64], s: int, price: float
) -> float:
    for j in range(ptr[s], ptr[s + 1]):
        if tpx[j] == price:
            return vol[j]
    return 0.0


@njit
def _qfunc(x: float, model: int, power: float) -> float:
    if x < 0.0:
        x = 0.0
    if model == _M_POW:
        return x**power
    return math.log1p(x)


@njit
def _simulate_limit(
    side: NDArray[np.int8],
    price: NDArray[np.float64],
    qty: NDArray[np.float64],
    place: NDArray[np.int64],
    latency: NDArray[np.int64],
    max_wait: int,
    model: int,
    power: float,
    bid_px: NDArray[np.float64],
    bid_qty: NDArray[np.float64],
    ask_px: NDArray[np.float64],
    ask_qty: NDArray[np.float64],
    ptr: NDArray[np.int64],
    tpx: NDArray[np.float64],
    buy_vol: NDArray[np.float64],
    sell_vol: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int8],
]:
    m = side.shape[0]
    n = bid_px.shape[0]
    levels = bid_px.shape[1]
    filled_o = np.zeros(m)
    avg_o = np.full(m, np.nan)
    taker_o = np.zeros(m)
    first_o = np.full(m, -1, np.int64)
    done_o = np.full(m, -1, np.int64)
    status_o = np.zeros(m, np.int8)
    for i in range(m):
        e = place[i] + latency[i]
        if e < 0 or e >= n:
            continue
        buy = side[i] > 0
        P = price[i]
        Q = qty[i]
        if buy:
            opx, oqty, mpx, mqty = ask_px, ask_qty, bid_px, bid_qty
        else:
            opx, oqty, mpx, mqty = bid_px, bid_qty, ask_px, ask_qty
        if not (mpx[e, 0] == mpx[e, 0] and opx[e, 0] == opx[e, 0]):
            continue
        filled = 0.0
        notional = 0.0
        first = -1
        # 도착: 반대 호가 스윕(테이커)
        for k in range(levels):
            p = opx[e, k]
            if p != p:
                continue
            if (buy and p <= P) or ((not buy) and p >= P):
                take = min(Q - filled, oqty[e, k])
                if take > 0:
                    filled += take
                    notional += take * p
        taker = filled
        if filled > 0:
            first = e
        done = -1
        if filled >= Q:
            done = e
        else:
            prev = _qty_at(mpx[e], mqty[e], P)
            best = mpx[e, 0]
            improves = (buy and P > best) or ((not buy) and P < best)
            if taker > 0 or improves:
                prev = 0.0
            if model == _M_TOUCH:
                front = 0.0
                known = True
            elif model == _M_THROUGH:
                front = np.inf
                known = True
            elif prev == prev:
                front = prev
                known = True
            else:
                front = np.inf
                known = False
            end = min(e + max_wait, n - 1)
            for s in range(e + 1, end + 1):
                rem = Q - filled
                # 1) 관통
                if buy:
                    through = (lo[s] == lo[s] and lo[s] < P) or (
                        opx[s, 0] == opx[s, 0] and opx[s, 0] <= P
                    )
                else:
                    through = (hi[s] == hi[s] and hi[s] > P) or (
                        opx[s, 0] == opx[s, 0] and opx[s, 0] >= P
                    )
                if through:
                    filled += rem
                    notional += rem * P
                    if first < 0:
                        first = s
                    done = s
                    break
                # 2) 내 가격 체결(매수 주문은 매도 주도 체결이 대기열을 줄인다)
                tv = _trade_vol(ptr, tpx, sell_vol if buy else buy_vol, s, P)
                if tv > 0 and model != _M_THROUGH and known:
                    if model == _M_TOUCH:
                        ex = min(rem, tv)
                    else:
                        front -= tv
                        ex = 0.0
                        if front < 0:
                            ex = min(rem, -front)
                            front = 0.0
                    if ex > 0:
                        filled += ex
                        notional += ex * P
                        if first < 0:
                            first = s
                        if filled >= Q:
                            done = s
                            break
                # 3) 스냅샷 잔량 변화
                new = _qty_at(mpx[s], mqty[s], P)
                if new != new:
                    b = mpx[s, 0]
                    if b == b and ((buy and P > b) or ((not buy) and P < b)):
                        new = 0.0
                if model == _M_RA or model == _M_POW or model == _M_LOG:
                    if new == new:
                        if not known:
                            front = new
                            known = True
                        elif model == _M_RA or prev != prev:
                            front = min(front, new)
                        else:
                            chg = prev - new - tv
                            if chg < 0:
                                front = min(front, new)
                            else:
                                # 체결로 줄기 전 앞 잔량은 front + tv 였다 — 그만큼은 뒤가 아니다.
                                back = prev - tv - front
                                if back < 0:
                                    back = 0.0
                                fb = _qfunc(back, model, power)
                                ff = _qfunc(front, model, power)
                                pr = fb / (fb + ff) if fb + ff > 0 else 1.0
                                if not math.isfinite(pr):
                                    pr = 1.0
                                est = front - (1.0 - pr) * chg + min(back - pr * chg, 0.0)
                                if est < 0:
                                    est = 0.0
                                front = min(est, new)
                prev = new
            if done < 0:
                done = end
        filled_o[i] = filled
        taker_o[i] = taker
        first_o[i] = first
        done_o[i] = done
        if filled > 0:
            avg_o[i] = notional / filled
        if filled >= Q:
            status_o[i] = STATUS_FILLED
        elif filled > 0:
            status_o[i] = STATUS_PARTIAL
        else:
            status_o[i] = STATUS_CANCELED
    return filled_o, avg_o, taker_o, first_o, done_o, status_o


def _vec(a: Any, m: int, dtype: Any, name: str) -> NDArray[Any]:
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim == 0:
        return np.full(m, arr, dtype=dtype)
    arr = np.ascontiguousarray(arr.reshape(-1))
    if arr.shape[0] != m:
        raise ValueError(f"{name} must be scalar or length {m}")
    return arr


def simulate_limit_orders(
    side: Any,
    price: Any,
    qty: Any,
    place_sec: Any,
    book: BookGrid,
    trades: LevelTrades,
    *,
    max_wait: int,
    latency: Any = 1,
    queue_model: QueueModel = "risk_averse",
    power: float = 2.0,
) -> LimitFillResult:
    """지정가 주문 묶음을 서로 독립으로 시뮬레이션(내 주문끼리 대기열을 다투지 않는다).

    Args:
        side: ``+1`` 매수 / ``-1`` 매도(스칼라 또는 주문별).
        price: 지정가(유효 호가여야 한다 — 스냅샷 가격과 정확히 같아야 대기열을 찾는다).
        qty: 주문 수량(주).
        place_sec: 주문을 낸 격자 초.
        book, trades: :func:`~.book.build_book_grid`·:func:`~.book.build_level_trades`.
        max_wait: 도착 뒤 최대 대기 초. 넘기면 남은 수량 취소.
        latency: 주문 → 도착 지연(정수 초, 스칼라 또는 주문별 — 실측 분포에서 뽑아 넣는다).
        queue_model: :data:`QUEUE_MODELS` 중 하나.
        power: ``prob_power`` 의 지수.
    """
    if queue_model not in QUEUE_MODELS:
        raise ValueError(f"queue_model must be one of {QUEUE_MODELS}: {queue_model!r}")
    if max_wait < 0:
        raise ValueError("max_wait must be >= 0")
    if trades.ptr.shape[0] != book.n + 1:
        raise ValueError("book and trades must share the same second grid")
    sides = np.atleast_1d(np.asarray(side))
    prices = np.atleast_1d(np.asarray(price))
    places = np.atleast_1d(np.asarray(place_sec))
    m = max(sides.shape[0], prices.shape[0], places.shape[0], np.atleast_1d(qty).shape[0])
    sd = _vec(side, m, np.int8, "side")
    if not np.isin(sd, (-1, 1)).all():
        raise ValueError("side must be +1 or -1")
    lat = _vec(latency, m, np.int64, "latency")
    if (lat < 0).any():
        raise ValueError("latency must be >= 0")
    out = _simulate_limit(
        sd,
        _vec(price, m, np.float64, "price"),
        _vec(qty, m, np.float64, "qty"),
        _vec(place_sec, m, np.int64, "place_sec"),
        lat,
        int(max_wait),
        QUEUE_MODELS.index(queue_model),
        float(power),
        book.bid_px,
        book.bid_qty,
        book.ask_px,
        book.ask_qty,
        trades.ptr,
        trades.price,
        trades.buy_vol,
        trades.sell_vol,
        trades.lo,
        trades.hi,
    )
    return LimitFillResult(*out)
