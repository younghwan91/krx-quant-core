"""KRX 호가단위(틱) 유틸.

**표 자체는 여기서 다시 정의하지 않는다.** 정본은 ``kiwoom_client.tick_size`` 다 —
scalp-it·daytrade-it·krx-signal-engine 이 같은 표를 각자 들고 있다가 경계 하나가
어긋나면 주문 거부나 상한가 오판이 조용히 생기므로, 실제로 주문을 내는
kiwoom-client 로 모았다(2026-09-12). 이 모듈은 그 위의 파생 연산만 둔다.

두 갈래 API 가 있다:

- ``Decimal`` 계열(:func:`tick_size`·:func:`round_to_tick`·:func:`shift_ticks` 등) —
  주문가·체결가처럼 한 원도 틀리면 안 되는 경로(daytrade-it 비용모델·브로커).
- ``float``/``int`` 계열(:func:`tick_size_int`·:func:`ticks_in_float`) — scalp-it
  실시간 경로는 가격을 plain float 로 다루고, ``Decimal`` 과 float 를 섞으면 바로
  ``TypeError`` 가 나므로 반환형을 int/float 로 고정한다. scalp-it
  ``ticksize.py`` 와 **경계 하나도** 동작이 같아야 한다(verify_* 재현성).
"""

from __future__ import annotations

import bisect
import importlib
import math
from decimal import ROUND_CEILING, Decimal
from typing import Any

import numpy as np
from kiwoom_client.tick_size import round_to_tick, tick_size, ticks_in
from numpy.typing import NDArray

__all__ = [
    "ETF_TICK_BANDS",
    "KRX_LOT_SIZE",
    "is_tick_valid",
    "round_to_tick",
    "round_to_tick_up",
    "shift_ticks",
    "tick_size",
    "tick_size_array",
    "tick_size_int",
    "ticks_in",
    "ticks_in_float",
]

#: 보통주 매매단위 1주. 단주 전용 거래가 없어졌고 KOSPI·KOSDAQ·KONEX 정규 주문은
#: 1주 단위다(daytrade-it ``tick_size.py`` 에서 옮김).
KRX_LOT_SIZE = 1

_Num = Decimal | int | float


def _load_bands() -> tuple[tuple[float, ...], tuple[int, ...]]:
    """정본 밴드 표를 float 경계·int 틱으로 옮기고 경계 양쪽을 정본 함수와 대조한다.

    정본 표가 바뀌면 여기가 조용히 어긋나지 않고 import 가 실패한다(``lob.ticks`` 와 같은 방식).
    """
    mod = importlib.import_module("kiwoom_client.tick_size")
    bounds = tuple(float(b) for b, _ in mod._TICK_BANDS)
    ticks = (*(int(t) for _, t in mod._TICK_BANDS), int(mod._TOP_TICK))
    for b in bounds:
        for p in (b - 1.0, b - 0.5, b, b + 0.5):
            i = bisect.bisect_right(bounds, p)
            if ticks[i] != int(tick_size(p)):
                raise RuntimeError(f"tick table drifted from kiwoom_client at price {p}")
    return bounds, ticks


#: 주식 밴드(정본에서 읽음): ``price < _BOUNDS[i]`` 인 첫 ``i`` 의 ``_TICKS[i]``,
#: 어느 경계도 아니면 ``_TICKS[-1]``.
_BOUNDS, _TICKS = _load_bands()

#: ETF·ETN 호가단위 — 2,000원 미만 1원, 이상 5원.
#: kiwoom-client 정본에 아직 없다(``lob.ticks`` 참고).
ETF_TICK_BANDS: tuple[tuple[float, ...], tuple[int, ...]] = ((2000.0,), (1, 5))


def round_to_tick_up(price: _Num) -> Decimal:
    """``price`` 를 그 가격대 호가단위로 **올림**.

    올린 결과가 윗 가격대로 넘어가도(예 1,999.5 → 2,000) 괜찮다 — 밴드 경계
    (2천·5천·2만·5만·20만·50만)는 모두 윗 밴드 틱의 배수라 여전히 유효한 호가다.
    """
    p = Decimal(str(price))
    tick = tick_size(p)
    return (p / tick).to_integral_value(rounding=ROUND_CEILING) * tick


def is_tick_valid(price: _Num) -> bool:
    """``price`` 가 실제로 낼 수 있는 호가인가(양수이고 그 가격대 틱의 배수)."""
    p = Decimal(str(price))
    if p <= 0:
        return False
    return p % tick_size(p) == 0


def shift_ticks(price: _Num, n: int) -> Decimal:
    """유효 호가 ``price`` 에서 ``n`` 틱 위(양수)/아래(음수)의 호가.

    가격대 경계를 넘을 때 틱이 바뀌는 것을 한 칸씩 반영한다. 예: 2,000 에서 한 틱
    아래는 1,995 가 아니라 **1,999**(아랫 밴드 틱 1원), 1,999 에서 한 틱 위는 2,000,
    2,000 에서 한 틱 위는 2,005.

    아래로 한 칸은 "``p`` 바로 아래 가격(``p-1``)의 틱"을 뺀다 — 모든 호가가 정수이고
    최소 틱이 1원이라 ``p-1`` 은 항상 아랫 틱이 적용되는 가격대에 있다.

    Raises:
        ValueError: ``price`` 가 유효 호가가 아니거나(어느 쪽으로 맞출지 호출부 의도를
            추측하지 않는다), 이동 결과가 0 이하가 될 때.
    """
    p = Decimal(str(price))
    if not is_tick_valid(p):
        raise ValueError(f"price is not a valid KRX tick: {price!r}")
    step = 1 if n > 0 else -1
    for _ in range(abs(n)):
        if step > 0:
            p += tick_size(p)
        else:
            if p <= 1:
                raise ValueError(f"cannot shift {price!r} by {n} ticks: below minimum price")
            p -= tick_size(p - 1)
    return p


def tick_size_int(price: float) -> int:
    """호가단위를 plain ``int`` 로(scalp-it ``ticksize.tick_size`` 와 동일).

    ``price <= 0`` 이면 ``kiwoom_client`` 처럼 ``ValueError`` 를 던지지 않고 **조용히
    1** 을 돌려준다 — scalp-it 실시간 경로에 초기화 전 price=0 을 거르지 않는
    호출부가 있을 수 있어서, 그 동작을 그대로 보존한다.

    유한한 양수는 float 밴드 표를 이분 탐색한다(호출마다 ``Decimal`` 을 만들지 않는다 — 실시간
    틱마다 불린다). nan·inf 는 예전처럼 정본 함수로 보내 같은 예외를 낸다.
    """
    if price <= 0:
        return 1
    if not math.isfinite(price):
        return int(tick_size(price))
    return _TICKS[bisect.bisect_right(_BOUNDS, price)]


def tick_size_array(prices: Any, *, etf: bool = False) -> NDArray[np.float64]:
    """벡터 호가단위(float). 0 이하·nan 은 nan. ``etf`` 면 :data:`ETF_TICK_BANDS`.

    배열 경로(연구 스크립트의 ``[TICK(x) for x in ask]``)를 대신한다. 값은 :func:`tick_size_int`
    와 같다(테스트가 밴드 경계마다 대조).
    """
    p = np.asarray(prices, dtype=np.float64)
    bounds, ticks = ETF_TICK_BANDS if etf else (_BOUNDS, _TICKS)
    flat = np.atleast_1d(p)
    out = np.asarray(ticks, np.float64)[np.searchsorted(np.asarray(bounds), flat, side="right")]
    out[~(flat > 0)] = np.nan
    return out.reshape(p.shape)


def ticks_in_float(width: float, price: float) -> float:
    """가격 ``price`` 대에서 폭 ``width``(원)가 몇 틱인지, 반올림 없는 ``float``.

    scalp-it ``ticksize.ticks_in`` 과 동일. 반올림하지 않는 이유: "2.2틱"과
    "3.0틱"의 차이가 곧 노이즈 청산 위험의 차이라 버림하면 판정이 뭉개진다.
    """
    return abs(width) / tick_size_int(price)
