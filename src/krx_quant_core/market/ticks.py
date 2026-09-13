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

from decimal import ROUND_CEILING, Decimal

from kiwoom_client.tick_size import round_to_tick, tick_size, ticks_in

__all__ = [
    "KRX_LOT_SIZE",
    "is_tick_valid",
    "round_to_tick",
    "round_to_tick_up",
    "shift_ticks",
    "tick_size",
    "tick_size_int",
    "ticks_in",
    "ticks_in_float",
]

#: 보통주 매매단위 1주. 단주 전용 거래가 없어졌고 KOSPI·KOSDAQ·KONEX 정규 주문은
#: 1주 단위다(daytrade-it ``tick_size.py`` 에서 옮김).
KRX_LOT_SIZE = 1

_Num = Decimal | int | float


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
    """
    if price <= 0:
        return 1
    return int(tick_size(price))


def ticks_in_float(width: float, price: float) -> float:
    """가격 ``price`` 대에서 폭 ``width``(원)가 몇 틱인지, 반올림 없는 ``float``.

    scalp-it ``ticksize.ticks_in`` 과 동일. 반올림하지 않는 이유: "2.2틱"과
    "3.0틱"의 차이가 곧 노이즈 청산 위험의 차이라 버림하면 판정이 뭉개진다.
    """
    return abs(width) / tick_size_int(price)
