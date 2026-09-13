"""호가 두께 — 10호가 소진 VWAP, 시장가 왕복충격(bp), 얇은 호가 비중 배수.

scalp-it ``realtime/pair_detector.py`` 의 ``_sweep_vwap``·``roundtrip_bp``·
``liquidity_size_multiplier`` 를 **식 그대로** 옮겼다. 백테스트 체결 시뮬레이션과
실시간 필터가 같은 충격 추정을 써야 한다 — 둘이 어긋나면 백테에서 통과한 종목을
실시간이 배제하거나 그 반대가 된다.

호가 한 단계는 ``(가격, 잔량)`` 튜플이고, 리스트는 최우선호가부터 바깥쪽 순서다
(매도호가는 ask1→ask10 오름차순, 매수호가는 bid1→bid10 내림차순).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["Level", "liquidity_size_multiplier", "roundtrip_bp", "sweep_vwap"]

Level = tuple[float, float]


def sweep_vwap(levels: Sequence[Level], krw: float) -> float | None:
    """``krw`` 원을 호가 단계별로 순서대로 소진했을 때의 체결 VWAP.

    잔량을 다 긁어도 금액을 못 채우면 None — 그 종목은 시장가로 치면 호가가 튄다
    (만쥬 §4-1: "상하 10호가를 합쳐도 소화 못 하면 재료가 좋아도 즉시 제외").
    가격·잔량이 0 이하인 단계는 건너뛴다(빈 호가 슬롯).
    """
    if krw <= 0:
        return None
    remain, shares = krw, 0.0
    for px, qty in levels:
        px, qty = float(px), float(qty)
        if px <= 0 or qty <= 0:
            continue
        take = min(px * qty, remain)
        shares += take / px
        remain -= take
        if remain <= 0:
            break
    if remain > 0 or shares <= 0:
        return None
    return krw / shares


def liquidity_size_multiplier(bp: float | None, *, thin_bp: float,
                              thin_multiplier: float) -> float:
    """호가 두께로 비중 배수를 정한다 — 만쥬 A-2 Thin Orderbook Restriction.

    > Thick Orderbook: Normal allocation.
    > Thin Orderbook: Cap size to prevent slippage (**Max 10-20% of normal size**).

    ``bp`` 는 :func:`roundtrip_bp` 가 잰 왕복 충격이다. 얇으면 **배제가 아니라 축소**한다
    — 얇아도 기회면 소액으로 잡는다. 호가를 못 봤으면(``None``) 얇은 쪽으로 본다
    (모르면 작게 가는 쪽이 안전하다). ``thin_bp`` 가 0 이하면 끈다. 경계값(``bp == thin_bp``)은
    아직 두꺼운 쪽이다.
    """
    if thin_bp <= 0:
        return 1.0
    if bp is None:
        return thin_multiplier
    return thin_multiplier if bp > thin_bp else 1.0


def roundtrip_bp(bids: Sequence[Level], asks: Sequence[Level], krw: float) -> float | None:
    """``krw`` 원 시장가 **왕복** 충격(bp). 호가로 못 채우면 ``inf``, 호가가 없으면 None.

    매수는 ask1→ask10, 매도는 bid1→bid10 을 순서대로 소진한 VWAP 을 쓰고
    ``(VWAP_buy − VWAP_sell) / mid × 10⁴`` 로 잰다. scalp-it 실측(2026-08-24, 80종목)에서
    1000만원 기준 중앙 왕복충격이 5.1bp~195.9bp 로 39배 벌어졌고, 두께 5분위 충격후
    손익이 Q1 −22bp vs Q5 −219bp 였다 — 두께가 곧 손익이다.
    """
    if not bids or not asks:
        return None
    mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
    if mid <= 0:
        return None
    buy = sweep_vwap(asks, krw)
    sell = sweep_vwap(bids, krw)
    if buy is None or sell is None:
        return math.inf  # 호가로도 못 채움 = 호가 튐 → 즉시 배제
    return (buy - sell) / mid * 1e4
