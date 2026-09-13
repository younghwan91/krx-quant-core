"""지정가 체결 판정 — ``touch``(낙관) vs ``through``(보수) 두 가정.

scalp-it 은 같은 판정을 두 곳에 들고 있다: 가상 포지션 회계
(``realtime/risk_guard.py`` ``RiskGuard._advance``)와 dry-run 주문 실행기
(``realtime/order_executor.py`` ``_is_buy_filled``·``_check_sell_fill``). 두 곳의 기준이
갈리면 같은 틱열에서 회계와 실행기가 서로 다른 체결을 기록한다(2026-08-25 사고).
그래서 판정식을 순수 함수 하나로 올린다.

틱 체결가만 보고 내 지정가가 채워졌는지 추정한다 — 호가 큐 위치는 모른다:

- **touch(낙관)**: 체결가가 내 지정가에 **닿기만** 하면 체결. 큐 맨 앞이라는 가정.
  매수 ``trade <= limit``, 매도 ``trade >= limit``.
- **through(보수, 기본)**: 체결가가 지정가를 **관통**해야 체결. 그 가격의 큐가 다
  비워져야 내 주문까지 온다는 가정. 매수 ``trade < limit``, 매도 ``trade > limit``.
  scalp-it 감사가 "through 를 실전 기대로" 요구해 기본값이다.

부동소수 비교는 scalp-it 과 같은 ``1e-9`` 여유를 둔다(through 는 더 엄격한 쪽으로,
touch 는 더 너그러운 쪽으로).

원본과 한 가지 다르다: 알 수 없는 ``basis`` 문자열. 실행기는 조용히 through 로,
회계는 조용히 touch 로 읽어 **서로 반대**였다. 여기서는 ``ValueError`` 를 낸다 — 두
원본 모두 CLI 에서 ``choices=("through","touch")`` 로 막혀 있어 유효 입력에서는 동일하다.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["EPS", "FILL_BASES", "FillBasis", "limit_buy_filled", "limit_sell_filled"]

FillBasis = Literal["through", "touch"]
FILL_BASES: tuple[str, ...] = ("through", "touch")

#: scalp-it ``risk_guard._EPS`` / ``order_executor._EPS`` 와 같은 값.
EPS = 1e-9


def _check(basis: str) -> None:
    if basis not in FILL_BASES:
        raise ValueError(f"fill basis 는 {FILL_BASES} 중 하나여야 한다: {basis!r}")


def limit_buy_filled(limit: float, trade_price: float, basis: FillBasis = "through") -> bool:
    """매수 지정가 ``limit`` 이 체결가 ``trade_price`` 틱으로 채워졌다고 볼 수 있나.

    through: ``trade_price < limit - EPS`` · touch: ``trade_price <= limit + EPS``.
    """
    _check(basis)
    if basis == "through":
        return trade_price < limit - EPS
    return trade_price <= limit + EPS


def limit_sell_filled(limit: float, trade_price: float, basis: FillBasis = "through") -> bool:
    """매도 지정가 ``limit`` 이 채워졌다고 볼 수 있나 — 매수와 대칭(가격이 **위로** 가야 붙는다).

    through: ``trade_price > limit + EPS`` · touch: ``trade_price >= limit - EPS``.
    """
    _check(basis)
    if basis == "through":
        return trade_price > limit + EPS
    return trade_price >= limit - EPS
