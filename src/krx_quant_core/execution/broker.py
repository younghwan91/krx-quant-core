"""``Broker`` 프로토콜 — 페이퍼·키움 어댑터가 공통으로 따르는 최소 인터페이스.

`OrderManager`(Task 6)가 이 프로토콜 하나만 알면 되게 해서, 전략·주문관리 코드가
모의매매·실매매에서 어댑터만 바꿔 그대로 돈다(스펙 3.7 ``EngineCore`` 의 전제).
"""

from __future__ import annotations

from typing import Protocol

from .events import Fill, Holding, OpenOrder
from .orders import OrderIntent, OrderResult

__all__ = ["Broker"]


class Broker(Protocol):
    """주문 제출·취소·조회의 공통 창구."""

    def submit(self, intent: OrderIntent) -> OrderResult:
        """주문을 낸다."""
        ...

    def cancel(self, ord_no: str, intent: OrderIntent) -> OrderResult:
        """미체결 주문을 취소한다."""
        ...

    def open_orders(self) -> list[OpenOrder]:
        """현재 미체결 주문 목록."""
        ...

    def holdings(self) -> dict[str, Holding]:
        """계좌 보유 종목(종목코드 → :class:`Holding`)."""
        ...

    def poll_fills(self) -> list[Fill]:
        """지난 호출 이후 새로 발생한 체결만 돌려준다."""
        ...
