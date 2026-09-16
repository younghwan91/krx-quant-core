"""체결·보유·미체결 값 객체 — :class:`Broker` 프로토콜과 :class:`PositionBook` 이 주고받는 모양.

여기 값 객체들은 HTTP·키움 응답 파싱과 무관한 **순수** 자료형이다. 어댑터
(``paper.py``·``kiwoom_broker.py``)가 각자의 원천에서 이 모양으로 변환해 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

__all__ = ["Fill", "Holding", "OpenOrder", "OrderStatus"]


class OrderStatus(StrEnum):
    """주문 하나의 생애주기 상태. 로그·사유 문자열에 그대로 쓴다."""

    BLOCKED = "blocked"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Fill:
    """체결 1건. :class:`~krx_quant_core.execution.book.PositionBook.apply` 의 입력."""

    ord_no: str
    code: str
    side: Literal["buy", "sell"]
    qty: int
    price: int
    ts: datetime


@dataclass(frozen=True)
class Holding:
    """보유 1종목. 평단은 매수 fill 로 가중평균한 값."""

    code: str
    qty: int
    avg_price: float


@dataclass(frozen=True)
class OpenOrder:
    """미체결 주문 1건."""

    ord_no: str
    code: str
    side: str
    qty: int
    remaining: int
    price: int
