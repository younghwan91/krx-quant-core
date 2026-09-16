"""체결·보유·미체결 값 객체 — :class:`Broker` 프로토콜과 :class:`PositionBook` 이 주고받는 모양.

여기 값 객체들은 HTTP·키움 응답 파싱과 무관한 **순수** 자료형이다. 어댑터
(``paper.py``·``kiwoom_broker.py``)가 각자의 원천에서 이 모양으로 변환해 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

__all__ = ["Fill", "Holding", "OpenOrder", "OrderStatus", "normalize_ord_no"]


class OrderStatus(StrEnum):
    """주문 하나의 생애주기 상태. 로그·사유 문자열에 그대로 쓴다."""

    BLOCKED = "blocked"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    #: 주문이 나갔는지 모른다 — 재시도 금지, 미체결/잔고로 확인. 브로커 호출이 예외로
    #: 끝났거나(타임아웃 등) 제출 응답에 ``return_code`` 가 없을 때. 거부(REJECTED)로
    #: 읽으면 호출부가 "안 나갔다"고 믿고 다시 내 이중 주문이 된다.
    UNKNOWN = "unknown"


def normalize_ord_no(s: str) -> str:
    """주문번호 비교 키 — 선행 0 을 뗀다(``"0000123"`` → ``"123"``).

    키움은 같은 주문번호를 응답마다 0 패딩을 달리해 줄 수 있어, 원문 비교로는 "내 주문"
    판별(:class:`~.oms.OrderManager`)과 누적 체결 추적(``KiwoomBroker``)이 조용히 어긋난다.
    전부 0 인 입력은 지워지지 않게 원문(공백 제거)을 돌려준다. ``PaperBroker`` 의
    ``"P000001"`` 같은 번호는 0 으로 시작하지 않으니 그대로다.
    """
    text = str(s if s is not None else "").strip()
    return text.lstrip("0") or text


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
