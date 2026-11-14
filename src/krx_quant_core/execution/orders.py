"""주문 값 객체 — 낼 주문(:class:`OrderIntent`)과 그 결과(:class:`OrderResult`).

scalp-it ``adapters/kiwoom_order.py`` 의 순수 부분을 **동작 그대로** 옮겼다. HTTP·토큰·
잔고조회는 여기 없다 — 그건 각 레포의 클라이언트 몫이다.

원본과 다른 점은 둘뿐이고, 둘 다 계좌·키움 클라이언트에 묶인 것을 코어 밖으로 뺀
것이다:

* 종목명 표시에 쓰던 ``HELD_STOCK_NAMES``(scalp-it 계좌의 보유 3종목)는 코어에 없다.
  대신 :attr:`OrderIntent.code_names` 클래스 변수로 받는다 — scalp-it 은 서브클래스에서
  자기 표를 넣으면 ``describe()``·``to_record()``·가드 사유 문자열이 글자 하나 안 바뀐다.
* 알 수 없는 매매구분에서 원본은 ``KiwoomError`` 를 던졌다. 코어는 키움 클라이언트
  예외를 모르므로 **같은 메시지의** ``ValueError`` 를 던진다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar

from .kiwoom_spec import (
    BUY_API_ID,
    CANCEL_API_ID,
    DEFAULT_EXCHANGE,
    MODIFY_API_ID,
    SELL_API_ID,
    TRDE_TP_LIMIT,
    validate_order_body,
)

__all__ = ["OrderBlocked", "OrderIntent", "OrderResult", "cancel_body", "modify_body"]


class OrderBlocked(Exception):
    """안전 가드가 주문을 막았다. 실계좌 사고 방지의 마지막 벽.

    코어에서는 평범한 ``Exception`` 이다. scalp-it 은 기존 ``except KiwoomError`` 가
    계속 잡도록 ``class OrderBlocked(core.OrderBlocked, KiwoomError)`` 로 다중상속하면 된다.
    """


def cancel_body(
    org_ord_no: str, code: str, qty: int, exchange: str = DEFAULT_EXCHANGE
) -> dict[str, str]:
    """취소(``kt10003``) 바디. 원주문번호 필수, 수량은 문자열(``"0"`` = 잔량 전부).

    필드명·순서는 키움 공식 스펙(``Kiwoom-Securities/Kiwoom-REST-API``
    ``kiwoom/_data/kiwoom_api_spec.json`` c7a3d25)의 kt10003 바디 그대로다:
    ``dmst_stex_tp`` · ``orig_ord_no`` · ``stk_cd`` · ``cncl_qty``.
    전에는 ``org_ord_no``·``ord_qty`` 를 보냈고 실주문 취소 14건이 모두 ``return_code == 2``
    였다(scalp-it ``data/pair_orders.log``). 새 필드명은 **실주문 취소로는 아직 확인 전**이다 —
    :mod:`.kiwoom_spec` 머리말 참고.

    인자 이름 ``org_ord_no`` 는 호출부 호환을 위해 그대로 둔다. 만든 바디는
    :func:`~.kiwoom_spec.validate_order_body` 로 스펙 필드를 확인한 뒤 돌려준다.
    """
    body = {
        "dmst_stex_tp": exchange,
        "orig_ord_no": str(org_ord_no),
        "stk_cd": code,
        "cncl_qty": str(int(qty)),
    }
    validate_order_body(CANCEL_API_ID, body)
    return body


def modify_body(
    org_ord_no: str,
    code: str,
    qty: int,
    price: int,
    exchange: str = DEFAULT_EXCHANGE,
    cond_price: int | None = None,
) -> dict[str, str]:
    """정정(``kt10002``) 바디. 수량·가격은 문자열(``qty=0`` → ``"0"`` = 잔량 전부 정정).

    필드명·순서는 공식 스펙 kt10002 그대로다: ``dmst_stex_tp`` · ``orig_ord_no`` ·
    ``stk_cd`` · ``mdfy_qty`` · ``mdfy_uv`` (+ 선택 ``mdfy_cond_uv``, ``cond_price`` 가
    있을 때만). **실주문으로는 확인 전**이다 — :mod:`.kiwoom_spec` 머리말 참고.
    """
    body = {
        "dmst_stex_tp": exchange,
        "orig_ord_no": str(org_ord_no),
        "stk_cd": code,
        "mdfy_qty": str(int(qty)),
        "mdfy_uv": str(int(price)),
    }
    if cond_price is not None:
        body["mdfy_cond_uv"] = str(int(cond_price))
    validate_order_body(MODIFY_API_ID, body)
    return body


@dataclass(frozen=True)
class OrderIntent:
    """낼(또는 낼 뻔한) 주문 하나. 종목·수량·가격·매수/매도."""

    #: 로그·사유 문자열에 붙일 종목명 표. 코어는 비어 있다 — 계좌별 표는 서브클래스가 둔다.
    code_names: ClassVar[Mapping[str, str]] = MappingProxyType({})

    side: str                           # "buy" | "sell" (취소 기록용 "cancel")
    code: str                           # 6자리
    qty: int
    price: int                          # 지정가(원)
    exchange: str = DEFAULT_EXCHANGE

    def api_id(self) -> str:
        if self.side == "buy":
            return BUY_API_ID
        if self.side == "sell":
            return SELL_API_ID
        raise ValueError(f"알 수 없는 매매구분: {self.side!r}")

    def body(self) -> dict[str, str]:
        """키움 주문 바디. 수량·가격은 문자열. **지정가만** 낸다."""
        return {
            "dmst_stex_tp": self.exchange,
            "stk_cd": self.code,
            "ord_qty": str(int(self.qty)),
            "ord_uv": str(int(self.price)),
            "trde_tp": TRDE_TP_LIMIT,
        }

    def cancel_body(self, org_ord_no: str) -> dict[str, str]:
        """이 intent 의 종목·수량·거래소로 만든 취소 바디."""
        return cancel_body(org_ord_no, self.code, self.qty, self.exchange)

    def modify_body(self, org_ord_no: str) -> dict[str, str]:
        """이 intent 의 종목·수량·가격·거래소로 만든 정정 바디(수량·가격 = 정정 후 값)."""
        return modify_body(org_ord_no, self.code, self.qty, self.price, self.exchange)

    def describe(self) -> str:
        kind = "매수" if self.side == "buy" else "매도"
        name = self.code_names.get(self.code, "")
        tag = f"{name}({self.code})" if name else self.code
        return f"{kind} 지정가  {tag}  {self.qty}주 @ {self.price:,}원 [{self.exchange}]"


@dataclass
class OrderResult:
    """주문 제출 결과. dry-run 이면 ``submitted=False`` 이고 POST 안 함."""

    intent: OrderIntent
    dry_run: bool
    submitted: bool                     # 실제 POST 했는가
    blocked: bool = False
    blocked_reason: str = ""
    ord_no: str | None = None
    return_code: int | None = None
    return_msg: str = ""

    @property
    def ok(self) -> bool:
        """실주문이 성공했거나(dry 아님·rc0) dry-run 으로 통과했는가."""
        if self.blocked:
            return False
        if self.dry_run:
            return True
        return self.submitted and self.return_code == 0

    def to_record(self, ts: datetime | None = None) -> dict[str, Any]:
        """주문 로그 한 줄. **비밀(토큰·계좌번호)은 없다.**

        ``ts`` 기본값은 원본 그대로 naive 로컬시각이다(호스트가 KST 라는 가정).
        """
        return {
            "ts": (ts or datetime.now()).isoformat(sep=" ", timespec="seconds"),  # noqa: DTZ005
            "side": self.intent.side,
            "code": self.intent.code,
            "name": self.intent.code_names.get(self.intent.code, ""),
            "qty": self.intent.qty,
            "price": self.intent.price,
            "exchange": self.intent.exchange,
            "dry_run": self.dry_run,
            "submitted": self.submitted,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "ord_no": self.ord_no,
            "return_code": self.return_code,
            # 거부 사유가 있을 때만 — 없으면 키를 넣지 않아 기존 로그 줄은 그대로다.
            **({"return_msg": self.return_msg} if self.return_msg else {}),
        }
