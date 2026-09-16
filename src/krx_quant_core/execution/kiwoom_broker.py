"""``KiwoomBroker`` — kiwoom-client 위의 실매매 :class:`~.broker.Broker` 어댑터.

trader 데몬(scalp-it·daytrade-it)이 ``kiwoom_client.KiwoomAPI`` 호환 객체
(``api.order.buy_order(**body)``·``api.account.evaluation_balance_detail(**kw)`` 등)를
주면, 이 클래스가 :class:`~.broker.Broker` 프로토콜로 감싼다. 테스트는 진짜
``kiwoom_client`` 없이 ``SimpleNamespace(order=..., account=...)`` 가짜 api 로 돈다.

실계좌 안전 규칙(바꾸지 않는다)
==============================

* **제출(주문·취소)은 절대 재시도하지 않는다.** 응답을 못 받아도 다시 보내지
  않는다 — 이중 주문이 실포지션 사고로 이어진다. 예외는 ``OrderResult``로
  감싸 돌려줄 뿐, 여기서 삼키거나 재시도하지 않는다.
* **조회(holdings·open_orders·poll_fills)는 예외 시 1회만 재시도한다**
  (``_read``). 재시도해도 이중 주문 위험이 없다.
* **토큰을 폐기(logout/revoke)하는 메서드는 만들지 않는다.**
  :data:`~.kiwoom_spec.TOKEN_SHARING_WARNING` — scalp-it·daytrade-it 이 같은
  appkey 로 동시에 실매매하므로, 한쪽이 revoke 하면 다른 쪽 세션이 죽는다.
  토큰 갱신은 kiwoom-client 가 알아서 한다.
* ``dry_run=True``(기본)면 주문 제출·취소는 POST 하지 않고
  ``OrderResult(dry_run=True, submitted=False)`` 를 돌려준다. 조회 메서드는
  dry-run 과 무관하게 정상 동작한다(읽기는 계좌에 해를 끼치지 않는다).

미검증 필드 — kiwoom-client 실호출로 확인 안 됨
================================================

* ``ka10075``(미체결) 매수/매도 판별 필드 — 응답에 지정가·수량은 확실하지만
  매매구분을 어느 필드로 읽는지는 확인 안 됐다. ``io_tp_nm``(사람이 읽는
  "매수"/"매도" 텍스트로 추정) 다음 ``trde_tp`` 순서로 본다(:func:`_first`).
  텍스트에 ``"매도"`` 가 들어 있으면 매도, 그 외엔 매수로 본다 — 코드값
  (``trde_tp``)이 오면 오판할 수 있다는 뜻이라 여기 적어 둔다.
* ``ka10075`` 지정가 필드명 ``ord_pric`` — 브리프 지시값, 실호출 미확인.
* ``ka10076``(체결) 목록 키 ``cntr``, 주문번호 ``ord_no``, 누적 체결수량
  ``cntr_qty``, 체결단가 ``cntr_pric`` — 전부 브리프 지시값이고 실호출로 확인
  안 됐다. 시각 필드도 미확인이라 없으면 ``now_kst()`` 로 채운다.
* ``poll_fills`` 의 ``filled_orders`` 조회 바디(``all_stk_tp``·``trde_tp``·
  ``stex_tp``)는 ``ka10075`` 미체결 조회 바디를 그대로 가져다 썼다 — ``ka10076``
  전용 바디는 확인 안 됐다.
"""

from __future__ import annotations

from typing import Any

from ..market.codes import normalize_code
from ..market.session import now_kst
from .events import Fill, Holding, OpenOrder
from .kiwoom_spec import BALANCE_ROWS_KEY, UNFILLED_ROWS_KEY, strip_sign
from .orders import OrderIntent, OrderResult

__all__ = ["KiwoomBroker"]

#: ``ka10076`` 체결 목록 키. **미검증**(브리프 지시값) — 모듈 docstring 참고.
FILLED_ROWS_KEY = "cntr"


def _first(row: dict[str, Any], *keys: str) -> Any:
    """``row`` 에서 ``keys`` 를 순서대로 봐서 처음 값이 있는 것을 돌려준다.

    필드명이 미검증인 응답에서 후보 키를 차례로 시도할 때 쓴다. 값이
    ``None``·빈 문자열이면 없는 것으로 치고 다음 후보로 넘어간다.
    """
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _side_from_row(row: dict[str, Any]) -> str:
    """미체결/체결 행에서 매수/매도를 판별한다. **미검증** — 모듈 docstring 참고."""
    text = str(_first(row, "io_tp_nm", "trde_tp") or "")
    return "sell" if "매도" in text else "buy"


class KiwoomBroker:
    """``kiwoom_client.KiwoomAPI`` 호환 객체를 감싸는 실매매 :class:`Broker` 어댑터."""

    def __init__(self, api: Any, *, dry_run: bool = True) -> None:
        self._api = api
        self._dry_run = dry_run
        #: 주문번호 → 지금까지 본 누적 체결수량. ``poll_fills`` 증분 계산용.
        self._cum_filled: dict[str, int] = {}

    # ----- 제출/취소 — 재시도 없음 -------------------------------------------

    def submit(self, intent: OrderIntent) -> OrderResult:
        """주문을 낸다. **재시도 없음.** ``dry_run`` 이면 POST 하지 않는다."""
        if self._dry_run:
            return OrderResult(intent=intent, dry_run=True, submitted=False)
        fn = self._api.order.buy_order if intent.side == "buy" else self._api.order.sell_order
        try:
            resp = fn(**intent.body())
        except Exception as exc:  # noqa: BLE001 — 어떤 예외든 결과로 감싼다. 재시도 안 함.
            return OrderResult(
                intent=intent,
                dry_run=False,
                submitted=True,
                return_code=None,
                return_msg=f"{type(exc).__name__}: {exc}",
            )
        return self._result_from_response(intent, resp, fallback_ord_no=None)

    def cancel(self, ord_no: str, intent: OrderIntent) -> OrderResult:
        """미체결 주문을 취소한다. **재시도 없음.** ``dry_run`` 이면 POST 하지 않는다."""
        if self._dry_run:
            return OrderResult(intent=intent, dry_run=True, submitted=False)
        try:
            resp = self._api.order.cancel_order(**intent.cancel_body(ord_no))
        except Exception as exc:  # noqa: BLE001 — 재시도 안 함, 결과로 감싼다.
            return OrderResult(
                intent=intent,
                dry_run=False,
                submitted=True,
                return_code=None,
                return_msg=f"{type(exc).__name__}: {exc}",
            )
        return self._result_from_response(intent, resp, fallback_ord_no=str(ord_no))

    @staticmethod
    def _result_from_response(
        intent: OrderIntent, resp: dict[str, Any], *, fallback_ord_no: str | None
    ) -> OrderResult:
        rc = resp.get("return_code")
        raw_ord_no = resp.get("ord_no")
        ord_no = str(raw_ord_no).strip() if raw_ord_no not in (None, "") else fallback_ord_no
        return OrderResult(
            intent=intent,
            dry_run=False,
            submitted=True,
            ord_no=ord_no,
            return_code=int(rc) if rc is not None else None,
            return_msg=str(resp.get("return_msg") or ""),
        )

    # ----- 조회 — 예외 시 1회 재시도 -----------------------------------------

    def _read(self, fn: Any, **kwargs: Any) -> dict[str, Any]:
        """조회 호출. 예외면 1회만 재시도한다(제출과 달리 재시도해도 안전하다)."""
        try:
            return fn(**kwargs)
        except Exception:  # noqa: BLE001 — 1회 재시도, 두 번째도 실패하면 올린다.
            return fn(**kwargs)

    def holdings(self) -> dict[str, Holding]:
        """``kt00018`` 계좌평가잔고내역 → 보유 종목(종목코드 → :class:`Holding`)."""
        resp = self._read(
            self._api.account.evaluation_balance_detail, qry_tp="1", dmst_stex_tp="KRX"
        )
        out: dict[str, Holding] = {}
        for row in resp.get(BALANCE_ROWS_KEY) or []:
            code = normalize_code(row.get("stk_cd"))
            if not code:
                continue
            qty = strip_sign(row.get("rmnd_qty"))
            avg_price = float(strip_sign(row.get("pur_pric")))
            out[code] = Holding(code=code, qty=qty, avg_price=avg_price)
        return out

    def open_orders(self) -> list[OpenOrder]:
        """``ka10075`` 미체결요청 → 미체결 주문 목록."""
        resp = self._read(
            self._api.account.unfilled_orders, all_stk_tp="0", trde_tp="0", stex_tp="0"
        )
        out: list[OpenOrder] = []
        for row in resp.get(UNFILLED_ROWS_KEY) or []:
            ord_no = str(row.get("ord_no") or "").strip()
            if not ord_no:
                continue
            out.append(
                OpenOrder(
                    ord_no=ord_no,
                    code=normalize_code(row.get("stk_cd")),
                    side=_side_from_row(row),
                    qty=strip_sign(row.get("ord_qty")),
                    remaining=strip_sign(_first(row, "oso_qty", "ord_remnq")),
                    price=strip_sign(row.get("ord_pric")),
                )
            )
        return out

    def poll_fills(self) -> list[Fill]:
        """``ka10076`` 체결요청 → 지난 호출 이후 새로 늘어난 체결만 :class:`Fill` 로.

        키움 응답은 주문별 **누적** 체결수량을 준다(추정 — 모듈 docstring 미검증
        목록 참고). 지난번 누적보다 늘어난 만큼만 ``Fill`` 로 낸다. 시각 필드가
        없으면 :func:`~krx_quant_core.market.session.now_kst` 로 채운다.
        """
        resp = self._read(
            self._api.account.filled_orders, all_stk_tp="0", trde_tp="0", stex_tp="0"
        )
        fills: list[Fill] = []
        for row in resp.get(FILLED_ROWS_KEY) or []:
            ord_no = str(row.get("ord_no") or "").strip()
            if not ord_no:
                continue
            cum_qty = strip_sign(row.get("cntr_qty"))
            prev_qty = self._cum_filled.get(ord_no, 0)
            delta = cum_qty - prev_qty
            if delta > 0:
                fills.append(
                    Fill(
                        ord_no=ord_no,
                        code=normalize_code(row.get("stk_cd")),
                        side=_side_from_row(row),
                        qty=delta,
                        price=strip_sign(row.get("cntr_pric")),
                        ts=now_kst(),
                    )
                )
            self._cum_filled[ord_no] = max(cum_qty, prev_qty)
        return fills
