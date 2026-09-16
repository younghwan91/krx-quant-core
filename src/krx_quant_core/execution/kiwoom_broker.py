"""``KiwoomBroker`` — kiwoom-client 위의 실매매 :class:`~.broker.Broker` 어댑터.

trader 데몬(scalp-it·daytrade-it)이 ``kiwoom_client.KiwoomAPI`` 호환 객체
(``api.order.buy_order(**body)``·``api.account.evaluation_balance_detail(**kw)`` 등)를
주면, 이 클래스가 :class:`~.broker.Broker` 프로토콜로 감싼다. 테스트는 진짜
``kiwoom_client`` 없이 ``SimpleNamespace(order=..., account=...)`` 가짜 api 로 돈다.

실계좌 안전 규칙(바꾸지 않는다)
==============================

* **제출(주문·취소)은 절대 재시도하지 않는다.** 응답을 못 받아도 다시 보내지
  않는다 — 이중 주문이 실포지션 사고로 이어진다. 예외는 ``OrderResult``로
  감싸 돌려줄 뿐, 여기서 삼키거나 재시도하지 않는다. ``kiwoom_client`` 자체가
  HTTP 429·401·``return_code==5``(요청개수초과)에서 **주문이 접수되기 전에**
  내부적으로 재시도하는 것은 별개다(``base.py`` ``BaseClient.request``) —
  그건 아직 접수 안 된 요청의 전송 재시도이지, 접수된 주문을 다시 내는 게
  아니라서 이중 주문 위험이 없다. 여기서 금지하는 건 **응답을 이미 받았거나
  모호한 상태에서 우리가 다시 부르는 것**뿐이다.
* **조회(holdings·open_orders·poll_fills)는 예외 시 1회만 재시도한다**
  (``_read``). 재시도해도 이중 주문 위험이 없다.
* **토큰을 폐기(logout/revoke)하는 메서드는 만들지 않는다.**
  :data:`~.kiwoom_spec.TOKEN_SHARING_WARNING` — scalp-it·daytrade-it 이 같은
  appkey 로 동시에 실매매하므로, 한쪽이 revoke 하면 다른 쪽 세션이 죽는다.
  토큰 갱신은 kiwoom-client 가 알아서 한다.
* ``dry_run=True``(기본)면 주문 제출·취소는 POST 하지 않고
  ``OrderResult(dry_run=True, submitted=False)`` 를 돌려준다. 조회 메서드는
  dry-run 과 무관하게 정상 동작한다(읽기는 계좌에 해를 끼치지 않는다).
* ``cancel`` 은 ``intent.qty``(원주문 수량)를 그대로 취소바디에 보낸다
  (``OrderIntent.cancel_body``) — 부분체결 후 남은 수량이 아니다. 호출부가
  남은 수량으로 ``intent`` 를 다시 만들어 넘겨야 한다.

예외 → ``OrderResult`` 매핑
============================

``kiwoom_client`` 는 ``return_code`` 가 0이 아니면 ``KiwoomAPIError(code,
message, response)`` 를 던진다(``base.py`` ``_check_return_code``) — 즉
"거부"도 파이썬 예외로 온다. 이걸 타임아웃 같은 진짜 전송 실패와 구분 못 하면
``return_code`` 가 늘 ``None`` 이 되어 재시도 판단·로그가 무의미해진다. 그래서
예외에서 ``.code``·``.response``·``.message`` 속성을 덕타이핑으로 읽는다
(``KiwoomAPIError`` 를 직접 import 하지 않는다 — 테스트 가짜 예외도 같은
속성만 있으면 된다). 없으면(순수 네트워크 예외 등) 기존처럼
``f"{type(exc).__name__}: {exc}"`` 로 감싼다.

성공 응답이라도 ``return_code`` 가 숫자가 아니거나 응답 자체가 dict 가 아니면
(``resp.get``/``int()`` 가 터지면) 파싱 예외가 호출부로 새는 대신
``submitted=True, return_code=None, return_msg=repr(원본 응답)`` 로 떨어진다
— 주문은 이미 나갔을 수 있으니 예외를 던져 ``ord_no`` 를 잃어버리면 안 된다.

미검증 필드 — kiwoom-client 실호출로 확인 안 됨
================================================

* 매수/매도 판별 필드(``ka10075`` 미체결·``ka10076`` 체결 공통) — 실호출로
  확인 안 됐다. ``io_tp_nm``·``sell_tp_nm``(사람이 읽는 "매수"/"매도" 텍스트로
  추정) 다음 ``sell_tp``(코드값 — 키움 관례상 ``"1"``\\=매도·``"2"``\\=매수로
  추정, 이것도 미확인)를 순서대로 본다(:func:`_side_from_row`). **어느 후보도
  판별이 안 되면 그 행은 조용히 버린다(절대 추측하지 않는다)** —
  ``KiwoomBroker.skipped_rows`` 로 몇 건 버렸는지 셀 수 있다. ``trde_tp`` 는
  후보에서 뺐다 — 이건 매매구분이 아니라 **주문유형**(``"0"``\\=보통/지정가
  같은) 텍스트("보통" 등)라 매수/매도 판별에 못 쓴다.
* ``ka10075`` 지정가 필드명 ``ord_pric`` — 브리프 지시값, 실호출 미확인.
* ``ka10076``(체결) 목록 키 ``cntr``, 주문번호 ``ord_no``, 누적 체결수량
  ``cntr_qty``, 체결단가 ``cntr_pric`` — 전부 브리프 지시값이고 실호출로 확인
  안 됐다. 시각 필드도 미확인이라 없으면 ``now_kst()`` 로 채운다. ``cntr_pric``
  는 한 번의 응답에 같은 주문의 체결 행이 여러 개 묶여 오면(:func:`_diff_fills`
  가 합산) 마지막 행 값을 그대로 쓴다 — 진짜 평균단가가 아니라 근사치다.
* ``poll_fills``/``prime`` 의 ``filled_orders`` 조회 바디
  (``stk_cd=""``·``qry_tp="0"``·``sell_tp="0"``·``ord_no=""``·``stex_tp="0"``)
  는 키움 REST 문서상 ``ka10076`` 필드명을 참고한 추정값이다 —
  ``kiwoom-client`` 패키지 자체에는 ``ka10076`` 바디 예시가 없다(``**kwargs``
  그대로 전달하는 얇은 래퍼라 필드명 정보가 없다 — ``domestic/account.py``).
  **모의계좌로 실호출 확인 전에는 ``poll_fills`` 를 실주문 데몬에 쓰지 말 것.**
  그래서 ``fills_verified=False``(기본)면 ``poll_fills``·``prime`` 은 조회하지 않고
  ``[]`` 를 돌려주며 처음 한 번 경고 로그를 남긴다 — :class:`~.engine.EngineCore` 가
  이벤트마다 ``oms.sync()`` 로 부르는데, 틀린 필드로 읽은 체결이 장부·킬스위치에
  들어가는 편보다 체결이 안 들어오는 편(대사로 드러난다)이 덜 위험하다.
* ``ka10075``(미체결) 행이 있는데 **전부** 매수/매도 판별에 실패하면 경고 로그를
  남긴다. 조용히 빈 목록이 되면 걸린 매도 잔량 예약이 0 이 되어 중복 청산 주문이
  나간다 — 이게 실패 모드다.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..market.codes import normalize_code
from ..market.session import now_kst
from .events import Fill, Holding, OpenOrder, normalize_ord_no
from .kiwoom_spec import BALANCE_ROWS_KEY, UNFILLED_ROWS_KEY, strip_sign
from .orders import OrderIntent, OrderResult

__all__ = ["KiwoomBroker"]

_log = logging.getLogger(__name__)

#: ``ka10076`` 체결 목록 키. **미검증**(브리프 지시값) — 모듈 docstring 참고.
FILLED_ROWS_KEY = "cntr"

#: ``ka10076`` 조회 바디. **미검증** — 모듈 docstring 참고.
_FILLED_QUERY_BODY: dict[str, str] = {
    "stk_cd": "",
    "qry_tp": "0",
    "sell_tp": "0",
    "ord_no": "",
    "stex_tp": "0",
}

#: 매수/매도 코드값 판별(``sell_tp``). 키움 관례상 ``1``\\=매도·``2``\\=매수로
#: 추정 — 실호출 미확인(모듈 docstring 참고).
_SELL_TP_CODES = {"1": "sell", "2": "buy"}


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


def _side_from_row(row: dict[str, Any]) -> str | None:
    """미체결/체결 행에서 매수/매도를 판별한다. 모르면 ``None``(절대 추측 안 함).

    **미검증** — 모듈 docstring "미검증 필드" 참고.
    """
    for key in ("io_tp_nm", "sell_tp_nm"):
        text = row.get(key)
        if text not in (None, ""):
            text = str(text)
            if "매도" in text:
                return "sell"
            if "매수" in text:
                return "buy"
    code = row.get("sell_tp")
    if code not in (None, ""):
        side = _SELL_TP_CODES.get(str(code).strip())
        if side is not None:
            return side
    return None


class KiwoomBroker:
    """``kiwoom_client.KiwoomAPI`` 호환 객체를 감싸는 실매매 :class:`Broker` 어댑터.

    ``fills_verified`` 는 ``ka10076`` 체결 조회 필드를 모의계좌 실호출로 확인했다는
    호출부의 선언이다. ``False``(기본)면 ``poll_fills``·``prime`` 은 ``[]``/무동작이다
    (모듈 docstring "미검증 필드").
    """

    def __init__(self, api: Any, *, dry_run: bool = True, fills_verified: bool = False) -> None:
        self._api = api
        self._dry_run = dry_run
        self._fills_verified = fills_verified
        self._warned_unverified = False
        #: 정규화 주문번호 → 지금까지 본 누적 체결수량. ``poll_fills`` 증분 계산용.
        self._cum_filled: dict[str, int] = {}
        #: 마지막으로 ``_cum_filled`` 를 채운 날짜(KST). 날짜가 바뀌면 초기화한다
        #: (키움 체결 조회는 당일 기준이라, 전날 누적을 들고 있으면 다음날 첫
        #: 체결의 증분이 음수/0 으로 계산될 수 있다).
        self._fills_date: date | None = None
        #: 매수/매도를 판별 못 해 버린 행 수(디버깅·모니터링용 카운터).
        self.skipped_rows: int = 0

    @property
    def dry_run(self) -> bool:
        """``True`` 면 주문·취소를 POST 하지 않는다. ``OrderManager`` 가 차단 결과에 옮겨 적는다."""
        return self._dry_run

    # ----- 제출/취소 — 재시도 없음 -------------------------------------------

    def submit(self, intent: OrderIntent) -> OrderResult:
        """주문을 낸다. **재시도 없음.** ``dry_run`` 이면 POST 하지 않는다."""
        if self._dry_run:
            return OrderResult(intent=intent, dry_run=True, submitted=False)
        fn = self._api.order.buy_order if intent.side == "buy" else self._api.order.sell_order
        try:
            resp = fn(**intent.body())
        except Exception as exc:  # noqa: BLE001 — 어떤 예외든 결과로 감싼다. 재시도 안 함.
            return self._result_from_exception(intent, exc, fallback_ord_no=None)
        return self._result_from_response(intent, resp, fallback_ord_no=None)

    def cancel(self, ord_no: str, intent: OrderIntent) -> OrderResult:
        """미체결 주문을 취소한다. **재시도 없음.** ``dry_run`` 이면 POST 하지 않는다."""
        if self._dry_run:
            return OrderResult(intent=intent, dry_run=True, submitted=False)
        try:
            resp = self._api.order.cancel_order(**intent.cancel_body(ord_no))
        except Exception as exc:  # noqa: BLE001 — 재시도 안 함, 결과로 감싼다.
            return self._result_from_exception(intent, exc, fallback_ord_no=str(ord_no))
        return self._result_from_response(intent, resp, fallback_ord_no=str(ord_no))

    @staticmethod
    def _result_from_exception(
        intent: OrderIntent, exc: Exception, *, fallback_ord_no: str | None
    ) -> OrderResult:
        """``kiwoom_client`` 의 ``KiwoomAPIError``(``.code``·``.message``·
        ``.response``)를 덕타이핑으로 풀어 ``return_code`` 를 살린다. 그런
        속성이 없는 예외(네트워크 오류 등)는 기존처럼 문자열로만 감싼다.
        """
        code = getattr(exc, "code", None)
        if code is None:
            return OrderResult(
                intent=intent,
                dry_run=False,
                submitted=True,
                ord_no=fallback_ord_no,
                return_code=None,
                return_msg=f"{type(exc).__name__}: {exc}",
            )
        try:
            return_code: int | None = int(code)
        except (TypeError, ValueError):
            return_code = None
        response = getattr(exc, "response", None)
        ord_no = fallback_ord_no
        if isinstance(response, dict):
            raw_ord_no = response.get("ord_no")
            if raw_ord_no not in (None, ""):
                ord_no = str(raw_ord_no).strip()
        message = getattr(exc, "message", None)
        return_msg = str(message) if message not in (None, "") else f"{type(exc).__name__}: {exc}"
        return OrderResult(
            intent=intent,
            dry_run=False,
            submitted=True,
            ord_no=ord_no,
            return_code=return_code,
            return_msg=return_msg,
        )

    @staticmethod
    def _result_from_response(
        intent: OrderIntent, resp: Any, *, fallback_ord_no: str | None
    ) -> OrderResult:
        """응답이 예상 모양(dict, 숫자 ``return_code``)이 아니어도 예외를 올리지
        않는다 — 주문이 이미 나갔을 수 있어, 파싱 실패로 ``ord_no`` 를 통째로
        잃어버리면 안 된다.
        """
        try:
            rc = resp.get("return_code")
            raw_ord_no = resp.get("ord_no")
            ord_no = str(raw_ord_no).strip() if raw_ord_no not in (None, "") else fallback_ord_no
            return_code = int(rc) if rc is not None else None
            return_msg = str(resp.get("return_msg") or "")
        except (AttributeError, TypeError, ValueError):
            return OrderResult(
                intent=intent,
                dry_run=False,
                submitted=True,
                ord_no=fallback_ord_no,
                return_code=None,
                return_msg=repr(resp),
            )
        return OrderResult(
            intent=intent,
            dry_run=False,
            submitted=True,
            ord_no=ord_no,
            return_code=return_code,
            return_msg=return_msg,
        )

    # ----- 조회 — 예외 시 1회 재시도 -----------------------------------------

    def _read(self, fn: Any, **kwargs: Any) -> dict[str, Any]:
        """조회 호출. 예외면 1회만 재시도한다(제출과 달리 재시도해도 안전하다)."""
        try:
            return fn(**kwargs)
        except Exception:  # noqa: BLE001 — 1회 재시도, 두 번째도 실패하면 올린다.
            return fn(**kwargs)

    def holdings(self) -> dict[str, Holding]:
        """``kt00018`` 계좌평가잔고내역 → 보유 종목(종목코드 → :class:`Holding`).

        수량 0인 행(완전 청산됐지만 계좌 응답에 잔존하는 행)은 뺀다.
        """
        resp = self._read(
            self._api.account.evaluation_balance_detail, qry_tp="1", dmst_stex_tp="KRX"
        )
        out: dict[str, Holding] = {}
        for row in resp.get(BALANCE_ROWS_KEY) or []:
            code = normalize_code(row.get("stk_cd"))
            if not code:
                continue
            qty = strip_sign(row.get("rmnd_qty"))
            if qty == 0:
                continue
            avg_price = _strip_sign_float(row.get("pur_pric"))
            out[code] = Holding(code=code, qty=qty, avg_price=avg_price)
        return out

    def open_orders(self) -> list[OpenOrder]:
        """``ka10075`` 미체결요청 → 미체결 주문 목록.

        종목코드가 비었거나 매수/매도 판별이 안 되는 행은 버린다
        (``skipped_rows`` 에 판별 실패분만 센다).
        """
        resp = self._read(
            self._api.account.unfilled_orders, all_stk_tp="0", trde_tp="0", stex_tp="0"
        )
        out: list[OpenOrder] = []
        unknown_side = 0
        for row in resp.get(UNFILLED_ROWS_KEY) or []:
            ord_no = str(row.get("ord_no") or "").strip()
            if not ord_no:
                continue
            code = normalize_code(row.get("stk_cd"))
            if not code:
                continue
            side = _side_from_row(row)
            if side is None:
                self.skipped_rows += 1
                unknown_side += 1
                continue
            out.append(
                OpenOrder(
                    ord_no=ord_no,
                    code=code,
                    side=side,
                    qty=strip_sign(row.get("ord_qty")),
                    remaining=strip_sign(_first(row, "oso_qty", "ord_remnq")),
                    price=strip_sign(row.get("ord_pric")),
                )
            )
        if unknown_side and not out:
            # 조용한 0 이 실패 모드다 — 걸린 매도 잔량 예약이 사라져 중복 청산이 나간다.
            _log.warning(
                "open_orders: ka10075 미체결 %d행 전부 매수/매도 판별 실패 — 빈 목록을 돌려준다"
                "(매도 잔량 예약 0). 판별 필드(io_tp_nm·sell_tp_nm·sell_tp) 확인 필요",
                unknown_side,
            )
        return out

    def prime(self) -> None:
        """``poll_fills`` 의 누적 체결수량 기준선을 **Fill 을 내지 않고** 채운다.

        데몬 재시작 직후, ``poll_fills`` 를 처음 부르기 **전에 딱 한 번**
        호출한다. 안 하면 재시작 전에 이미 체결된 수량이 "새 체결"로 오인돼
        중복 처리된다(재시작 시점의 누적 체결수량을 ``0`` 으로 보기 때문).
        ``fills_verified=False`` 면 무동작이다(경고 1회).
        """
        if not self._fills_enabled():
            return
        now = now_kst()
        self._reset_if_new_day(now)
        resp = self._read(self._api.account.filled_orders, **_FILLED_QUERY_BODY)
        self._diff_fills(resp, emit=False, ts=now)

    def poll_fills(self) -> list[Fill]:
        """``ka10076`` 체결요청 → 지난 호출 이후 새로 늘어난 체결만 :class:`Fill` 로.

        키움 응답은 주문별 **누적** 체결수량을 준다고 가정한다(추정 — 모듈
        docstring 미검증 목록 참고). 같은 응답 안에 같은 주문번호 행이 여러 개면
        (한 번에 여러 체결 기록이 나뉘어 온 것으로 보고) 그 ``cntr_qty`` 를
        합산한 값을 그 주문의 이번 누적으로 삼는다. 지난번 누적보다 늘어난
        만큼만 ``Fill`` 로 낸다. 매일 첫 호출에서 날짜가 바뀐 걸 감지하면 누적을
        초기화한다(당일 기준 조회라서). 재시작 후 첫 호출 전엔 :meth:`prime` 을
        불러 기준선부터 잡을 것.

        ``fills_verified=False``(기본)면 조회하지 않고 ``[]`` 를 돌려준다(경고 1회).
        """
        if not self._fills_enabled():
            return []
        now = now_kst()
        self._reset_if_new_day(now)
        resp = self._read(self._api.account.filled_orders, **_FILLED_QUERY_BODY)
        return self._diff_fills(resp, emit=True, ts=now)

    def _fills_enabled(self) -> bool:
        if self._fills_verified:
            return True
        if not self._warned_unverified:
            self._warned_unverified = True
            _log.warning(
                "KiwoomBroker: ka10076 체결 조회가 미검증이라 poll_fills/prime 을 끈다"
                "(fills_verified=False). 모의계좌 실호출 확인 후 fills_verified=True 로 켤 것"
            )
        return False

    def _reset_if_new_day(self, now: Any) -> None:
        today = now.date()
        if self._fills_date != today:
            self._cum_filled.clear()
            self._fills_date = today

    def _diff_fills(self, resp: dict[str, Any], *, emit: bool, ts: Any) -> list[Fill]:
        groups: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for row in resp.get(FILLED_ROWS_KEY) or []:
            raw_ord_no = str(row.get("ord_no") or "").strip()
            if not raw_ord_no:
                continue
            key = normalize_ord_no(raw_ord_no)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)

        fills: list[Fill] = []
        for key in order:
            rows = groups[key]
            raw_ord_no = str(rows[0].get("ord_no") or "").strip()
            cum_qty = sum(strip_sign(r.get("cntr_qty")) for r in rows)
            prev_qty = self._cum_filled.get(key, 0)
            delta = cum_qty - prev_qty
            if emit and delta > 0:
                last = rows[-1]
                code = normalize_code(last.get("stk_cd"))
                side = _side_from_row(last)
                if not code or side is None:
                    self.skipped_rows += 1
                else:
                    fills.append(
                        Fill(
                            ord_no=raw_ord_no,
                            code=code,
                            side=side,
                            qty=delta,
                            price=strip_sign(last.get("cntr_pric")),
                            ts=ts,
                        )
                    )
            self._cum_filled[key] = max(cum_qty, prev_qty)
        return fills


def _strip_sign_float(value: Any) -> float:
    """:func:`~.kiwoom_spec.strip_sign` 의 실수 버전.

    평단가 같은 필드는 소수점을 가질 수 있어 ``strip_sign`` 의 ``int()`` 로
    자르면 안 된다. 천단위 구분자(``","``)도 뗀다 — daytrade-it
    ``kiwoom_broker.py`` 의 ``_to_decimal`` 이 실계좌 응답에서 본 형태.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return 0.0
    text = text.lstrip("+-").replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"숫자로 읽을 수 없는 값이다: {value!r}") from exc
