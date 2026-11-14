"""키움 REST 주문·계좌 프로토콜 상수 — 실주문으로 확인된 것과 아닌 것을 갈라 적는다.

scalp-it ``adapters/kiwoom_order.py`` 와 daytrade-it
``infrastructure/brokers/kiwoom_broker.py`` 가 같은 값을 따로 들고 있던 것을 여기로
모았다. 값 하나가 틀리면 실계좌 주문이 거부되거나(그나마 다행) 엉뚱하게 나간다 —
그래서 **어디서 어떻게 확인됐는지**를 값 옆에 남긴다.

확인된 사실
==========

* 조회 API(``ka00001``·``kt00018``·``ka10001``·``ka10075``)는 scalp-it 이
  2026-08-25 실계좌로 호출해 경로·응답 키를 확인했다.
* 주문 API(``kt10000``/``kt10001``/``kt10003``)는 처음엔 키움 문서 +
  kiwoom-client README 로만 짰으나, scalp-it 이 2026-08-28 부터 실계좌로 지정가를
  냈다. ``data/pair_orders.log`` 의 실발주 22건 중 **1주 매수 4·매도 4 = 8건만
  ``return_code == 0`` + ``ord_no``** 이고(전부 09:09~09:19), **취소 14건은 모두
  ``return_code == 2``** 다 — 취소 성공 사례가 없다(2026-11-02 로그 전수, 전에 여기 적었던
  "22건 전부 rc=0" 은 틀렸다). 즉 매수·매도 **바디 필드명**(``dmst_stex_tp``·``stk_cd``·
  ``ord_qty``·``ord_uv``·``trde_tp``)과 **값** ``dmst_stex_tp="KRX"``(``"01"`` 아님),
  ``trde_tp="0"``(지정가, ``"00"`` 아님)은 실주문으로 확인됐고, **취소(``kt10003``)는
  확인되지 않았다.** 동시호가 주문과 2주 이상 주문 이력도 없다.
* 취소 14건이 실패한 바디는 ``{dmst_stex_tp, stk_cd, org_ord_no, ord_qty}`` 였다. 키움 공식
  스펙 JSON(``Kiwoom-Securities/Kiwoom-REST-API`` c7a3d25, 2026-11-14 대조)의 kt10003 은
  ``{dmst_stex_tp, orig_ord_no, stk_cd, cncl_qty}``(``cncl_qty="0"`` = 잔량 전부), 정정 kt10002 는
  ``{dmst_stex_tp, orig_ord_no, stk_cd, mdfy_qty, mdfy_uv[, mdfy_cond_uv]}`` 다.
  코어는 공식 필드명으로 바꿨지만 **실주문 취소 스모크로 확인되기 전까지 "미확인"** 이다.
  같은 실수가 다시 나가지 않게 :data:`ORDER_BODY_FIELDS`(공식 스펙에서 옮긴 TR 별 필드 표)와
  :func:`validate_order_body` 를 두고, 바디 빌더와 ``KiwoomBroker`` 가 POST 전에 부른다 —
  스펙에 없는 필드나 빠진 필수 필드는 네트워크 전에 :class:`InvalidOrderBody` 로 막힌다.
* 주문 바디에는 계좌번호가 **안 들어간다** — 토큰이 계좌에 묶여 있다.

확인 안 된 것 — 시장가 매매구분
==============================

두 레포의 기록이 다르고, **어느 쪽도 실호출로 확인된 적이 없다**:

* scalp-it docstring: ``"3"=시장가`` (키움 문서/래퍼 README 를 읽고 적은 값).
* daytrade-it ``_ORDER_TYPE_MARKET = "03"`` — 스스로 UNVERIFIED 라고 적어 두었고,
  그래서 daytrade-it ``execute_trade`` 는 MARKET 주문을 거부한다.

여기서 하나를 골라 조용히 정본으로 만들지 않는다. :data:`TRDE_TP_MARKET` 은 현재
daytrade-it 이 실제로 쓰는 ``"03"`` 을 그대로 두되(동작 보존), 이름 옆
:data:`TRDE_TP_MARKET_VERIFIED` 가 ``False`` 이고 scalp-it 문서값은
:data:`TRDE_TP_MARKET_SCALP_DOC` 로 따로 남긴다. 실호출(모의 포함)로 확인하기 전엔
시장가를 쓰지 말 것.

토큰 공유 경고 — 절대 revoke 하지 말 것
=====================================

키움은 한 appkey 에 대해 유효기간 동안 **모든 프로세스에 같은 토큰**을 준다
(2026-09-13 확인: 두 번 발급해도 동일 토큰, 첫 토큰도 계속 유효). scalp-it 과
daytrade-it 은 **같은 appkey 로 동시에 실매매**한다. 한쪽이 종료·재시작·점검 스크립트
끝에서 토큰을 폐기(logout/revoke)하면 다른 쪽 실매매 세션이 죽는다 — scalp-it 은
401 에서만 갱신하는데 키움은 잘못된 토큰을 HTTP 200 + ``8005`` 로 돌려주고, 주문
POST 는 재시도도 없다. 토큰은 스스로 만료되게 둔다.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

__all__ = [
    "ACCOUNT_NO_API_ID",
    "ACCOUNT_PATH",
    "BALANCE_API_ID",
    "BALANCE_ROWS_KEY",
    "BUY_API_ID",
    "CANCEL_API_ID",
    "DEFAULT_EXCHANGE",
    "InvalidOrderBody",
    "KIWOOM_BASE",
    "KIWOOM_MOCK_BASE",
    "MODIFY_API_ID",
    "ORDER_BODY_FIELDS",
    "ORDER_PATH",
    "PRICE_API_ID",
    "SELL_API_ID",
    "STKINFO_PATH",
    "TOKEN_SHARING_WARNING",
    "TRDE_TP_LIMIT",
    "TRDE_TP_MARKET",
    "TRDE_TP_MARKET_SCALP_DOC",
    "TRDE_TP_MARKET_VERIFIED",
    "UNFILLED_API_ID",
    "UNFILLED_ROWS_KEY",
    "strip_sign",
    "validate_order_body",
]

#: 실계좌 REST 베이스.
KIWOOM_BASE = "https://api.kiwoom.com"
#: 모의투자 REST 베이스 — scalp-it ``--mock`` 이 주문도 이쪽으로 강제한다.
KIWOOM_MOCK_BASE = "https://mockapi.kiwoom.com"

# ----- 경로 ------------------------------------------------------------------
#: 주문 4종(매수·매도·정정·취소) 공통 경로. api-id 헤더로 구분한다.
ORDER_PATH = "/api/dostk/ordr"
#: 계좌 조회(계좌번호·잔고·미체결) 경로.
ACCOUNT_PATH = "/api/dostk/acnt"
#: 종목 기본정보(현재가) 경로.
STKINFO_PATH = "/api/dostk/stkinfo"

# ----- api-id ----------------------------------------------------------------
#: 현금 매수. 실주문 확인.
BUY_API_ID = "kt10000"
#: 현금 매도. 실주문 확인.
SELL_API_ID = "kt10001"
#: 정정. 바디 ``{dmst_stex_tp, orig_ord_no, stk_cd, mdfy_qty, mdfy_uv[, mdfy_cond_uv]}`` —
#: 공식 스펙 기준, 실주문 미확인.
MODIFY_API_ID = "kt10002"
#: 취소. 바디 ``{dmst_stex_tp, orig_ord_no, stk_cd, cncl_qty}`` — 공식 스펙 기준, 실주문 미확인.
CANCEL_API_ID = "kt10003"
#: 계좌번호조회. 빈 바디, 응답 ``acctNo``(10자리).
ACCOUNT_NO_API_ID = "ka00001"
#: 계좌평가잔고내역. 바디 ``{qry_tp:"1", dmst_stex_tp:"KRX"}``.
BALANCE_API_ID = "kt00018"
#: 주식기본정보(현재가). 바디 ``{stk_cd}``, 응답 ``cur_prc`` 는 부호 붙음.
PRICE_API_ID = "ka10001"
#: 미체결. 바디 ``{all_stk_tp:"0", trde_tp:"0", stex_tp:"0"}``.
UNFILLED_API_ID = "ka10075"

# ----- 응답 리스트 키 ----------------------------------------------------------
#: ``kt00018`` 보유 리스트. 행의 ``stk_cd`` 는 ``"A041830"`` 처럼 접두가 붙는다.
BALANCE_ROWS_KEY = "acnt_evlt_remn_indv_tot"
#: ``ka10075`` 미체결 리스트.
UNFILLED_ROWS_KEY = "oso"

# ----- 바디 값 -----------------------------------------------------------------
#: 매매구분 — 보통(지정가). 실주문 8건(1주 매수·매도)으로 확인(``"00"`` 아님).
TRDE_TP_LIMIT = "0"
#: 매매구분 — 시장가. **미확인.** daytrade-it 이 들고 있는 값 그대로(모듈 docstring).
TRDE_TP_MARKET = "03"
#: scalp-it docstring 이 키움 문서에서 옮겨 적은 시장가 값. 역시 미확인.
TRDE_TP_MARKET_SCALP_DOC = "3"
#: 시장가 매매구분이 실호출로 확인됐는가. 확인되면 이 값과 위 둘을 함께 고친다.
TRDE_TP_MARKET_VERIFIED = False
#: 거래소구분 기본. 실주문 확인(``"01"`` 아님). ``"NXT"``/``"SOR"`` 도 문서상 가능.
DEFAULT_EXCHANGE = "KRX"

# ----- 주문 바디 필드 표 ----------------------------------------------------------
#: TR 별 요청 바디 필드 ``(필드명, 필수 여부)`` — 공식 스펙 JSON 의 순서 그대로.
#: 출처: ``Kiwoom-Securities/Kiwoom-REST-API`` ``kiwoom/_data/kiwoom_api_spec.json`` c7a3d25
#: (sha256 ``42a7b391…cba79``) 의 ``apis[*].request.body``. 테스트
#: ``tests/execution/data/kiwoom_order_spec_c7a3d25.json``(같은 스펙에서 네 TR 만 뽑은 사본)과
#: 글자 하나까지 대조한다. 옛 취소 바디의 ``org_ord_no``·``ord_qty``(kt10003)는 여기 없어서
#: :func:`validate_order_body` 가 거부한다.
ORDER_BODY_FIELDS: Mapping[str, tuple[tuple[str, bool], ...]] = MappingProxyType({
    BUY_API_ID: (
        ("dmst_stex_tp", True), ("stk_cd", True), ("ord_qty", True),
        ("ord_uv", False), ("trde_tp", True), ("cond_uv", False),
    ),
    SELL_API_ID: (
        ("dmst_stex_tp", True), ("stk_cd", True), ("ord_qty", True),
        ("ord_uv", False), ("trde_tp", True), ("cond_uv", False),
    ),
    MODIFY_API_ID: (
        ("dmst_stex_tp", True), ("orig_ord_no", True), ("stk_cd", True),
        ("mdfy_qty", True), ("mdfy_uv", True), ("mdfy_cond_uv", False),
    ),
    CANCEL_API_ID: (
        ("dmst_stex_tp", True), ("orig_ord_no", True), ("stk_cd", True),
        ("cncl_qty", True),
    ),
})


class InvalidOrderBody(ValueError):
    """주문·정정·취소 바디가 공식 스펙 필드와 안 맞는다 — 네트워크로 나가기 전에 막는다."""


def validate_order_body(api_id: str, body: Mapping[str, Any]) -> None:
    """``body`` 가 ``api_id`` 의 공식 요청 필드와 맞는지 본다. 안 맞으면 :class:`InvalidOrderBody`.

    * 표에 없는 TR → 거부(검사 못 한 바디를 내보내지 않는다).
    * 스펙에 없는 필드(예: 취소의 옛 ``org_ord_no``·``ord_qty``) → 거부.
    * 필수 필드가 없거나 빈 문자열·``None`` → 거부. 값 ``"0"`` 은 유효하다
      (``cncl_qty``·``mdfy_qty`` 의 ``"0"`` = 잔량 전부).

    값의 의미(가격 범위·호가단위 등)는 보지 않는다 — 그건 가드 몫이다.
    """
    fields = ORDER_BODY_FIELDS.get(api_id)
    if fields is None:
        raise InvalidOrderBody(f"필드 표에 없는 주문 TR: {api_id!r}")
    known = {name for name, _ in fields}
    unknown = [k for k in body if k not in known]
    if unknown:
        raise InvalidOrderBody(
            f"{api_id} 바디에 스펙에 없는 필드: {unknown} (허용: {[n for n, _ in fields]})"
        )
    missing = [name for name, req in fields if req and body.get(name) in (None, "")]
    if missing:
        raise InvalidOrderBody(f"{api_id} 바디에 필수 필드 누락·빈 값: {missing}")


#: 로그·코드리뷰에서 인용하라고 둔 한 줄 경고.
TOKEN_SHARING_WARNING = (
    "키움 토큰은 appkey 단위로 공유된다 — scalp-it·daytrade-it 이 같은 appkey 로 "
    "동시에 실매매하므로 절대 revoke/logout 하지 말 것(상대 세션이 죽는다)."
)


def strip_sign(value: Any) -> int:
    """``+274500`` / ``-258000`` 같은 부호 붙은 값을 정수로 만든다.

    부호는 전일 대비 방향 표시라 값의 음양이 아니다. 그래서 떼고 읽는다.
    빈 값은 0 이다 — 거래 없는 분봉이 실제로 온다.

    scalp-it ``adapters/kiwoom.py`` 에서 옮겼다. 원본은 읽을 수 없는 값에
    ``KiwoomError`` 를 던지지만 코어는 키움 클라이언트 예외를 모르므로 같은 메시지의
    ``ValueError`` 를 던진다(호출부가 감싸서 원래 예외로 바꿀 수 있게).
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return 0
    text = text.lstrip("+-")
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"숫자로 읽을 수 없는 값이다: {value!r}") from exc
