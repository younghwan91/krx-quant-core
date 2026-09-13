"""키움 프로토콜 상수 — 실주문으로 확인된 값이 조용히 바뀌지 않게 못박는다."""

from __future__ import annotations

import pytest

from krx_quant_core.execution import kiwoom_spec as spec
from krx_quant_core.execution import strip_sign


def test_live_verified_order_constants():
    assert spec.ORDER_PATH == "/api/dostk/ordr"
    assert (spec.BUY_API_ID, spec.SELL_API_ID, spec.CANCEL_API_ID) == (
        "kt10000", "kt10001", "kt10003")
    assert spec.TRDE_TP_LIMIT == "0"
    assert spec.DEFAULT_EXCHANGE == "KRX"


def test_account_query_constants():
    assert spec.ACCOUNT_PATH == "/api/dostk/acnt"
    assert spec.STKINFO_PATH == "/api/dostk/stkinfo"
    assert spec.ACCOUNT_NO_API_ID == "ka00001"
    assert spec.BALANCE_API_ID == "kt00018"
    assert spec.PRICE_API_ID == "ka10001"
    assert spec.UNFILLED_API_ID == "ka10075"
    assert spec.BALANCE_ROWS_KEY == "acnt_evlt_remn_indv_tot"
    assert spec.UNFILLED_ROWS_KEY == "oso"


def test_market_order_type_is_recorded_as_unverified():
    # 두 레포 기록이 다르고 둘 다 실호출 확인 전이다 — 확인되면 이 테스트를 고친다.
    assert spec.TRDE_TP_MARKET == "03"
    assert spec.TRDE_TP_MARKET_SCALP_DOC == "3"
    assert spec.TRDE_TP_MARKET_VERIFIED is False


def test_bases():
    assert spec.KIWOOM_BASE == "https://api.kiwoom.com"
    assert spec.KIWOOM_MOCK_BASE == "https://mockapi.kiwoom.com"
    assert "revoke" in spec.TOKEN_SHARING_WARNING


@pytest.mark.parametrize(
    ("raw", "want"),
    [("+274500", 274500), ("-258000", 258000), ("000000062900", 62900), ("", 0),
     (None, 0), ("-", 0), ("  +10  ", 10), ("12.9", 12), (1234, 1234)],
)
def test_strip_sign(raw, want):
    assert strip_sign(raw) == want


def test_strip_sign_rejects_garbage():
    with pytest.raises(ValueError, match="숫자로 읽을 수 없는 값이다"):
        strip_sign("abc")
