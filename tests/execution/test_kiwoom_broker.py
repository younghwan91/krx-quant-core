"""``KiwoomBroker`` 단위 테스트 — 가짜 kiwoom-client api(``SimpleNamespace``)로 돈다.

호출 기록은 각 가짜 함수가 자기 호출 리스트에 ``kwargs`` 를 append 해서 확인한다.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from krx_quant_core.execution.events import Holding, OpenOrder
from krx_quant_core.execution.kiwoom_broker import KiwoomBroker
from krx_quant_core.execution.orders import OrderIntent


def _recorder(responses: list[Any] | None = None):
    """호출마다 ``calls`` 에 kwargs 를 남기고, ``responses`` 를 순서대로 돌려주는 가짜 함수.

    ``responses`` 항목이 ``Exception`` 인스턴스면 그 예외를 던진다(재시도 시나리오용).
    """
    calls: list[dict[str, Any]] = []
    responses = list(responses or [])

    def fn(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if responses:
            item = responses.pop(0)
        else:
            item = {}
        if isinstance(item, Exception):
            raise item
        return item

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def _make_api(*, buy=None, sell=None, cancel=None, balance=None, unfilled=None, filled=None):
    order_ns = SimpleNamespace(
        buy_order=buy or _recorder(),
        sell_order=sell or _recorder(),
        cancel_order=cancel or _recorder(),
    )
    account_ns = SimpleNamespace(
        evaluation_balance_detail=balance or _recorder([{}]),
        unfilled_orders=unfilled or _recorder([{}]),
        filled_orders=filled or _recorder([{}]),
    )
    return SimpleNamespace(order=order_ns, account=account_ns)


def _buy_intent(code="005930", qty=10, price=70000) -> OrderIntent:
    return OrderIntent(side="buy", code=code, qty=qty, price=price)


def _sell_intent(code="005930", qty=10, price=70000) -> OrderIntent:
    return OrderIntent(side="sell", code=code, qty=qty, price=price)


# ----- dry-run -----------------------------------------------------------------


def test_dry_run_submit_does_not_call_order_api():
    buy = _recorder()
    api = _make_api(buy=buy)
    broker = KiwoomBroker(api, dry_run=True)

    result = broker.submit(_buy_intent())

    assert buy.calls == []
    assert result.dry_run is True
    assert result.submitted is False
    assert result.ord_no is None


def test_dry_run_cancel_does_not_call_order_api():
    cancel = _recorder()
    api = _make_api(cancel=cancel)
    broker = KiwoomBroker(api, dry_run=True)

    result = broker.cancel("123", _sell_intent())

    assert cancel.calls == []
    assert result.dry_run is True
    assert result.submitted is False


def test_dry_run_holdings_still_reads():
    balance = _recorder([{"acnt_evlt_remn_indv_tot": [
        {"stk_cd": "A005930", "rmnd_qty": "10", "pur_pric": "70000"}
    ]}])
    api = _make_api(balance=balance)
    broker = KiwoomBroker(api, dry_run=True)

    holdings = broker.holdings()

    assert len(balance.calls) == 1
    assert holdings["005930"] == Holding(code="005930", qty=10, avg_price=70000.0)


# ----- submit: 성공/예외, 재시도 없음 --------------------------------------------


def test_submit_buy_success_returns_ord_no_and_ok():
    buy = _recorder([{"return_code": 0, "return_msg": "정상", "ord_no": "0000123"}])
    api = _make_api(buy=buy)
    broker = KiwoomBroker(api, dry_run=False)

    result = broker.submit(_buy_intent())

    assert len(buy.calls) == 1
    assert buy.calls[0] == _buy_intent().body()
    assert result.submitted is True
    assert result.ord_no == "0000123"
    assert result.return_code == 0
    assert result.ok is True


def test_submit_sell_calls_sell_order():
    sell = _recorder([{"return_code": 0, "ord_no": "5"}])
    api = _make_api(sell=sell)
    broker = KiwoomBroker(api, dry_run=False)

    broker.submit(_sell_intent())

    assert len(sell.calls) == 1
    assert sell.calls[0] == _sell_intent().body()


def test_submit_exception_no_retry_and_wraps_result():
    def raise_once(**kwargs):
        raise_once.calls.append(kwargs)
        raise ConnectionError("boom")

    raise_once.calls = []
    api = _make_api(buy=raise_once)
    broker = KiwoomBroker(api, dry_run=False)

    result = broker.submit(_buy_intent())

    assert len(raise_once.calls) == 1  # 재시도 없음 — 딱 한 번만 호출
    assert result.submitted is True
    assert result.return_code is None
    assert result.return_msg == "ConnectionError: boom"
    assert result.ok is False


def test_cancel_success():
    cancel = _recorder([{"return_code": 0, "ord_no": "77"}])
    api = _make_api(cancel=cancel)
    broker = KiwoomBroker(api, dry_run=False)

    result = broker.cancel("77", _sell_intent())

    assert len(cancel.calls) == 1
    assert cancel.calls[0] == _sell_intent().cancel_body("77")
    assert result.ord_no == "77"
    assert result.return_code == 0


def test_cancel_exception_no_retry():
    def raise_once(**kwargs):
        raise_once.calls.append(kwargs)
        raise TimeoutError("timeout")

    raise_once.calls = []
    api = _make_api(cancel=raise_once)
    broker = KiwoomBroker(api, dry_run=False)

    result = broker.cancel("9", _sell_intent())

    assert len(raise_once.calls) == 1
    assert result.return_code is None
    assert result.return_msg == "TimeoutError: timeout"


# ----- holdings: 파싱, 부호·접두 제거 ---------------------------------------------


def test_holdings_parses_code_prefix_and_signed_fields():
    balance = _recorder([{"acnt_evlt_remn_indv_tot": [
        {"stk_cd": "A005930", "rmnd_qty": "+000010", "pur_pric": "70000"},
        {"stk_cd": "A000660", "rmnd_qty": "5", "pur_pric": "-120000"},
    ]}])
    api = _make_api(balance=balance)
    broker = KiwoomBroker(api, dry_run=False)

    holdings = broker.holdings()

    assert holdings["005930"] == Holding(code="005930", qty=10, avg_price=70000.0)
    assert holdings["000660"] == Holding(code="000660", qty=5, avg_price=120000.0)


# ----- open_orders -----------------------------------------------------------


def test_open_orders_parses_rows():
    unfilled = _recorder([{"oso": [
        {
            "ord_no": "0000321",
            "stk_cd": "A005930",
            "io_tp_nm": "매수",
            "ord_qty": "10",
            "oso_qty": "4",
            "ord_pric": "70000",
        },
        {
            "ord_no": "0000322",
            "stk_cd": "A000660",
            "io_tp_nm": "매도정정",
            "ord_qty": "3",
            "oso_qty": "3",
            "ord_pric": "120000",
        },
    ]}])
    api = _make_api(unfilled=unfilled)
    broker = KiwoomBroker(api, dry_run=False)

    orders = broker.open_orders()

    assert orders == [
        OpenOrder(ord_no="0000321", code="005930", side="buy", qty=10, remaining=4, price=70000),
        OpenOrder(ord_no="0000322", code="000660", side="sell", qty=3, remaining=3, price=120000),
    ]


# ----- poll_fills: 증분만 -------------------------------------------------------


def test_poll_fills_returns_only_new_increment_across_two_calls():
    row1 = {
        "ord_no": "1", "stk_cd": "A005930", "io_tp_nm": "매수",
        "cntr_qty": "4", "cntr_pric": "70000",
    }
    row2 = {
        "ord_no": "1", "stk_cd": "A005930", "io_tp_nm": "매수",
        "cntr_qty": "10", "cntr_pric": "70100",
    }
    filled = _recorder([{"cntr": [row1]}, {"cntr": [row2]}])
    api = _make_api(filled=filled)
    broker = KiwoomBroker(api, dry_run=False)

    first = broker.poll_fills()
    second = broker.poll_fills()

    assert len(first) == 1
    assert first[0].qty == 4
    assert first[0].code == "005930"
    assert first[0].side == "buy"
    assert first[0].price == 70000
    assert first[0].ord_no == "1"

    assert len(second) == 1
    assert second[0].qty == 6  # 10 - 4 만 신규
    assert second[0].price == 70100


def test_poll_fills_no_new_fill_yields_empty_list():
    row = {"ord_no": "1", "stk_cd": "A005930", "cntr_qty": "10", "cntr_pric": "70000"}
    filled = _recorder([{"cntr": [row]}, {"cntr": [row]}])
    api = _make_api(filled=filled)
    broker = KiwoomBroker(api, dry_run=False)

    broker.poll_fills()
    second = broker.poll_fills()

    assert second == []


# ----- 조회 1회 재시도 -----------------------------------------------------------


def test_holdings_retries_once_after_exception_then_succeeds():
    row = {"stk_cd": "A005930", "rmnd_qty": "1", "pur_pric": "1000"}
    balance = _recorder(
        [ConnectionError("network blip"), {"acnt_evlt_remn_indv_tot": [row]}]
    )
    api = _make_api(balance=balance)
    broker = KiwoomBroker(api, dry_run=False)

    holdings = broker.holdings()

    assert len(balance.calls) == 2
    assert holdings["005930"].qty == 1


def test_holdings_raises_after_two_failures():
    balance = _recorder([ConnectionError("first"), ConnectionError("second")])
    api = _make_api(balance=balance)
    broker = KiwoomBroker(api, dry_run=False)

    with pytest.raises(ConnectionError):
        broker.holdings()

    assert len(balance.calls) == 2


def test_open_orders_retries_once():
    unfilled = _recorder([TimeoutError("blip"), {"oso": []}])
    api = _make_api(unfilled=unfilled)
    broker = KiwoomBroker(api, dry_run=False)

    orders = broker.open_orders()

    assert orders == []
    assert len(unfilled.calls) == 2


def test_poll_fills_retries_once():
    filled = _recorder([TimeoutError("blip"), {"cntr": []}])
    api = _make_api(filled=filled)
    broker = KiwoomBroker(api, dry_run=False)

    fills = broker.poll_fills()

    assert fills == []
    assert len(filled.calls) == 2


# ----- 빈 응답 방어 -----------------------------------------------------------


def test_holdings_empty_response_returns_empty_dict():
    api = _make_api(balance=_recorder([{}]))
    broker = KiwoomBroker(api, dry_run=False)

    assert broker.holdings() == {}


def test_open_orders_skips_rows_without_ord_no():
    row = {"stk_cd": "A005930", "ord_qty": "1", "oso_qty": "1", "ord_pric": "1"}
    unfilled = _recorder([{"oso": [row]}])
    api = _make_api(unfilled=unfilled)
    broker = KiwoomBroker(api, dry_run=False)

    assert broker.open_orders() == []
