"""``PaperBroker`` — 같은 ``Broker`` 프로토콜의 모의 체결.

마켓어블 즉시 체결(테이커) vs 대기 후 지정가 판정(``through``/``touch``), latency,
보유 부족 매도 차단, 취소, ``poll_fills`` 증분을 검증한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from krx_quant_core.execution.events import Holding
from krx_quant_core.execution.orders import OrderIntent
from krx_quant_core.execution.paper import PaperBroker

CODE = "005930"


def _ts(sec: int = 0) -> datetime:
    return datetime(2026, 9, 16, 9, 0, 0) + timedelta(seconds=sec)


def _buy(qty: int = 10, price: int = 10_000) -> OrderIntent:
    return OrderIntent(side="buy", code=CODE, qty=qty, price=price)


def _sell(qty: int = 10, price: int = 10_000) -> OrderIntent:
    return OrderIntent(side="sell", code=CODE, qty=qty, price=price)


def test_marketable_buy_fills_immediately_at_ask():
    broker = PaperBroker()
    result = broker.submit(_buy(qty=10, price=10_000))
    assert result.submitted and not result.blocked
    assert result.ord_no == "P000001"
    assert result.return_code == 0

    fills = broker.on_quote(CODE, _ts(0), bid=9_990, ask=9_995, last=9_993)

    assert len(fills) == 1
    fill = fills[0]
    assert fill.ord_no == "P000001"
    assert fill.side == "buy"
    assert fill.qty == 10
    assert fill.price == 9_995
    assert broker.open_orders() == []
    assert broker.holdings()[CODE].qty == 10
    assert broker.holdings()[CODE].avg_price == 9_995.0


def test_queued_buy_through_basis_last_equal_limit_not_filled_then_below_fills():
    broker = PaperBroker(fill_basis="through")
    broker.submit(_buy(qty=10, price=10_000))

    # ask 는 아직 지정가 위 — 마켓어블 아님. last == limit 은 through 에서 미체결.
    fills1 = broker.on_quote(CODE, _ts(0), bid=9_900, ask=10_050, last=10_000)
    assert fills1 == []
    assert len(broker.open_orders()) == 1

    fills2 = broker.on_quote(CODE, _ts(1), bid=9_900, ask=10_050, last=9_999)
    assert len(fills2) == 1
    assert fills2[0].price == 10_000
    assert broker.open_orders() == []


def test_queued_buy_touch_basis_last_equal_limit_fills():
    broker = PaperBroker(fill_basis="touch")
    broker.submit(_buy(qty=10, price=10_000))

    fills = broker.on_quote(CODE, _ts(0), bid=9_900, ask=10_050, last=10_000)
    assert len(fills) == 1
    assert fills[0].price == 10_000


def test_latency_blocks_quote_before_arrival():
    broker = PaperBroker(latency_sec=2.0)
    # 이 종목 시세를 먼저 한 번 본다 — 도착시각 기준(last_seen_ts)이 생긴다.
    broker.on_quote(CODE, _ts(0), bid=9_900, ask=10_050, last=10_020)

    broker.submit(_buy(qty=10, price=10_000))  # 도착시각 = ts(0) + 2s = ts(2)

    # 1초 뒤 — 마켓어블이어도 아직 도착 전이라 체결 안 됨.
    fills = broker.on_quote(CODE, _ts(1), bid=9_900, ask=9_995, last=9_990)
    assert fills == []
    assert len(broker.open_orders()) == 1

    # 2초 뒤 — 도착. 마켓어블이라 체결.
    fills2 = broker.on_quote(CODE, _ts(2), bid=9_900, ask=9_995, last=9_990)
    assert len(fills2) == 1
    assert fills2[0].price == 9_995


def test_sell_blocked_when_qty_exceeds_holdings():
    broker = PaperBroker(holdings={CODE: Holding(code=CODE, qty=5, avg_price=10_000.0)})
    result = broker.submit(_sell(qty=10, price=10_000))

    assert result.blocked
    assert result.submitted is False
    assert result.blocked_reason == "paper: 보유 부족"
    assert broker.open_orders() == []


def test_sell_blocked_when_already_queued_sell_uses_up_holdings():
    broker = PaperBroker(holdings={CODE: Holding(code=CODE, qty=10, avg_price=10_000.0)})
    first = broker.submit(_sell(qty=10, price=10_500))
    assert first.submitted and not first.blocked

    second = broker.submit(_sell(qty=1, price=10_500))
    assert second.blocked
    assert second.blocked_reason == "paper: 보유 부족"


def test_sell_marketable_fills_at_bid_and_reduces_holdings():
    broker = PaperBroker(holdings={CODE: Holding(code=CODE, qty=10, avg_price=9_000.0)})
    broker.submit(_sell(qty=10, price=10_000))

    fills = broker.on_quote(CODE, _ts(0), bid=10_100, ask=10_150, last=10_120)

    assert len(fills) == 1
    assert fills[0].side == "sell"
    assert fills[0].price == 10_100
    assert CODE not in broker.holdings()


def test_cancel_removes_pending_order_and_no_fill_afterwards():
    broker = PaperBroker()
    result = broker.submit(_buy(qty=10, price=10_000))
    assert result.ord_no is not None

    cancel_result = broker.cancel(result.ord_no, result.intent)
    assert cancel_result.return_code == 0
    assert broker.open_orders() == []

    fills = broker.on_quote(CODE, _ts(0), bid=9_990, ask=9_995, last=9_993)
    assert fills == []


def test_cancel_unknown_order_returns_error():
    broker = PaperBroker()
    intent = _buy()
    result = broker.cancel("P999999", intent)

    assert result.return_code == -1
    assert result.return_msg == "paper: 주문 없음"


def test_buy_taker_skipped_when_ask_not_positive():
    broker = PaperBroker()
    broker.submit(_buy(qty=10, price=10_000))

    # ask==0 은 "호가 없음"(book 센티널) — 공짜로 체결시키면 안 된다. last 도 없다.
    fills = broker.on_quote(CODE, _ts(0), bid=9_990, ask=0, last=None)

    assert fills == []
    assert len(broker.open_orders()) == 1


def test_sell_taker_skipped_when_bid_not_positive():
    broker = PaperBroker(holdings={CODE: Holding(code=CODE, qty=10, avg_price=9_000.0)})
    broker.submit(_sell(qty=10, price=10_000))

    fills = broker.on_quote(CODE, _ts(0), bid=0, ask=10_100, last=None)

    assert fills == []
    assert len(broker.open_orders()) == 1


def test_resting_through_rule_fills_even_when_bid_ask_are_nan():
    broker = PaperBroker(fill_basis="through")
    broker.submit(_buy(qty=10, price=10_000))

    fills = broker.on_quote(CODE, _ts(0), bid=float("nan"), ask=float("nan"), last=9_999)

    assert len(fills) == 1
    assert fills[0].price == 10_000


def test_latency_zero_still_fills_on_first_quote_seen_after_submit():
    broker = PaperBroker(latency_sec=0.0)
    broker.submit(_buy(qty=10, price=10_000))  # 이 종목 시세를 아직 한 번도 못 봤다.

    fills = broker.on_quote(CODE, _ts(0), bid=9_990, ask=9_995, last=9_993)

    assert len(fills) == 1


def test_latency_positive_excludes_the_anchoring_quote_itself():
    broker = PaperBroker(latency_sec=2.0)
    broker.submit(_buy(qty=10, price=10_000))  # 이 종목 시세를 아직 한 번도 못 봤다.

    # 첫 시세가 도착시각(ts(0)+2s) 을 고정한다 — 마켓어블이어도 이 시세 자신은 제외.
    fills0 = broker.on_quote(CODE, _ts(0), bid=9_990, ask=9_995, last=9_993)
    assert fills0 == []
    assert len(broker.open_orders()) == 1

    # 도착 전.
    fills1 = broker.on_quote(CODE, _ts(1), bid=9_990, ask=9_995, last=9_993)
    assert fills1 == []

    # 도착.
    fills2 = broker.on_quote(CODE, _ts(2), bid=9_990, ask=9_995, last=9_993)
    assert len(fills2) == 1


def test_poll_fills_drains_incrementally():
    broker = PaperBroker()
    broker.submit(_buy(qty=10, price=10_000))
    broker.on_quote(CODE, _ts(0), bid=9_990, ask=9_995, last=9_993)

    first_poll = broker.poll_fills()
    assert len(first_poll) == 1

    second_poll = broker.poll_fills()
    assert second_poll == []
