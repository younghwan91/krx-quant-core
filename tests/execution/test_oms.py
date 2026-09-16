"""``OrderManager``·``InstanceLock``·``ReconcileReport`` — 매수는 킬·가드, 청산은 보유만 본다.

브로커는 :class:`PaperBroker` 를 쓴다(실주문 없음). 시계는 가드와 매니저가 같은
가짜 시계를 공유한다 — 둘의 날짜가 어긋나면 일일 카운터 리셋 시점이 달라진다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.events import Fill, Holding, OrderStatus
from krx_quant_core.execution.guards import OrderGuard, OrderGuardConfig
from krx_quant_core.execution.oms import (
    AlreadyRunning,
    InstanceLock,
    OrderManager,
    ReconcileReport,
)
from krx_quant_core.execution.orders import OrderIntent, OrderResult
from krx_quant_core.execution.paper import PaperBroker
from krx_quant_core.risk.killswitch import KillSwitch, KillSwitchConfig

CODE = "005930"
T0 = datetime(2026, 9, 16, 9, 0, 0)


class _Clock:
    def __init__(self, t: datetime = T0) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def _guard(clock: _Clock, **kw) -> OrderGuard:
    cfg = OrderGuardConfig(**{"max_qty": 10, "price_band_pct": 0.03, **kw})
    return OrderGuard(cfg, clock=clock)


def _oms(
    *,
    broker=None,
    kill: KillSwitch | None = None,
    order_log=None,
    guard_kw: dict | None = None,
    book: PositionBook | None = None,
) -> tuple[OrderManager, PaperBroker, OrderGuard, PositionBook, _Clock]:
    clock = _Clock()
    broker = broker if broker is not None else PaperBroker()
    guard = _guard(clock, **(guard_kw or {}))
    book = book if book is not None else PositionBook()
    om = OrderManager(broker, guard=guard, book=book, kill=kill, clock=clock, order_log=order_log)
    return om, broker, guard, book, clock


# ------------------------------------------------------------------ buy


def test_buy_passes_and_counts():
    om, broker, guard, _, _ = _oms()
    mr = om.buy(CODE, 5, 10_000, ref_price=10_000)
    assert mr.status is OrderStatus.SUBMITTED
    assert mr.result.ok and mr.result.ord_no == "P000001"
    assert guard.count_total == 1 and guard.count_by_code == {CODE: 1}
    assert len(broker.open_orders()) == 1


def test_kill_blocks_buy_but_not_sell():
    kill = KillSwitch()
    kill.force_stop()
    broker = PaperBroker(holdings={CODE: Holding(CODE, 10, 10_000.0)})
    om, broker, guard, _, _ = _oms(broker=broker, kill=kill)

    mr = om.buy(CODE, 1, 10_000, ref_price=10_000)
    assert mr.status is OrderStatus.BLOCKED
    assert mr.result.blocked and mr.result.blocked_reason == "kill:manual_stop"
    assert not mr.result.submitted
    assert guard.count_total == 0
    assert broker.open_orders() == []

    sell = om.sell(CODE, 10, 9_000)
    assert sell.status is OrderStatus.SUBMITTED
    assert sell.result.ok


def test_guard_reason_string_is_identical():
    om, broker, guard, _, _ = _oms()
    # 가격 밴드 밖 지정가
    intent = OrderIntent(side="buy", code=CODE, qty=1, price=11_000)
    expected = guard.reason(intent, 10_000)
    assert expected  # 사유가 있어야 이 테스트가 의미 있다
    mr = om.buy(CODE, 1, 11_000, ref_price=10_000)
    assert mr.status is OrderStatus.BLOCKED
    assert mr.result.blocked_reason == expected
    assert guard.count_total == 0
    assert broker.open_orders() == []


def test_guard_lists_and_counts_do_not_block_exit():
    """블록리스트·화이트리스트·횟수 상한은 청산을 막지 않는다."""
    broker = PaperBroker(holdings={CODE: Holding(CODE, 3, 10_000.0)})
    om, _, guard, _, _ = _oms(
        broker=broker,
        guard_kw={
            "blocklist": frozenset({CODE}),
            "whitelist": frozenset({"000660"}),
            "max_orders_total": 1,
        },
    )
    guard.record_order("000660")  # 총량 상한 소진
    mr = om.sell(CODE, 3, 10_000)
    assert mr.status is OrderStatus.SUBMITTED
    assert guard.count_total == 1  # 청산은 카운트하지 않는다


class _DryBroker(PaperBroker):
    def submit(self, intent: OrderIntent) -> OrderResult:
        return OrderResult(intent=intent, dry_run=True, submitted=False)


class _RejectBroker(PaperBroker):
    def submit(self, intent: OrderIntent) -> OrderResult:
        return OrderResult(
            intent=intent, dry_run=False, submitted=True, return_code=1, return_msg="거부"
        )


def test_dry_run_pass_counts():
    om, _, guard, _, _ = _oms(broker=_DryBroker())
    mr = om.buy(CODE, 1, 10_000, ref_price=10_000)
    assert mr.result.dry_run and mr.result.ok
    assert mr.status is OrderStatus.SUBMITTED
    assert guard.count_total == 1


def test_rejected_broker_response_does_not_count():
    om, _, guard, _, _ = _oms(broker=_RejectBroker())
    mr = om.buy(CODE, 1, 10_000, ref_price=10_000)
    assert mr.status is OrderStatus.REJECTED
    assert guard.count_total == 0


def test_broker_blocked_is_blocked_status():
    # PaperBroker 는 보유 부족 매도를 blocked 로 돌려준다. 장부는 보유가 있다고 믿는 상황.
    book = PositionBook()
    book.apply(Fill("X1", CODE, "buy", 5, 10_000, T0))
    om, _, _, _, _ = _oms(book=book)  # 브로커 보유는 비어 있음
    mr = om.sell(CODE, 5, 10_000)
    assert mr.status is OrderStatus.BLOCKED
    assert mr.result.blocked_reason == "paper: 보유 부족"


# ------------------------------------------------------------------ sell


def test_sell_over_holdings_blocked_uses_book_first():
    book = PositionBook()
    book.apply(Fill("X1", CODE, "buy", 3, 10_000, T0))
    broker = PaperBroker(holdings={CODE: Holding(CODE, 100, 10_000.0)})
    om, broker, _, _, _ = _oms(broker=broker, book=book)
    mr = om.sell(CODE, 4, 10_000)
    assert mr.status is OrderStatus.BLOCKED
    assert mr.result.blocked_reason == "청산 수량이 보유 초과: 4 > 3"
    assert broker.open_orders() == []


def test_sell_falls_back_to_broker_holdings_when_book_empty():
    broker = PaperBroker(holdings={CODE: Holding(CODE, 2, 10_000.0)})
    om, _, _, _, _ = _oms(broker=broker)
    assert om.sell(CODE, 3, 10_000).result.blocked_reason == "청산 수량이 보유 초과: 3 > 2"
    assert om.sell(CODE, 2, 10_000).status is OrderStatus.SUBMITTED


def test_sell_nothing_held_blocked():
    om, _, _, _, _ = _oms()
    mr = om.sell(CODE, 1, 10_000)
    assert mr.result.blocked_reason == "청산 수량이 보유 초과: 1 > 0"


def test_sell_non_positive_price_blocked():
    broker = PaperBroker(holdings={CODE: Holding(CODE, 2, 10_000.0)})
    om, _, _, _, _ = _oms(broker=broker)
    mr = om.sell(CODE, 1, 0)
    assert mr.status is OrderStatus.BLOCKED
    assert mr.result.blocked_reason == "지정가가 0 이하: 0"


# ------------------------------------------------------------------ cancel


def test_cancel_ok_and_missing():
    om, broker, _, _, _ = _oms()
    mr = om.buy(CODE, 1, 9_000, ref_price=9_000)
    intent = mr.result.intent
    c = om.cancel(mr.result.ord_no, intent)
    assert c.status is OrderStatus.CANCELED
    assert broker.open_orders() == []
    c2 = om.cancel("P999999", intent)
    assert c2.status is OrderStatus.REJECTED


# ------------------------------------------------------------------ sync


def test_sync_updates_book_and_triggers_consecutive_loss_kill():
    kill = KillSwitch(KillSwitchConfig(max_consecutive_losses=2))
    om, broker, _, book, _ = _oms(kill=kill)

    for i in range(2):
        t = T0.replace(minute=i * 2)
        assert om.buy(CODE, 1, 10_000, ref_price=10_000).status is OrderStatus.SUBMITTED
        broker.on_quote(CODE, t, bid=9_990, ask=10_000)
        fills = om.sync()
        assert [f.side for f in fills] == ["buy"]
        assert book.position(CODE).qty == 1
        assert om.sell(CODE, 1, 9_700).status is OrderStatus.SUBMITTED
        broker.on_quote(CODE, t.replace(second=30), bid=9_700, ask=9_710)
        fills = om.sync()
        assert [f.side for f in fills] == ["sell"]
        assert book.position(CODE) is None

    assert kill.consecutive_losses == 2
    assert kill.killed and kill.kill_reason == "consecutive_losses"
    blocked = om.buy(CODE, 1, 10_000, ref_price=10_000)
    assert blocked.result.blocked_reason == "kill:consecutive_losses"


def test_sync_without_kill():
    om, broker, _, book, _ = _oms()
    om.buy(CODE, 2, 10_000, ref_price=10_000)
    broker.on_quote(CODE, T0, bid=9_990, ask=10_000)
    assert len(om.sync()) == 1
    assert book.position(CODE).qty == 2
    assert om.sync() == []


# ------------------------------------------------------------------ reconcile


def test_reconcile_reports_three_differences_without_fixing():
    book = PositionBook()
    book.apply(Fill("a", "000001", "buy", 5, 1_000, T0))  # 장부에만
    book.apply(Fill("b", "000002", "buy", 3, 1_000, T0))  # 수량 불일치
    book.apply(Fill("c", "000003", "buy", 7, 1_000, T0))  # 일치
    broker = PaperBroker(
        holdings={
            "000002": Holding("000002", 4, 1_000.0),
            "000003": Holding("000003", 7, 1_000.0),
            "000004": Holding("000004", 9, 1_000.0),  # 브로커에만
        }
    )
    om, broker, _, book, _ = _oms(broker=broker, book=book)
    rep = om.reconcile()
    assert rep == ReconcileReport(
        only_book={"000001": 5},
        only_broker={"000004": 9},
        qty_mismatch={"000002": (3, 4)},
    )
    assert not rep.ok
    # 보고만 한다 — 장부·브로커 그대로
    assert book.position("000001").qty == 5
    assert "000004" not in book.positions()
    assert broker.holdings()["000002"].qty == 4


def test_reconcile_ok_when_equal():
    rep = _oms()[0].reconcile()
    assert rep.ok and rep == ReconcileReport({}, {}, {})


# ------------------------------------------------------------------ order_log


def test_order_log_jsonl(tmp_path):
    log = tmp_path / "sub" / "orders.jsonl"
    kill = KillSwitch()
    om, _, _, _, clock = _oms(order_log=log)
    om.buy(CODE, 1, 10_000, ref_price=10_000)
    om.sell(CODE, 1, 10_000)
    om2 = OrderManager(PaperBroker(), guard=_guard(clock), book=PositionBook(), kill=kill,
                       clock=clock, order_log=log)
    kill.force_stop()
    om2.buy(CODE, 1, 10_000, ref_price=10_000)

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in rows] == ["submitted", "blocked", "blocked"]
    assert rows[0]["ts"] == "2026-09-16 09:00:00"
    assert rows[0]["ord_no"] == "P000001" and rows[0]["side"] == "buy"
    assert rows[1]["blocked_reason"] == "청산 수량이 보유 초과: 1 > 0"
    assert rows[2]["blocked_reason"] == "kill:manual_stop"


# ------------------------------------------------------------------ InstanceLock


def test_instance_lock_second_acquire_raises(tmp_path):
    path = tmp_path / "run" / "daemon.lock"
    first = InstanceLock(path)
    first.acquire()
    try:
        assert path.read_text().strip() == str(os.getpid())
        with pytest.raises(AlreadyRunning, match="이미 실행 중"):
            InstanceLock(path).acquire()
    finally:
        first.release()
    # 풀리면 다시 잡힌다
    with InstanceLock(path) as again:
        assert again is not None
        with pytest.raises(AlreadyRunning):
            with InstanceLock(path):
                pass
    first.release()  # 두 번 풀어도 안전
