"""``EngineCore`` — 같은 전략 코드가 ``PaperBroker``·``KiwoomBroker`` 위에서 똑같이 불린다.

페이퍼 위에서는 이벤트를 ``on_quote`` 로 먹이고, 실매매 어댑터 위에서는 ``sync`` 만 한다.
Bar 는 호가가 없으니 bid/ask 를 NaN 으로 넘겨 종가 기준 대기 판정만 받게 한다(보수적).
"""

from __future__ import annotations

import math
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.engine import Bar, EngineCore, Quote, StrategyContext, Trade
from krx_quant_core.execution.guards import OrderGuard, OrderGuardConfig
from krx_quant_core.execution.kiwoom_broker import KiwoomBroker
from krx_quant_core.execution.oms import OrderManager
from krx_quant_core.execution.paper import PaperBroker

CODE = "005930"
T = [datetime(2026, 9, 16, 9, 0, i) for i in range(10)]


class _SpyPaper(PaperBroker):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.quotes: list[tuple] = []

    def on_quote(self, code, ts, bid, ask, last=None):
        self.quotes.append((code, ts, bid, ask, last))
        return super().on_quote(code, ts, bid, ask, last)


class _Recorder:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.started = 0
        self.ended = 0
        self.seen_now: list[datetime | None] = []

    def on_start(self, ctx: StrategyContext) -> None:
        self.started += 1

    def on_event(self, ev, ctx: StrategyContext) -> None:
        self.events.append(ev)
        self.seen_now.append(ctx.now)

    def on_end(self, ctx: StrategyContext) -> None:
        self.ended += 1


def _oms(broker) -> OrderManager:
    clock = lambda: T[0]  # noqa: E731
    guard = OrderGuard(OrderGuardConfig(max_qty=10, price_band_pct=0.0), clock=clock)
    return OrderManager(broker, guard=guard, book=PositionBook(), clock=clock)


def test_feed_updates_ctx_and_calls_strategy_after_quote():
    broker = _SpyPaper()
    strat = _Recorder()
    eng = EngineCore(strat, _oms(broker))
    eng.start()
    q = Quote(T[1], CODE, 70_000, 70_100)
    eng.feed(q)
    eng.end()

    assert strat.started == 1 and strat.ended == 1
    assert strat.events == [q]
    assert strat.seen_now == [T[1]]
    assert eng.ctx.last[CODE] is q
    assert broker.quotes == [(CODE, T[1], 70_000, 70_100, None)]


def test_strategy_without_start_end_hooks_is_fine():
    class Bare:
        def __init__(self) -> None:
            self.n = 0

        def on_event(self, ev, ctx) -> None:
            self.n += 1

    s = Bare()
    eng = EngineCore(s, _oms(PaperBroker()))
    eng.start()
    eng.feed(Quote(T[1], CODE, 1, 2))
    eng.end()
    assert s.n == 1


def test_trade_uses_last_quote_bid_ask_and_price_as_last():
    broker = _SpyPaper()
    eng = EngineCore(_Recorder(), _oms(broker))
    eng.feed(Quote(T[1], CODE, 99, 101))
    eng.feed(Bar(T[2], CODE, 100, 102, 98, 100, 1000))  # 직전 호가를 덮어쓰면 안 된다
    eng.feed(Trade(T[3], CODE, 100, 5))
    code, ts, bid, ask, last = broker.quotes[-1]
    assert (code, ts, bid, ask, last) == (CODE, T[3], 99, 101, 100)


def test_trade_without_prior_quote_passes_nan_bid_ask():
    broker = _SpyPaper()
    eng = EngineCore(_Recorder(), _oms(broker))
    eng.feed(Trade(T[1], CODE, 100, 5))
    _, _, bid, ask, last = broker.quotes[-1]
    assert math.isnan(bid) and math.isnan(ask) and last == 100


def test_bar_passes_nan_bid_ask_and_close_as_last():
    broker = _SpyPaper()
    eng = EngineCore(_Recorder(), _oms(broker))
    eng.feed(Bar(T[1], CODE, 100, 110, 90, 105, 1000))
    _, ts, bid, ask, last = broker.quotes[-1]
    assert ts == T[1] and math.isnan(bid) and math.isnan(ask) and last == 105


def test_bar_does_not_taker_fill_limit_at_close_under_through():
    """bid=ask=close 로 먹이면 종가와 같은 매수 지정가가 테이커로 체결된다 — 그러면 안 된다."""
    broker = PaperBroker(fill_basis="through")
    oms = _oms(broker)
    eng = EngineCore(_Recorder(), oms)
    eng.feed(Bar(T[1], CODE, 100, 100, 100, 100, 1))
    oms.buy(CODE, 1, 100, ref_price=100)
    assert eng.feed(Bar(T[2], CODE, 100, 101, 99, 100, 1)) == []  # 종가 == 지정가: through 미체결
    fills = eng.feed(Bar(T[3], CODE, 100, 100, 98, 99, 1))  # 종가가 뚫음
    assert [(f.side, f.qty, f.price, f.ts) for f in fills] == [("buy", 1, 100, T[3])]


def test_feed_returns_synced_fills_and_strategy_sees_book_updated():
    broker = PaperBroker()
    oms = _oms(broker)
    seen: list[int] = []

    class S:
        def on_event(self, ev, ctx) -> None:
            pos = ctx.oms.book.position(CODE)
            seen.append(pos.qty if pos else 0)

    eng = EngineCore(S(), oms)
    eng.feed(Quote(T[1], CODE, 70_000, 70_100))
    oms.buy(CODE, 2, 70_100, ref_price=70_100)
    fills = eng.feed(Quote(T[2], CODE, 70_000, 70_100))
    assert [(f.side, f.qty, f.price) for f in fills] == [("buy", 2, 70_100)]
    assert seen == [0, 2]


# ----- 실매매 어댑터: on_quote 없이 sync 만 -------------------------------------


def _fake_kiwoom(filled_responses: list[dict]) -> SimpleNamespace:
    calls: list[dict] = []
    responses = list(filled_responses)

    def filled_orders(**kw):
        calls.append(kw)
        return responses.pop(0) if responses else {}

    def empty(**kw):
        return {}

    order_ns = SimpleNamespace(buy_order=empty, sell_order=empty, cancel_order=empty)
    account_ns = SimpleNamespace(
        evaluation_balance_detail=empty, unfilled_orders=empty, filled_orders=filled_orders
    )
    api = SimpleNamespace(order=order_ns, account=account_ns)
    api.filled_calls = calls
    return api


def test_engine_on_kiwoom_broker_only_syncs_and_calls_same_strategy():
    row = {
        "ord_no": "1",
        "stk_cd": "A005930",
        "io_tp_nm": "매수",
        "cntr_qty": "3",
        "cntr_pric": "70000",
    }
    api = _fake_kiwoom([{"cntr": []}, {"cntr": [row]}])
    broker = KiwoomBroker(api, dry_run=False)
    strat = _Recorder()
    oms = _oms(broker)
    eng = EngineCore(strat, oms)

    q1 = Quote(T[1], CODE, 70_000, 70_100)
    t2 = Trade(T[2], CODE, 70_000, 3)
    assert eng.feed(q1) == []
    fills = eng.feed(t2)

    assert len(api.filled_calls) == 2  # 이벤트마다 sync 한 번
    assert [(f.side, f.qty, f.price) for f in fills] == [("buy", 3, 70_000)]
    assert strat.events == [q1, t2]
    assert oms.book.position(CODE).qty == 3
