"""``run_replay``·``merge_events`` — 리플레이는 결정론적이고, 비용은 장부가 뺀다."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from krx_quant_core.backtest.replay import ReplayResult, merge_events, run_replay
from krx_quant_core.costs.model import CostModelConfig, KoreanCostModel
from krx_quant_core.execution.engine import Bar, Quote, StrategyContext, Trade

CODE = "005930"
T = [datetime(2026, 9, 16, 9, 0, i) for i in range(10)]


class BuyThenSell:
    """첫 Quote 에 매도1호가로 매수, 보유가 생기면 매수1호가로 매도(둘 다 마켓어블 지정가)."""

    def __init__(self) -> None:
        self.bought = False
        self.sold = False
        self.ended = False

    def on_event(self, ev, ctx: StrategyContext) -> None:
        if not isinstance(ev, Quote):
            return
        if not self.bought:
            ctx.oms.buy(ev.code, 1, int(ev.ask), ref_price=ev.ask)
            self.bought = True
        elif not self.sold and ctx.oms.book.position(ev.code) is not None:
            ctx.oms.sell(ev.code, 1, int(ev.bid))
            self.sold = True

    def on_end(self, ctx: StrategyContext) -> None:
        self.ended = True


def _events() -> list:
    return [
        Quote(T[0], CODE, 70_000, 70_100),
        Quote(T[1], CODE, 70_000, 70_100),  # 매수 체결 @70100 → 매도 지정가 70000 주문
        Quote(T[2], CODE, 71_000, 71_100),  # 매도 테이커 체결 @매수1호가 71000
        Quote(T[3], CODE, 71_000, 71_100),
    ]


def test_toy_strategy_round_trip_pnl_net_of_costs():
    strat = BuyThenSell()
    res = run_replay(strat, _events())

    assert isinstance(res, ReplayResult)
    assert strat.ended
    assert len(res.fills) == 2
    assert list(res.fills["side"]) == ["buy", "sell"]
    assert list(res.fills["price"]) == [70_100, 71_000]
    assert len(res.trades) == 1

    cm = KoreanCostModel(
        CostModelConfig(kospi_slippage_pct=Decimal("0"), kosdaq_slippage_pct=Decimal("0"))
    )
    sell = cm.cost_of_trade(Decimal(71_000), Decimal(1), "SELL", "KOSPI", T[2].date())
    expected = (
        Decimal(71_000 - 70_100) - cm.commission(Decimal(70_100)) - sell.commission - sell.tax
    )
    assert res.trades["pnl"].iloc[0] == pytest.approx(float(expected))
    assert res.trades["pnl"].iloc[0] < 900
    assert res.book.realized_krw == pytest.approx(float(expected))
    assert [o["side"] for o in res.orders] == ["buy", "sell"]
    assert all(o["status"] == "submitted" for o in res.orders)
    assert res.orders[0]["ts"] == "2026-09-16 09:00:00"  # 이벤트 시각 시계


def test_same_input_twice_gives_equal_fills():
    a = run_replay(BuyThenSell(), _events())
    b = run_replay(BuyThenSell(), _events())
    pd.testing.assert_frame_equal(a.fills, b.fills)
    assert a.orders == b.orders


def test_open_position_stays_in_book_at_end():
    res = run_replay(BuyThenSell(), _events()[:2])
    assert len(res.fills) == 1
    assert res.trades.empty
    assert res.book.position(CODE).qty == 1


def test_default_guard_caps_qty_at_10():
    class Big:
        def on_event(self, ev, ctx) -> None:
            ctx.oms.buy(ev.code, 11, int(ev.ask), ref_price=ev.ask)

    res = run_replay(Big(), _events()[:2])
    assert res.fills.empty
    assert all(o["blocked"] for o in res.orders)


def test_merge_events_same_ts_orders_quote_trade_bar_stably():
    quotes = pd.DataFrame(
        {"ts": [T[1], T[0]], "code": ["A", "B"], "bid": [1.0, 2.0], "ask": [3.0, 4.0]}
    )
    trades = pd.DataFrame(
        {"ts": [T[1], T[1]], "code": ["C", "D"], "price": [10.0, 11.0], "qty": [1.0, 2.0]}
    )
    bars = pd.DataFrame(
        {
            "ts": [T[1]],
            "code": ["E"],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100.0],
        }
    )
    evs = merge_events(bars=bars, trades=trades, quotes=quotes)
    assert [type(e).__name__ + e.code for e in evs] == [
        "QuoteB",
        "QuoteA",
        "TradeC",
        "TradeD",
        "BarE",
    ]
    assert evs[0] == Quote(T[0], "B", 2.0, 4.0)
    assert isinstance(evs[0].ts, datetime)
    assert evs[-1] == Bar(T[1], "E", 1.0, 2.0, 0.5, 1.5, 100.0)
    assert evs[2] == Trade(T[1], "C", 10.0, 1.0)


def test_merge_events_none_gives_empty():
    assert merge_events() == []


def test_bar_replay_resting_limit_needs_close_through():
    class LimitBuy:
        def __init__(self) -> None:
            self.done = False

        def on_event(self, ev, ctx) -> None:
            if not self.done:
                ctx.oms.buy(ev.code, 1, 100, ref_price=100)
                self.done = True

    bars = [
        Bar(T[0], CODE, 100, 100, 100, 100, 1),
        Bar(T[1], CODE, 101, 101, 99, 100, 1),  # 저가는 뚫었지만 종가 == 지정가
        Bar(T[2], CODE, 100, 100, 98, 99, 1),
    ]
    res = run_replay(LimitBuy(), bars)
    assert list(res.fills["ts"]) == [T[2]]
    touch = run_replay(LimitBuy(), bars, fill_basis="touch")
    assert list(touch.fills["ts"]) == [T[1]]
