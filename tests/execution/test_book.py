"""``PositionBook`` — 평단·실현손익(비용 차감)·journal 재생.

체결 값 객체(:class:`Fill`)를 넣으면 평단이 갱신되고, 매도는 :class:`KoreanCostModel`
로 계산한 수수료·세금을 뺀 순손익을 돌려준다. journal 을 재생하면 같은 상태로
복원돼야 한다(재기록 없이).
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

import pytest

from krx_quant_core.costs.model import CostModelConfig, KoreanCostModel
from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.events import Fill

CODE = "251270"  # 넷마블 — KOSDAQ 으로 취급(테스트용 시장 매핑)


def _kosdaq(_code: str) -> str:
    return "KOSDAQ"


def _zero_slippage_cost_model() -> KoreanCostModel:
    return KoreanCostModel(
        CostModelConfig(kospi_slippage_pct=Decimal("0"), kosdaq_slippage_pct=Decimal("0"))
    )


def test_buy_twice_averages_price():
    book = PositionBook(market_of=_kosdaq)
    book.apply(Fill("1", CODE, "buy", 100, 10_000, datetime(2026, 9, 16, 9, 0, 0)))
    realized = book.apply(Fill("2", CODE, "buy", 100, 10_200, datetime(2026, 9, 16, 9, 5, 0)))

    assert realized == 0.0
    pos = book.position(CODE)
    assert pos is not None
    assert pos.qty == 200
    assert pos.avg_price == pytest.approx(10_100.0)


def test_sell_partial_realized_pnl_matches_cost_model():
    """2026-09-16 KOSDAQ 세율 — 순손익 = (price-avg)*qty - 매수측·매도측 commission - 매도세."""
    book = PositionBook(market_of=_kosdaq)
    book.apply(Fill("1", CODE, "buy", 100, 10_000, datetime(2026, 9, 16, 9, 0, 0)))
    book.apply(Fill("2", CODE, "buy", 100, 10_200, datetime(2026, 9, 16, 9, 5, 0)))

    avg = Decimal("10100")  # (10000*100 + 10200*100) / 200
    sell_qty = 50
    sell_price = 10_500
    trade_date = datetime(2026, 9, 16, 10, 0, 0).date()

    cost_model = _zero_slippage_cost_model()
    buy_cost = cost_model.cost_of_trade(avg, Decimal(sell_qty), "BUY", "KOSDAQ", trade_date)
    sell_cost = cost_model.cost_of_trade(
        Decimal(sell_price), Decimal(sell_qty), "SELL", "KOSDAQ", trade_date
    )
    expected = (
        (Decimal(sell_price) - avg) * sell_qty
        - buy_cost.commission
        - sell_cost.commission
        - sell_cost.tax
    )

    realized = book.apply(
        Fill("3", CODE, "sell", sell_qty, sell_price, datetime(2026, 9, 16, 10, 0, 0))
    )

    assert realized == pytest.approx(float(expected))
    assert book.realized_krw == pytest.approx(float(expected))
    assert book.n_round_trips == 1
    pos = book.position(CODE)
    assert pos is not None
    assert pos.qty == 150
    assert pos.avg_price == pytest.approx(10_100.0)  # 매도는 평단을 바꾸지 않는다


def test_sell_non_tick_aligned_average_price_commission_not_snapped():
    """평단이 호가단위가 아니어도 매수측 수수료가 snap 된 가격으로 계산되면 안 된다.

    round 1 리뷰 finding #1: ``cost_of_trade`` 경로는 슬리피지 0 이어도
    ``round_to_tick`` 으로 가격을 스냅한다 — 매수측은 실제로 낸 적 없는 가상
    청산이므로 ``commission()`` 을 평단 그대로의 notional 에 직접 적용해야 한다.
    """
    book = PositionBook(market_of=_kosdaq)
    book.apply(Fill("1", CODE, "buy", 1, 10_000, datetime(2026, 9, 16, 9, 0, 0)))
    book.apply(Fill("2", CODE, "buy", 2, 10_050, datetime(2026, 9, 16, 9, 5, 0)))

    avg = (Decimal(10_000) * 1 + Decimal(10_050) * 2) / Decimal(3)  # 10033.33...
    assert avg != avg.to_integral_value()  # 평단이 정수(따라서 호가단위)가 아님을 확인

    sell_qty = 3
    sell_price = 10_100
    trade_date = datetime(2026, 9, 16, 10, 0, 0).date()

    cost_model = _zero_slippage_cost_model()
    buy_commission = cost_model.commission(avg * sell_qty)
    sell_cost = cost_model.cost_of_trade(
        Decimal(sell_price), Decimal(sell_qty), "SELL", "KOSDAQ", trade_date
    )
    expected = (
        (Decimal(sell_price) - avg) * sell_qty
        - buy_commission
        - sell_cost.commission
        - sell_cost.tax
    )

    realized = book.apply(
        Fill("3", CODE, "sell", sell_qty, sell_price, datetime(2026, 9, 16, 10, 0, 0))
    )

    assert realized == pytest.approx(float(expected))

    # 회귀 가드: 평단을 호가단위로 스냅한 값으로 계산한 수수료와는 달라야 한다
    # (스냅되면 대금이 조용히 바뀌어 수수료도 달라진다 — 그게 원래 버그였다).
    from krx_quant_core.market.ticks import round_to_tick

    snapped_avg = round_to_tick(avg)
    assert snapped_avg != avg
    wrong_commission = cost_model.commission(snapped_avg * sell_qty)
    assert buy_commission != wrong_commission


def test_sell_to_zero_then_buy_starts_fresh_position():
    book = PositionBook(market_of=_kosdaq)
    t0 = datetime(2026, 9, 16, 9, 0, 0)
    book.apply(Fill("1", CODE, "buy", 100, 10_000, t0))
    book.apply(Fill("2", CODE, "sell", 100, 10_500, datetime(2026, 9, 16, 9, 30, 0)))

    assert book.position(CODE) is None

    t1 = datetime(2026, 9, 16, 10, 0, 0)
    book.apply(Fill("3", CODE, "buy", 50, 9_800, t1))

    pos = book.position(CODE)
    assert pos is not None
    assert pos.qty == 50
    assert pos.avg_price == pytest.approx(9_800.0)

    book.apply(Fill("4", CODE, "sell", 50, 9_900, datetime(2026, 9, 16, 10, 30, 0)))
    frame = book.to_frame()
    last_row = frame.iloc[-1]
    assert last_row["entry_ts"] == t1  # 새 포지션의 entry_ts 는 이전 청산과 무관하게 새로 시작


def test_two_codes_stay_isolated():
    book = PositionBook(market_of=_kosdaq)
    book.apply(Fill("1", "AAA001", "buy", 100, 10_000, datetime(2026, 9, 16, 9, 0, 0)))
    book.apply(Fill("2", "BBB002", "buy", 50, 20_000, datetime(2026, 9, 16, 9, 0, 0)))

    assert book.position("AAA001").qty == 100
    assert book.position("BBB002").qty == 50

    book.apply(Fill("3", "AAA001", "sell", 100, 10_500, datetime(2026, 9, 16, 9, 30, 0)))

    assert book.position("AAA001") is None
    pos_bbb = book.position("BBB002")
    assert pos_bbb is not None
    assert pos_bbb.qty == 50
    assert set(book.positions()) == {"BBB002"}


def test_sell_more_than_held_raises_value_error():
    book = PositionBook(market_of=_kosdaq)
    book.apply(Fill("1", CODE, "buy", 100, 10_000, datetime(2026, 9, 16, 9, 0, 0)))

    with pytest.raises(ValueError):
        book.apply(Fill("2", CODE, "sell", 101, 10_000, datetime(2026, 9, 16, 9, 5, 0)))


def test_sell_without_any_holding_raises_value_error():
    book = PositionBook(market_of=_kosdaq)

    with pytest.raises(ValueError):
        book.apply(Fill("1", CODE, "sell", 1, 10_000, datetime(2026, 9, 16, 9, 5, 0)))


def test_journal_replay_restores_same_state(tmp_path):
    journal = tmp_path / "fills.jsonl"

    book = PositionBook(journal=journal, market_of=_kosdaq)
    book.apply(Fill("1", CODE, "buy", 100, 10_000, datetime(2026, 9, 16, 9, 0, 0)))
    book.apply(Fill("2", CODE, "buy", 100, 10_200, datetime(2026, 9, 16, 9, 5, 0)))
    book.apply(Fill("3", CODE, "sell", 50, 10_500, datetime(2026, 9, 16, 10, 0, 0)))

    # journal 은 append-only JSONL. 한 줄에 fill 하나.
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert set(row) == {"ord_no", "code", "side", "qty", "price", "ts"}

    restored = PositionBook.load(journal, market_of=_kosdaq)

    assert restored.positions() == book.positions()
    assert restored.realized_krw == pytest.approx(book.realized_krw)
    assert restored.n_round_trips == book.n_round_trips

    # 재생은 재기록하지 않는다 — 줄 수가 그대로여야 한다.
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 3


def test_journal_creates_missing_parent_directory(tmp_path):
    journal = tmp_path / "nested" / "does" / "not" / "exist" / "fills.jsonl"

    book = PositionBook(journal=journal, market_of=_kosdaq)
    book.apply(Fill("1", CODE, "buy", 10, 10_000, datetime(2026, 9, 16, 9, 0, 0)))

    assert journal.exists()
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_to_frame_row_count_equals_sell_fill_count():
    book = PositionBook(market_of=_kosdaq)
    book.apply(Fill("1", CODE, "buy", 100, 10_000, datetime(2026, 9, 16, 9, 0, 0)))
    book.apply(Fill("2", CODE, "sell", 30, 10_100, datetime(2026, 9, 16, 9, 5, 0)))
    book.apply(Fill("3", CODE, "sell", 30, 10_200, datetime(2026, 9, 16, 9, 10, 0)))

    frame = book.to_frame()

    assert len(frame) == 2
    assert list(frame.columns) == [
        "code",
        "entry_ts",
        "exit_ts",
        "qty",
        "entry_price",
        "exit_price",
        "pnl",
    ]
    assert frame["qty"].tolist() == [30, 30]
