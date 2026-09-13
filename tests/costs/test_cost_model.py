"""거래비용 모델 — daytrade-it tests/unit/domain/test_cost_model.py 이식 + 일자별 세율."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

import pytest

from krx_quant_core.costs.model import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_KOSDAQ_SLIPPAGE_PCT,
    DEFAULT_KOSDAQ_TAX_RATE,
    DEFAULT_KOSPI_SLIPPAGE_PCT,
    DEFAULT_KOSPI_TAX_RATE,
    CostModelConfig,
    KoreanCostModel,
    round_trip_cost,
)
from krx_quant_core.market.codes import Market
from krx_quant_core.market.ticks import round_to_tick


class Exchange(StrEnum):
    """daytrade-it domain/entities/stock.py 의 Exchange 와 같은 모양."""

    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"
    KONEX = "KONEX"


def test_defaults_unchanged_from_daytrade():
    assert DEFAULT_KOSPI_TAX_RATE == Decimal("0.0020")
    assert DEFAULT_KOSDAQ_TAX_RATE == Decimal("0.0020")
    assert DEFAULT_COMMISSION_RATE == Decimal("0.00015")
    assert DEFAULT_KOSPI_SLIPPAGE_PCT == Decimal("0.0005")
    assert DEFAULT_KOSDAQ_SLIPPAGE_PCT == Decimal("0.0015")


class TestTransactionTax:
    def test_buy_has_no_tax(self) -> None:
        model = KoreanCostModel()
        assert model.transaction_tax(Decimal("1000000"), side="BUY", market=Market.KOSPI) == 0

    def test_sell_kospi_tax_is_020_pct(self) -> None:
        model = KoreanCostModel()
        notional = Decimal("1000000")
        tax = model.transaction_tax(notional, side="SELL", market=Market.KOSPI)
        assert tax == notional * DEFAULT_KOSPI_TAX_RATE
        assert tax == Decimal("2000.00")

    def test_sell_kosdaq_tax_is_020_pct(self) -> None:
        model = KoreanCostModel()
        notional = Decimal("1000000")
        tax = model.transaction_tax(notional, side="SELL", market=Market.KOSDAQ)
        assert tax == notional * DEFAULT_KOSDAQ_TAX_RATE
        assert tax == Decimal("2000.00")

    def test_side_is_case_insensitive(self) -> None:
        model = KoreanCostModel()
        notional = Decimal("1000000")
        assert model.transaction_tax(notional, side="sell") == model.transaction_tax(
            notional, side="SELL"
        )

    def test_explicit_default_ignores_trade_date(self) -> None:
        """기본 설정은 daytrade 동작 그대로 — 2025년 거래라도 0.20%."""
        model = KoreanCostModel()
        tax = model.transaction_tax(
            Decimal("1000000"), "SELL", Market.KOSPI, trade_date=date(2025, 5, 1)
        )
        assert tax == Decimal("2000.00")


class TestDatedTax:
    def _model(self) -> KoreanCostModel:
        return KoreanCostModel(CostModelConfig(kospi_tax_rate=None, kosdaq_tax_rate=None))

    @pytest.mark.parametrize(
        "on,market,expected",
        [
            (date(2025, 5, 1), Market.KOSPI, "1500"),
            (date(2025, 5, 1), Market.KOSDAQ, "1500"),
            (date(2024, 5, 1), Market.KOSDAQ, "1800"),
            (date(2021, 5, 1), Market.KOSPI, "2300"),
            (date(2026, 2, 1), Market.KOSPI, "2000"),
            (date(2024, 5, 1), Market.KONEX, "1800"),
        ],
    )
    def test_uses_schedule_when_config_none(self, on, market, expected) -> None:
        tax = self._model().transaction_tax(Decimal("1000000"), "SELL", market, trade_date=on)
        assert tax == Decimal(expected)

    def test_no_date_uses_today_kst(self, monkeypatch) -> None:
        from datetime import datetime

        import krx_quant_core.costs.model as m

        monkeypatch.setattr(m, "now_kst", lambda: datetime(2025, 7, 1, 10, 0))
        tax = self._model().transaction_tax(Decimal("1000000"), "SELL", Market.KOSPI)
        assert tax == Decimal("1500")

    def test_partial_none_only_affects_that_market(self) -> None:
        model = KoreanCostModel(CostModelConfig(kosdaq_tax_rate=None))
        on = date(2025, 5, 1)
        assert model.transaction_tax(Decimal("1000000"), "SELL", "KOSDAQ", on) == Decimal("1500")
        assert model.transaction_tax(Decimal("1000000"), "SELL", "KOSPI", on) == Decimal("2000")

    def test_cost_of_trade_passes_trade_date(self) -> None:
        cost = self._model().cost_of_trade(
            Decimal("10000"), Decimal("100"), "SELL", Market.KOSDAQ, trade_date=date(2025, 5, 1)
        )
        assert cost.tax == cost.fill_price * Decimal("100") * Decimal("0.0015")


class TestCommission:
    def test_commission_applies_to_notional(self) -> None:
        model = KoreanCostModel()
        assert model.commission(Decimal("1000000")) == Decimal("1000000") * DEFAULT_COMMISSION_RATE

    def test_min_commission_floor(self) -> None:
        config = CostModelConfig(commission_rate=Decimal("0.00015"), min_commission=Decimal("100"))
        assert KoreanCostModel(config).commission(Decimal("10")) == Decimal("100")

    def test_commission_is_configurable_not_hardcoded(self) -> None:
        model = KoreanCostModel(CostModelConfig(commission_rate=Decimal("0.001")))
        assert model.commission(Decimal("1000000")) == Decimal("1000")


class TestSlippage:
    def test_buy_slips_up(self) -> None:
        model = KoreanCostModel()
        price = Decimal("70000")
        filled = model.apply_slippage(price, side="BUY", market=Market.KOSPI)
        assert filled == round_to_tick(price * (1 + DEFAULT_KOSPI_SLIPPAGE_PCT))
        assert filled >= price

    def test_sell_slips_down(self) -> None:
        model = KoreanCostModel()
        price = Decimal("70000")
        filled = model.apply_slippage(price, side="SELL", market=Market.KOSPI)
        assert filled == round_to_tick(price * (1 - DEFAULT_KOSPI_SLIPPAGE_PCT))
        assert filled < price

    def test_kosdaq_slips_more_than_kospi(self) -> None:
        model = KoreanCostModel()
        price = Decimal("10000")
        kospi_fill = model.apply_slippage(price, side="BUY", market=Market.KOSPI)
        kosdaq_fill = model.apply_slippage(price, side="BUY", market=Market.KOSDAQ)
        assert (kosdaq_fill - price) > (kospi_fill - price)

    def test_konex_falls_back_to_kospi_rate(self) -> None:
        model = KoreanCostModel()
        price = Decimal("10000")
        assert model.apply_slippage(price, "BUY", Market.KONEX) == model.apply_slippage(
            price, "BUY", Market.KOSPI
        )

    def test_unknown_market_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            KoreanCostModel().apply_slippage(Decimal("10000"), "BUY", "NASDAQ")


class TestCostOfTrade:
    def test_buy_breakdown_has_no_tax(self) -> None:
        model = KoreanCostModel()
        cost = model.cost_of_trade(Decimal("70000"), Decimal("10"), "BUY", Market.KOSPI)
        expected_fill = round_to_tick(Decimal("70000") * (1 + DEFAULT_KOSPI_SLIPPAGE_PCT))
        assert cost.fill_price == expected_fill
        assert cost.tax == 0
        assert cost.commission == expected_fill * Decimal("10") * DEFAULT_COMMISSION_RATE
        assert cost.total_cost == cost.commission

    def test_sell_breakdown_known_inputs(self) -> None:
        model = KoreanCostModel()
        price, quantity = Decimal("50000"), Decimal("100")
        cost = model.cost_of_trade(price=price, quantity=quantity, side="SELL", market=Market.KOSPI)
        expected_fill = round_to_tick(price * (1 - DEFAULT_KOSPI_SLIPPAGE_PCT))
        notional = expected_fill * quantity
        assert cost.fill_price == expected_fill
        assert cost.commission == notional * DEFAULT_COMMISSION_RATE
        assert cost.tax == notional * DEFAULT_KOSPI_TAX_RATE
        assert cost.total_cost == cost.commission + cost.tax

    def test_round_trip_costs_reduce_net_pnl(self) -> None:
        model = KoreanCostModel()
        price, quantity = Decimal("50000"), Decimal("10")
        buy = model.cost_of_trade(price, quantity, "BUY", Market.KOSPI)
        sell = model.cost_of_trade(price, quantity, "SELL", Market.KOSPI)
        buy_outlay = buy.fill_price * quantity + buy.commission + buy.tax
        sell_proceeds = sell.fill_price * quantity - sell.commission - sell.tax
        assert sell_proceeds < buy_outlay


class TestConfigOverride:
    def test_custom_config_used_end_to_end(self) -> None:
        config = CostModelConfig(
            kospi_tax_rate=Decimal("0.001"),
            kosdaq_tax_rate=Decimal("0.001"),
            commission_rate=Decimal("0.0005"),
            kospi_slippage_pct=Decimal("0.002"),
            kosdaq_slippage_pct=Decimal("0.004"),
        )
        cost = KoreanCostModel(config).cost_of_trade(
            Decimal("10000"), Decimal("1"), "SELL", Market.KOSPI
        )
        expected_fill = Decimal("10000") * (1 - Decimal("0.002"))
        assert cost.fill_price == expected_fill
        assert cost.tax == expected_fill * Decimal("0.001")
        assert cost.commission == expected_fill * Decimal("0.0005")


@pytest.mark.parametrize("market", [Market.KOSPI, Market.KOSDAQ])
def test_zero_quantity_does_not_error(market: Market) -> None:
    cost = KoreanCostModel().cost_of_trade(Decimal("10000"), Decimal("0"), "SELL", market)
    assert cost.commission == 0
    assert cost.tax == 0


@pytest.mark.parametrize("ex", list(Exchange))
def test_daytrade_exchange_members_pass_through(ex: Exchange) -> None:
    model = KoreanCostModel()
    via_exchange = model.cost_of_trade(Decimal("12345"), Decimal("7"), "SELL", ex)
    via_market = model.cost_of_trade(Decimal("12345"), Decimal("7"), "SELL", Market(ex.value))
    assert via_exchange == via_market


# ---- round_trip_cost --------------------------------------------------------


def test_round_trip_cost_components():
    assert round_trip_cost(date(2026, 3, 1), "KOSPI") == pytest.approx(0.0023)
    assert round_trip_cost(date(2025, 3, 1), "KOSDAQ") == pytest.approx(0.0018)
    assert round_trip_cost(date(2024, 3, 1), Market.KOSDAQ) == pytest.approx(0.0021)
    assert round_trip_cost(
        date(2026, 3, 1), "KOSDAQ", commission_rate=0.0, slippage_one_way=0.0015
    ) == pytest.approx(0.0050)


def test_scalp_0023_equals_2026_round_trip():
    """scalp-it 의 0.0023 은 2026(=2023) 세율 0.20% + 수수료 0.015%×2 와 같다."""
    assert round_trip_cost(date(2026, 1, 2), "KOSDAQ") == pytest.approx(0.0023)
    assert round_trip_cost(date(2023, 6, 1), "KOSPI") == pytest.approx(0.0023)
