"""일자별 증권거래세 스케줄."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from krx_quant_core.costs.tax import TAX_SCHEDULE, TaxRate, sell_tax_rate, tax_rate
from krx_quant_core.market.codes import Market


@pytest.mark.parametrize(
    "on,kospi_sec,kospi_total,kosdaq_total",
    [
        (date(2018, 12, 31), "0.0015", "0.0030", "0.0030"),
        (date(2019, 6, 2), "0.0015", "0.0030", "0.0030"),
        (date(2019, 6, 3), "0.0010", "0.0025", "0.0025"),
        (date(2020, 12, 31), "0.0010", "0.0025", "0.0025"),
        (date(2021, 1, 1), "0.0008", "0.0023", "0.0023"),
        (date(2022, 6, 1), "0.0008", "0.0023", "0.0023"),
        (date(2023, 1, 1), "0.0005", "0.0020", "0.0020"),
        (date(2024, 1, 2), "0.0003", "0.0018", "0.0018"),
        (date(2025, 1, 1), "0", "0.0015", "0.0015"),
        (date(2025, 12, 31), "0", "0.0015", "0.0015"),
        (date(2026, 1, 1), "0.0005", "0.0020", "0.0020"),
        (date(2026, 9, 14), "0.0005", "0.0020", "0.0020"),
    ],
)
def test_schedule_steps(on, kospi_sec, kospi_total, kosdaq_total):
    k = tax_rate(on, Market.KOSPI)
    assert k.securities_tax == Decimal(kospi_sec)
    assert k.rural_special_tax == Decimal("0.0015")
    assert k.total == Decimal(kospi_total)
    q = tax_rate(on, Market.KOSDAQ)
    assert q.rural_special_tax == 0
    assert q.total == Decimal(kosdaq_total)


def test_konex_modeled_as_kosdaq_unverified():
    for p in TAX_SCHEDULE:
        on = max(p.effective, date(2000, 1, 1))
        assert tax_rate(on, Market.KONEX) == tax_rate(on, Market.KOSDAQ)


def test_accepts_market_aliases_and_datetime():
    assert tax_rate(datetime(2025, 3, 1, 15, 30), "코스피") == tax_rate(date(2025, 3, 1), "KOSPI")
    with pytest.raises(ValueError):
        tax_rate(date(2026, 1, 1), "NYSE")


def test_sell_tax_rate_float():
    assert sell_tax_rate(date(2026, 1, 2), "KOSPI") == 0.002
    assert sell_tax_rate(date(2025, 6, 1), "KOSDAQ") == 0.0015
    assert type(sell_tax_rate(date(2025, 6, 1), "KOSDAQ")) is float


def test_schedule_is_sorted():
    eff = [p.effective for p in TAX_SCHEDULE]
    assert eff == sorted(eff)


def test_taxrate_total_default_rural_zero():
    assert TaxRate(Decimal("0.002")).total == Decimal("0.002")
