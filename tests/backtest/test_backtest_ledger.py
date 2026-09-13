"""backtest.ledger — Trade/apply_costs 와 daytrade ``_calculate_metrics`` 산식 고정.

daytrade-it 에는 ``_calculate_metrics`` 직접 테스트가 없어, 손으로 계산한 값으로 관례
(365.25 달력일 연환산·252 Sharpe·표본 stdev·profit_factor 999 캡·무승부 처리)를 고정한다.
"""

from __future__ import annotations

import statistics
from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from krx_quant_core.backtest.ledger import (
    PerformanceMetrics,
    Trade,
    TradeResult,
    apply_costs,
    performance_metrics,
)

T0 = datetime(2026, 1, 5, 9, 0)
START, END = datetime(2026, 1, 1), datetime(2026, 12, 31)


def _trade(entry, exit_, qty=10, side="long", hours=6.0, closed=True):
    return Trade(
        code="005930", side=side, entry_ts=T0, entry_price=entry,
        exit_ts=T0 + timedelta(hours=hours) if closed else None,
        exit_price=exit_ if closed else None, qty=qty,
    )


# --- Trade / apply_costs --------------------------------------------------------------

def test_trade_gross_pnl_long_and_short():
    assert _trade(100.0, 110.0).gross_pnl == pytest.approx(100.0)
    assert _trade(100.0, 110.0, side="short").gross_pnl == pytest.approx(-100.0)
    assert _trade(100.0, 110.0).gross_return == pytest.approx(0.10)
    assert _trade(100.0, None, closed=False).gross_pnl == 0.0


def test_trade_is_frozen():
    with pytest.raises(FrozenInstanceError):
        _trade(1.0, 2.0).qty = 5  # type: ignore[misc]


def test_apply_costs_subtracts_and_skips_open_trades():
    calls = []

    def cost_fn(t: Trade) -> float:
        calls.append(t)
        return 7.5

    closed, open_ = _trade(100.0, 110.0), _trade(100.0, None, closed=False)
    res = apply_costs([closed, open_], cost_fn)
    assert calls == [closed]  # 미청산엔 비용을 안 부른다
    assert res[0] == TradeResult(closed, 100.0, 7.5, 92.5, pytest.approx(0.10))
    assert res[0].entry_ts == T0 and res[0].exit_ts == closed.exit_ts
    assert (res[1].pnl, res[1].cost) == (0.0, 0.0)


# --- performance_metrics ---------------------------------------------------------------

def test_empty_trades_all_zero():
    m = performance_metrics([], [], Decimal("1000000"), start_date=START, end_date=END)
    assert m.total_trades == 0
    assert m.sharpe_ratio == Decimal("0") and m.profit_factor == Decimal("0")


@dataclass
class _DaytradeLike:
    """daytrade ``BacktestTrade`` 모양(entry_date/exit_date, Decimal pnl)."""

    pnl: Decimal
    entry_date: datetime
    exit_date: datetime | None


def _equity():
    eq = [1_000_000.0, 1_010_000.0, 1_005_000.0, 1_020_000.0, 1_000_000.0]
    peak, out = 0.0, []
    for v in eq:
        peak = max(peak, v)
        out.append({"equity": v, "drawdown": (peak - v) / peak})
    return eq, out


def test_metrics_hand_computed():
    trades = [
        _DaytradeLike(Decimal("30000.00"), T0, T0 + timedelta(hours=2)),
        _DaytradeLike(Decimal("-10000.00"), T0, T0 + timedelta(hours=4)),
        _DaytradeLike(Decimal("0"), T0, T0 + timedelta(hours=6)),       # 무승부
        _DaytradeLike(Decimal("99999"), T0, None),                        # 미청산 — 제외
    ]
    eq, curve = _equity()
    m = performance_metrics(trades, curve, Decimal("1000000"), start_date=START, end_date=END)

    assert isinstance(m, PerformanceMetrics)
    assert m.total_return == Decimal("20000.00")
    assert m.total_return_pct == Decimal("0.02")
    years = (END - START).days / 365.25
    assert m.annualized_return == Decimal(str(round(1.02 ** (1 / years) - 1, 4)))
    assert (m.total_trades, m.winning_trades, m.losing_trades) == (3, 1, 1)
    assert m.win_rate == Decimal("0.3333")  # 무승부도 분모에 들어간다
    assert m.profit_factor == Decimal("3.0")
    assert m.avg_win == Decimal("30000.0") and m.avg_loss == Decimal("10000.0")
    assert m.best_trade == Decimal("30000.0") and m.worst_trade == Decimal("-10000.0")
    assert m.avg_trade_duration_hours == Decimal("4.0")

    dd = max(c["drawdown"] for c in curve)
    assert m.max_drawdown_pct == Decimal(str(round(dd, 4)))
    assert m.max_drawdown == Decimal(str(round(dd * 1_000_000, 2)))  # 초기자본 기준 금액

    rets = [(eq[i] - eq[i - 1]) / eq[i - 1] for i in range(1, len(eq))]
    mu, sd = statistics.mean(rets), statistics.stdev(rets)
    neg = [r for r in rets if r < 0]
    assert m.sharpe_ratio == Decimal(str(round(mu * 252 / (sd * 252**0.5), 2)))
    assert m.sortino_ratio == Decimal(
        str(round(mu * 252 / (statistics.stdev(neg) * 252**0.5), 2)))


def test_profit_factor_capped_at_999_without_losses():
    trades = [_DaytradeLike(Decimal("100"), T0, T0 + timedelta(hours=1))]
    m = performance_metrics(trades, [], Decimal("1000"), start_date=START, end_date=END)
    assert m.profit_factor == Decimal("999")
    assert m.sharpe_ratio == Decimal("0")  # equity 점 1개 이하 → 0


def test_sortino_falls_back_to_total_std_with_one_negative_return():
    eq = [100.0, 110.0, 105.0, 120.0]
    curve = [{"equity": v, "drawdown": 0.0} for v in eq]
    trades = [_DaytradeLike(Decimal("1"), T0, T0 + timedelta(hours=1))]
    m = performance_metrics(trades, curve, Decimal("100"), start_date=START, end_date=END)
    assert m.sortino_ratio == m.sharpe_ratio


def test_float_inputs_match_decimal_inputs():
    """core TradeResult(float pnl) 와 daytrade 모양(Decimal pnl)이 같은 지표를 낸다."""
    closed = [_trade(100.0, 130.0, qty=1000), _trade(100.0, 90.0, qty=1000, hours=4)]
    res = apply_costs(closed, lambda t: 0.0)
    dt = [_DaytradeLike(Decimal(str(round(r.pnl, 2))), r.entry_ts, r.exit_ts) for r in res]
    _, curve = _equity()
    a = performance_metrics(res, curve, 1_000_000.0, start_date=START, end_date=END)
    b = performance_metrics(dt, curve, Decimal("1000000"), start_date=START, end_date=END)
    assert a == b


def test_total_loss_beyond_capital_disables_annualisation():
    trades = [_DaytradeLike(Decimal("-2000"), T0, T0 + timedelta(hours=1))]
    m = performance_metrics(trades, [], Decimal("1000"), start_date=START, end_date=END)
    assert m.total_return_pct == Decimal("-2.0")
    assert m.annualized_return == Decimal("0")
