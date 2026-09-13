"""트레이드 원장 — 체결 한 건(``Trade``), 비용 차감 순손익, 성과지표.

``performance_metrics`` 는 daytrade-it ``application/handlers/backtest.py``
``BacktestHandler._calculate_metrics`` 의 산식을 **그대로** 옮겼다. 반환형
:class:`PerformanceMetrics` 의 필드·반올림 자릿수·``Decimal`` 타입도 daytrade 의
``BacktestMetrics`` 와 같다 — 대시보드·MCP 도구가 그 모양을 이미 읽고 있어서, 산식을
옮기면서 모양을 바꾸면 소비자 쪽 비교 테스트가 무의미해진다.

그 산식의 관례(고치지 않고 보존한 것 — 바꾸면 기존 수치가 조용히 바뀐다):

- 연환산 수익률은 **달력일** 기준 ``(end-start).days / 365.25`` 년.
- Sharpe/Sortino 는 equity curve 의 일간 수익률로 ``mean·252 / (stdev·√252)``,
  **무위험수익률 차감 없음**, 표본표준편차(``statistics.stdev``, ddof=1).
- Sortino 의 하방편차는 음수 수익률들의 **표본표준편차**(0 기준 semideviation 이 아님).
  음수 수익률이 2개 미만이면 전체 표준편차로 대체.
- 손실 트레이드가 없으면 profit_factor 는 ``999`` 로 캡.
- 최대낙폭은 equity curve 각 점의 ``drawdown``(양수 비율) 최대값이고, 금액은
  ``max_dd × initial_capital`` (고점 자본이 아니라 초기자본 기준).
- 무승부(pnl == 0) 는 승도 패도 아니지만 ``total_trades`` 와 승률 분모에는 들어간다.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

__all__ = [
    "PerformanceMetrics",
    "Trade",
    "TradeResult",
    "apply_costs",
    "performance_metrics",
]

Side = Literal["long", "short"]


@dataclass(frozen=True)
class Trade:
    """체결된 트레이드 한 건. ``exit_ts``/``exit_price`` 가 None 이면 미청산.

    ``side`` 는 포지션 방향(``"long"``·``"short"``)이다 — 주문 방향(BUY/SELL)이 아니다.
    ``market`` 은 비용 함수가 KOSPI/KOSDAQ 세율을 고를 수 있게 실어 나른다.
    """

    code: str
    side: Side
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime | None
    exit_price: float | None
    qty: float
    market: str = "KOSPI"

    @property
    def closed(self) -> bool:
        return self.exit_ts is not None and self.exit_price is not None

    @property
    def gross_pnl(self) -> float:
        """비용 전 손익(원). 미청산이면 0."""
        if not self.closed:
            return 0.0
        sign = 1.0 if self.side == "long" else -1.0
        return (float(self.exit_price) - float(self.entry_price)) * float(self.qty) * sign

    @property
    def gross_return(self) -> float:
        """비용 전 가격 수익률. 미청산이거나 진입가 0 이면 0."""
        if not self.closed or not self.entry_price:
            return 0.0
        sign = 1.0 if self.side == "long" else -1.0
        return (float(self.exit_price) - float(self.entry_price)) / float(self.entry_price) * sign


@dataclass(frozen=True)
class TradeResult:
    """비용을 뗀 트레이드. ``pnl`` = ``gross_pnl - cost`` (원)."""

    trade: Trade
    gross_pnl: float
    cost: float
    pnl: float
    pnl_pct: float

    @property
    def entry_ts(self) -> datetime:
        return self.trade.entry_ts

    @property
    def exit_ts(self) -> datetime | None:
        return self.trade.exit_ts


def apply_costs(trades: Iterable[Trade], cost_fn: Callable[[Trade], float]) -> list[TradeResult]:
    """트레이드마다 ``cost_fn(trade)`` (왕복 총비용, 원 — 수수료·세금·슬리피지)을 떼 순손익을 낸다.

    비용 모델은 여기서 정하지 않는다(``krx_quant_core.costs`` 나 소비자 몫). 미청산
    트레이드는 비용을 부르지 않고 손익 0 으로 둔다 — 아직 안 판 것에 매도세를 물리면
    안 된다. ``pnl_pct`` 는 daytrade 관례대로 **비용 전** 가격 수익률이다.
    """
    out: list[TradeResult] = []
    for t in trades:
        if not t.closed:
            out.append(TradeResult(t, 0.0, 0.0, 0.0, 0.0))
            continue
        gross = t.gross_pnl
        cost = float(cost_fn(t))
        out.append(TradeResult(t, gross, cost, gross - cost, t.gross_return))
    return out


@dataclass(frozen=True)
class PerformanceMetrics:
    """daytrade-it ``BacktestMetrics`` 와 같은 필드·타입."""

    total_return: Decimal
    total_return_pct: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    avg_trade_duration_hours: Decimal
    best_trade: Decimal
    worst_trade: Decimal


def _dec(x: Any) -> Decimal:
    # daytrade 는 pnl 을 이미 Decimal(str(round(x, 2))) 로 들고 온다. float 가 들어와도
    # str 경유로 바꿔 Decimal 합산을 유지한다 — float 합으로 바꾸면 반올림 경계가 흔들린다.
    return x if isinstance(x, Decimal) else Decimal(str(x))


def _ts(t: Any, which: str) -> datetime | None:
    # core 는 entry_ts/exit_ts, daytrade BacktestTrade 는 entry_date/exit_date 다.
    v = getattr(t, f"{which}_ts", None)
    if v is None:
        v = getattr(t, f"{which}_date", None)
    return v


def performance_metrics(
    trades: Sequence[Any],
    equity_curve: Sequence[Mapping[str, Any]],
    initial_capital: Decimal | float,
    *,
    start_date: datetime,
    end_date: datetime,
) -> PerformanceMetrics:
    """트레이드 목록·equity curve → 성과지표 (daytrade ``_calculate_metrics`` 와 수치 동일).

    Args:
        trades: ``pnl`` (순손익)과 ``entry_ts``/``exit_ts`` (또는 daytrade 의
            ``entry_date``/``exit_date``) 속성을 가진 객체들. :class:`TradeResult` 가 맞다.
            청산 시각이 None 인 트레이드는 집계에서 빠진다.
        equity_curve: ``{"equity": float, "drawdown": float}`` 점들(시간순).
        initial_capital: 초기자본.
        start_date, end_date: 연환산 기간(달력일).
    """
    if not trades:
        zero = Decimal("0")
        return PerformanceMetrics(
            total_return=zero, total_return_pct=zero, annualized_return=zero,
            sharpe_ratio=zero, sortino_ratio=zero, max_drawdown=zero,
            max_drawdown_pct=zero, win_rate=zero, profit_factor=zero,
            total_trades=0, winning_trades=0, losing_trades=0,
            avg_win=zero, avg_loss=zero, avg_trade_duration_hours=zero,
            best_trade=zero, worst_trade=zero,
        )
    initial_capital = _dec(initial_capital)

    closed = [t for t in trades if _ts(t, "exit") is not None]
    pnl_of = {id(t): _dec(t.pnl) for t in closed}
    winning = [t for t in closed if pnl_of[id(t)] > 0]
    losing = [t for t in closed if pnl_of[id(t)] < 0]

    total_pnl = sum(pnl_of[id(t)] for t in closed)
    total_return_pct = total_pnl / initial_capital

    days = (end_date - start_date).days
    years = days / 365.25

    if years > 0 and total_return_pct > -1:
        annualized = (1 + float(total_return_pct)) ** (1 / years) - 1
    else:
        annualized = 0

    win_rate = len(winning) / len(closed) if closed else 0

    gross_profit = sum(pnl_of[id(t)] for t in winning)
    gross_loss = abs(sum(pnl_of[id(t)] for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("inf")

    avg_win = gross_profit / len(winning) if winning else Decimal("0")
    avg_loss = gross_loss / len(losing) if losing else Decimal("0")

    max_dd = max(e["drawdown"] for e in equity_curve) if equity_curve else 0
    max_dd_value = max_dd * float(initial_capital)

    durations = []
    for t in closed:
        entry, exit_ = _ts(t, "entry"), _ts(t, "exit")
        if entry and exit_:
            durations.append((exit_ - entry).total_seconds() / 3600)
    avg_duration = sum(durations) / len(durations) if durations else 0

    pnls = [float(pnl_of[id(t)]) for t in closed]
    best = max(pnls) if pnls else 0
    worst = min(pnls) if pnls else 0

    sharpe: float = 0
    sortino: float = 0
    if len(equity_curve) > 1:
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]["equity"]
            curr = equity_curve[i]["equity"]
            if prev > 0:
                daily_returns.append((curr - prev) / prev)
        if daily_returns:
            avg_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1
            sharpe = (avg_return * 252) / (std_return * (252**0.5)) if std_return > 0 else 0
            negative = [r for r in daily_returns if r < 0]
            downside_std = statistics.stdev(negative) if len(negative) > 1 else std_return
            sortino = (avg_return * 252) / (downside_std * (252**0.5)) if downside_std > 0 else 0

    return PerformanceMetrics(
        total_return=total_pnl,
        total_return_pct=Decimal(str(round(float(total_return_pct), 4))),
        annualized_return=Decimal(str(round(annualized, 4))),
        sharpe_ratio=Decimal(str(round(sharpe, 2))),
        sortino_ratio=Decimal(str(round(sortino, 2))),
        max_drawdown=Decimal(str(round(max_dd_value, 2))),
        max_drawdown_pct=Decimal(str(round(max_dd, 4))),
        win_rate=Decimal(str(round(win_rate, 4))),
        profit_factor=(
            Decimal(str(round(float(profit_factor), 2)))
            if profit_factor != Decimal("inf")
            else Decimal("999")
        ),
        total_trades=len(closed),
        winning_trades=len(winning),
        losing_trades=len(losing),
        avg_win=Decimal(str(round(float(avg_win), 2))),
        avg_loss=Decimal(str(round(float(avg_loss), 2))),
        avg_trade_duration_hours=Decimal(str(round(avg_duration, 1))),
        best_trade=Decimal(str(round(best, 2))),
        worst_trade=Decimal(str(round(worst, 2))),
    )
