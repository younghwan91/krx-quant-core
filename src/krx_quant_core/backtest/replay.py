"""``run_replay`` — 과거 이벤트를 ``PaperBroker`` 위 :class:`EngineCore` 로 다시 흘린다.

실매매와 **같은 전략 객체**를 그대로 받는다. 주문은 실매매와 같은 ``OrderManager``
(킬·가드·장부)를 거치고, 체결만 ``PaperBroker`` 가 시세로 판정한다. 비용은 장부
(``PositionBook``)가 청산 때 뺀다 — 백테스트 회계와 실매매 장부가 같은 코드다.

시계는 벽시계가 아니라 **이벤트 시각**이다. 벽시계를 쓰면 가드의 일일 카운터·킬스위치
날짜·주문 로그 시각이 리플레이하는 날과 어긋나고, 같은 입력을 두 번 돌려도 결과가
달라진다(결정론이 깨진다). 첫 이벤트 전에는 1970-01-01 KST 를 쓴다.

한 번의 ``run_replay`` 는 한 거래일·한 계좌다. 여러 날은 호출부가 나눠 병렬로 돌린다.
끝나도 미청산 포지션을 강제로 닫지 않는다 — 청산 규칙은 전략 몫이고, 남은 포지션은
``book`` 에 그대로 보인다.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import pandas as pd

from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.engine import Bar, EngineCore, Event, Quote, Strategy, Trade
from krx_quant_core.execution.events import Fill
from krx_quant_core.execution.guards import OrderGuard, OrderGuardConfig
from krx_quant_core.execution.oms import OrderManager
from krx_quant_core.execution.paper import PaperBroker
from krx_quant_core.market.session import KST
from krx_quant_core.risk.killswitch import KillSwitch, KillSwitchConfig

from .fills import FillBasis

__all__ = ["ReplayResult", "merge_events", "run_replay"]

_EPOCH = datetime(1970, 1, 1, tzinfo=KST)
_FILL_COLUMNS = [f.name for f in fields(Fill)]
#: 같은 시각이면 호가 → 체결 → 봉. 봉은 구간이 닫힌 뒤의 요약이라 가장 늦게 안다.
_KIND_ORDER: dict[type, int] = {Quote: 0, Trade: 1, Bar: 2}


def _default_guard_config() -> OrderGuardConfig:
    # 리플레이 기본값: 수량 상한만 두고 가격 밴드·횟수 상한은 끈다(전략 검증이 목적).
    return OrderGuardConfig(
        max_qty=10, price_band_pct=0.0, max_orders_per_code=0, max_orders_total=0
    )


@dataclass
class ReplayResult:
    """리플레이 결과. ``trades`` 는 장부의 청산 원장(매도 fill 한 건당 한 행)."""

    fills: pd.DataFrame
    orders: list[dict]
    book: PositionBook
    trades: pd.DataFrame


def _frame_events(df: pd.DataFrame | None, cls: type) -> list[Event]:
    if df is None:
        return []
    names = [f.name for f in fields(cls) if f.name in df.columns]
    out: list[Event] = []
    for row in df[names].to_dict("records"):
        kw = {}
        for name, value in row.items():
            if name == "ts":
                kw[name] = pd.Timestamp(value).to_pydatetime()
            elif name == "code":
                kw[name] = str(value)
            else:
                kw[name] = float(value)
        out.append(cls(**kw))
    return out


def merge_events(
    *,
    quotes: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    bars: pd.DataFrame | None = None,
) -> list[Event]:
    """DataFrame(열 이름 = 이벤트 필드명)을 시각순 이벤트 목록으로 합친다.

    정렬 키는 ``(ts, 종류 순서, 입력 순서)`` — 안정 정렬이라 같은 시각·같은 종류는
    입력 순서를 지킨다. 같은 입력이면 언제나 같은 순서(리플레이 결정론의 전제).
    """
    evs = _frame_events(quotes, Quote) + _frame_events(trades, Trade) + _frame_events(bars, Bar)
    return sorted(evs, key=lambda e: (e.ts, _KIND_ORDER[type(e)]))


def run_replay(
    strategy: Strategy,
    events: Iterable[Event],
    *,
    guard_config: OrderGuardConfig | None = None,
    kill_config: KillSwitchConfig | None = None,
    fill_basis: FillBasis = "through",
    latency_sec: float = 0.0,
    market_of: Callable[[str], str] = lambda c: "KOSPI",
) -> ReplayResult:
    """``events`` 를 순서대로 ``PaperBroker`` 위 엔진에 먹이고 결과를 모은다."""
    engine: EngineCore | None = None

    def clock() -> datetime:
        return (engine.ctx.now if engine is not None else None) or _EPOCH

    broker = PaperBroker(fill_basis=fill_basis, latency_sec=latency_sec)
    guard = OrderGuard(guard_config or _default_guard_config(), clock=clock)
    book = PositionBook(market_of=market_of)
    kill = KillSwitch(kill_config) if kill_config is not None else None

    fills: list[Fill] = []
    # 주문 로그는 OrderManager 의 공개 경로(order_log jsonl)를 그대로 쓰고 끝에 읽는다 —
    # 실매매 주문 로그와 같은 행 모양이 리플레이 결과에도 남는다.
    with tempfile.TemporaryDirectory(prefix="kqc-replay-") as tmp:
        log_path = Path(tmp) / "orders.jsonl"
        oms = OrderManager(
            broker, guard=guard, book=book, kill=kill, clock=clock, order_log=log_path
        )
        engine = EngineCore(strategy, oms)
        engine.start()
        for ev in events:
            fills.extend(engine.feed(ev))
        engine.end()
        orders: list[dict] = []
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                orders = [json.loads(line) for line in f if line.strip()]

    fills_df = pd.DataFrame([asdict(f) for f in fills], columns=_FILL_COLUMNS)
    return ReplayResult(fills=fills_df, orders=orders, book=book, trades=book.to_frame())
