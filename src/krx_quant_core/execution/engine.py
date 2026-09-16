"""``EngineCore`` — 실매매와 리플레이가 **같은 전략 코드**를 돌리는 이벤트 루프 한 겹.

NautilusTrader 의 원칙을 빌렸다: 전략은 시세 이벤트와 :class:`OrderManager` 만 보고,
어디서 체결되는지는 모른다. 실매매는 웹소켓 이벤트를 ``KiwoomBroker`` 위 엔진에,
리플레이(:mod:`krx_quant_core.backtest.replay`)는 과거 이벤트를 ``PaperBroker`` 위
엔진에 먹인다. 바뀌는 것은 브로커 어댑터뿐이라, 리플레이에서 검증한 코드가 그대로
실계좌로 간다(백테스트 전용 전략을 따로 짜다 실전과 어긋나는 사고를 막는다).

``feed`` 한 번의 순서는 고정이다: (페이퍼면) 시세로 대기 주문 판정 → ``oms.sync()`` 로
체결을 장부에 반영 → ``strategy.on_event``. 그래서 전략은 언제나 이번 이벤트까지의
체결이 반영된 장부를 본다. 이번 이벤트에 전략이 낸 주문은 **다음** 시세부터 판정된다
(같은 시세로 즉시 체결시키면 미래 정보를 쓰는 셈이다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from .events import Fill
from .oms import OrderManager
from .paper import PaperBroker

__all__ = ["Bar", "EngineCore", "Event", "Quote", "Strategy", "StrategyContext", "Trade"]

_NAN = float("nan")


@dataclass(frozen=True)
class Quote:
    """호가 1건(최우선 매수/매도 호가)."""

    ts: datetime
    code: str
    bid: float
    ask: float
    bid_qty: float = 0
    ask_qty: float = 0


@dataclass(frozen=True)
class Trade:
    """체결(틱) 1건."""

    ts: datetime
    code: str
    price: float
    qty: float


@dataclass(frozen=True)
class Bar:
    """봉 1개. ``ts`` 는 봉이 **닫힌** 시각이어야 한다 — 종가를 그 시각에 안다고 보기 때문."""

    ts: datetime
    code: str
    open: float
    high: float
    low: float
    close: float
    volume: float


Event = Quote | Trade | Bar


@dataclass
class StrategyContext:
    """전략이 보는 전부 — 주문 창구, 현재(이벤트) 시각, 종목별 마지막 이벤트."""

    oms: OrderManager
    now: datetime | None = None
    last: dict[str, Event] = field(default_factory=dict)


class Strategy(Protocol):
    """전략 프로토콜. ``on_start(ctx)``·``on_end(ctx)`` 는 선택 — 없으면 건너뛴다."""

    def on_event(self, ev: Event, ctx: StrategyContext) -> None: ...


class EngineCore:
    """이벤트 하나를 브로커 시세 → 체결 동기화 → 전략 순으로 흘린다."""

    def __init__(self, strategy: Strategy, oms: OrderManager) -> None:
        self.strategy = strategy
        self.oms = oms
        self.ctx = StrategyContext(oms=oms)
        #: 종목별 마지막 Quote. ``ctx.last`` 는 Trade·Bar 로 덮이므로 따로 든다 —
        #: Trade 를 페이퍼에 먹일 때 직전 호가가 필요하다.
        self._last_quote: dict[str, Quote] = {}

    def start(self) -> None:
        hook = getattr(self.strategy, "on_start", None)
        if hook is not None:
            hook(self.ctx)

    def feed(self, ev: Event) -> list[Fill]:
        """이벤트 1건을 처리하고, 이번에 동기화된 체결을 돌려준다."""
        self.ctx.now = ev.ts
        self.ctx.last[ev.code] = ev
        broker = self.oms.broker
        if isinstance(broker, PaperBroker):
            self._feed_paper(broker, ev)
        fills = self.oms.sync()
        self.strategy.on_event(ev, self.ctx)
        return fills

    def end(self) -> None:
        hook = getattr(self.strategy, "on_end", None)
        if hook is not None:
            hook(self.ctx)

    def _feed_paper(self, broker: PaperBroker, ev: Event) -> None:
        if isinstance(ev, Quote):
            self._last_quote[ev.code] = ev
            broker.on_quote(ev.code, ev.ts, ev.bid, ev.ask)
        elif isinstance(ev, Trade):
            # 체결 틱 자체에는 호가가 없다 — 직전 Quote 의 호가로 테이커 판정을 하고,
            # 체결가는 대기 주문의 through/touch 판정에만 쓴다. 호가를 본 적 없으면 NaN
            # (PaperBroker 가 "호가 없음"으로 취급한다).
            q = self._last_quote.get(ev.code)
            bid, ask = (q.bid, q.ask) if q is not None else (_NAN, _NAN)
            broker.on_quote(ev.code, ev.ts, bid, ask, last=ev.price)
        elif isinstance(ev, Bar):
            # 봉에는 호가가 없다. bid=ask=close 로 먹이면 종가 이상 매수 지정가가 전부
            # 종가에 테이커 체결돼 through/touch 구분이 사라진다(낙관적). 고가·저가로
            # 먹이면 봉 안의 순서를 모르는 채 가장 유리한 체결을 가정하게 된다. 그래서
            # 호가는 없음(NaN)으로 두고 종가 하나로만 대기 판정한다 — 보수적 체결.
            broker.on_quote(ev.code, ev.ts, _NAN, _NAN, last=ev.close)
        else:  # pragma: no cover — 타입 밖 입력
            raise TypeError(f"알 수 없는 이벤트: {type(ev).__name__}")
