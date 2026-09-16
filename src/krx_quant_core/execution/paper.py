"""``PaperBroker`` — 같은 :class:`~krx_quant_core.execution.broker.Broker` 프로토콜의 모의 체결.

리플레이 백테스트·페이퍼트레이딩이 실전(:mod:`kiwoom_broker`)과 **같은 전략 코드**로
돌게 하는 게 존재 이유다. 시세를 ``on_quote`` 로 먹이면, 대기 중인 주문을 그 시세로
판정해 체결시킨다. 체결가 판정은 :mod:`backtest.fills` 의 ``limit_buy_filled``/
``limit_sell_filled`` 를 그대로 쓴다 — 백테스트 회계와 페이퍼 체결이 다른 기준을
쓰던 사고(2026-08-25)를 여기서도 반복하지 않기 위해서다.

가정:

- **내 주문의 시장 영향 없음** — 체결이 다음 호가를 바꾸지 않는다.
- **부분 체결 없음** — 판정에 걸리면 잔량 전체가 한 번에 체결된다.
- **비용은 장부(``PositionBook``) 몫** — 여기서는 체결가를 그대로 낸다. 수수료·세금·
  슬리피지를 빼지 않는다.
- 도착시각 이전에 들어온 시세는 그 주문 판정에 안 쓴다(``latency_sec``). 그 종목의
  시세를 아직 한 번도 못 봤다면 latency 를 더할 기준이 없으니, 주문 뒤 처음 들어오는
  시세부터 바로 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from krx_quant_core.backtest.fills import FillBasis, limit_buy_filled, limit_sell_filled

from .events import Fill, Holding, OpenOrder
from .orders import OrderIntent, OrderResult

__all__ = ["PaperBroker"]


@dataclass
class _Pending:
    """대기 중인 주문 1건의 내부 상태."""

    intent: OrderIntent
    ord_no: str
    #: ``None`` 이면 "이 종목 시세를 아직 한 번도 못 본 채로 주문했다" — 다음에 들어오는
    #: 첫 시세부터 바로 판정한다(latency 를 더할 기준 시각이 없으므로).
    arrival_ts: datetime | None


class PaperBroker:
    """시세를 먹여 지정가를 판정하는 모의 브로커. ``Broker`` 프로토콜 전부를 구현한다."""

    def __init__(
        self,
        *,
        fill_basis: FillBasis = "through",
        latency_sec: float = 0.0,
        holdings: dict[str, Holding] | None = None,
    ) -> None:
        self._fill_basis = fill_basis
        self._latency_sec = latency_sec
        self._holdings: dict[str, Holding] = dict(holdings) if holdings else {}
        self._orders: dict[str, _Pending] = {}
        self._last_seen_ts: dict[str, datetime] = {}
        self._fills_buffer: list[Fill] = []
        self._next_ord_no = 1

    # -- Broker 프로토콜 ---------------------------------------------------

    def submit(self, intent: OrderIntent) -> OrderResult:
        """주문을 큐에 넣는다. 매도는 보유(− 이미 걸린 매도 잔량) 부족이면 막는다."""
        if intent.side == "sell":
            held = self._holdings.get(intent.code)
            held_qty = held.qty if held is not None else 0
            queued_sell = sum(
                o.intent.qty
                for o in self._orders.values()
                if o.intent.code == intent.code and o.intent.side == "sell"
            )
            available = held_qty - queued_sell
            if intent.qty > available:
                return OrderResult(
                    intent=intent,
                    dry_run=False,
                    submitted=False,
                    blocked=True,
                    blocked_reason="paper: 보유 부족",
                )

        ord_no = f"P{self._next_ord_no:06d}"
        self._next_ord_no += 1

        last_ts = self._last_seen_ts.get(intent.code)
        arrival_ts = last_ts + timedelta(seconds=self._latency_sec) if last_ts is not None else None
        self._orders[ord_no] = _Pending(intent=intent, ord_no=ord_no, arrival_ts=arrival_ts)

        return OrderResult(
            intent=intent, dry_run=False, submitted=True, ord_no=ord_no, return_code=0
        )

    def cancel(self, ord_no: str, intent: OrderIntent) -> OrderResult:
        """대기 중이면 큐에서 뺀다. 없으면 실패로 돌려준다."""
        if ord_no in self._orders:
            del self._orders[ord_no]
            return OrderResult(intent=intent, dry_run=False, submitted=True, return_code=0)
        return OrderResult(
            intent=intent,
            dry_run=False,
            submitted=False,
            return_code=-1,
            return_msg="paper: 주문 없음",
        )

    def open_orders(self) -> list[OpenOrder]:
        """현재 대기 중인 주문 전부(부분 체결이 없으니 ``qty == remaining``)."""
        return [
            OpenOrder(
                ord_no=o.ord_no,
                code=o.intent.code,
                side=o.intent.side,
                qty=o.intent.qty,
                remaining=o.intent.qty,
                price=o.intent.price,
            )
            for o in self._orders.values()
        ]

    def holdings(self) -> dict[str, Holding]:
        """현재 보유(종목코드 → :class:`Holding`)."""
        return dict(self._holdings)

    def poll_fills(self) -> list[Fill]:
        """지난 호출 이후 새로 쌓인 체결만 비워서 돌려준다."""
        fills = self._fills_buffer
        self._fills_buffer = []
        return fills

    # -- 시세 주입 -----------------------------------------------------------

    def on_quote(
        self, code: str, ts: datetime, bid: float, ask: float, last: float | None = None
    ) -> list[Fill]:
        """이 종목의 새 시세로 대기 주문을 판정한다. 이번 호출에서 난 체결만 돌려준다."""
        fills: list[Fill] = []
        for ord_no, pending in list(self._orders.items()):
            intent = pending.intent
            if intent.code != code:
                continue
            if pending.arrival_ts is not None and ts < pending.arrival_ts:
                continue

            price = self._judge_fill(intent, bid, ask, last)
            if price is None:
                continue

            fill = Fill(
                ord_no=ord_no, code=code, side=intent.side, qty=intent.qty, price=price, ts=ts
            )
            self._apply_fill(fill)
            fills.append(fill)
            del self._orders[ord_no]

        self._last_seen_ts[code] = ts
        return fills

    def _judge_fill(
        self, intent: OrderIntent, bid: float, ask: float, last: float | None
    ) -> int | None:
        """이번 시세로 ``intent`` 가 채워지면 체결가(정수)를, 아니면 ``None`` 을 돌려준다."""
        if intent.side == "buy":
            if ask <= intent.price:
                return int(round(ask))
            if last is not None and limit_buy_filled(intent.price, last, self._fill_basis):
                return int(intent.price)
            return None
        if bid >= intent.price:
            return int(round(bid))
        if last is not None and limit_sell_filled(intent.price, last, self._fill_basis):
            return int(intent.price)
        return None

    def _apply_fill(self, fill: Fill) -> None:
        """체결로 보유를 갱신한다. 매수는 가중평균, 매도는 수량만 줄이고(0 이면 제거)."""
        if fill.side == "buy":
            held = self._holdings.get(fill.code)
            if held is None or held.qty == 0:
                new_qty = fill.qty
                new_avg = float(fill.price)
            else:
                new_qty = held.qty + fill.qty
                new_avg = (held.avg_price * held.qty + fill.price * fill.qty) / new_qty
            self._holdings[fill.code] = Holding(code=fill.code, qty=new_qty, avg_price=new_avg)
        else:
            held = self._holdings.get(fill.code)
            held_qty = held.qty if held is not None else 0
            remaining = held_qty - fill.qty
            if remaining <= 0:
                self._holdings.pop(fill.code, None)
            else:
                assert held is not None
                self._holdings[fill.code] = Holding(
                    code=fill.code, qty=remaining, avg_price=held.avg_price
                )
        self._fills_buffer.append(fill)
