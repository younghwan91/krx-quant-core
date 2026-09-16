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
- **매수 현금 잔고 확인 없음** — 이 브로커는 보유 수량만 안다. 계좌 현금이 충분한지는
  호출자(``OrderManager``) 또는 장부(``PositionBook``) 몫이다.
- 도착시각 이전에 들어온 시세는 그 주문 판정에 안 쓴다(``latency_sec``). 그 종목의
  시세를 주문 시점까지 한 번도 못 봤다면, 주문 뒤 처음 들어오는 시세에 도착시각을
  고정한다(그 시세 ts + latency) — latency 가 0 이면 그 시세 자신이 바로 도착이고,
  0 보다 크면 그 시세는 도착 전이라 판정에서 빠진다(다음 시세부터).
- 테이커 체결가는 ``int(round(...))`` 로 반올림한다 — KRX 시세는 원래 정수(원) 호가라
  일반적으로는 그대로지만, 방어적으로 반올림한다.
- 호가(``bid``/``ask``)가 없거나(0 이하) 유한하지 않으면(NaN/inf) 그 쪽은 "없다"로
  본다 — 테이커 판정에 안 쓴다. ``last`` 도 같은 기준으로 없으면 대기 판정에 안 쓴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from krx_quant_core.backtest.fills import FillBasis, limit_buy_filled, limit_sell_filled

from .events import Fill, Holding, OpenOrder
from .orders import OrderIntent, OrderResult

__all__ = ["PaperBroker"]


def _is_valid_quote(value: float | None) -> bool:
    """호가/체결가 하나가 "있다"고 볼 수 있나 — 유한하고 0 보다 커야 한다.

    호가창이 비었을 때 호출자가 0 을 채워 보낼 수 있다(book 센티널). 0 을 진짜
    매도 1호가로 읽으면 공짜로 체결시키는 사고가 난다. NaN/inf 도 마찬가지로 걸러낸다.
    """
    return value is not None and math.isfinite(value) and value > 0


@dataclass
class _Pending:
    """대기 중인 주문 1건의 내부 상태."""

    intent: OrderIntent
    ord_no: str
    #: ``None`` 이면 "아직 도착시각을 못 정했다" — 주문 시점에 이 종목 시세를 한 번도
    #: 못 봐서다. 다음에 들어오는 첫 시세에서 ``ts + latency_sec`` 로 고정한다(그
    #: 시세 자신은 latency 가 0 초과면 도착 전이라 판정 대상이 아니다).
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
        """대기 중이면 큐에서 뺀다. 없으면 실패로 돌려준다.

        ``submitted=True`` 는 "취소 요청을 받아들였다"는 뜻이다 — 실제 주문을 낸
        것과는 별개다(``OrderResult.submitted`` 의 원래 의미는 신규 주문 제출 기준).
        """
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
            if pending.arrival_ts is None:
                # 주문 시점에 이 종목 시세를 못 봤다 — 지금 이 시세로 도착시각을
                # 고정한다. latency_sec > 0 이면 이 시세 자신은 도착 전이라 빠진다.
                pending.arrival_ts = ts + timedelta(seconds=self._latency_sec)
            if ts < pending.arrival_ts:
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
        """이번 시세로 ``intent`` 가 채워지면 체결가(정수)를, 아니면 ``None`` 을 돌려준다.

        호가창이 비어(0 이하) 있거나 값이 유한하지 않으면(NaN/inf) 그 쪽은 "없다"로
        보고 테이커 판정을 건너뛴다 — 0 을 진짜 호가로 오인해 공짜로 체결시키지
        않기 위해서다. ``last`` 도 같은 기준으로 없으면 대기(through/touch) 판정을
        건너뛴다.
        """
        if intent.side == "buy":
            if _is_valid_quote(ask) and ask <= intent.price:
                return int(round(ask))
            if _is_valid_quote(last) and limit_buy_filled(intent.price, last, self._fill_basis):
                return int(intent.price)
            return None
        if _is_valid_quote(bid) and bid >= intent.price:
            return int(round(bid))
        if _is_valid_quote(last) and limit_sell_filled(intent.price, last, self._fill_basis):
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
