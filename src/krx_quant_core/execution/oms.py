"""``OrderManager`` — 브로커 위의 주문관리 한 겹. 매수는 킬·가드, 청산은 보유만 본다.

scalp-it·daytrade-it 데몬이 실계좌(``KiwoomBroker``)나 모의(``PaperBroker``) 위에 같은
코드로 두르는 층이다. 핵심 비대칭:

* **매수(신규 진입)** 는 킬스위치 → :meth:`OrderGuard.reason` 순으로 막는다. 사유
  문자열은 가드 것을 한 글자도 바꾸지 않는다(주문 로그로 사고를 추적하므로).
* **매도(청산)** 는 킬스위치·블록리스트·화이트리스트·횟수 상한으로 막지 **않는다**.
  킬이 걸린 날일수록 들고 있는 포지션은 빠져나가야 한다 — 청산을 막으면 실포지션이
  감시도 손절도 없이 남는다(가드 N-1 과 같은 이유). 막는 것은 셋뿐이다: 1주 미만,
  가용 보유(보유 − 걸린 매도 잔량)보다 많이 파는 것(공매도·중복 청산 사고), 0 이하 지정가.

:class:`InstanceLock` 은 같은 계좌를 도는 데몬이 두 번 뜨는 것을 막는다(두 프로세스가
같은 신호로 주문을 두 번 낸다). :meth:`OrderManager.reconcile` 은 장부와 브로커 보유를
비교해 **보고만** 한다 — 어느 쪽이 맞는지 코드가 판단해 고치면 실계좌에서 엉뚱한
주문이 나갈 수 있으니, 판단은 사람 몫으로 남긴다.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType

from ..market.session import now_kst
from ..risk.killswitch import KillSwitch
from .book import PositionBook
from .broker import Broker
from .events import Fill, OrderStatus
from .guards import OrderGuard
from .orders import OrderIntent, OrderResult

__all__ = [
    "AlreadyRunning",
    "InstanceLock",
    "ManagedResult",
    "OrderManager",
    "ReconcileReport",
]


class AlreadyRunning(RuntimeError):
    """같은 잠금 파일을 다른 인스턴스가 이미 잡고 있다."""


class InstanceLock:
    """``flock(LOCK_EX|LOCK_NB)`` 기반 단일 인스턴스 잠금. 컨텍스트 매니저로도 쓴다.

    flock 은 **열린 파일 기술(open file description)** 단위라, 같은 프로세스에서도
    ``InstanceLock`` 객체 두 개가 같은 경로를 잡으면 충돌한다. 프로세스가 죽으면 커널이
    잠금을 풀어 주므로 낡은 잠금 파일이 재기동을 막지 않는다. 파일 안의 pid 는 사람이
    "누가 잡고 있나"를 볼 때 쓰는 참고값일 뿐 판정에 쓰지 않는다.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> InstanceLock:
        if self._fd is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise AlreadyRunning(f"이미 실행 중: {self.path}") from None
        except BaseException:
            os.close(fd)
            raise
        # 잠금을 잡은 뒤에만 비운다 — 먼저 비우면 실행 중인 쪽의 pid 를 지운다.
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return self

    def release(self) -> None:
        """잠금을 풀고 닫는다. 잡고 있지 않으면 무동작(두 번 불러도 안전)."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> InstanceLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


@dataclass(frozen=True)
class ReconcileReport:
    """장부 vs 브로커 보유 차이. 수량은 주 단위, ``qty_mismatch`` 값은 ``(장부, 브로커)``."""

    only_book: dict[str, int]
    only_broker: dict[str, int]
    qty_mismatch: dict[str, tuple[int, int]]

    @property
    def ok(self) -> bool:
        return not (self.only_book or self.only_broker or self.qty_mismatch)


@dataclass
class ManagedResult:
    """브로커 결과 + 매니저가 매긴 상태."""

    result: OrderResult
    status: OrderStatus


def _status_of(result: OrderResult) -> OrderStatus:
    if result.blocked:
        return OrderStatus.BLOCKED
    if result.ok:
        return OrderStatus.SUBMITTED
    return OrderStatus.REJECTED


class OrderManager:
    """킬·가드·장부·브로커를 묶는 주문 창구.

    가드의 일일 횟수 카운트(:meth:`OrderGuard.record_order`) 규칙: **매수가 가드를
    통과했고, dry-run 으로 통과했거나 실제 제출이 성공(``result.ok``)했을 때만** 센다.
    브로커가 거부(rc≠0)하거나 막은 주문은 세지 않는다 — 나가지 않은 주문으로 상한이
    닳으면 정작 필요한 진입이 막힌다. 청산 매도는 가드를 거치지 않으므로 세지 않는다
    (세면 청산이 늘수록 매수 한도가 줄어드는 엉뚱한 결합이 생긴다).

    ``clock`` 은 킬스위치 날짜와 주문 로그 시각에 쓴다. 가드도 같은 시계로 만들어야
    일일 리셋 시점이 어긋나지 않는다.
    """

    def __init__(
        self,
        broker: Broker,
        *,
        guard: OrderGuard,
        book: PositionBook,
        kill: KillSwitch | None = None,
        clock: Callable[[], datetime] = now_kst,
        order_log: Path | str | None = None,
    ) -> None:
        self.broker = broker
        self.guard = guard
        self.book = book
        self.kill = kill
        self._clock = clock
        self._order_log = Path(order_log) if order_log is not None else None
        #: 장부가 적용하지 못한 체결(``PositionBook.apply`` 가 ``ValueError``). 대개 장부
        #: 밖에서 이미 들고 있던 보유(브로커 보유로 청산한 경우)나 중복 체결이다. 킬스위치에는
        #: 반영하지 않았다 — :meth:`reconcile` 결과와 함께 호출부·사람이 확인한다.
        self.unmatched_fills: list[Fill] = []
        #: 종목별로 "포지션이 열린 뒤 지금까지" 분할 청산 실현손익 합. 완전 청산 때 리셋.
        #: 날짜로는 리셋하지 않는다 — 장중 데몬은 매일 청산하고 끝난다는 전제(오버나잇
        #: 분할 청산이면 전날 조각이 다음 날 승패 판정에 섞인다).
        self._open_realized: dict[str, float] = {}

    @property
    def _dry_run(self) -> bool:
        return bool(getattr(self.broker, "dry_run", False))

    # ----- 주문 --------------------------------------------------------------

    def buy(self, code: str, qty: int, price: int, *, ref_price: float | None) -> ManagedResult:
        """신규 진입. 킬스위치 → 가드 → 제출 순. 먼저 걸린 사유 하나로 막는다.

        브로커 호출이 예외를 던지면 REJECTED 로 기록하고 돌려준다(데몬을 죽이지 않는다).
        이때 주문이 실제로 나갔는지 알 수 없으므로 횟수는 **보수적으로 센다**.
        """
        intent = OrderIntent(side="buy", code=code, qty=qty, price=price)
        if self.kill is not None:
            kill_reason = self.kill.check(self._clock().date())
            if kill_reason:
                return self._finish(self._blocked(intent, f"kill:{kill_reason}"))
        guard_reason = self.guard.reason(intent, ref_price)
        if guard_reason:
            return self._finish(self._blocked(intent, guard_reason))
        try:
            result = self.broker.submit(intent)
        except Exception as e:  # 어떤 브로커 예외든 기록하고 계속 돈다
            self.guard.record_order(code)
            return self._finish(self._errored(intent, e), OrderStatus.REJECTED)
        if result.ok:  # dry-run 통과 또는 실제 제출 성공
            self.guard.record_order(code)
        return self._finish(result)

    def sell(
        self, code: str, qty: int, price: int, *, ref_price: float | None = None
    ) -> ManagedResult:
        """청산. **킬·가드로 막지 않는다** — 막으면 실포지션이 무감시로 남는다.

        막는 것: 1주 미만, 가용 보유 초과, 0 이하 지정가. 가용 보유 = 보유 − 브로커에
        이미 걸린 매도 잔량(같은 포지션을 두 번 청산 주문하는 사고 방지). 잔고·미체결
        조회가 예외면 order_log 에 남기고 그 검사만 건너뛴다(청산 경로는 죽지 않는다).

        보유는 **장부가 있으면 장부가 우선**이고, 장부에 없을 때만 브로커 보유를 본다
        (재시작 직후 장부가 비어 있어도 실계좌 포지션은 청산할 수 있어야 한다). 그래서
        장부 밖에서 들고 있던 주식은 장부 포지션이 있는 동안 청산 한도에 안 잡히고,
        :meth:`reconcile` 의 ``qty_mismatch`` 로 드러난다. 브로커 보유로 낸 청산의 체결은
        장부가 적용하지 못할 수 있다 — :meth:`sync` 가 :attr:`unmatched_fills` 로 뺀다.

        ``ref_price`` 는 매수와 호출 모양을 맞추려고 받을 뿐 판정에 쓰지 않는다 — 급락 중
        청산 지정가가 밴드 밖이라고 막히면 안 된다.
        """
        del ref_price
        intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
        if qty < 1:
            return self._finish(self._blocked(intent, f"청산 수량이 1주 미만: {qty}"))
        held = self._held_for_exit(code)
        if held is not None:
            available = held - self._pending_sell_qty(code)
            if qty > available:
                return self._finish(
                    self._blocked(intent, f"청산 수량이 보유 초과: {qty} > {available}")
                )
        if price <= 0:
            return self._finish(self._blocked(intent, f"지정가가 0 이하: {price}"))
        try:
            result = self.broker.submit(intent)
        except Exception as e:
            return self._finish(self._errored(intent, e), OrderStatus.REJECTED)
        return self._finish(result)

    def cancel(self, ord_no: str, intent: OrderIntent) -> ManagedResult:
        """미체결 취소. 킬·가드와 무관(취소는 위험을 줄이는 쪽). 예외는 REJECTED 로."""
        try:
            result = self.broker.cancel(ord_no, intent)
        except Exception as e:
            return self._finish(self._errored(intent, e), OrderStatus.REJECTED)
        if result.blocked:
            status = OrderStatus.BLOCKED
        elif result.ok:
            status = OrderStatus.CANCELED
        else:
            status = OrderStatus.REJECTED
        return self._finish(result, status)

    # ----- 체결·대사 ----------------------------------------------------------

    def sync(self) -> list[Fill]:
        """새 체결을 장부에 반영하고, 청산 손익을 킬스위치에 기록한다. 폴링한 fill 전부를 돌려준다.

        ``poll_fills`` 는 이미 커서를 넘겼으므로 한 fill 이 실패해도 나머지는 **각각**
        처리한다 — 예외로 빠져나가면 남은 체결이 장부·킬스위치에서 영영 사라진다.
        장부가 적용 못 한 fill 은 :attr:`unmatched_fills` 에 쌓고 킬스위치는 건드리지 않는다.

        킬스위치 기록: 분할 청산 중간 조각은 ``record_realized``(손익만, 승패 없음), 포지션이
        닫히는 조각에서 ``record_trade`` 한 번 — 승패는 그 포지션의 분할 손익 **합계**로
        가른다. 조각마다 한 거래로 세면 3분할 손절 한 번이 연속손절 3회가 된다.
        ``record_trade`` 도 손익을 더하므로 마지막 조각 손익만 넘긴다(이중 합산 방지).
        """
        fills = self.broker.poll_fills()
        for fill in fills:
            try:
                realized = self.book.apply(fill)
            except ValueError as e:
                self.unmatched_fills.append(fill)
                self._log_row(
                    {
                        "ts": self._clock().isoformat(sep=" ", timespec="seconds"),
                        "event": "unmatched_fill",
                        "ord_no": fill.ord_no,
                        "code": fill.code,
                        "side": fill.side,
                        "qty": fill.qty,
                        "price": fill.price,
                        "fill_ts": fill.ts.isoformat(),
                        "error": str(e),
                    }
                )
                continue
            if fill.side != "sell":
                continue
            cum = self._open_realized.get(fill.code, 0.0) + realized
            on = fill.ts.date()
            if self.book.position(fill.code) is not None:
                self._open_realized[fill.code] = cum
                if self.kill is not None:
                    self.kill.record_realized(realized, on=on)
            else:
                self._open_realized.pop(fill.code, None)
                if self.kill is not None:
                    self.kill.record_trade(realized, on=on, is_loss=cum < 0)
        return fills

    def reconcile(self) -> ReconcileReport:
        """장부 vs 브로커 보유. **고치지 않고 보고만 한다.**"""
        book = {c: h.qty for c, h in self.book.positions().items()}
        broker = {c: h.qty for c, h in self.broker.holdings().items() if h.qty > 0}
        return ReconcileReport(
            only_book={c: q for c, q in book.items() if c not in broker},
            only_broker={c: q for c, q in broker.items() if c not in book},
            qty_mismatch={
                c: (q, broker[c]) for c, q in book.items() if c in broker and broker[c] != q
            },
        )

    # ----- 내부 --------------------------------------------------------------

    def _held_for_exit(self, code: str) -> int | None:
        """청산 한도용 보유. 장부 우선, 없으면 브로커. 브로커 조회 실패면 ``None``.

        ``None`` 이면 보유 검사를 건너뛴다 — 잔고 조회가 실패했다고 청산을 막으면
        포지션이 무감시로 남는다. 실제 초과 매도는 브로커(증권사)가 보유 부족으로 거부한다.
        """
        pos = self.book.position(code)
        if pos is not None:
            return pos.qty
        try:
            broker_pos = self.broker.holdings().get(code)
        except Exception as e:
            self._log_failure("holdings_failed", code, e)
            return None
        return broker_pos.qty if broker_pos is not None else 0

    def _pending_sell_qty(self, code: str) -> int:
        """브로커에 걸린 이 종목 매도 잔량. 조회 실패면 0(예약 없이 내보낸다 — 이유는 위와 같다)."""
        try:
            return sum(
                o.remaining
                for o in self.broker.open_orders()
                if o.code == code and o.side == "sell"
            )
        except Exception as e:
            self._log_failure("open_orders_failed", code, e)
            return 0

    def _log_failure(self, event: str, code: str, exc: Exception) -> None:
        self._log_row(
            {
                "ts": self._clock().isoformat(sep=" ", timespec="seconds"),
                "event": event,
                "code": code,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    def _blocked(self, intent: OrderIntent, reason: str) -> OrderResult:
        return OrderResult(
            intent=intent,
            dry_run=self._dry_run,
            submitted=False,
            blocked=True,
            blocked_reason=reason,
        )

    def _errored(self, intent: OrderIntent, exc: Exception) -> OrderResult:
        # submitted=True: 요청이 브로커에 닿았는지 모른다 — 안 나갔다고 가정하지 않는다.
        # dry_run=False: 예외는 "아무것도 확인되지 않았다"는 뜻이다. dry-run 브로커라도
        # dry_run=True 로 두면 OrderResult.ok 가 True 가 되어 실패가 통과로 읽힌다.
        return OrderResult(
            intent=intent,
            dry_run=False,
            submitted=True,
            return_code=None,
            return_msg=f"{type(exc).__name__}: {exc}",
        )

    def _finish(self, result: OrderResult, status: OrderStatus | None = None) -> ManagedResult:
        managed = ManagedResult(result=result, status=status or _status_of(result))
        row = {**result.to_record(ts=self._clock()), "status": str(managed.status)}
        if result.return_msg:
            row["return_msg"] = result.return_msg
        self._log_row(row)
        return managed

    def _log_row(self, row: dict[str, object]) -> None:
        if self._order_log is None:
            return
        self._order_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self._order_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
