"""``OrderManager`` — 브로커 위의 주문관리 한 겹. 매수는 킬·가드, 청산은 보유만 본다.

scalp-it·daytrade-it 데몬이 실계좌(``KiwoomBroker``)나 모의(``PaperBroker``) 위에 같은
코드로 두르는 층이다. 핵심 비대칭:

* **매수(신규 진입)** 는 킬스위치 → :meth:`OrderGuard.reason` 순으로 막는다. 사유
  문자열은 가드 것을 한 글자도 바꾸지 않는다(주문 로그로 사고를 추적하므로).
* **매도(청산)** 는 킬스위치·블록리스트·화이트리스트·횟수 상한으로 막지 **않는다**.
  킬이 걸린 날일수록 들고 있는 포지션은 빠져나가야 한다 — 청산을 막으면 실포지션이
  감시도 손절도 없이 남는다(가드 N-1 과 같은 이유). 막는 것은 둘뿐이다: 보유보다
  많이 파는 것(공매도·중복 청산 사고), 0 이하 지정가.

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


def _blocked(intent: OrderIntent, reason: str) -> OrderResult:
    return OrderResult(
        intent=intent, dry_run=False, submitted=False, blocked=True, blocked_reason=reason
    )


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

    # ----- 주문 --------------------------------------------------------------

    def buy(self, code: str, qty: int, price: int, *, ref_price: float | None) -> ManagedResult:
        """신규 진입. 킬스위치 → 가드 → 제출 순. 먼저 걸린 사유 하나로 막는다."""
        intent = OrderIntent(side="buy", code=code, qty=qty, price=price)
        if self.kill is not None:
            kill_reason = self.kill.check(self._clock().date())
            if kill_reason:
                return self._finish(_blocked(intent, f"kill:{kill_reason}"))
        guard_reason = self.guard.reason(intent, ref_price)
        if guard_reason:
            return self._finish(_blocked(intent, guard_reason))
        result = self.broker.submit(intent)
        if result.ok:  # dry-run 통과 또는 실제 제출 성공
            self.guard.record_order(code)
        return self._finish(result)

    def sell(
        self, code: str, qty: int, price: int, *, ref_price: float | None = None
    ) -> ManagedResult:
        """청산. **킬·가드로 막지 않는다** — 막으면 실포지션이 무감시로 남는다.

        보유는 장부를 먼저 보고, 장부에 없으면 브로커 보유를 본다(재시작 직후 장부가
        비어 있어도 실계좌 포지션은 청산할 수 있어야 한다). ``ref_price`` 는 매수와
        호출 모양을 맞추려고 받을 뿐 판정에 쓰지 않는다 — 급락 중 청산 지정가가 밴드
        밖이라고 막히면 안 된다.
        """
        del ref_price
        intent = OrderIntent(side="sell", code=code, qty=qty, price=price)
        pos = self.book.position(code)
        if pos is not None:
            held = pos.qty
        else:
            broker_pos = self.broker.holdings().get(code)
            held = broker_pos.qty if broker_pos is not None else 0
        if qty > held:
            return self._finish(_blocked(intent, f"청산 수량이 보유 초과: {qty} > {held}"))
        if price <= 0:
            return self._finish(_blocked(intent, f"지정가가 0 이하: {price}"))
        return self._finish(self.broker.submit(intent))

    def cancel(self, ord_no: str, intent: OrderIntent) -> ManagedResult:
        """미체결 취소. 킬·가드와 무관(취소는 위험을 줄이는 쪽)."""
        result = self.broker.cancel(ord_no, intent)
        if result.blocked:
            status = OrderStatus.BLOCKED
        elif result.ok:
            status = OrderStatus.CANCELED
        else:
            status = OrderStatus.REJECTED
        return self._finish(result, status)

    # ----- 체결·대사 ----------------------------------------------------------

    def sync(self) -> list[Fill]:
        """새 체결을 장부에 반영하고, 매도 체결의 실현손익을 킬스위치에 기록한다."""
        fills = self.broker.poll_fills()
        for fill in fills:
            realized = self.book.apply(fill)
            if fill.side == "sell" and self.kill is not None:
                self.kill.record_trade(realized, on=fill.ts.date())
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

    def _finish(self, result: OrderResult, status: OrderStatus | None = None) -> ManagedResult:
        managed = ManagedResult(result=result, status=status or _status_of(result))
        if self._order_log is not None:
            row = {**result.to_record(ts=self._clock()), "status": str(managed.status)}
            self._order_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self._order_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        return managed
