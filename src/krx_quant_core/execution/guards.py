"""주문 안전 가드 — 실계좌에 나가기 전 마지막 판정. 순수 로직, 시계 주입.

scalp-it ``adapters/kiwoom_order.py`` 의 ``OrderGuardConfig`` 와
``KiwoomOrderClient.guard_reason``·일일 횟수 카운터를 **판정 순서·사유 문자열까지
그대로** 옮겼다. 사유 문자열은 주문 로그(``pair_orders.log``)에 남고 사람이 그걸로
사고를 추적하므로, 글자를 바꾸면 그것도 동작 변경이다.

판정 순서(먼저 걸린 사유 하나만 돌려준다)
======================================

0. 종목코드 형식(6자)
1. 블록리스트 — **매수·매도 양방향** 차단
1b. 동적 보유 병합 — **매수에만**(N-1: 방금 산 종목의 청산 매도까지 막으면 실포지션이
    무감시로 방치된다)
2. 화이트리스트 — 비어 있지 않으면 그 외 전부 차단
3. 수량 — 1주 미만 → 하드캡 → 설정 상한
4. 가격 — 0 이하 → 참조가 대비 밴드
5. 일일 횟수 — 총량 → 종목당

계좌 보유 종목(scalp-it 의 인바디·GS·심텍)은 코어에 박지 않는다. 대신
:attr:`OrderGuardConfig.always_blocked` 클래스 변수로 받는다 — scalp-it 은
``always_blocked = HELD_STOCK_CODES`` 인 서브클래스를 두면 기본 생성·병합 결과가
원본과 정확히 같다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import ClassVar

from ..market.codes import normalize_code
from ..market.session import now_kst
from .orders import OrderIntent

__all__ = [
    "OrderGuard",
    "OrderGuardConfig",
    "count_limit_reason",
    "evaluate_order",
    "holdings_block_reason",
]


@dataclass(frozen=True)
class OrderGuardConfig:
    """주문 안전 가드 파라미터. 보수적 기본값."""

    #: **무슨 설정을 줘도** 블록리스트에 병합되는 코드. 코어는 비어 있다 —
    #: 계좌 보유 종목은 레포별 서브클래스가 채운다(모듈 docstring).
    always_blocked: ClassVar[frozenset[str]] = frozenset()

    max_qty: int = 1                    # 주문당 수량 상한(기본 1주)
    max_qty_hard_cap: int = 10          # 절대 하드캡 — 설정으로도 못 넘음
    price_band_pct: float = 0.03        # 참조가 대비 ±3% 밖 지정가 거부. 0 = 끔
    max_orders_per_code: int = 5        # 종목당 일일 주문 횟수 상한. 0 = 끔
    max_orders_total: int = 20          # 일일 총 주문 횟수 상한. 0 = 끔
    whitelist: frozenset[str] = frozenset()   # 허용 종목. 비면 화이트리스트 미적용
    blocklist: frozenset[str] = field(default_factory=frozenset)  # 항상 차단(양방향)

    def __post_init__(self) -> None:
        # 하드캡을 넘는 max_qty 는 하드캡으로 강제(불변).
        cap = max(1, int(self.max_qty_hard_cap))
        object.__setattr__(self, "max_qty_hard_cap", cap)
        object.__setattr__(self, "max_qty", max(1, min(int(self.max_qty), cap)))
        # always_blocked 는 무슨 일이 있어도 블록리스트에 포함(정규화).
        merged = {normalize_code(c) for c in self.blocklist} | set(type(self).always_blocked)
        object.__setattr__(self, "blocklist", frozenset(c for c in merged if c))
        object.__setattr__(
            self,
            "whitelist",
            frozenset(normalize_code(c) for c in self.whitelist if normalize_code(c)),
        )

    def with_allowed(self, codes: Iterable[str]) -> OrderGuardConfig:
        """화이트리스트에 종목을 **더한** 사본.

        장중에 대장 스캔·돌발재료로 붙은 종목은 장전 화이트리스트에 없다. 그대로
        두면 감지기는 신호를 내는데 주문은 전부 차단된다(scalp-it 2026-08-27 실측).
        """
        merged = set(self.whitelist) | {normalize_code(c) for c in codes}
        return replace(self, whitelist=frozenset(c for c in merged if c))

    def with_blocked(self, codes: Iterable[str]) -> OrderGuardConfig:
        """코드를 블록리스트(양방향)에 병합한 새 설정."""
        merged = set(self.blocklist) | {normalize_code(c) for c in codes if normalize_code(c)}
        return replace(self, blocklist=frozenset(merged))


# --------------------------------------------------------------------- 순수 판정

def _names_for(intent: OrderIntent, code_names: Mapping[str, str] | None) -> Mapping[str, str]:
    return code_names if code_names is not None else intent.code_names


def _static_reason(
    intent: OrderIntent,
    ref_price: float | None,
    config: OrderGuardConfig,
    dynamic_buy_block: frozenset[str] | set[str],
    names: Mapping[str, str],
) -> str | None:
    """횟수 상한을 뺀 판정(0~4단계). 상태가 필요 없는 부분."""
    g = config
    code = intent.code
    if not code or len(code) != 6:
        return f"잘못된 종목코드: {code!r}"
    # (1) 블록리스트 — 최우선. **매수·매도 양방향**.
    if code in g.blocklist:
        name = names.get(code, "")
        return f"블록리스트 차단(보유/금지 종목{': ' + name if name else ''}) — {code}"
    # (1b) 동적 보유 병합 — **매수에만** 적용(N-1). 청산 매도는 막지 않는다.
    if intent.side == "buy" and code in dynamic_buy_block:
        name = names.get(code, "")
        return f"실시간 보유종목 차단(매수){': ' + name if name else ''} — {code}"
    # (2) 화이트리스트 — 비어있지 않으면 그 외 전부 차단.
    if g.whitelist and code not in g.whitelist:
        return f"화이트리스트 외 종목 — {code} (등록 쌍 후속주만 허용)"
    # (3) 수량 상한.
    if intent.qty < 1:
        return f"수량이 1주 미만: {intent.qty}"
    if intent.qty > g.max_qty_hard_cap:
        return f"수량 하드캡({g.max_qty_hard_cap}) 초과: {intent.qty}"
    if intent.qty > g.max_qty:
        return f"수량 상한({g.max_qty}) 초과: {intent.qty}"
    # (4) 가격 범위.
    if intent.price <= 0:
        return f"지정가가 0 이하: {intent.price}"
    if g.price_band_pct and ref_price and ref_price > 0:
        dev = abs(intent.price / float(ref_price) - 1.0)
        if dev > g.price_band_pct + 1e-9:
            return (f"지정가 {intent.price:,} 가 참조가 {ref_price:,.0f} 대비 "
                    f"{dev:.2%} (±{g.price_band_pct:.1%} 밖)")
    return None


def count_limit_reason(
    config: OrderGuardConfig,
    code: str,
    orders_total: int,
    orders_by_code: Mapping[str, int],
) -> str | None:
    """일일 횟수 상한 위반이면 사유 문자열, 아니면 None. 총량을 먼저 본다."""
    g = config
    if g.max_orders_total and orders_total >= g.max_orders_total:
        return f"일일 주문 횟수 상한({g.max_orders_total}) 초과"
    if g.max_orders_per_code and orders_by_code.get(code, 0) >= g.max_orders_per_code:
        return f"종목당 주문 횟수 상한({g.max_orders_per_code}) 초과 — {code}"
    return None


def evaluate_order(
    intent: OrderIntent,
    ref_price: float | None,
    config: OrderGuardConfig,
    *,
    dynamic_buy_block: frozenset[str] | set[str] = frozenset(),
    orders_total: int = 0,
    orders_by_code: Mapping[str, int] | None = None,
    code_names: Mapping[str, str] | None = None,
) -> str | None:
    """주문을 막을 사유. None 이면 통과. **dry-run 에서도 항상 돌려야 한다.**

    완전한 순수 함수다 — 오늘 이미 낸 횟수는 호출부가 넘긴다(날짜 리셋 포함).
    ``code_names`` 를 안 주면 ``intent`` 클래스의 :attr:`OrderIntent.code_names` 를 쓴다.
    """
    names = _names_for(intent, code_names)
    return _static_reason(intent, ref_price, config, dynamic_buy_block, names) or (
        count_limit_reason(config, intent.code, orders_total, orders_by_code or {})
    )


def holdings_block_reason(
    intent: OrderIntent,
    config: OrderGuardConfig,
    held_or_dynamic: frozenset[str] | set[str],
    *,
    code_names: Mapping[str, str] | None = None,
) -> str | None:
    """주문 직전 실시간 잔고조회 결과로 내는 차단 사유(scalp-it ``_preorder_holdings_guard``).

    조회 자체와 fail-closed 처리는 호출부 몫이다. 여기선 조회가 **성공한 뒤** 병합된
    집합으로 판정만 한다. 매수가 아니면 None.
    """
    if intent.side != "buy":
        return None
    if intent.code in config.blocklist or intent.code in held_or_dynamic:
        name = _names_for(intent, code_names).get(intent.code, "")
        return f"실시간 보유종목 차단{': ' + name if name else ''} — {intent.code}"
    return None


# --------------------------------------------------------------------- 상태 보유 가드

class OrderGuard:
    """설정 + 동적 매수차단 집합 + 일일 횟수 카운터를 들고 있는 가드.

    scalp-it ``KiwoomOrderClient`` 에서 HTTP 를 뺀 판정 상태 전부다. 시계는 주입한다
    (기본 :func:`~krx_quant_core.market.session.now_kst`; scalp-it 은 자기 naive 시계를
    넘기면 된다). 날짜 확인은 원본처럼 **횟수 판정·기록 시점에만** 한다.
    """

    def __init__(
        self,
        config: OrderGuardConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        code_names: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or OrderGuardConfig()
        self._clock = clock or now_kst
        self._code_names = code_names
        #: 매수 전용 차단 집합(N-1). config.blocklist(양방향)와 섞지 않는다.
        self.dynamic_buy_block: frozenset[str] = frozenset()
        self.count_day: date | None = None
        self.count_total = 0
        self.count_by_code: dict[str, int] = {}

    # ----- 카운터 ------------------------------------------------------------

    def observe_day(self) -> None:
        d = self._clock().date()
        if self.count_day != d:
            self.count_day = d
            self.count_total = 0
            self.count_by_code = {}

    def count_reason(self, code: str) -> str | None:
        """횟수 상한 위반이면 사유 문자열, 아니면 None."""
        self.observe_day()
        return count_limit_reason(self.config, code, self.count_total, self.count_by_code)

    def record_order(self, code: str) -> None:
        """가드 통과한 주문 1건을 센다(dry-run 도 리허설로 센다 — 원본 규약)."""
        self.observe_day()
        self.count_total += 1
        self.count_by_code[code] = self.count_by_code.get(code, 0) + 1

    # ----- 집합 갱신 ---------------------------------------------------------

    def allow_codes(self, codes: Iterable[str]) -> None:
        """장중에 붙은 종목을 화이트리스트에 더한다. 비었으면 무동작."""
        codes = list(codes)
        if codes:
            self.config = self.config.with_allowed(set(codes))

    def block_buys(self, codes: Iterable[str]) -> set[str]:
        """계좌 보유분을 **매수 전용** 차단 집합에 병합. 병합한(정규화된) 코드를 돌려준다."""
        held = {normalize_code(c) for c in codes if normalize_code(c)}
        self.dynamic_buy_block = self.dynamic_buy_block | held
        return held

    # ----- 판정 --------------------------------------------------------------

    def reason(self, intent: OrderIntent, ref_price: float | None) -> str | None:
        """주문을 막을 사유. None 이면 통과. (scalp-it ``guard_reason`` 과 동일)"""
        names = _names_for(intent, self._code_names)
        return _static_reason(
            intent, ref_price, self.config, self.dynamic_buy_block, names
        ) or self.count_reason(intent.code)

    def holdings_reason(self, intent: OrderIntent, held: Iterable[str]) -> str | None:
        """실시간 잔고조회 성공 후: 보유분을 매수차단에 병합하고 사유를 판정한다."""
        if intent.side != "buy":
            return None
        held = set(held)
        if held:
            self.block_buys(held)
        return holdings_block_reason(
            intent, self.config, self.dynamic_buy_block, code_names=self._code_names
        )
