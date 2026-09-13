"""주문 가드·intent·result — scalp-it tests/test_kiwoom_order.py·test_leader_scan.py 이식.

원본은 KiwoomOrderClient 를 MockTransport 로 돌렸다. 여기선 HTTP 를 뺀 판정만
남았으니 :class:`OrderGuard` 에 직접 태운다. 판정 결과(차단 여부·사유 문구)는 같다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar

import pytest

from krx_quant_core.execution import (
    OrderBlocked,
    OrderGuard,
    OrderGuardConfig,
    OrderIntent,
    OrderResult,
    cancel_body,
    count_limit_reason,
    evaluate_order,
    holdings_block_reason,
)

HELD = frozenset({"041830", "078930", "222800"})
HELD_NAMES = {"041830": "인바디", "078930": "GS", "222800": "심텍"}


class HeldGuardConfig(OrderGuardConfig):
    """scalp-it 이 쓸 서브클래스 모양 — 보유 3종목을 항상 병합."""

    always_blocked: ClassVar[frozenset[str]] = HELD


class NamedIntent(OrderIntent):
    code_names = HELD_NAMES


def _buy(code: str, qty: int = 1, price: int = 10000, cls: type[OrderIntent] = OrderIntent):
    return cls(side="buy", code=code, qty=qty, price=price)


def _sell(code: str, qty: int = 1, price: int = 10000):
    return OrderIntent(side="sell", code=code, qty=qty, price=price)


class _Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def _submit(g: OrderGuard, intent: OrderIntent, ref: float | None = 10000) -> str | None:
    """KiwoomOrderClient.submit 의 가드 부분: 통과하면 카운트."""
    reason = g.reason(intent, ref)
    if reason is None:
        g.record_order(intent.code)
    return reason


# ------------------------------------------------------------------ 설정 기본값

def test_defaults_match_scalp_it():
    g = OrderGuardConfig()
    assert (g.max_qty, g.max_qty_hard_cap, g.price_band_pct) == (1, 10, 0.03)
    assert (g.max_orders_per_code, g.max_orders_total) == (5, 20)
    assert g.whitelist == frozenset()
    assert g.blocklist == frozenset(), "코어 기본은 계좌 보유 종목을 모른다"


def test_subclass_reproduces_held_default_exactly():
    g = HeldGuardConfig()
    assert g.blocklist == HELD
    # 설정으로 blocklist 를 줘도 보유 3종목은 빠지지 않는다(원본 __post_init__).
    g2 = HeldGuardConfig(blocklist=frozenset({"A005930"}))
    assert g2.blocklist == HELD | {"005930"}
    # replace 계열도 서브클래스를 유지해 병합이 계속된다.
    g3 = g2.with_allowed({"294570"}).with_blocked({"0155E0"})
    assert type(g3) is HeldGuardConfig
    assert HELD <= g3.blocklist and "0155E0" in g3.blocklist
    assert g3.whitelist == frozenset({"294570"})


def test_qty_hard_cap_cannot_be_exceeded_by_config():
    assert OrderGuardConfig(max_qty=100, max_qty_hard_cap=10).max_qty == 10
    assert OrderGuardConfig(max_qty=0).max_qty == 1
    assert OrderGuardConfig(max_qty_hard_cap=0).max_qty_hard_cap == 1


def test_lists_are_normalized():
    g = OrderGuardConfig(whitelist=frozenset({"A294570", "", " 41830 "}),
                         blocklist=frozenset({"a078930", ""}))
    assert g.whitelist == frozenset({"294570", "041830"})
    assert g.blocklist == frozenset({"078930"})


def test_with_allowed_keeps_the_original_entries():
    g = OrderGuardConfig(whitelist=frozenset({"000001"}))
    assert g.with_allowed({"000002"}).whitelist == frozenset({"000001", "000002"})
    assert g.whitelist == frozenset({"000001"}), "원본은 그대로여야 한다(frozen)"


def test_blocklist_matches_the_real_holding_not_a_lookalike():
    g = OrderGuardConfig(whitelist=frozenset()).with_blocked({"0155E0"})
    assert "0155E0" in g.blocklist
    assert "001550" not in g.blocklist, "엉뚱한 회사를 막으면 안 된다"


# ------------------------------------------------------------------ 보유종목 차단

@pytest.mark.parametrize("code", ["041830", "078930", "222800"])
def test_held_stocks_hard_blocked(code):
    g = OrderGuard(HeldGuardConfig())
    r = g.reason(_buy(code), 10000)
    assert r is not None and ("블록리스트" in r or "보유" in r)


def test_hardcoded_held_blocks_sell_too():
    g = OrderGuard(HeldGuardConfig())
    assert g.reason(_sell("041830"), 10000) is not None


def test_block_message_with_and_without_names():
    cfg = HeldGuardConfig()
    assert (evaluate_order(_buy("041830"), 10000, cfg)
            == "블록리스트 차단(보유/금지 종목) — 041830")
    assert (evaluate_order(_buy("041830", cls=NamedIntent), 10000, cfg)
            == "블록리스트 차단(보유/금지 종목: 인바디) — 041830")
    # 명시 code_names 가 intent 클래스 표보다 우선.
    assert (evaluate_order(_buy("041830"), 10000, cfg, code_names={"041830": "X"})
            == "블록리스트 차단(보유/금지 종목: X) — 041830")


# ------------------------------------------------------------------ 동적 보유(N-1)

def test_dynamic_holdings_block_buy_but_not_sell():
    g = OrderGuard(OrderGuardConfig(whitelist=frozenset({"294570"})))
    assert g.block_buys(["A294570"]) == {"294570"}
    assert g.reason(_buy("294570"), 10000) == "실시간 보유종목 차단(매수) — 294570"
    assert g.reason(_sell("294570"), 10000) is None
    assert "294570" not in g.config.blocklist, "양방향 블록리스트를 건드리면 안 된다"


def test_holdings_reason_merges_and_blocks_buy_only():
    g = OrderGuard(OrderGuardConfig(whitelist=frozenset({"294570"})))
    assert g.holdings_reason(_sell("294570"), {"294570"}) is None
    assert g.dynamic_buy_block == frozenset(), "매도는 병합도 안 한다(원본 조기 반환)"
    assert g.holdings_reason(_buy("294570"), {"294570"}) == "실시간 보유종목 차단 — 294570"
    assert "294570" in g.dynamic_buy_block
    assert g.holdings_reason(_buy("033340"), set()) is None


def test_holdings_block_reason_pure_with_name():
    cfg = HeldGuardConfig()
    assert (holdings_block_reason(_buy("041830", cls=NamedIntent), cfg, frozenset())
            == "실시간 보유종목 차단: 인바디 — 041830")


# ------------------------------------------------------------------ 화이트리스트

def test_whitelist_blocks_unlisted():
    g = OrderGuard(OrderGuardConfig(whitelist=frozenset({"294570"})))
    assert g.reason(_buy("999999"), 10000) == (
        "화이트리스트 외 종목 — 999999 (등록 쌍 후속주만 허용)")
    assert g.reason(_buy("294570"), 10000) is None


def test_allow_codes_extends_whitelist():
    g = OrderGuard(OrderGuardConfig(whitelist=frozenset({"000001"})))
    g.allow_codes(["000002", "000003"])
    assert g.config.whitelist == frozenset({"000001", "000002", "000003"})
    before = g.config
    g.allow_codes([])
    assert g.config is before


# ------------------------------------------------------------------ 수량·가격

def test_qty_checks_in_order():
    g = OrderGuard(OrderGuardConfig(max_qty=1, whitelist=frozenset({"294570"})))
    assert g.reason(_buy("294570", qty=0), 10000) == "수량이 1주 미만: 0"
    assert g.reason(_buy("294570", qty=11), 10000) == "수량 하드캡(10) 초과: 11"
    assert g.reason(_buy("294570", qty=2), 10000) == "수량 상한(1) 초과: 2"
    assert g.reason(_buy("294570", qty=1), 10000) is None


def test_price_band_rejects_outside():
    g = OrderGuard(OrderGuardConfig(price_band_pct=0.03, whitelist=frozenset({"294570"})))
    assert g.reason(_buy("294570", price=10500), 10000) == (
        "지정가 10,500 가 참조가 10,000 대비 5.00% (±3.0% 밖)")
    assert g.reason(_buy("294570", price=10200), 10000) is None
    assert g.reason(_buy("294570", price=10300), 10000) is None, "경계는 통과(eps)"
    assert g.reason(_buy("294570", price=0), 10000) == "지정가가 0 이하: 0"


def test_price_band_skipped_without_ref_or_when_off():
    cfg = OrderGuardConfig(whitelist=frozenset({"294570"}))
    assert evaluate_order(_buy("294570", price=99999), None, cfg) is None
    assert evaluate_order(_buy("294570", price=99999), 0, cfg) is None
    off = OrderGuardConfig(price_band_pct=0, whitelist=frozenset({"294570"}))
    assert evaluate_order(_buy("294570", price=99999), 10000, off) is None


def test_bad_code():
    assert evaluate_order(_buy(""), 10000, OrderGuardConfig()) == "잘못된 종목코드: ''"
    assert evaluate_order(_buy("12345"), 10000, OrderGuardConfig()) == "잘못된 종목코드: '12345'"


def test_check_order_blocklist_before_whitelist_before_qty():
    cfg = HeldGuardConfig(whitelist=frozenset({"294570"}))
    # 보유종목 + 화이트리스트 밖 + 수량 초과 → 블록리스트 사유가 이긴다.
    assert evaluate_order(_buy("041830", qty=50, price=0), 1, cfg).startswith("블록리스트")
    assert evaluate_order(_buy("999999", qty=50), 1, cfg).startswith("화이트리스트")


# ------------------------------------------------------------------ 횟수 상한

def test_order_count_limits():
    g = OrderGuard(OrderGuardConfig(max_orders_per_code=2, max_orders_total=3,
                                    whitelist=frozenset({"294570", "033340"})),
                   clock=_Clock(datetime(2026, 9, 14, 9, 30)))
    assert _submit(g, _buy("294570")) is None
    assert _submit(g, _buy("294570")) is None
    assert _submit(g, _buy("294570")) == "종목당 주문 횟수 상한(2) 초과 — 294570"
    assert _submit(g, _buy("033340")) is None
    assert _submit(g, _buy("033340")) == "일일 주문 횟수 상한(3) 초과"


def test_blocked_orders_are_not_counted():
    g = OrderGuard(OrderGuardConfig(max_orders_total=1, whitelist=frozenset({"294570"})))
    assert _submit(g, _buy("999999")) is not None
    assert g.count_total == 0
    assert _submit(g, _buy("294570")) is None


def test_counters_reset_on_new_day():
    clock = _Clock(datetime(2026, 9, 14, 15, 0))
    g = OrderGuard(OrderGuardConfig(max_orders_total=1, whitelist=frozenset({"294570"})),
                   clock=clock)
    assert _submit(g, _buy("294570")) is None
    assert _submit(g, _buy("294570")) == "일일 주문 횟수 상한(1) 초과"
    clock.t += timedelta(days=1)
    assert _submit(g, _buy("294570")) is None
    assert g.count_day == clock.t.date()


def test_clock_only_consulted_when_counts_are_checked():
    """원본은 앞 단계에서 막히면 날짜를 안 본다 — 시계 호출 횟수까지 같다."""
    calls = []

    def clock():
        calls.append(1)
        return datetime(2026, 9, 14, 10, 0)

    g = OrderGuard(OrderGuardConfig(whitelist=frozenset({"294570"})), clock=clock)
    g.reason(_buy("999999"), 10000)
    assert calls == []
    g.reason(_buy("294570"), 10000)
    assert len(calls) == 1


def test_default_clock_is_kst_aware():
    g = OrderGuard()
    g.observe_day()
    assert g.count_day is not None


def test_count_limit_reason_zero_disables():
    cfg = OrderGuardConfig(max_orders_per_code=0, max_orders_total=0)
    assert count_limit_reason(cfg, "294570", 10_000, {"294570": 10_000}) is None


def test_evaluate_order_uses_passed_counts():
    cfg = OrderGuardConfig(max_orders_per_code=2, whitelist=frozenset({"294570"}))
    assert evaluate_order(_buy("294570"), 10000, cfg, orders_by_code={"294570": 2}) == (
        "종목당 주문 횟수 상한(2) 초과 — 294570")
    assert evaluate_order(_buy("294570"), 10000, cfg, orders_total=20) == (
        "일일 주문 횟수 상한(20) 초과")


# ------------------------------------------------------------------ intent·result

def test_intent_api_id_and_body():
    intent = OrderIntent(side="buy", code="294570", qty=1, price=1928)
    assert intent.api_id() == "kt10000"
    assert OrderIntent(side="sell", code="294570", qty=1, price=1).api_id() == "kt10001"
    assert intent.body() == {"dmst_stex_tp": "KRX", "stk_cd": "294570",
                             "ord_qty": "1", "ord_uv": "1928", "trde_tp": "0"}
    assert list(intent.body()) == ["dmst_stex_tp", "stk_cd", "ord_qty", "ord_uv", "trde_tp"]
    with pytest.raises(ValueError, match="알 수 없는 매매구분: 'weird'"):
        OrderIntent(side="weird", code="294570", qty=1, price=1).api_id()


def test_cancel_body_exact():
    body = cancel_body("0539055", "294570", 1)
    assert body == {"dmst_stex_tp": "KRX", "stk_cd": "294570",
                    "org_ord_no": "0539055", "ord_qty": "1"}
    assert list(body) == ["dmst_stex_tp", "stk_cd", "org_ord_no", "ord_qty"]
    intent = OrderIntent(side="cancel", code="294570", qty=3, price=0, exchange="NXT")
    assert intent.cancel_body(539055) == {"dmst_stex_tp": "NXT", "stk_cd": "294570",
                                          "org_ord_no": "539055", "ord_qty": "3"}


def test_describe_with_and_without_names():
    assert (OrderIntent(side="buy", code="294570", qty=1, price=12500).describe()
            == "매수 지정가  294570  1주 @ 12,500원 [KRX]")
    assert (NamedIntent(side="sell", code="041830", qty=2, price=62900).describe()
            == "매도 지정가  인바디(041830)  2주 @ 62,900원 [KRX]")


def test_result_ok():
    it = _buy("294570")
    assert OrderResult(it, dry_run=True, submitted=False).ok is True
    assert OrderResult(it, dry_run=True, submitted=False, blocked=True).ok is False
    assert OrderResult(it, dry_run=False, submitted=True, return_code=0).ok is True
    assert OrderResult(it, dry_run=False, submitted=True, return_code=1).ok is False
    assert OrderResult(it, dry_run=False, submitted=False, return_code=0).ok is False
    assert OrderResult(it, dry_run=False, submitted=True, return_code=None).ok is False


def test_result_to_record():
    r = OrderResult(NamedIntent(side="buy", code="041830", qty=1, price=10000),
                    dry_run=False, submitted=True, ord_no="0539055", return_code=0)
    rec = r.to_record(datetime(2026, 9, 14, 9, 1, 2, 999))
    assert rec == {
        "ts": "2026-09-14 09:01:02", "side": "buy", "code": "041830", "name": "인바디",
        "qty": 1, "price": 10000, "exchange": "KRX", "dry_run": False, "submitted": True,
        "blocked": False, "blocked_reason": "", "ord_no": "0539055", "return_code": 0,
    }
    assert "ts" in OrderResult(_buy("294570"), True, False).to_record()


def test_order_blocked_is_plain_exception_and_mixable():
    assert issubclass(OrderBlocked, Exception)

    class KiwoomError(Exception):
        pass

    class ShimBlocked(OrderBlocked, KiwoomError):
        pass

    err = ShimBlocked("x")
    assert isinstance(err, KiwoomError) and isinstance(err, OrderBlocked)
