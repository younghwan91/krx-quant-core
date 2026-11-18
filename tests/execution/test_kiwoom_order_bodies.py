"""주문·정정·취소 바디 ↔ 키움 공식 스펙 대조 — 네트워크 없음.

정답지는 ``data/kiwoom_order_spec_c7a3d25.json`` — 공식 ``Kiwoom-Securities/Kiwoom-REST-API``
``kiwoom/_data/kiwoom_api_spec.json`` c7a3d25 에서 kt10000~kt10003 의 ``request.body`` 만 뽑은
사본이다. 코어의 필드 표(:data:`ORDER_BODY_FIELDS`)를 손으로 옮겨 적었으니, 그 표가 아니라
이 사본과 비교해야 옮겨 적은 실수가 잡힌다. 취소 14/14 ``rc=2`` 는 ``org_ord_no``·``ord_qty``
(스펙에 없는 이름)를 보낸 탓으로 보인다 — 실주문 스모크로는 아직 확인 전이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from krx_quant_core.execution import (
    InvalidOrderBody,
    KiwoomBroker,
    OrderIntent,
    OrderResult,
    cancel_body,
    modify_body,
    validate_order_body,
)
from krx_quant_core.execution import kiwoom_spec as spec

SPEC = json.loads((Path(__file__).parent / "data" / "kiwoom_order_spec_c7a3d25.json").read_text())
TRS = SPEC["trs"]


def _spec_fields(api_id: str) -> list[tuple[str, bool]]:
    return [(f["element"], f["required"] == "Y") for f in TRS[api_id]["body"]]


def _required(api_id: str) -> list[str]:
    return [name for name, req in _spec_fields(api_id) if req]


def test_fixture_is_the_official_c7a3d25_extract():
    assert "Kiwoom-REST-API/blob/c7a3d25/" in SPEC["source"]
    assert SPEC["sha256"] == "42a7b3912c9d9588c83bdc2db7779c8d2e038703a2b5562e54ef46ae905cba79"
    assert sorted(TRS) == ["kt10000", "kt10001", "kt10002", "kt10003"]
    assert {tr["url"] for tr in TRS.values()} == {spec.ORDER_PATH}


@pytest.mark.parametrize("api_id", ["kt10000", "kt10001", "kt10002", "kt10003"])
def test_core_field_table_matches_official_spec(api_id):
    assert list(spec.ORDER_BODY_FIELDS[api_id]) == _spec_fields(api_id)


def test_api_ids():
    assert spec.MODIFY_API_ID == "kt10002"
    assert spec.CANCEL_API_ID == "kt10003"
    assert set(spec.ORDER_BODY_FIELDS) == {
        spec.BUY_API_ID, spec.SELL_API_ID, spec.MODIFY_API_ID, spec.CANCEL_API_ID}


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_order_body_has_spec_required_fields(side):
    intent = OrderIntent(side=side, code="005930", qty=1, price=70000)
    body = intent.body()
    api_id = intent.api_id()
    assert set(_required(api_id)) <= set(body)
    assert set(body) <= {name for name, _ in _spec_fields(api_id)}
    validate_order_body(api_id, body)


def test_cancel_body_matches_spec_required_fields_exactly():
    body = cancel_body("0000140", "005930", 0)  # 0 = 잔량 전부 취소(스펙 description)
    assert list(body) == _required("kt10003") == [
        "dmst_stex_tp", "orig_ord_no", "stk_cd", "cncl_qty"]
    assert body == {"dmst_stex_tp": "KRX", "orig_ord_no": "0000140",
                    "stk_cd": "005930", "cncl_qty": "0"}
    assert "잔량 전부 취소" in TRS["kt10003"]["body"][3]["description"]


def test_modify_body_matches_spec_required_fields():
    body = modify_body("0000139", "005930", 1, 199700)
    assert list(body) == _required("kt10002") == [
        "dmst_stex_tp", "orig_ord_no", "stk_cd", "mdfy_qty", "mdfy_uv"]
    assert body == {"dmst_stex_tp": "KRX", "orig_ord_no": "0000139", "stk_cd": "005930",
                    "mdfy_qty": "1", "mdfy_uv": "199700"}
    with_cond = modify_body("0000139", "005930", 1, 199700, exchange="NXT", cond_price=199000)
    assert list(with_cond) == [name for name, _ in _spec_fields("kt10002")]
    assert with_cond["dmst_stex_tp"] == "NXT" and with_cond["mdfy_cond_uv"] == "199000"
    intent = OrderIntent(side="buy", code="005930", qty=2, price=70100)
    assert intent.modify_body("0000139") == {
        "dmst_stex_tp": "KRX", "orig_ord_no": "0000139", "stk_cd": "005930",
        "mdfy_qty": "2", "mdfy_uv": "70100"}


@pytest.mark.parametrize("api_id", ["kt10002", "kt10003"])
def test_validation_rejects_old_org_ord_no_and_ord_qty(api_id):
    old = {"dmst_stex_tp": "KRX", "stk_cd": "005930", "org_ord_no": "0000140", "ord_qty": "1"}
    with pytest.raises(InvalidOrderBody, match="스펙에 없는 필드: \\['org_ord_no', 'ord_qty'\\]"):
        validate_order_body(api_id, old)


def test_validation_rejects_missing_empty_unknown_tr():
    good = cancel_body("0000140", "005930", 1)
    for name in _required("kt10003"):
        missing = {k: v for k, v in good.items() if k != name}
        with pytest.raises(InvalidOrderBody, match=f"필수 필드 누락·빈 값: \\['{name}'\\]"):
            validate_order_body("kt10003", missing)
        with pytest.raises(InvalidOrderBody, match="필수 필드 누락"):
            validate_order_body("kt10003", {**good, name: ""})
    with pytest.raises(InvalidOrderBody, match="필드 표에 없는 주문 TR: 'kt99999'"):
        validate_order_body("kt99999", good)
    with pytest.raises(InvalidOrderBody, match="필수 필드 누락·빈 값: \\['mdfy_uv'\\]"):
        validate_order_body("kt10002", {**modify_body("1", "005930", 1, 1), "mdfy_uv": ""})
    # 선택 필드는 비어도 된다(공식 예시가 ord_uv·cond_uv 를 "" 로 보낸다).
    validate_order_body("kt10000", {"dmst_stex_tp": "KRX", "stk_cd": "005930", "ord_qty": "1",
                                    "ord_uv": "", "trde_tp": "3", "cond_uv": ""})
    assert issubclass(InvalidOrderBody, ValueError)


class _OldCancelIntent(OrderIntent):
    """옛 취소 바디를 그대로 들고 있는 서브클래스 — 소비 레포가 이렇게 덮어쓸 수 있다."""

    def cancel_body(self, org_ord_no: str) -> dict[str, str]:
        return {"dmst_stex_tp": self.exchange, "stk_cd": self.code,
                "org_ord_no": str(org_ord_no), "ord_qty": str(self.qty)}


class _OldBuyIntent(OrderIntent):
    def body(self) -> dict[str, str]:
        return {**super().body(), "ord_dvsn": "00"}


def _api(calls: list):
    def rec(**kw):
        calls.append(kw)
        return {"return_code": 0, "ord_no": "0000141"}

    return SimpleNamespace(order=SimpleNamespace(buy_order=rec, sell_order=rec, cancel_order=rec))


@pytest.mark.parametrize("dry_run", [False, True])
def test_broker_rejects_bad_bodies_before_network(dry_run):
    calls: list = []
    broker = KiwoomBroker(_api(calls), dry_run=dry_run)
    with pytest.raises(InvalidOrderBody, match="org_ord_no"):
        broker.cancel("0000140", _OldCancelIntent(side="sell", code="005930", qty=1, price=0))
    with pytest.raises(InvalidOrderBody, match="ord_dvsn"):
        broker.submit(_OldBuyIntent(side="buy", code="005930", qty=1, price=70000))
    assert calls == []


def test_broker_cancel_sends_official_body():
    calls: list = []
    broker = KiwoomBroker(_api(calls), dry_run=False)
    result = broker.cancel("0000140", OrderIntent(side="sell", code="005930", qty=1, price=0))
    assert result.ok
    assert calls == [{"dmst_stex_tp": "KRX", "orig_ord_no": "0000140",
                      "stk_cd": "005930", "cncl_qty": "1"}]


def test_return_code_and_msg_kept_in_record():
    def reject(**kw):
        return {"return_code": 2, "return_msg": "[2000](RC4007:필수입력값 누락)"}

    api = SimpleNamespace(order=SimpleNamespace(cancel_order=reject))
    result = KiwoomBroker(api, dry_run=False).cancel(
        "0000140", OrderIntent(side="sell", code="005930", qty=1, price=0))
    rec = result.to_record()
    assert rec["return_code"] == 2
    assert rec["return_msg"] == "[2000](RC4007:필수입력값 누락)"
    ok = OrderResult(OrderIntent(side="buy", code="005930", qty=1, price=1),
                     dry_run=False, submitted=True, return_code=0, return_msg="정상처리")
    assert ok.to_record()["return_code"] == 0
    assert ok.to_record()["return_msg"] == "정상처리"
