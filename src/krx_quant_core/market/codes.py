"""KRX 종목코드와 시장 구분.

``normalize_code`` 는 scalp-it(``adapters/kiwoom_order.py``)과 daytrade-it
(``brokers/kiwoom_broker.py``)에 같은 구현이 두 벌 있던 것을 여기로 모았다.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

__all__ = ["Market", "is_valid_code", "normalize_code"]

_CODE_RE = re.compile(r"^[0-9A-Z]{6}$")


class Market(StrEnum):
    """KRX 주식 시장. 값은 영문 대문자라 로그·설정에서 그대로 쓴다."""

    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"
    KONEX = "KONEX"

    @classmethod
    def parse(cls, value: Any) -> Market:
        """``"코스닥"``·``"kosdaq"``·``"거래소"`` 같은 표기를 :class:`Market` 으로.

        quant-airflow ``stocks.market`` 은 한글(``거래소``/``코스닥``/``코넥스``)이고
        daytrade-it 은 영문 enum 을 쓴다 — 둘 다 받는다. 모르는 값은 ``ValueError``.
        """
        if isinstance(value, cls):
            return value
        text = str(value).strip().upper()
        aliases = {
            "KOSPI": cls.KOSPI, "거래소": cls.KOSPI, "유가증권": cls.KOSPI, "코스피": cls.KOSPI,
            "KOSDAQ": cls.KOSDAQ, "코스닥": cls.KOSDAQ,
            "KONEX": cls.KONEX, "코넥스": cls.KONEX,
        }
        try:
            return aliases[text]
        except KeyError:
            raise ValueError(f"unknown KRX market: {value!r}") from None


def normalize_code(code: Any) -> str:
    """종목코드를 6자리로 정규화한다.

    kt00018 응답은 ``"A041830"`` 처럼 알파벳 **접두**가 붙는다. 그것만 떼고 6자리를
    그대로 둔다.

    **알파벳을 무조건 지우면 안 된다.** KRX 코드에는 가운데에 알파벳이 들어가는
    것이 실재한다(2026-08-27 기준 55종목). 숫자만 남기는 옛 구현은
    ``0155E0``(해치텍) → ``001550``(조비) 로 **완전히 다른 회사 코드**를 만들었다.
    보유종목 차단이 걸린 실주문 경로라 조용히 넘길 자리가 아니다.
    """
    text = str(code if code is not None else "").strip().upper()
    text = "".join(ch for ch in text if ch.isalnum())
    if not text:
        return ""
    # 7자리 + 알파벳 접두 = kt00018 형식("A041830"). 접두 한 글자만 뗀다.
    if len(text) == 7 and text[0].isalpha():
        text = text[1:]
    if len(text) > 6:
        text = text[-6:]
    # 숫자만인 코드는 6자리로 채운다("41830" → "041830"). 알파벳이 섞인 코드는
    # 이미 6자리 규격이라 그대로 둔다.
    return text.zfill(6) if text.isdigit() else text


def is_valid_code(code: Any) -> bool:
    """정규화 **없이** 이미 6자리 KRX 코드 형식인지(숫자·대문자 6자)."""
    return isinstance(code, str) and bool(_CODE_RE.match(code))
