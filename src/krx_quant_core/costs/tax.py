"""증권거래세 일자별 스케줄(매도 시에만 부과).

세율이 몇 년 새 여러 번 바뀌었다(0.25% → 0.23% → 0.20% → 0.18% → 0.15% → 0.20%).
백테스트에 "현재 세율" 하나를 박으면 2024년 거래는 과대, 2025년 거래는 과소 비용이
된다. 그래서 **거래일 기준**으로 세율을 고른다.

KOSPI 는 증권거래세 + 농어촌특별세(0.15%, 매도대금 기준)로 나뉘고, KOSDAQ 은
농특세 없이 증권거래세만 있다.

출처와 신뢰도(솔직하게):

- **2026-01-01 인상**(KOSPI 0.05%+0.15%, KOSDAQ 0.20%): 기획재정부 증권거래세법
  시행령 개정 보도로 확인 — 헤럴드경제 "내년부터 증권거래세율 0.05%P 상향",
  머니투데이 2025-12-30 "손절해도 '세금'...증권거래세 부활에 두 번 우는 단타 개미들".
  그래도 법령 원문(law.go.kr) 대조는 아니다.
- **그 이전 단계**(2019-06-03·2021·2023·2024·2025): 나무위키 "증권거래세" 연혁표
  기준의 **2차 출처**다. 1차 법령 텍스트로 확인하지 않았다.
- **KONEX**: 실제 세율(역사적으로 0.10% 로 알려짐)을 확인하지 못해 **KOSDAQ 과 같게
  모델링했다(미검증).** KONEX 를 실제로 거래·검증하기 전에 반드시 확인할 것.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from krx_quant_core.market.codes import Market

__all__ = ["TAX_SCHEDULE", "TaxRate", "sell_tax_rate", "tax_rate"]


@dataclass(frozen=True)
class TaxRate:
    """매도대금 대비 세율(분수, ``0.0015`` = 0.15%)."""

    securities_tax: Decimal
    rural_special_tax: Decimal = Decimal("0")

    @property
    def total(self) -> Decimal:
        return self.securities_tax + self.rural_special_tax


@dataclass(frozen=True)
class _TaxPeriod:
    effective: date
    kospi: TaxRate
    kosdaq: TaxRate


def _bp(securities: str, rural: str = "0") -> TaxRate:
    """퍼센트 문자열 → 분수 ``TaxRate`` (``"0.05"`` → ``Decimal("0.0005")``)."""
    return TaxRate(Decimal(securities) / 100, Decimal(rural) / 100)


#: (시행일, KOSPI, KOSDAQ) — 시행일 오름차순. 첫 행의 ``date.min`` 은 "그 이전 전부".
TAX_SCHEDULE: tuple[_TaxPeriod, ...] = (
    _TaxPeriod(date.min, _bp("0.15", "0.15"), _bp("0.30")),
    _TaxPeriod(date(2019, 6, 3), _bp("0.10", "0.15"), _bp("0.25")),
    _TaxPeriod(date(2021, 1, 1), _bp("0.08", "0.15"), _bp("0.23")),
    _TaxPeriod(date(2023, 1, 1), _bp("0.05", "0.15"), _bp("0.20")),
    _TaxPeriod(date(2024, 1, 1), _bp("0.03", "0.15"), _bp("0.18")),
    _TaxPeriod(date(2025, 1, 1), _bp("0.00", "0.15"), _bp("0.15")),
    _TaxPeriod(date(2026, 1, 1), _bp("0.05", "0.15"), _bp("0.20")),
)
_EFFECTIVE = [p.effective for p in TAX_SCHEDULE]


def tax_rate(on: date, market: Any) -> TaxRate:
    """``on`` 날짜 매도에 적용되는 세율. ``market`` 은 ``Market.parse`` 가 받는 무엇이든.

    ``datetime`` 을 넘기면 날짜 부분만 쓴다. KONEX 는 KOSDAQ 세율(미검증, 모듈 설명 참고).
    """
    day = on.date() if hasattr(on, "date") and callable(on.date) else on
    period = TAX_SCHEDULE[bisect.bisect_right(_EFFECTIVE, day) - 1]
    m = Market.parse(market)
    return period.kospi if m is Market.KOSPI else period.kosdaq


def sell_tax_rate(on: date, market: Any) -> float:
    """리서치 코드용 float 총 세율(증권거래세 + 농특세)."""
    return float(tax_rate(on, market).total)
