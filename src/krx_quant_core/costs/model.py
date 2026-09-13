"""한국 주식 거래비용 모델 — 세금 + 수수료 + 슬리피지.

두 갈래를 한곳에 둔다.

1. :class:`KoreanCostModel` — daytrade-it ``domain/services/cost_model.py`` 의 이식.
   백테스트와 모의 브로커가 같은 숫자를 쓰게 하려고 만든 ``Decimal`` API 를 **그대로**
   유지한다(메서드 이름·기본값·틱 반올림까지). 달라진 점은 둘뿐이다.

   - 시장 인자로 daytrade 의 ``Exchange`` 대신 :class:`~krx_quant_core.market.codes.Market`
     을 쓰되, ``Market.parse`` 가 받는 무엇이든 받는다. ``Exchange`` 는 값이
     ``"KOSPI"/"KOSDAQ"/"KONEX"`` 인 ``StrEnum`` 이라 그대로 통과한다.
   - ``CostModelConfig`` 의 세율을 ``None`` 으로 두면 :mod:`~krx_quant_core.costs.tax`
     의 일자별 스케줄을 쓴다(``trade_date`` 인자). **기본값은 daytrade 와 같은 명시적
     0.0020** 이라, 설정을 안 바꾼 기존 호출부는 동작이 한 원도 달라지지 않는다.

2. :func:`round_trip_cost` — 수익률에서 바로 빼는 리서치용 float 헬퍼.

기본값 신뢰도(daytrade-it 조사 기록 요약):

- 수수료 0.015% 는 온라인 증권사 대표 수준의 **자리표시값**이다(실제 수수료표 아님).
- 슬리피지 KOSPI 0.05% / KOSDAQ 0.15% 는 실제 호가 스프레드로 **보정하지 않은** 평탄한
  자리표시값이다. KONEX 는 전용 데이터가 없어 KOSPI 값을 쓴다.
- 세율 출처는 :mod:`krx_quant_core.costs.tax` 참고(2차 출처 위주).

scalp-it 에 흩어진 비용 상수와의 관계(**값은 바꾸지 않는다** — 사전등록 규칙과 과거
검증 결과의 의미가 그 숫자에 묶여 있다):

- ``0.0023`` (``closebet/backtest.py`` 왕복, ``validate/*`` 의 ``cost_one_way``):
  세 0.20% + 수수료 0.015%×2 = 0.23% 와 정확히 같다 — 2023년 또는 2026년 세율 기준
  :func:`round_trip_cost` 값이다. ``validate/*`` 는 이름이 ``one_way`` 지만 수치는
  왕복 한 번과 같다는 점에 주의.
- ``0.0034`` (``cli_cbet_watch.COST_ROUND_TRIP``, 여러 sanity 스크립트 "왕복 34bp"):
  세율 0.18%(2024년) 시절에 수수료·기타를 넉넉히 얹어 잡은 반올림 가정치다. 현재
  스케줄로 분해되지 않는다(2024년 :func:`round_trip_cost` 는 0.21%) — 16bp 정도의
  여유분(스프레드·수수료 보수 가정)이 들어 있다고 보는 게 맞다. 주의: 보관된 PEAD
  문서(``docs/archive/pead-migrated-to-kr-quant/combo_book.py``)는 같은 0.0034 를
  "편도 34bp(왕복 68bp)" 로 적었다 — 이 숫자가 편도인지 왕복인지는 파일마다 확인할 것.
- ``0.0064`` (``cli_pair_detect --cost-rate``, ``verify_pair.COST_BASE``):
  위 34bp + 슬리피지 왕복 30bp. ``0.0094`` 는 슬리피지 60bp 비관 시나리오.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from krx_quant_core.costs.tax import sell_tax_rate, tax_rate
from krx_quant_core.market.codes import Market
from krx_quant_core.market.session import now_kst
from krx_quant_core.market.ticks import round_to_tick

__all__ = [
    "DEFAULT_COMMISSION_RATE",
    "DEFAULT_KOSDAQ_SLIPPAGE_PCT",
    "DEFAULT_KOSDAQ_TAX_RATE",
    "DEFAULT_KOSPI_SLIPPAGE_PCT",
    "DEFAULT_KOSPI_TAX_RATE",
    "CostModelConfig",
    "KoreanCostModel",
    "TradeCost",
    "round_trip_cost",
]

#: ``OrderSide`` 같은 인프라 enum 에 의존하지 않으려고 ``"BUY"/"SELL"`` 문자열을 받는다.
_SELL = "SELL"

#: 2026-01-01 개정 KOSPI 0.05% + 농특세 0.15%. 2차 출처 — ``costs/tax.py`` 참고.
DEFAULT_KOSPI_TAX_RATE = Decimal("0.0020")
#: 2026-01-01 개정 KOSDAQ 0.20%(농특세 없음). 2차 출처 — ``costs/tax.py`` 참고.
DEFAULT_KOSDAQ_TAX_RATE = Decimal("0.0020")
#: 대표 온라인 수수료 자리표시값. 매수·매도 양쪽에 붙는다.
DEFAULT_COMMISSION_RATE = Decimal("0.00015")
#: KOSPI(대형주급 유동성) 평탄 슬리피지 자리표시값. 미보정.
DEFAULT_KOSPI_SLIPPAGE_PCT = Decimal("0.0005")
#: KOSDAQ(소형주급 유동성) 평탄 슬리피지 자리표시값. KOSPI 보다 크게, 그러나 미보정.
DEFAULT_KOSDAQ_SLIPPAGE_PCT = Decimal("0.0015")


@dataclass(frozen=True)
class CostModelConfig:
    """:class:`KoreanCostModel` 설정. 모든 비율은 분수(``0.001`` = 0.1%).

    ``kospi_tax_rate``/``kosdaq_tax_rate`` 를 ``None`` 으로 두면 거래일 기준 일자별
    세율 스케줄을 쓴다. KONEX 는 KOSPI 쪽 필드를 쓴다(daytrade-it 동작 보존) — 단
    ``None`` 이면 스케줄상 KONEX 세율(=KOSDAQ, 미검증)을 쓴다.
    """

    kospi_tax_rate: Decimal | None = DEFAULT_KOSPI_TAX_RATE
    kosdaq_tax_rate: Decimal | None = DEFAULT_KOSDAQ_TAX_RATE
    commission_rate: Decimal = DEFAULT_COMMISSION_RATE
    min_commission: Decimal = field(default_factory=lambda: Decimal("0"))
    kospi_slippage_pct: Decimal = DEFAULT_KOSPI_SLIPPAGE_PCT
    kosdaq_slippage_pct: Decimal = DEFAULT_KOSDAQ_SLIPPAGE_PCT


@dataclass(frozen=True)
class TradeCost:
    """체결 1건의 비용 내역.

    ``fill_price`` 는 이미 불리한 방향 슬리피지가 반영된 가격이고, 수수료·세금은
    그 **슬리피지 반영 후** 대금(``fill_price * quantity``)에 붙는다 — 실제 브로커도
    체결가 기준으로 떼기 때문이다.
    """

    fill_price: Decimal
    commission: Decimal
    tax: Decimal

    @property
    def total_cost(self) -> Decimal:
        """수수료 + 세금(슬리피지는 ``fill_price`` 에 들어 있다)."""
        return self.commission + self.tax


class KoreanCostModel:
    """한국 시장 거래비용 모델(세금 + 수수료 + 슬리피지)."""

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self._config = config or CostModelConfig()

    @property
    def config(self) -> CostModelConfig:
        return self._config

    def _slippage_pct(self, market: Any) -> Decimal:
        if Market.parse(market) is Market.KOSDAQ:
            return self._config.kosdaq_slippage_pct
        # KOSPI·KONEX(전용 데이터 없음)는 KOSPI 값. KONEX 는 실제로 더 얇은 시장이라
        # 보수적이지도 않은, 확인 안 된 기본값이다.
        return self._config.kospi_slippage_pct

    def _tax_rate(self, market: Any, trade_date: date | None) -> Decimal:
        m = Market.parse(market)
        configured = (
            self._config.kosdaq_tax_rate if m is Market.KOSDAQ else self._config.kospi_tax_rate
        )
        if configured is not None:
            return configured
        # 날짜를 안 주면 오늘(KST) 세율 — 실시간 모의매매가 이 경로다. 백테스트는
        # 반드시 trade_date 를 넘겨야 과거 세율이 적용된다.
        on = trade_date if trade_date is not None else now_kst().date()
        return tax_rate(on, m).total

    def apply_slippage(self, price: Decimal, side: str, market: Any = Market.KOSPI) -> Decimal:
        """불리한 방향(매수는 위, 매도는 아래)으로 슬리피지를 반영한 가격.

        결과는 호가단위로 **내림**한다 — 50,123.4원 같은 가격은 KRX 에서 체결될 수
        없다. 틱 반올림 때문에 작은 매수 슬리피지는 아예 사라질 수 있다(>= 가 불변식).

        Raises:
            ValueError: ``market`` 을 알 수 없을 때(daytrade 의 ``round_to_tick`` 과 동일).
        """
        slip = price * self._slippage_pct(market)
        raw_price = price - slip if side.upper() == _SELL else price + slip
        return round_to_tick(raw_price)

    def commission(self, notional: Decimal) -> Decimal:
        """위탁수수료 — 매수·매도 양쪽."""
        return max(notional * self._config.commission_rate, self._config.min_commission)

    def transaction_tax(
        self,
        notional: Decimal,
        side: str,
        market: Any = Market.KOSPI,
        trade_date: date | None = None,
    ) -> Decimal:
        """증권거래세(+농특세): 매도에만, 매수는 0."""
        if side.upper() != _SELL:
            return Decimal("0")
        return notional * self._tax_rate(market, trade_date)

    def cost_of_trade(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market: Any = Market.KOSPI,
        trade_date: date | None = None,
    ) -> TradeCost:
        """체결 1건의 전체 비용 내역: 슬리피지(체결가에 반영) + 수수료 + 세금."""
        fill_price = self.apply_slippage(price, side, market)
        notional = fill_price * quantity
        return TradeCost(
            fill_price=fill_price,
            commission=self.commission(notional),
            tax=self.transaction_tax(notional, side, market, trade_date),
        )


def round_trip_cost(
    on: date,
    market: Any,
    *,
    commission_rate: float = 0.00015,
    slippage_one_way: float = 0.0,
) -> float:
    """왕복(매수+매도) 비용률 = 수수료×2 + 매도세 + 슬리피지×2.

    수익률에서 바로 빼는 근사다 — 매도 대금이 매수 대금과 같다고 본다. ``on`` 은
    **매도일**(세율 결정일)을 넘긴다.
    """
    return 2 * commission_rate + sell_tax_rate(on, market) + 2 * slippage_one_way
