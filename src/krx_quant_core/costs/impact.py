"""크기 의존 거래비용 — 제곱근 충격 법칙 + 정가 수수료·일자별 매도세·반스프레드.

:mod:`.model` 의 비용은 주문 크기와 무관한 평탄값이다. 다수 종목을 ADV 대비 크게 사고파는 북은
충격 비용이 주문 크기에 따라 늘어나므로, 여기서 제곱근 법칙을 더한다.

- 충격: ``Y · σ_daily · √(Q / ADV)`` (Tóth et al. 2011, "Anomalous price impact and the critical
  nature of liquidity in financial markets" — Y 는 시장·기간을 가리지 않고 O(1)). 기본 ``y=1.0`` 은
  문헌 중앙값을 가져온 **보정 전 자리표시값**이다 — KRX 체결 데이터로 맞춘 숫자가 아니다.
  소비 레포는 ``y`` 를 1·2배로 돌려 결론이 비용 가정에 버티는지 본다.
- 수수료: :data:`~krx_quant_core.costs.model.DEFAULT_COMMISSION_RATE`(정가 자리표시값, 매수·매도
  양쪽). 매도세: :func:`~krx_quant_core.costs.tax.sell_tax_rate`(거래일 기준). 할인·면제·리베이트는
  넣지 않는다(정가 원칙) — 세율·수수료를 여기서 다시 정의하지 않는다.

단위(사고가 났던 곳이라 함수가 막는다):

- ``notional``·``adv_value`` 는 **원**. ``daily_bars.trade_value`` 는 **백만원**이라 받는 쪽이
  ``×1e6`` 해야 한다. 그대로 넣은 사고가 두 번 있었다(quant-airflow CLAUDE.md §3) — ADV 가
  1e6 원(=100만원) 미만이면 단위 실수로 보고 ``ValueError``.
- ``daily_vol`` 은 일간 수익률 표준편차의 **분수**(``0.02`` = 2%). 퍼센트(``2.0``)를 넣는
  실수를 막으려고 1 을 넘으면 ``ValueError`` — KRX 가격제한폭(±30%) 아래에서 일간 변동성
  100% 는 나올 수 없다.
- 반환 bp 는 **거래대금 대비**(``1 bp = 1e-4``). 원화 비용 = ``bp / 1e4 · notional``.

NaN 은 지우지 않고 그대로 흘린다(ADV 를 모르는 종목의 비용은 NaN — 0 이 아니다).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from krx_quant_core.costs.model import DEFAULT_COMMISSION_RATE
from krx_quant_core.costs.tax import sell_tax_rate

__all__ = ["MIN_ADV_VALUE", "participation", "sqrt_impact_bp", "trade_cost_bp"]

#: ADV(원) 하한 가드. 이보다 작으면 백만원 단위를 그대로 넣은 것으로 본다.
MIN_ADV_VALUE = 1e6

_BP = 1e4


def _adv(adv_value: Any) -> np.ndarray:
    adv = np.asarray(adv_value, dtype=np.float64)
    # NaN 비교는 False 라 결측은 통과하고 결과에서 NaN 으로 남는다.
    bad = adv < MIN_ADV_VALUE
    if np.any(bad):
        raise ValueError(
            f"adv_value 는 원 단위여야 한다 — {MIN_ADV_VALUE:.0e} 원 미만 {int(np.sum(bad))}개"
            f"(최솟값 {np.nanmin(adv[bad]):.6g}). "
            "daily_bars.trade_value(백만원)는 ×1e6 해서 넘긴다."
        )
    return adv


def _vol(daily_vol: Any) -> np.ndarray:
    vol = np.asarray(daily_vol, dtype=np.float64)
    if np.any(vol < 0):
        raise ValueError("daily_vol 은 0 이상이어야 한다")
    if np.any(vol > 1.0):
        raise ValueError(
            "daily_vol 은 분수(0.02 = 2%)여야 한다 — 1 을 넘는 값이 있다(퍼센트로 넣었나?)"
        )
    return vol


def participation(notional: Any, adv_value: Any) -> np.ndarray:
    """참여율 ``|notional| / ADV`` (분수, ``0.05`` = ADV 의 5%).

    최적화의 참여율 제약(:func:`krx_quant_core.portfolio.optimize`)과 사후 보고에 같은 식을 쓴다.

    Raises:
        ValueError: ``adv_value`` 에 1e6 원 미만 값이 있을 때(백만원 단위 실수).
    """
    adv = _adv(adv_value)
    return np.abs(np.asarray(notional, dtype=np.float64)) / adv


def sqrt_impact_bp(
    notional: Any,
    adv_value: Any,
    daily_vol: Any,
    *,
    y: float = 1.0,
) -> np.ndarray:
    """제곱근 법칙 충격 비용(bp, 거래대금 대비) ``Y · σ_daily · √(|Q| / ADV) · 1e4``.

    매수·매도 대칭이고 ``notional`` 의 부호는 보지 않는다. 동차성: 주문을 ``s`` 배 하면 bp 는
    ``√s`` 배, 원화 비용은 ``s^{3/2}`` 배 — 최적화의 ``k_i |Δw_i|^{3/2}`` 항이 이 식이다.

    Args:
        notional: 주문 대금(원). 스칼라·배열 모두 브로드캐스트.
        adv_value: 평균 일 거래대금(원).
        daily_vol: 일간 수익률 표준편차(분수).
        y: 충격 계수 ``Y``. 기본 1.0 은 **보정 전 자리표시값**(모듈 설명).

    Raises:
        ValueError: ADV 가 1e6 원 미만·``daily_vol`` 이 음수이거나 1 초과·``y`` 가 음수일 때.
    """
    if y < 0:
        raise ValueError("y 는 0 이상이어야 한다")
    adv = _adv(adv_value)
    vol = _vol(daily_vol)
    q = np.abs(np.asarray(notional, dtype=np.float64))
    return y * vol * np.sqrt(q / adv) * _BP


def trade_cost_bp(
    side: Any,
    notional: Any,
    adv_value: Any,
    daily_vol: Any,
    market: Any,
    trade_date: date,
    *,
    half_spread_bp: Any,
    y: float = 1.0,
) -> np.ndarray:
    """편도 거래비용(bp, 거래대금 대비) = 수수료 + 매도세 + 반스프레드 + 충격.

    - 수수료 :data:`~krx_quant_core.costs.model.DEFAULT_COMMISSION_RATE` — 매수·매도 양쪽.
    - 매도세 :func:`~krx_quant_core.costs.tax.sell_tax_rate` ``(trade_date, market)`` —
      ``SELL`` 에만.
    - 반스프레드 ``half_spread_bp`` — 호출부가 준다(종목별 배열 가능). 기본값이 없는 것은 일부러다:
      0 을 기본으로 두면 스프레드를 잊은 백테스트가 조용히 낙관적이 된다.
    - 충격 :func:`sqrt_impact_bp`.

    Args:
        side: ``"BUY"``/``"SELL"``(대소문자 무관). 스칼라 또는 종목별 배열.
        notional: 주문 대금(원, 부호 무시).
        adv_value: 평균 일 거래대금(원).
        daily_vol: 일간 수익률 표준편차(분수).
        market: ``Market.parse`` 가 받는 값. 스칼라 또는 종목별 배열.
        trade_date: 체결일(세율 결정일).
        half_spread_bp: 반스프레드(bp, 0 이상).
        y: 충격 계수(:func:`sqrt_impact_bp`).

    Raises:
        ValueError: ``side`` 가 BUY/SELL 이 아니거나 ``half_spread_bp`` 가 음수일 때, 그리고
            :func:`sqrt_impact_bp` 의 단위 가드.
    """
    sides = np.char.upper(np.asarray(side, dtype=str))
    unknown = ~np.isin(sides, ("BUY", "SELL"))
    if np.any(unknown):
        raise ValueError(f"side 는 'BUY'/'SELL' 이어야 한다: {np.unique(sides[unknown]).tolist()}")
    hs = np.asarray(half_spread_bp, dtype=np.float64)
    if np.any(hs < 0):
        raise ValueError("half_spread_bp 는 0 이상이어야 한다")

    markets = np.asarray(market, dtype=object)
    if markets.ndim == 0:
        tax_bp: np.ndarray | float = sell_tax_rate(trade_date, markets.item()) * _BP
    else:
        flat = markets.ravel().tolist()
        rates = {m: sell_tax_rate(trade_date, m) * _BP for m in set(flat)}
        tax_bp = np.array([rates[m] for m in flat], dtype=np.float64).reshape(markets.shape)

    commission_bp = float(DEFAULT_COMMISSION_RATE) * _BP
    impact = sqrt_impact_bp(notional, adv_value, daily_vol, y=y)
    return commission_bp + np.where(sides == "SELL", tax_bp, 0.0) + hs + impact
