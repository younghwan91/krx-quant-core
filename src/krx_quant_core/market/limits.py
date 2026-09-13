"""KRX 가격제한폭(상·하한가).

규칙(2015-06-15 이후 ±30%): 기준가격(보통 전일종가) × 30% 를 더하고 뺀 값을
**기준가격 쪽으로** 호가단위에 맞춘다. 즉 상한가는 틱 내림, 하한가는 틱 올림이다 —
제한폭이 30% 를 넘지 않게 하려는 것이다. 호가단위는 계산된 가격이 속한 가격대의
틱을 쓴다.

float 로 계산하는 이유: scalp-it 실시간 감지기와 ``scripts/limitup_timing.py`` 가
float 식으로 상한가 검증 결과를 만들었고, 여기 값이 **한 치라도** 달라지면 그
검증과 실시간 판정이 어긋나 백테스트가 무의미해진다. 그래서 ``1e-9`` 보정까지
그대로 옮겼다.

기준가격이 전일종가가 아닌 경우(권리락·배당락·신규상장·정리매매 등)는 여기서 다루지
않는다 — 호출부가 올바른 기준가격을 넘겨야 한다.
"""

from __future__ import annotations

import math

from krx_quant_core.market.ticks import tick_size_int

__all__ = ["LIMIT_RATE", "limit_down_price", "limit_up_price", "near_limit_up"]

#: KOSPI·KOSDAQ 일일 가격제한폭(±30%).
LIMIT_RATE = 0.30


def limit_up_price(prev_close: float) -> float:
    """전일종가 × 1.3 을 그 가격대 호가단위로 **내림** = 상한가.

    scalp-it ``realtime/pair_detector.py::limit_up_price`` 의 정확한 이식이다.
    ``1.3`` 리터럴과 ``+1e-9`` 는 바꾸지 않는다: ``1 + LIMIT_RATE`` 로 쓰면 float
    오차 패턴이 달라질 수 있고, ``1e-9`` 는 ``raw/tick`` 이 정수 바로 아래(예
    ``12.999999999``)로 떨어져 한 틱 덜 계산되는 것을 막는다.
    """
    raw = float(prev_close) * 1.3
    tick = tick_size_int(raw)
    return math.floor(raw / tick + 1e-9) * tick


def limit_down_price(prev_close: float) -> float:
    """하한가 = 기준가 − (기준가 × 0.3 을 **기준가의** 호가단위로 절사한 폭).

    상한가(:func:`limit_up_price`)와 **대칭이 아니다.** 처음엔 "기준가×0.7 을 그
    가격대 틱으로 올림"으로 대칭 구현했는데, 2023-02 이후 ``daily_bars`` 에서 저가가
    기준가의 −29~−31% 인 879건에 대조하니 두 규칙이 갈리는 곳에서 이 규칙이 105건,
    대칭 규칙이 25건 맞았다(2026-09-14). 예: 239,000 → 실제 하한가 167,500
    (대칭 규칙은 167,300), 24,250 → 17,000(대칭 16,980).

    기준가가 유효 호가이면 결과도 유효 호가다 — 호가단위는 아래 밴드 틱의 배수라
    기준가 틱 배수만큼 빼도 아래 밴드에서 여전히 유효하다.
    """
    base = float(prev_close)
    tick = tick_size_int(base)
    return base - math.floor(base * LIMIT_RATE / tick + 1e-9) * tick


def near_limit_up(price: float, prev_close: float, ticks: int = 3) -> bool:
    """``price`` 가 상한가 ``ticks`` 호가 이내(상한가 이상 포함)인가.

    scalp-it ``morning_report.in_trigger_zone`` 의 거리 판정
    ``price >= limit - ticks * tick_size(limit)`` 와 같다. 차이는 **이미 상한가에
    닿은 가격도 True** 라는 점 — "아직 안 잠김" 조건은 전략 판단이라 호출부가
    ``price < limit_up_price(prev_close)`` 를 따로 붙인다. 가격·전일종가가 0 이하이면
    False.
    """
    if prev_close <= 0 or price <= 0:
        return False
    limit = limit_up_price(prev_close)
    return price >= limit - ticks * tick_size_int(limit)
