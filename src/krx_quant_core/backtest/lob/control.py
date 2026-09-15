"""무작위 진입 대조군 — 같은 종목·날, 같은 보유 시간, 진입 초만 무작위.

scalp-it 82번 ``common.random_control`` 의 이식(사전등록 §6-6). 전략 거래와 짝을 지어
"신호가 진입 시점을 고른 값어치"만 떼어 본다 — 보유 시간 분포와 종목·날 구성은 그대로다.

난수 소비 순서가 원본과 같다: 거래 순서대로 한 번씩 ``rng.choice``, 후보가 없으면 뽑지
않는다. 그래서 같은 시드면 원본과 같은 대조 수익률이 나온다.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["DecisionBook", "random_entry_control"]


class DecisionBook(Protocol):
    """한 에피소드(종목·날)의 결정 행. 세 배열 길이가 같고 ``secs`` 는 오름차순."""

    @property
    def secs(self) -> NDArray[np.integer]: ...  # 결정 초
    @property
    def buy(self) -> NDArray[np.floating]: ...  # 그 행에서 사면 체결가(다음 초 매도1호가)
    @property
    def sell(self) -> NDArray[np.floating]: ...  # 그 행에서 팔면 체결가(다음 초 매수1호가)


def random_entry_control(
    keys: Sequence[Hashable],
    holds: Sequence[int] | NDArray[np.integer],
    books: Mapping[Hashable, DecisionBook],
    *,
    cost: float,
    rng: np.random.Generator | int = 82,
) -> NDArray[np.float64]:
    """거래마다 대조 순수익 하나.

    거래 ``k`` 에 대해 ``books[keys[k]]`` 에서 ``secs[i] + hold <= secs[-1]`` 인 행 ``i`` 를
    무작위로 고르고, ``secs[i] + hold`` 이상인 첫 행 ``j`` 에서 판다:
    ``sell[j] / buy[i] − 1 − cost``. 후보가 없으면 nan.

    Args:
        keys: 거래별 에피소드 키(예 ``(day, code)``), 거래 순서대로.
        holds: 거래별 보유 초.
        books: 키 → :class:`DecisionBook`.
        cost: 왕복 비용률(:func:`krx_quant_core.costs.round_trip_cost`).
        rng: ``Generator`` 또는 시드(원본 82).
    """
    if len(keys) != len(holds):
        raise ValueError("keys and holds must have the same length")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    out = np.full(len(keys), np.nan)
    for k, (key, hold_) in enumerate(zip(keys, holds, strict=True)):
        book = books[key]
        secs = np.asarray(book.secs)
        hold = int(hold_)
        cand = np.nonzero(secs + hold <= secs[-1])[0]
        if len(cand) == 0:
            continue
        i = int(gen.choice(cand))
        j = min(int(np.searchsorted(secs, secs[i] + hold)), len(secs) - 1)
        out[k] = float(book.sell[j]) / float(book.buy[i]) - 1.0 - cost
    return out
