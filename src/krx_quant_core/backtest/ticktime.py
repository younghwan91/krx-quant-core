"""불규칙 틱 시각 위의 창 연산 — 초 격자로 모으기 전 원시 체결열에서 바로.

:mod:`.lob` 은 초 격자(빈 초 포함 고정 간격) 위 커널이다. 체결 단위 연구(scalp-it
``tick_sanity``·``filter_counterfactual``·``volume_spike_scalp``)는 틱 시각 그대로 "직전 W초"·
"다음 H초" 창을 쓰는데, 그 두 포인터·덱 루프가 순수 파이썬이라 하루 233종목에 수십 초가 걸렸다.
같은 알고리즘·같은 덧셈 순서를 numba 로 옮겨 **비트 단위로 같은 값**을 낸다(scalp-it 원본 대조
테스트, 13만 틱 기준 직전 창 21×·미래 창 119×).

시각 ``ts`` 는 오름차순 정수·실수(초)면 된다 — :class:`krx_quant_core.data.CodeDay` 의 ``sec``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .lob._jit import njit

__all__ = ["forward_max_last", "trailing_sums"]


@njit
def _trailing(
    ts: NDArray[np.float64], vals: NDArray[np.float64], window: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    k, n = vals.shape
    out = np.empty((k, n))
    count = np.zeros(n, np.int64)
    acc = np.zeros(k)
    left = 0
    for i in range(n):
        for j in range(k):
            acc[j] += vals[j, i]
        lo = ts[i] - window
        while ts[left] <= lo:
            for j in range(k):
                acc[j] -= vals[j, left]
            left += 1
        count[i] = i - left + 1
        for j in range(k):
            out[j, i] = acc[j]
    return out, count


def trailing_sums(
    ts: Any, *values: Any, window: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """각 틱 ``i`` 로 끝나는 창 ``(ts[i] - window, ts[i]]`` 의 값 합과 건수.

    누적 덧셈·창 밖 뺄셈을 원본(scalp-it ``tick_sanity.trailing_features``)과 같은 순서로 하므로
    합이 비트 단위로 같다 — 평균·비율은 호출부가 ``sums / count`` 처럼 같은 식으로 낸다.

    Returns:
        ``(sums, count)`` — ``sums`` 는 ``(len(values), n)``, ``count`` 는 int64 ``(n,)``.
    """
    t = np.ascontiguousarray(ts, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("ts must be 1-D")
    if len(t) > 1 and (np.diff(t) < 0).any():
        raise ValueError("ts must be non-decreasing")
    v = np.ascontiguousarray(np.vstack([np.asarray(x, np.float64) for x in values]))
    if v.shape[1] != len(t):
        raise ValueError("every value array must have len(ts) entries")
    return _trailing(t, v, float(window))  # type: ignore[no-any-return]


@njit
def _forward(
    ts: NDArray[np.float64], price: NDArray[np.float64], horizon: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    n = ts.shape[0]
    fmax = np.full(n, np.nan)
    last = np.full(n, -1, np.int64)
    dq = np.empty(n, np.int64)  # (i, r] 구간 인덱스, price 내림차순
    head = 0
    tail = 0
    r = 0
    for i in range(n):
        hi = ts[i] + horizon
        while r < n and ts[r] <= hi:
            if r > i:
                while tail > head and price[dq[tail - 1]] <= price[r]:
                    tail -= 1
                dq[tail] = r
                tail += 1
            r += 1
        while tail > head and dq[head] <= i:
            head += 1
        if r - 1 > i:
            last[i] = r - 1
            if tail > head:
                fmax[i] = price[dq[head]]
    return fmax, last


def forward_max_last(
    ts: Any, price: Any, horizon: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """각 틱 ``i`` 의 미래 창 ``(ts[i], ts[i] + horizon]`` 최고가와 마지막 틱 인덱스. O(n).

    창에 틱이 없으면 ``nan``·``-1``. MFE·종가수익은 ``fmax / entry - 1``,
    ``price[last] / entry - 1`` (scalp-it ``tick_sanity.forward_labels_all`` 과 같은 값).
    **미래를 본다** — 라벨 전용.
    """
    t = np.ascontiguousarray(ts, dtype=np.float64)
    p = np.ascontiguousarray(price, dtype=np.float64)
    if t.shape != p.shape or t.ndim != 1:
        raise ValueError("ts and price must be 1-D of equal length")
    if len(t) > 1 and (np.diff(t) < 0).any():
        raise ValueError("ts must be non-decreasing")
    return _forward(t, p, float(horizon))  # type: ignore[no-any-return]
