"""미래 창 최댓값·최솟값 — 단조 덱 O(n) 커널(창 길이와 무관).

``out[t] = nanmax(x[t+lo : t+lo+h])`` (뒤는 ``n`` 에서 자른다, 창 안이 전부 nan 이거나 비면
nan). 원본들은 지평 ``h`` 만큼 배열 전체를 ``fmax`` 로 겹쳐 O(n·h) 였다 — 7,200초 지평이면
하루 한 종목에 배열 패스 7,200번이다. 최댓값은 **고르는** 연산이라 더하는 순서가 없어서
덱으로 바꿔도 결과가 비트 단위로 같다(골든 테스트가 원본 식과 대조한다).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._jit import njit

__all__ = ["forward_extreme"]


@njit
def _forward_extreme(x: NDArray[np.float64], lo: int, h: int, largest: bool) -> NDArray[np.float64]:
    n = x.shape[0]
    out = np.full(n, np.nan)
    if h <= 0:
        return out
    dq = np.empty(n, np.int64)  # 창 안 후보 인덱스. head 가 가장 큰 인덱스(먼저 만료)
    head = 0
    tail = 0  # [head, tail) — tail 쪽이 가장 최근에 넣은(작은) 인덱스
    # t 를 거꾸로 돌며 창 [t+lo, t+lo+h-1] 을 왼쪽으로 민다.
    for t in range(n - 1, -1, -1):
        i = t + lo
        if i < n:
            v = x[i]
            if v == v:
                if largest:
                    while tail > head and x[dq[tail - 1]] <= v:
                        tail -= 1
                else:
                    while tail > head and x[dq[tail - 1]] >= v:
                        tail -= 1
                dq[tail] = i
                tail += 1
        last = t + lo + h - 1
        while tail > head and dq[head] > last:
            head += 1
        if tail > head:
            out[t] = x[dq[head]]
    return out


def forward_extreme(
    x: NDArray[np.float64], lo: int, h: int, *, largest: bool
) -> NDArray[np.float64]:
    """``out[t]`` = ``x[t+lo : t+lo+h]`` 의 nan 무시 최댓값(``largest``)/최솟값. 없으면 nan."""
    if lo < 0:
        raise ValueError("lo must be >= 0")
    arr = np.ascontiguousarray(x, dtype=np.float64)
    return _forward_extreme(arr, int(lo), int(h), bool(largest))  # type: ignore[no-any-return]
