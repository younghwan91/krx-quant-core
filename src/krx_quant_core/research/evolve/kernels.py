"""진화 적합도용 일반 커널 — 해시 난수·월 합·이동 블록 부트스트랩 하단(numba 선택).

swing-it ``research/signals/ga_fast.py`` 의 ``_h32_py``·``_block_boot_py`` 와 그 뒤의
``np.percentile(·, q)``(numpy 기본 linear) 를 **한 번 호출**로 묶었다. 숫자는 비트 단위로 같다
(시험이 numpy 와 대조). 파이썬 접착(``np.unique``·``bincount``·``np.percentile``)이 평가 한 번에
0.1ms 넘게 쓰던 것을 없애는 게 목적이다.

- 난수는 정수 해시(lowbias32 변형, 곱셈 상수 < 2³¹ 라 int64 곱이 넘치지 않는다) — numba 와 파이썬이
  같은 표본을 낸다.
- 분위는 numpy ``_quantile``(method="linear") 와 같은 식: 가상 위치 ``(n-1)·q``, 아래 이웃
  ``floor``, 보간 ``a + (b-a)·γ`` (γ ≥ 0.5 면 ``b - (b-a)·(1-γ)``). 표본에 NaN 이 있으면 NaN.
"""

from __future__ import annotations

import math

import numpy as np

from krx_quant_core._jit import HAVE_NUMBA, njit

__all__ = [
    "HAVE_NUMBA",
    "block_boot",
    "block_boot_lower",
    "block_boot_lower_nb",
    "h32",
    "monthly_block_lower",
    "monthly_sums",
    "monthly_sums_nb",
    "percentile_linear",
    "percentile_nb",
]

M32 = 0xFFFFFFFF


@njit
def h32(x):
    """32비트 정수 해시(lowbias32 변형). swing-it ``ga_fast._h32_py`` 와 같다."""
    x = x & M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x5BD1E995) & M32
    x ^= x >> 16
    return x


@njit
def block_boot(monthly, block, n_boot, seed):
    """월 수익열의 이동 블록 부트스트랩(블록 ``block`` 개월) → 연환산 평균(×12) 표본 ``n_boot`` 개.

    swing-it ``ga_fast._block_boot_py`` 와 같은 난수·같은 덧셈 순서."""
    M = monthly.shape[0]
    out = np.empty(n_boot)
    nstart = M - block + 1
    nb = (M + block - 1) // block
    hs = h32(seed * 0x2545F491 + 0x1B873593)
    for b in range(n_boot):
        hb = h32(hs ^ h32(b + 0x27D4EB2F))
        s = 0.0
        cnt = 0
        for k in range(nb):
            u = h32(hb ^ h32(k * 3 + 1))
            st = (u * nstart) >> 32
            for j in range(block):
                if cnt < M:
                    s += monthly[st + j]
                    cnt += 1
        out[b] = s / M * 12.0
    return out


@njit
def _pct_sorted_pair(s2, g):
    a = s2[0]
    b = s2[1]
    d = b - a
    if g >= 0.5:
        return b - d * (1.0 - g)
    return a + d * g


@njit
def _pct_sorted(s, q):
    n = s.shape[0]
    quant = q / 100.0
    v = (n - 1) * quant
    if v != v:
        return math.nan
    if v >= n - 1:
        return s[n - 1]
    if v < 0:
        return s[0]
    lo = int(math.floor(v))
    g = v - math.floor(v)
    a = s[lo]
    b = s[lo + 1]
    d = b - a
    if g >= 0.5:
        return b - d * (1.0 - g)
    return a + d * g


@njit
def _percentile(x, q):
    n = x.shape[0]
    for i in range(n):
        if x[i] != x[i]:
            return math.nan
    v = (n - 1) * (q / 100.0)
    if v >= n - 1 or v < 0:
        return _pct_sorted(np.sort(x), q)
    lo = int(math.floor(v))
    p = np.partition(x, lo)                      # 정렬 없이 lo 번째·그다음 원소(같은 값)
    s2 = np.empty(2)
    s2[0] = p[lo]
    s2[1] = np.min(p[lo + 1:])
    return _pct_sorted_pair(s2, v - math.floor(v))


def percentile_linear(x: np.ndarray, q: float) -> float:
    """``np.percentile(x, q)`` (method="linear") 와 비트 동일. 빈 배열이면 ``ValueError``."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("빈 배열의 백분위")
    return float(_percentile(x, float(q)))


@njit
def _block_boot_lower(monthly, block, n_boot, seed, q):
    return _percentile(block_boot(monthly, block, n_boot, seed), q)


def block_boot_lower(monthly: np.ndarray, block: int, n_boot: int, seed: int,
                     q: float = 2.5) -> float:
    """``np.percentile(ga_fast.block_boot(monthly, block, n_boot, seed), q)`` 와 비트 동일한 한 번
    호출."""
    m = np.ascontiguousarray(monthly, dtype=np.float64)
    if m.shape[0] < block or n_boot < 1:
        raise ValueError(f"월 {m.shape[0]}개 < 블록 {block} 이거나 n_boot < 1")
    return float(_block_boot_lower(m, int(block), int(n_boot), int(seed), float(q)))


@njit
def _monthly_sums(x, month_of, t_lo, sums, ids):
    n = x.shape[0]
    k = -1
    prev = -1
    for i in range(n):
        m = month_of[t_lo + i]
        if k >= 0 and m < prev:
            return -1                            # 비감소가 아니다
        if k < 0 or m != prev:
            k += 1
            sums[k] = 0.0
            ids[k] = m
            prev = m
        sums[k] += x[i]
    return k + 1


def monthly_sums(x: np.ndarray, month_of: np.ndarray, t_lo: int) -> tuple[np.ndarray, np.ndarray]:
    """일별 ``x`` 의 월 합 — ``np.unique(month_of[t_lo:t_lo+len(x)], return_inverse=True)`` +
    ``np.bincount(inv, weights=x)`` 와 같은 값·같은 순서. ``month_of`` 는 비감소여야 한다(거래일
    순)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if t_lo < 0 or t_lo + x.shape[0] > len(month_of):
        raise ValueError("month_of 범위를 넘는다")
    seg = np.ascontiguousarray(month_of[t_lo:t_lo + x.shape[0]], dtype=np.int64)   # 창만 변환
    sums = np.empty(x.shape[0])
    ids = np.empty(x.shape[0], np.int64)
    k = _monthly_sums(x, seg, 0, sums, ids)
    if k < 0:
        raise ValueError("month_of 가 비감소가 아니다")
    return sums[:k], ids[:k]


def monthly_block_lower(x: np.ndarray, month_of: np.ndarray, t_lo: int, block: int, n_boot: int,
                        seed: int, q: float = 2.5, *, min_months: int = 6,
                        short: float = -1.0) -> float:
    """일별 초과 ``x`` → 월 합 → 블록 부트스트랩 연환산 평균의 ``q`` 백분위. 월 < ``min_months`` 면
    ``short``.

    swing-it ``Engine.evaluate`` 의 ``lower`` 와 같은 정의(월 ≥ 6 이 아니면 −1.0)."""
    mo, _ = monthly_sums(x, month_of, t_lo)
    if mo.shape[0] < min_months:
        return float(short)
    return block_boot_lower(mo, block, n_boot, seed, q)


# --- 소비자 njit 커널 안에서 부를 수 있는 판(검사 없음 — 인자는 호출자가 보장한다) ---
#: ``percentile_nb(x, q)`` — ``np.percentile`` 비트 동일(NaN 이면 NaN, 빈 배열 금지).
percentile_nb = _percentile
#: ``block_boot_lower_nb(monthly, block, n_boot, seed, q)`` —
#: :func:`block_boot_lower` 의 검사 없는 판.
block_boot_lower_nb = _block_boot_lower
#: ``monthly_sums_nb(x, month_of, t_lo, sums, ids) -> k`` —
#: ``sums``·``ids`` (길이 ≥ len(x)) 앞 k 칸을 채운다. ``month_of`` 가 비감소가 아니면 −1.
monthly_sums_nb = _monthly_sums
