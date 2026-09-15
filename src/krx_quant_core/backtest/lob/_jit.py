"""numba 선택 가속 — ``pip install "krx-quant-core[fast]"`` 이면 njit, 아니면 순수 파이썬.

커널은 numba 가 받는 부분집합(스칼라 산술·배열 인덱싱·루프)으로만 쓴다. 그래서 numba 가
없을 때 같은 함수를 그대로 파이썬으로 돌려도 **같은 숫자**가 나온다(느릴 뿐). 나눗셈은
:func:`div` 로 한다 — numba 는 ``error_model="numpy"`` 로 0 나눗셈이 nan/inf 가 되지만
파이썬 float 는 ``ZeroDivisionError`` 를 던지므로 둘을 맞춘다.

``KRX_QUANT_CORE_DISABLE_NUMBA=1`` 이면 numba 가 깔려 있어도 폴백 경로를 쓴다.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

HAVE_NUMBA: bool
try:
    if os.environ.get("KRX_QUANT_CORE_DISABLE_NUMBA"):
        raise ImportError("disabled by KRX_QUANT_CORE_DISABLE_NUMBA")
    import numba as _numba

    HAVE_NUMBA = True
except ImportError:
    _numba = None
    HAVE_NUMBA = False


def njit(fn: F) -> F:
    """numba 가 있으면 ``njit(cache=True, error_model="numpy")``, 없으면 그대로.

    컴파일된 함수의 원본 파이썬은 ``fn.py_func`` 로 남는다(동일성 테스트가 쓴다).
    """
    if _numba is None:
        return fn
    return _numba.njit(cache=True, error_model="numpy", nogil=True)(fn)  # type: ignore[no-any-return]


@njit
def div(a: float, b: float) -> float:
    """numpy 의미의 나눗셈: ``x/0`` → ±inf, ``0/0``·nan → nan."""
    if b == 0.0:
        if a != a or a == 0.0:
            return math.nan
        return math.inf if (a > 0.0) == (math.copysign(1.0, b) > 0.0) else -math.inf
    return a / b
