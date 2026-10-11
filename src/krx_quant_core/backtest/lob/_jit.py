"""호환 경로 — numba 선택 가속은 :mod:`krx_quant_core._jit` 로 옮겼다.

lob 패키지를 올리지 않고도 쓰게 하려고 옮겼다."""

from krx_quant_core._jit import HAVE_NUMBA, div, njit

__all__ = ["HAVE_NUMBA", "div", "njit"]
