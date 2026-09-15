"""njit 커널용 호가단위 표 — **값은 kiwoom-client 정본에서 읽는다**.

``kiwoom_client.tick_size`` 는 ``Decimal`` 함수라 njit 루프에서 부를 수 없다. 그래서 그
모듈의 밴드 표를 float 배열로 옮기고, import 시점에 정본 함수와 경계마다 대조한다 —
정본이 바뀌면 여기가 조용히 어긋나지 않고 import 가 실패한다.

ETF 호가단위(2,000원 미만 1원·이상 5원)는 kiwoom-client 0.4.0 에 **없다**. 정본에 추가를
요청해 두었고, 그 전까지 ETF 를 돌리는 호출부는 :class:`TickTable` 을 직접 만들어 넘긴다.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from krx_quant_core.market.ticks import tick_size

from ._jit import njit

__all__ = ["TickTable", "stock_tick_table", "tick_of"]


@dataclass(frozen=True)
class TickTable:
    """``price < bounds[k]`` 인 첫 ``k`` 의 ``ticks[k]``, 어느 경계도 아니면 ``top``."""

    bounds: NDArray[np.float64]
    ticks: NDArray[np.float64]
    top: float

    def __post_init__(self) -> None:
        b = np.ascontiguousarray(self.bounds, dtype=np.float64)
        t = np.ascontiguousarray(self.ticks, dtype=np.float64)
        if b.shape != t.shape or b.ndim != 1:
            raise ValueError("bounds and ticks must be 1-D arrays of equal length")
        if len(b) > 1 and not np.all(np.diff(b) > 0):
            raise ValueError("bounds must be strictly increasing")
        object.__setattr__(self, "bounds", b)
        object.__setattr__(self, "ticks", t)
        object.__setattr__(self, "top", float(self.top))


@njit
def tick_of(
    price: float, bounds: NDArray[np.float64], ticks: NDArray[np.float64], top: float
) -> float:
    """njit 안에서 쓰는 호가단위 조회. ``price`` 가 nan 이면 nan."""
    if price != price:
        return np.nan
    for k in range(bounds.shape[0]):
        if price < bounds[k]:
            return ticks[k]
    return top


def _load_stock_table() -> TickTable:
    mod = importlib.import_module("kiwoom_client.tick_size")
    bands = mod._TICK_BANDS
    table = TickTable(
        bounds=np.array([float(b) for b, _ in bands]),
        ticks=np.array([float(t) for _, t in bands]),
        top=float(mod._TOP_TICK),
    )
    # 정본 함수와 경계 양쪽을 대조한다(표 구조가 바뀌면 여기서 멈춘다).
    probes = [1.0]
    for b in table.bounds:
        probes += [b - 1.0, b - 0.5, b, b + 0.5]
    for p in probes:
        got = tick_of(p, table.bounds, table.ticks, table.top)
        if got != float(tick_size(p)):
            raise RuntimeError(f"tick table drifted from kiwoom_client at price {p}")
    return table


_STOCK = _load_stock_table()


def stock_tick_table() -> TickTable:
    """KOSPI·KOSDAQ 주식 호가단위(kiwoom-client 정본)."""
    return _STOCK
