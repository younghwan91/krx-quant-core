"""지정가 대기열 시뮬레이터의 입력 — 초 격자 10단계 호가창과 초·가격별 체결량.

키움 수집 데이터는 시각이 **1초 단위**(``ts`` 초 절삭)이고 호가는 종목·초당 스냅샷 하나
(``PRIMARY KEY (code, ts)``)다. 초 안의 체결·호가 순서는 알 수 없으므로 여기서 만드는
구조도 초 단위다. 스냅샷은 그 초가 **끝났을 때**의 호가창으로 본다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .grid import SESSION_END_SEC, SESSION_START_SEC, _sec_of_day

__all__ = ["BookGrid", "LevelTrades", "build_book_grid", "build_level_trades"]


@dataclass(frozen=True)
class BookGrid:
    """``(n초, levels)`` 호가창. 스냅샷이 없는 초는 직전 유효 스냅샷을 채운다(장 초반은 nan).

    ``bid_px[s, 0]`` 이 매수1호가. 잔량은 주 단위 float.
    """

    bid_px: NDArray[np.float64]
    bid_qty: NDArray[np.float64]
    ask_px: NDArray[np.float64]
    ask_qty: NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.bid_px.shape[0])

    @property
    def levels(self) -> int:
        return int(self.bid_px.shape[1])


@dataclass(frozen=True)
class LevelTrades:
    """초·가격별 체결량(CSR). 초 ``s`` 의 행은 ``ptr[s]:ptr[s+1]``, 가격 오름차순.

    ``buy_vol`` 은 매수 주도(매도호가를 친) 체결, ``sell_vol`` 은 매도 주도(매수호가를 친)
    체결 주수. ``side`` 가 0 이나 결측이면 양쪽에 모두 넣는다(어느 대기열을 줄였는지 모름).
    ``lo``/``hi`` 는 그 초 체결가 최저·최고(체결 없으면 nan).
    """

    ptr: NDArray[np.int64]
    price: NDArray[np.float64]
    buy_vol: NDArray[np.float64]
    sell_vol: NDArray[np.float64]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]


def build_book_grid(
    quotes: pd.DataFrame,
    *,
    levels: int = 10,
    start_sec: int = SESSION_START_SEC,
    end_sec: int = SESSION_END_SEC,
) -> BookGrid:
    """``quotes``(``ts, bid1..L, bidqty1..L, ask1..L, askqty1..L``, ts 정렬) → :class:`BookGrid`.

    ``bid1 > 0`` 且 ``ask1 > 0`` 인 스냅샷만 쓴다(:func:`~.grid.aggregate_seconds` 와 같은 기준).
    가격 0 인 깊은 단계는 nan(그 단계 없음)으로 둔다.
    """
    n = end_sec - start_sec
    cols = [f"{p}{i}" for p in ("bid", "bidqty", "ask", "askqty") for i in range(1, levels + 1)]
    miss = [c for c in ["ts", *cols] if c not in quotes.columns]
    if miss:
        raise ValueError(f"quotes missing columns: {miss}")
    out = np.full((4, n, levels), np.nan)
    if len(quotes):
        sec = _sec_of_day(quotes["ts"], start_sec)
        m = (sec >= 0) & (sec < n)
        q = quotes[m]
        sec = sec[m]
        vals = q[cols].to_numpy(float).reshape(len(q), 4, levels)
        ok = (vals[:, 0, 0] > 0) & (vals[:, 2, 0] > 0)
        vals = vals[ok]
        sec = sec[ok]
        # 가격 0 이하 단계는 없는 단계
        for side in (0, 2):
            bad = ~(vals[:, side, :] > 0)
            vals[:, side, :][bad] = np.nan
            vals[:, side + 1, :][bad] = np.nan
        out[:, sec, :] = vals.transpose(1, 0, 2)  # 같은 초는 마지막 스냅샷이 남는다
        have = np.zeros(n, dtype=bool)
        have[sec] = True
        idx = np.where(have, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = out[:, idx, :]
        if have.any():
            filled[:, : int(np.argmax(have)), :] = np.nan
        else:
            filled[:] = np.nan
        out = filled
    return BookGrid(
        bid_px=np.ascontiguousarray(out[0]),
        bid_qty=np.ascontiguousarray(out[1]),
        ask_px=np.ascontiguousarray(out[2]),
        ask_qty=np.ascontiguousarray(out[3]),
    )


def build_level_trades(
    ticks: pd.DataFrame,
    *,
    start_sec: int = SESSION_START_SEC,
    end_sec: int = SESSION_END_SEC,
) -> LevelTrades:
    """``ticks``(``ts, price, volume, side``) → :class:`LevelTrades`."""
    miss = [c for c in ("ts", "price", "volume", "side") if c not in ticks.columns]
    if miss:
        raise ValueError(f"ticks missing columns: {miss}")
    n = end_sec - start_sec
    sec = _sec_of_day(ticks["ts"], start_sec)
    m = (sec >= 0) & (sec < n)
    sec = sec[m]
    tk = ticks[m]
    price = tk.price.to_numpy(float)
    vol = tk.volume.to_numpy(float)
    side = np.nan_to_num(tk.side.to_numpy(float), nan=0.0)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    if len(tk) == 0:
        z = np.zeros(0)
        return LevelTrades(np.zeros(n + 1, np.int64), z, z.copy(), z.copy(), lo, hi)
    df = pd.DataFrame(
        {
            "sec": sec,
            "price": price,
            "buy": np.where(side >= 0, vol, 0.0),
            "sell": np.where(side <= 0, vol, 0.0),
        }
    )
    g = df.groupby(["sec", "price"], sort=True)[["buy", "sell"]].sum().reset_index()
    counts = np.bincount(g["sec"].to_numpy(), minlength=n)
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=ptr[1:])
    by = df.groupby("sec").price
    lo_s, hi_s = by.min(), by.max()
    lo[lo_s.index.to_numpy()] = lo_s.to_numpy()
    hi[hi_s.index.to_numpy()] = hi_s.to_numpy()
    return LevelTrades(
        ptr=ptr,
        price=g["price"].to_numpy(float),
        buy_vol=g["buy"].to_numpy(float),
        sell_vol=g["sell"].to_numpy(float),
        lo=lo,
        hi=hi,
    )
