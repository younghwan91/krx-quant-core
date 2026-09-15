"""틱·호가 DataFrame → 장중 초 격자(특징 + 리플레이 경로).

scalp-it 80번 ``scripts/orderflow_80/of_build.py::build_code`` 의 이식이다. 숫자는 원본과
같다(골든 테스트) — 특징 계산은 :mod:`.features` 의 커널을 쓰고, 여기서는 틱·호가를 초
단위 원시 입력으로 모으는 일만 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .features import INPUT_COLUMNS, STEP_COLUMNS, compute_features, forward_labels
from .ticks import TickTable

__all__ = [
    "FEATURE_COLUMNS",
    "PATH_KEYS",
    "SESSION_END_SEC",
    "SESSION_START_SEC",
    "aggregate_seconds",
    "build_second_grid",
]

#: 격자 시작(09:00:00)·끝(15:20:00, 미포함) — 하루 00:00 기준 초. 동시호가 이후는 없다.
SESSION_START_SEC = 9 * 3600
SESSION_END_SEC = 15 * 3600 + 20 * 60

#: :func:`build_second_grid` 특징 열 순서(라벨 ``net{h}``·``mfe{h}`` 는 뒤에 붙는다).
FEATURE_COLUMNS: tuple[str, ...] = (
    "sec", "cnt", "acc", "v30", "v600", "buyshare30", "buyshare10", "ret30", "ret300",
    "ofi10", "imb3", "spread_bp", "strength", "dstr10", "dstr30", "dstr60", "spread_ticks",
    "bid", "ask",
)  # fmt: skip

#: 리플레이 경로 키 — 모두 격자 길이 float32.
PATH_KEYS: tuple[str, ...] = ("bid", "ask", "lo", "hi", "bidval1", "bidval3", "ofi", "strength")

_TICK_COLS = ("ts", "price", "volume", "side", "strength", "best_bid", "best_ask")
_QUOTE_COLS = (
    "ts",
    "bid1",
    "ask1",
    "bidqty1",
    "bidqty2",
    "bidqty3",
    "askqty1",
    "askqty2",
    "askqty3",
)


def _sec_of_day(ts: pd.Series, start: int) -> NDArray[np.int64]:
    return (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy() - start  # type: ignore[no-any-return]


def _need(df: pd.DataFrame, cols: tuple[str, ...], what: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{what} missing columns: {miss}")


def aggregate_seconds(
    ticks: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    start_sec: int = SESSION_START_SEC,
    end_sec: int = SESSION_END_SEC,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """틱·호가를 초 격자 원시 입력(:data:`~.features.INPUT_COLUMNS`)으로.

    ``ticks`` 는 ``(ts, seq)`` 순, ``quotes`` 는 ``ts`` 순으로 **정렬돼 있어야** 한다 —
    같은 초 안에서는 마지막 값이 남는다. 반환 ``(inputs, extra)``; ``extra`` 는 커널에 안
    들어가는 경로 값(초당 체결가 ``lo``/``hi``).

    같은 초에 유효 호가 스냅샷(bid1>0 且 ask1>0)이 있으면 1호가는 스냅샷, 없으면 그 초 마지막
    틱의 best_bid/best_ask 다. 잔량·3호가 대금은 스냅샷에서만 온다.
    """
    _need(ticks, _TICK_COLS, "ticks")
    n = end_sec - start_sec
    sec_t = _sec_of_day(ticks["ts"], start_sec)
    m = (sec_t >= 0) & (sec_t < n)
    tk = ticks[m]
    sec_t = sec_t[m]
    val = (tk.price * tk.volume).to_numpy(float)
    side = tk.side.to_numpy()
    X = np.full((n, len(INPUT_COLUMNS)), np.nan)
    X[:, 0] = np.bincount(sec_t, weights=np.where(side > 0, val, 0), minlength=n)
    X[:, 1] = np.bincount(sec_t, weights=np.where(side < 0, val, 0), minlength=n)
    X[:, 2] = np.bincount(sec_t, minlength=n)
    X[sec_t, 5] = tk.strength.to_numpy(float)
    X[sec_t, 3] = tk.best_bid.to_numpy(float)
    X[sec_t, 4] = tk.best_ask.to_numpy(float)

    if len(quotes):
        _need(quotes, _QUOTE_COLS, "quotes")
        sec_q = _sec_of_day(quotes["ts"], start_sec)
        mq = (sec_q >= 0) & (sec_q < n)
        qt = quotes[mq]
        sec_q = sec_q[mq]
        b1 = qt.bid1.to_numpy(float)
        a1 = qt.ask1.to_numpy(float)
        ok = (b1 > 0) & (a1 > 0)
        so = sec_q[ok]
        X[so, 3] = b1[ok]
        X[so, 4] = a1[ok]
        X[so, 6] = qt.bidqty1.to_numpy(float)[ok]
        X[so, 7] = qt.askqty1.to_numpy(float)[ok]
        bsum = qt[["bidqty1", "bidqty2", "bidqty3"]].sum(axis=1).to_numpy(float)
        asum = qt[["askqty1", "askqty2", "askqty3"]].sum(axis=1).to_numpy(float)
        X[so, 8] = (bsum * b1)[ok]
        X[so, 9] = (asum * a1)[ok]

    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    if len(tk):
        pr = pd.Series(tk.price.to_numpy(float)).groupby(sec_t)
        lo_s, hi_s = pr.min(), pr.max()
        lo[lo_s.index.to_numpy()] = lo_s.to_numpy()
        hi[hi_s.index.to_numpy()] = hi_s.to_numpy()
    return X, {"lo": lo, "hi": hi}


def build_second_grid(
    ticks: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    start_sec: int = SESSION_START_SEC,
    end_sec: int = SESSION_END_SEC,
    warmup_sec: int = 300,
    tail_sec: int = 600,
    label_horizons: tuple[int, ...] = (),
    label_cost: float | None = None,
    tick_table: TickTable | None = None,
) -> tuple[pd.DataFrame, dict[str, NDArray[np.float32]]]:
    """한 종목·하루 → ``(features, path)``. 원본 ``build_code`` 와 같은 숫자.

    Args:
        ticks: ``ts, price, volume, side(+1 매수/−1 매도), strength, best_bid, best_ask``,
            ``(ts, seq)`` 정렬.
        quotes: ``ts, bid1, ask1, bidqty1..3, askqty1..3``, ``ts`` 정렬. 비어도 된다.
        warmup_sec, tail_sec: 특징 행을 ``[warmup_sec, n - tail_sec]`` 초로 자른다
            (원본 09:05~15:10).
        label_horizons: 주면 원본 80번 라벨 ``net{h}``·``mfe{h}`` 를 붙인다(미래를 본다).
            원본은 ``(5, 15, 30, 60, 120, 300)``.
        label_cost: 라벨 왕복 비용률 — ``label_horizons`` 를 주면 필수.
            :func:`krx_quant_core.costs.round_trip_cost` 로 구한다(원본 0.0023 = 2026년 세율).
        tick_table: ``spread_ticks`` 호가단위. 기본은 주식(kiwoom-client 정본).

    Returns:
        ``features``: 체결이 있고(``cnt>0``) 양쪽 호가가 있고 ``ask>bid`` 이며 ``acc`` 가 정의된
        초만 남긴 DataFrame(:data:`FEATURE_COLUMNS` [+ 라벨]; float 는 float32, ``sec``·``cnt`` 는
        int64). ``path``: 격자 전체 길이의 :data:`PATH_KEYS` float32 배열.
    """
    if label_horizons and label_cost is None:
        raise ValueError("label_cost is required when label_horizons is given")
    X, extra = aggregate_seconds(ticks, quotes, start_sec=start_sec, end_sec=end_sec)
    n = X.shape[0]
    F = compute_features(X, tick_table=tick_table)
    col = {c: F[:, i] for i, c in enumerate(STEP_COLUMNS)}

    f: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "sec": np.arange(n),
        "cnt": X[:, 2].astype(np.int64),
    }
    for c in FEATURE_COLUMNS[2:]:
        f[c] = col[c]
    if label_horizons:
        assert label_cost is not None
        f.update(forward_labels(col["bid"], col["ask"], label_horizons, cost=label_cost))

    path = {
        "bid": col["bid"].astype(np.float32),
        "ask": col["ask"].astype(np.float32),
        "lo": extra["lo"].astype(np.float32),
        "hi": extra["hi"].astype(np.float32),
        "bidval1": col["bidval1"].astype(np.float32),
        "bidval3": col["bidval3"].astype(np.float32),
        "ofi": col["ofi"].astype(np.float32),
        "strength": col["strength"].astype(np.float32),
    }
    df = pd.DataFrame(f)
    sec = np.arange(n)
    keep = (X[:, 2] > 0) & (sec >= warmup_sec) & (sec <= n - tail_sec) & (col["book_ok"] > 0)
    df = df[keep]
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df, path
