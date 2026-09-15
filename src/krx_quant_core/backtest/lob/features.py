"""초 단위 호가·체결 흐름 특징 — 연구 배치와 실시간 누적 갱신이 **같은 커널**을 쓴다.

한 초의 원시 입력(:data:`INPUT_COLUMNS`)을 받아 그 초의 특징(:data:`STEP_COLUMNS`)을 내는
:func:`step` 하나가 전부다. 배치(:func:`compute_features`)는 격자 전체에 ``step`` 을
루프로 돌리고, 실시간(:class:`SecondFeatureStream`)은 초가 끝날 때마다 한 번 부른다.
상태는 float 배열 두 개(스칼라 ``state``, 링버퍼 ``hist``)라 numba 로 그대로 넘어간다.

숫자는 scalp-it 80번 ``of_build.build_code`` 와 같게 짰다.

- 롤링 합은 원본처럼 **누적합의 차**(``c[t+1] - c[t+1-w]``)로 낸다. 증분 갱신도 누적합
  자체를 링버퍼에 담아 같은 뺄셈을 하므로 배치·실시간·원본이 비트 단위로 같다.
- 예외 하나: ``ofi10`` 분모의 60초 평균 호가 두께. 원본은 ``pandas.rolling.mean`` 인데
  pandas 는 보정합(Kahan)으로 더해서 누적합 차와 마지막 한두 ulp 가 다를 수 있다.
- 원본의 ``np.roll`` 은 하루 앞머리(``t < k``)에서 장 끝 값을 끌어오는 wrap-around 가 있다.
  여기서는 nan 이다. 그 행들은 원본에서도 ``keep``(09:05 이후)으로 버려져 결과는 같다.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ._jit import div, njit
from .ticks import TickTable, stock_tick_table, tick_of

__all__ = [
    "INPUT_COLUMNS",
    "STEP_COLUMNS",
    "SecondFeatureStream",
    "compute_features",
    "forward_labels",
    "new_state",
    "step",
]

#: 한 초의 원시 입력. 그 초에 갱신이 없으면 nan(``buy``·``sell``·``cnt`` 는 0).
#:
#: - ``buy``/``sell``: 그 초 매수/매도 체결대금 합(가격×수량).
#: - ``cnt``: 그 초 체결 건수.
#: - ``bid``/``ask``: 그 초 마지막 유효 호가 스냅샷(bid1>0 且 ask1>0)의 1호가, 없으면 그 초
#:   마지막 틱의 best_bid/best_ask.
#: - ``strength``: 그 초 마지막 틱의 체결강도.
#: - ``bidqty1``/``askqty1``: 호가 스냅샷 1호가 잔량.
#: - ``bid_depth3``/``ask_depth3``: (1~3호가 잔량 합) × 1호가 — 원본 정의 그대로(대금 근사).
INPUT_COLUMNS: tuple[str, ...] = (
    "buy", "sell", "cnt", "bid", "ask", "strength",
    "bidqty1", "askqty1", "bid_depth3", "ask_depth3",
)  # fmt: skip

#: :func:`step` 출력 한 줄. 마지막 ``keep`` 은 0/1(원본 필터 중 ``cnt``·시각 조건 제외).
STEP_COLUMNS: tuple[str, ...] = (
    "acc", "v30", "v600", "buyshare30", "buyshare10", "ret30", "ret300", "ofi10", "imb3",
    "spread_bp", "strength", "dstr10", "dstr30", "dstr60", "spread_ticks", "bid", "ask",
    "bidval1", "bidval3", "ofi", "mid", "book_ok",
)  # fmt: skip

# 입력 인덱스
_BUY, _SELL, _CNT, _BB, _BA, _STG, _BQ1, _AQ1, _BD3, _AD3 = range(10)
# state 인덱스: 전방채움 7개, 직전 초(OFI) 4개, 누적합 4개, 경과 초 1개
_S_FF = 0  # bb, ba, stg, bq1, aq1, bd3, ad3 → 0..6
_S_PB, _S_PA, _S_PBQ, _S_PAQ = 7, 8, 9, 10
_S_CBUY, _S_CTOT, _S_CE, _S_CDEP = 11, 12, 13, 14
_S_T = 15
_N_STATE = 16
# hist 행: 누적합 4개 + 지연 조회용 v600·mid·strength
_H_CBUY, _H_CTOT, _H_CE, _H_CDEP, _H_V600, _H_MID, _H_STG = range(7)
_N_HIST = 7
#: 링버퍼 길이. 가장 긴 창(600초 합) 이 ``c[t-600]`` 을 읽으므로 601.
RING = 601
_N_OUT = len(STEP_COLUMNS)
_DEPTH_WIN = 60


def new_state() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """빈 상태(장 시작 전). ``(state, hist)``."""
    state = np.zeros(_N_STATE, dtype=np.float64)
    state[_S_FF : _S_FF + 7] = np.nan
    state[_S_PB : _S_PAQ + 1] = np.nan
    hist = np.full((_N_HIST, RING), np.nan, dtype=np.float64)
    return state, hist


@njit
def _lag(hist: NDArray[np.float64], row: int, t: int, k: int) -> float:
    """``t-k`` 초의 값. 아직 없으면 nan."""
    j = t - k
    if j < 0:
        return np.nan
    return hist[row, j % RING]


@njit
def _roll_sum(hist: NDArray[np.float64], row: int, total: float, t: int, w: int) -> float:
    """원본 ``roll_sum``: ``c[t+1] - c[t+1-w]``, ``t < w-1`` 이면 nan."""
    if t < w - 1:
        return np.nan
    j = t - w
    prev = 0.0 if j < 0 else hist[row, j % RING]
    return total - prev


@njit
def step(
    x: NDArray[np.float64],
    state: NDArray[np.float64],
    hist: NDArray[np.float64],
    out: NDArray[np.float64],
    bounds: NDArray[np.float64],
    ticks: NDArray[np.float64],
    top: float,
) -> None:
    """한 초 전진. ``x``(:data:`INPUT_COLUMNS`) → ``out``(:data:`STEP_COLUMNS`), 상태 갱신."""
    t = int(state[_S_T])
    # 전방채움(원시값 기준 — 0 이하 가격도 채운 뒤에 nan 으로 바꾼다, 원본 순서)
    for k in range(7):
        v = x[_BB + k]
        if v == v:
            state[_S_FF + k] = v
    bb = state[0]
    ba = state[1]
    if bb <= 0.0:
        bb = np.nan
    if ba <= 0.0:
        ba = np.nan
    stg = state[2]
    bq1 = state[3]
    aq1 = state[4]
    bd3 = state[5]
    ad3 = state[6]
    mid = (bb + ba) / 2.0

    # OFI(Cont) 1호가. 원본은 bool×잔량 네 항의 합을 nan_to_num 하므로 잔량 넷 중 하나라도
    # nan 이면 0 이다(비교가 nan 이면 False 라 가격 nan 은 항을 0 으로 만들 뿐).
    if t == 0:
        e = 0.0
    else:
        pb = state[_S_PB]
        pa = state[_S_PA]
        pbq = state[_S_PBQ]
        paq = state[_S_PAQ]
        if bq1 != bq1 or pbq != pbq or aq1 != aq1 or paq != paq:
            e = 0.0
        else:
            t1 = bq1 if bb >= pb else 0.0
            t2 = pbq if bb <= pb else 0.0
            t3 = aq1 if ba <= pa else 0.0
            t4 = paq if ba >= pa else 0.0
            e = t1 - t2 - t3 + t4
    state[_S_PB] = bb
    state[_S_PA] = ba
    state[_S_PBQ] = bq1
    state[_S_PAQ] = aq1
    depth = (bq1 + aq1) / 2.0
    if depth != depth:
        depth = 0.0

    buy = x[_BUY]
    tot = buy + x[_SELL]
    c_buy = state[_S_CBUY] + buy
    c_tot = state[_S_CTOT] + tot
    c_e = state[_S_CE] + e
    c_dep = state[_S_CDEP] + depth

    v30 = _roll_sum(hist, _H_CTOT, c_tot, t, 30)
    v600 = _roll_sum(hist, _H_CTOT, c_tot, t, 600)
    v10 = _roll_sum(hist, _H_CTOT, c_tot, t, 10)
    base = _lag(hist, _H_V600, t, 30) / 20.0  # 30초 창 20개 평균(직전 30초 제외)
    jd = t - _DEPTH_WIN
    dprev = 0.0 if jd < 0 else hist[_H_CDEP, jd % RING]
    dmean = (c_dep - dprev) / min(t + 1, _DEPTH_WIN)

    out[0] = v30 / (base + 1e6)
    out[1] = v30
    out[2] = v600
    out[3] = _roll_sum(hist, _H_CBUY, c_buy, t, 30) / (v30 + 1.0)
    out[4] = _roll_sum(hist, _H_CBUY, c_buy, t, 10) / (v10 + 1.0)
    out[5] = mid / _lag(hist, _H_MID, t, 30) - 1.0
    out[6] = mid / _lag(hist, _H_MID, t, 300) - 1.0
    out[7] = _roll_sum(hist, _H_CE, c_e, t, 10) / (dmean + 1.0)
    out[8] = div(bd3 - ad3, bd3 + ad3)
    out[9] = (ba - bb) / mid * 1e4
    out[10] = stg
    out[11] = stg - _lag(hist, _H_STG, t, 10)
    out[12] = stg - _lag(hist, _H_STG, t, 30)
    out[13] = stg - _lag(hist, _H_STG, t, 60)
    out[14] = (ba - bb) / tick_of(ba if ba == ba else 1.0, bounds, ticks, top)
    out[15] = bb
    out[16] = ba
    bv1 = bq1 * bb
    out[17] = 0.0 if bv1 != bv1 else bv1
    out[18] = 0.0 if bd3 != bd3 else bd3
    out[19] = c_e
    out[20] = mid
    out[21] = 1.0 if (bb == bb and ba == ba and ba > bb and out[0] == out[0]) else 0.0

    slot = t % RING
    hist[_H_CBUY, slot] = c_buy
    hist[_H_CTOT, slot] = c_tot
    hist[_H_CE, slot] = c_e
    hist[_H_CDEP, slot] = c_dep
    hist[_H_V600, slot] = v600
    hist[_H_MID, slot] = mid
    hist[_H_STG, slot] = stg
    state[_S_CBUY] = c_buy
    state[_S_CTOT] = c_tot
    state[_S_CE] = c_e
    state[_S_CDEP] = c_dep
    state[_S_T] = t + 1


@njit
def _run(
    X: NDArray[np.float64],
    state: NDArray[np.float64],
    hist: NDArray[np.float64],
    bounds: NDArray[np.float64],
    ticks: NDArray[np.float64],
    top: float,
) -> NDArray[np.float64]:
    n = X.shape[0]
    out = np.empty((n, _N_OUT), dtype=np.float64)
    for i in range(n):
        step(X[i], state, hist, out[i], bounds, ticks, top)
    return out


def compute_features(
    inputs: NDArray[np.float64], *, tick_table: TickTable | None = None
) -> NDArray[np.float64]:
    """격자 전체 배치. ``inputs`` ``(n초, len(INPUT_COLUMNS))`` → ``(n초, len(STEP_COLUMNS))``."""
    X = np.ascontiguousarray(inputs, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(INPUT_COLUMNS):
        raise ValueError(f"inputs must be (n, {len(INPUT_COLUMNS)})")
    tt = tick_table or stock_tick_table()
    state, hist = new_state()
    return _run(X, state, hist, tt.bounds, tt.ticks, tt.top)  # type: ignore[no-any-return]


class SecondFeatureStream:
    """실시간 누적 갱신 — 초가 닫힐 때 :meth:`update` 한 번. 배치와 같은 :func:`step`.

    장중에 체결·호가가 없는 초도 **빠짐없이** 넣어야 한다(모든 입력 nan, 체결 0). 창 길이가
    "초 개수"라 한 초라도 건너뛰면 배치와 숫자가 달라진다.
    """

    def __init__(self, *, tick_table: TickTable | None = None) -> None:
        tt = tick_table or stock_tick_table()
        self._bounds, self._ticks, self._top = tt.bounds, tt.ticks, tt.top
        self._state, self._hist = new_state()
        self._x = np.empty(len(INPUT_COLUMNS), dtype=np.float64)
        self._out = np.empty(_N_OUT, dtype=np.float64)

    @property
    def seconds(self) -> int:
        """지금까지 넣은 초 수(= 다음 초의 격자 인덱스)."""
        return int(self._state[_S_T])

    def update(
        self,
        *,
        buy: float = 0.0,
        sell: float = 0.0,
        cnt: float = 0.0,
        bid: float = math.nan,
        ask: float = math.nan,
        strength: float = math.nan,
        bidqty1: float = math.nan,
        askqty1: float = math.nan,
        bid_depth3: float = math.nan,
        ask_depth3: float = math.nan,
    ) -> dict[str, float]:
        """한 초를 넣고 그 초의 특징을 돌려준다(값은 :data:`STEP_COLUMNS`)."""
        x = self._x
        x[:] = (buy, sell, cnt, bid, ask, strength, bidqty1, askqty1, bid_depth3, ask_depth3)
        return dict(zip(STEP_COLUMNS, self.update_array(x).tolist(), strict=True))

    def update_array(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """:meth:`update` 의 배열판. 반환 배열은 내부 버퍼라 다음 호출 때 덮인다."""
        step(
            np.ascontiguousarray(x, dtype=np.float64),
            self._state,
            self._hist,
            self._out,
            self._bounds,
            self._ticks,
            self._top,
        )
        return self._out


@njit
def _fwd_max_shifted(a: NDArray[np.float64], h: int) -> NDArray[np.float64]:
    """``out[t] = nanmax(a[t+2 .. t+1+h])`` — 원본 ``mfe`` 정렬(1초 지연 진입 뒤 h초)."""
    n = a.shape[0]
    out = np.full(n, np.nan)
    for t in range(n - 1):
        m = np.nan
        hi = min(t + 1 + h, n - 1)
        for s in range(t + 2, hi + 1):
            v = a[s]
            if v == v and (m != m or v > m):
                m = v
        out[t] = m
    return out


def forward_labels(
    bid: NDArray[np.float64],
    ask: NDArray[np.float64],
    horizons: tuple[int, ...],
    *,
    cost: float,
) -> dict[str, NDArray[np.float64]]:
    """원본 80번 라벨 — 1초 지연 매도1호가 진입, ``h`` 초 뒤 매수1호가 청산 순수익(``net{h}``)과
    그 사이 매수1호가 최고치 기준 순수익(``mfe{h}``). **미래를 본다** — 연구 전용.

    ``cost`` 는 왕복 비용률(:func:`krx_quant_core.costs.round_trip_cost`).
    """
    bb = np.ascontiguousarray(bid, dtype=np.float64)
    ba = np.ascontiguousarray(ask, dtype=np.float64)
    n = len(bb)
    ask_e = np.full(n, np.nan)
    ask_e[: n - 1] = ba[1:]
    out: dict[str, NDArray[np.float64]] = {}
    for h in horizons:
        exit_b = np.full(n, np.nan)
        if n - 1 - h > 0:
            exit_b[: n - 1 - h] = bb[1 + h :]
        out[f"net{h}"] = exit_b / ask_e - 1 - cost
        out[f"mfe{h}"] = _fwd_max_shifted(bb, int(h)) / ask_e - 1 - cost
    return out
