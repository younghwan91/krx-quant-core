"""초 단위 특징 v2 — scalp-it 83·84번 37피처를 연구 배치와 실시간 누적 갱신이 **같은 커널**로 낸다.

93·94번 강화학습 정책이 학습한 입력(scalp-it ``scripts/rl_93/features.json``)을 실매매에서
**같은 숫자**로 받으려고 만들었다. 기준은 원본 재현 하나다 — 정의를 "고치지" 않는다. 원본:
``scripts/flow_83/features.py::build_code``·``scripts/flow_84/library.py::_extra``·``build_day``.

구조는 v1(:mod:`.features`)과 같다. 한 초를 전진하는 :func:`step_v2` 하나를 배치
(:func:`compute_features_v2`)는 격자 전체에 루프로 돌리고, 실시간(:class:`SecondFeatureStreamV2`)은
초가 닫힐 때 한 번 부른다. 상태는 float 배열 두 개(스칼라 ``state``, 링버퍼 ``hist``)다.

비트 단위로 맞추려고 원본 연산 순서를 그대로 따른 곳들:

- 롤링 합(``rsum``)은 **누적합의 차**다. 1초 창(``n_tr1``·``buyval1``)도 그 초 값이 아니라
  ``c[t+1] - c[t]`` 라 반올림이 다를 수 있어서 그대로 뺀다. 링버퍼에는 누적합 자체를 담는다.
- ``ofiw{w}`` 분모의 60초 평균 호가 두께는 ``pandas.rolling(60, min_periods=1).mean`` 이다.
  pandas(3.0.x ``_libs/window/aggregations.pyx::roll_mean``)는 더할 때·뺄 때 보정항을 따로 둔
  Kahan 합에 "같은 값 연속이면 그 값", "전부 양수인데 음수면 0" 보정까지 한다 — 그 상태를 전부
  들고 같은 순서로 더하고 뺀다(v1 은 이걸 누적합 차로 근사해 ulp 차이를 허용했다).
- ``log1p`` 는 커널 밖에서 ``np.log1p`` 로 건다. numpy 는 CPU 에 따라 SIMD 수학 함수를 쓰는데
  numba·``math.log1p`` 는 libm 이라 마지막 ulp 가 다를 수 있다 — 원본이 ``np.log1p`` 였다.
- 횡단면 순위 두 열(``cum_rank``·``dayret_rank``)은 여러 종목이 필요해 이 커널이 내지 않는다
  (:data:`FEATURE_SET_V2_EXCLUDED`). 원본 식을 옮긴 :func:`cross_section_ranks_v2` 를 편의로 둔다.

출력 자료형은 **float64** 다. 원본이 계산한 피처 배열(``build_code``·``_extra`` 의 격자)이
float64 이고, 학습 캐시(flow84 pkl)는 그걸 저장 직전에 float32 로 바꿨다. 정책 입력이 필요하면
받는 쪽에서 ``.astype(np.float32)`` — 같은 float64 에서 같은 변환이라 학습 캐시와 같은 값이다
(골든 테스트가 float32 변환 결과도 대조한다). 배치·실시간이 같은 자료형이라 그대로 비교된다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._jit import div, njit
from .grid import SESSION_END_SEC, SESSION_START_SEC, _need, _sec_of_day
from .ticks import TickTable, stock_tick_table, tick_of

__all__ = [
    "FEATURE_SET_V2",
    "FEATURE_SET_V2_EXCLUDED",
    "INPUT_COLUMNS_V2",
    "STEP_COLUMNS_V2",
    "SecondFeatureStreamV2",
    "aggregate_seconds_v2",
    "compute_features_v2",
    "cross_section_ranks_v2",
    "new_state_v2",
    "step_v2",
]

#: 93·94번 정책 입력 37열(scalp-it ``scripts/rl_93/features.json``) 중 종목 하나로 계산되는 35열,
#: 정본 순서 그대로. 빠진 두 열은 :data:`FEATURE_SET_V2_EXCLUDED`.
FEATURE_SET_V2: tuple[str, ...] = (
    "sec", "log_cumval", "spread_bp", "n_tr10", "dist_lo300", "ret300",
    "same_size_val10", "sweep_ticks3", "dist_hi300", "micro_dev", "imb3", "ntr60", "n_tr5",
    "buyval3", "n_tr1", "burst_secs10", "buyval1", "sv20", "bv30", "buyval10", "sv60",
    "sellval5", "sellval3", "maxbuy5", "sweep10", "sellval1", "sweep_ticks1",
    "dstr3", "dstr5", "dS2", "ofiw20", "ofiw30", "ofiw60", "ofiw1", "ofiw2",
)  # fmt: skip

#: 정본 37열 중 이 모듈의 커널·스트림이 **내지 않는** 열(0·nan 으로 채우지도 않는다).
#:
#: - ``cum_rank``: 그 초까지 누적 체결대금의 **종목 간** 순위. 원본은 "그날 틱 500건 이상"
#:   종목끼리 매겼는데 이 대상 집합은 하루가 끝나야 정해진다(미래 정보) — 실시간 대상 집합은
#:   소비자가 정한다.
#: - ``dayret_rank``: 전일 종가(DB ``daily_bars``) 대비 등락률의 종목 간 순위.
#:
#: 둘 다 여러 종목과 외부 데이터가 필요해 종목 단위 커널 밖이다. 소비자가 채운다 — 원본 식은
#: :func:`cross_section_ranks_v2` 에 옮겨 두었다(보조 열 ``cumval``·``last`` 를 입력으로).
#: 재료 피처(공시·뉴스·테마)는 84번 후보 라이브러리에만 있고 정본 37열에는 없다.
FEATURE_SET_V2_EXCLUDED: tuple[str, ...] = ("cum_rank", "dayret_rank")

#: 한 초의 호가·체결강도 입력. 그 초에 갱신이 없으면 nan. 체결은 따로(가격·수량·방향 배열) 넣는다.
#:
#: - ``bid``/``ask``: 그 초 마지막 유효 호가 스냅샷(bid1>0 且 ask1>0)의 1호가, 없으면 그 초 마지막
#:   틱의 best_bid/best_ask(nan 이어도 마지막 틱 값).
#: - ``strength``: 그 초 마지막 틱의 체결강도.
#: - ``bidqty1``/``askqty1``·``bid_depth3``/``ask_depth3``: **유효** 스냅샷의 1호가 잔량,
#:   (1~3호가 잔량 합, nan 은 건너뜀)×1호가. OFI·``imb3`` 가 쓴다(83번).
#: - ``bidqty1_raw``/``askqty1_raw``: 유효 여부와 **상관없이** 그 초 마지막 스냅샷의 1호가 잔량.
#:   84번 ``_extra`` 가 호가를 다시 읽을 때 유효성 필터를 안 걸었다 — ``micro_dev``·``ofiw`` 분모가
#:   이 값을 쓴다. 고치지 않고 그대로 둔다.
INPUT_COLUMNS_V2: tuple[str, ...] = (
    "bid", "ask", "strength", "bidqty1", "askqty1", "bid_depth3", "ask_depth3",
    "bidqty1_raw", "askqty1_raw",
)  # fmt: skip

#: :func:`step_v2` 출력 한 줄: 35열 + 횡단면 순위용 보조 ``cumval``(누적 체결대금)·
#: ``last``(직전 체결가).
STEP_COLUMNS_V2: tuple[str, ...] = (*FEATURE_SET_V2, "cumval", "last")

_O = {c: i for i, c in enumerate(STEP_COLUMNS_V2)}
_N_OUT = len(STEP_COLUMNS_V2)
#: 커널 밖에서 ``np.log1p`` 를 거는 열(원본에서 ``np.log1p`` 였던 열).
_LOG_COLS = (
    "log_cumval", "same_size_val10", "buyval1", "buyval3", "buyval10", "sellval1", "sellval3",
    "sellval5", "maxbuy5", "sv20", "sv60", "bv30",
)  # fmt: skip
_LOG_IDX = np.array([_O[c] for c in _LOG_COLS], dtype=np.int64)
_I_CUMVAL = _O["cumval"]
_I_LAST = _O["last"]

# 출력 인덱스(numba 전역 상수)
_O_SEC = _O["sec"]
_O_LOGCUM = _O["log_cumval"]
_O_SPREAD = _O["spread_bp"]
_O_NTR10 = _O["n_tr10"]
_O_DLO300 = _O["dist_lo300"]
_O_RET300 = _O["ret300"]
_O_SAME = _O["same_size_val10"]
_O_SW3 = _O["sweep_ticks3"]
_O_DHI300 = _O["dist_hi300"]
_O_MICRO = _O["micro_dev"]
_O_IMB3 = _O["imb3"]
_O_NTR60 = _O["ntr60"]
_O_NTR5 = _O["n_tr5"]
_O_BV3 = _O["buyval3"]
_O_NTR1 = _O["n_tr1"]
_O_BURST10 = _O["burst_secs10"]
_O_BV1 = _O["buyval1"]
_O_SV20 = _O["sv20"]
_O_BV30 = _O["bv30"]
_O_BV10 = _O["buyval10"]
_O_SV60 = _O["sv60"]
_O_SV5 = _O["sellval5"]
_O_SV3 = _O["sellval3"]
_O_MB5 = _O["maxbuy5"]
_O_SW10 = _O["sweep10"]
_O_SV1 = _O["sellval1"]
_O_SW1 = _O["sweep_ticks1"]
_O_DSTR3 = _O["dstr3"]
_O_DSTR5 = _O["dstr5"]
_O_DS2 = _O["dS2"]
_O_OFIW20 = _O["ofiw20"]
_O_OFIW30 = _O["ofiw30"]
_O_OFIW60 = _O["ofiw60"]
_O_OFIW1 = _O["ofiw1"]
_O_OFIW2 = _O["ofiw2"]
_O_CUMVAL = _O["cumval"]
_O_LAST = _O["last"]

# 입력 인덱스
_BB, _BA, _STG, _BQ1, _AQ1, _BD3, _AD3, _BQR, _AQR = range(9)
# state: 0 경과 초, 1..10 전방채움(입력 9개 + 직전 체결가), 11..14 직전 초 1호가·잔량(OFI),
# 15..20 누적합, 21..27 pandas rolling mean 상태, 28..31 300초 최고·최저 단조 덱 머리·꼬리
_S_T = 0
_S_FF = 1
_S_LAST = 10
_S_PB, _S_PA, _S_PBQ, _S_PAQ = 11, 12, 13, 14
_S_CNTR, _S_CBUY, _S_CSELL, _S_CE, _S_CBURST, _S_CVAL = 15, 16, 17, 18, 19, 20
_S_RM_NOBS, _S_RM_SUM, _S_RM_CADD, _S_RM_CREM, _S_RM_NEG, _S_RM_NCONS, _S_RM_PREV = range(21, 28)
_S_HI_H, _S_HI_T, _S_LO_H, _S_LO_T = 28, 29, 30, 31
_N_STATE = 32
# hist 행
_H_CNTR, _H_CBUY, _H_CSELL, _H_CE, _H_CBURST = range(5)
_H_MID, _H_STG, _H_DEP, _H_SW, _H_MB, _H_DQHI, _H_DQLO = range(5, 12)
_N_HIST = 12
#: 링버퍼 길이. 가장 먼 조회는 ``ret300``·300초 덱의 ``t-300`` 이므로 301.
RING_V2 = 301
_EXT_WIN = 300
_DEPTH_WIN = 60
_SAME_WIN = 10
_FMAX = 1.7976931348623157e308


def new_state_v2() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """빈 상태(장 시작 전). ``(state, hist)``."""
    state = np.zeros(_N_STATE, dtype=np.float64)
    state[_S_FF : _S_LAST + 1] = np.nan
    state[_S_PB : _S_PAQ + 1] = np.nan
    hist = np.full((_N_HIST, RING_V2), np.nan, dtype=np.float64)
    return state, hist


@njit
def _nan_to_num(v: float) -> float:
    if v != v:
        return 0.0
    if v == math.inf:
        return _FMAX
    if v == -math.inf:
        return -_FMAX
    return v


@njit
def _rsum(hist: NDArray[np.float64], row: int, total: float, t: int, w: int) -> float:
    """원본 ``rsum``: ``c[t+1] - c[t+1-w]``, 앞머리(``t < w``)는 ``c[t+1] - 0.0``(= 부분합)."""
    j = t - w
    prev = 0.0 if j < 0 else hist[row, j % RING_V2]
    return total - prev


@njit
def _lag(hist: NDArray[np.float64], row: int, t: int, k: int) -> float:
    j = t - k
    if j < 0:
        return np.nan
    return hist[row, j % RING_V2]


@njit
def _roll_max(hist: NDArray[np.float64], row: int, t: int, w: int) -> float:
    """``pandas.rolling(w, min_periods=1).max`` — 입력에 nan 이 없는 열에만 쓴다."""
    m = hist[row, t % RING_V2]
    for k in range(1, w):
        j = t - k
        if j < 0:
            break
        v = hist[row, j % RING_V2]
        if v > m:
            m = v
    return m


@njit
def _mean_add(state: NDArray[np.float64], val: float) -> None:
    """pandas ``add_mean``(Kahan, 더하기 보정항)."""
    if val == val:
        state[_S_RM_NOBS] += 1.0
        s = state[_S_RM_SUM]
        y = val - state[_S_RM_CADD]
        t = s + y
        state[_S_RM_CADD] = t - s - y
        state[_S_RM_SUM] = t
        if math.copysign(1.0, val) < 0.0:
            state[_S_RM_NEG] += 1.0
        if val == state[_S_RM_PREV]:
            state[_S_RM_NCONS] += 1.0
        else:
            state[_S_RM_NCONS] = 1.0
        state[_S_RM_PREV] = val


@njit
def _mean_remove(state: NDArray[np.float64], val: float) -> None:
    """pandas ``remove_mean``(Kahan, 빼기 보정항은 따로)."""
    if val == val:
        state[_S_RM_NOBS] -= 1.0
        s = state[_S_RM_SUM]
        y = -val - state[_S_RM_CREM]
        t = s + y
        state[_S_RM_CREM] = t - s - y
        state[_S_RM_SUM] = t
        if math.copysign(1.0, val) < 0.0:
            state[_S_RM_NEG] -= 1.0


@njit
def _mean_value(state: NDArray[np.float64]) -> float:
    """pandas ``calc_mean`` (minp=1)."""
    nobs = state[_S_RM_NOBS]
    if nobs >= 1.0:
        result = state[_S_RM_SUM] / nobs
        if state[_S_RM_NCONS] >= nobs:
            result = state[_S_RM_PREV]
        elif state[_S_RM_NEG] == 0.0 and result < 0.0:
            result = 0.0
        elif state[_S_RM_NEG] == nobs and result > 0.0:
            result = 0.0
        return result
    return np.nan


@njit
def _extreme(
    hist: NDArray[np.float64],
    state: NDArray[np.float64],
    dq_row: int,
    s_head: int,
    s_tail: int,
    x: float,
    t: int,
    want_max: bool,
) -> float:
    """원본 ``_roll_extreme``(단조 덱) 한 걸음 — 최근 300초(포함) nan 아닌 mid 의 극값."""
    h = int(state[s_head])
    tl = int(state[s_tail])
    # 원본은 넣고 나서 앞을 버린다. 앞(창 밖 한 칸)을 먼저 버려도 남는 원소는 같고, 덱이 링버퍼
    # 한 바퀴(301칸)를 넘지 않는다.
    while tl > h and hist[dq_row, h % RING_V2] <= t - _EXT_WIN:
        h += 1
    if x == x:
        while tl > h:
            last = hist[_H_MID, int(hist[dq_row, (tl - 1) % RING_V2]) % RING_V2]
            if (want_max and last <= x) or ((not want_max) and last >= x):
                tl -= 1
            else:
                break
        hist[dq_row, tl % RING_V2] = t
        tl += 1
    while tl > h and hist[dq_row, h % RING_V2] <= t - _EXT_WIN:
        h += 1
    state[s_head] = h
    state[s_tail] = tl
    if tl > h:
        return hist[_H_MID, int(hist[dq_row, h % RING_V2]) % RING_V2]
    return np.nan


@njit
def _same_size_val(
    ptr: NDArray[np.int64],
    ptr_t0: int,
    price: NDArray[np.float64],
    vol: NDArray[np.float64],
    side: NDArray[np.float64],
    t: int,
) -> float:
    """83번 ``_window_trade_stats`` 의 ``same_val`` — 최근 10초 매수 체결 중 가장 많이 반복된
    수량(동률이면 작은 수량) × 반복 횟수 × 창 안 마지막 매수 체결가."""
    lo = ptr[max(0, t - (_SAME_WIN - 1)) - ptr_t0]
    hi = ptr[t + 1 - ptr_t0]
    if hi == lo:
        return 0.0
    vols = np.empty(hi - lo)
    nb = 0
    last_px = 0.0
    for i in range(lo, hi):
        if side[i] > 0:
            vols[nb] = vol[i]
            last_px = price[i]
            nb += 1
    if nb < 2:
        return 0.0
    vs = np.sort(vols[:nb])
    best = 1
    run = 1
    bestv = vs[0]
    for j in range(1, nb):
        if vs[j] == vs[j - 1]:
            run += 1
        else:
            run = 1
        if run > best:
            best = run
            bestv = vs[j]
    return best * bestv * last_px


@njit
def step_v2(
    x: NDArray[np.float64],
    ptr: NDArray[np.int64],
    ptr_t0: int,
    price: NDArray[np.float64],
    vol: NDArray[np.float64],
    side: NDArray[np.float64],
    state: NDArray[np.float64],
    hist: NDArray[np.float64],
    out: NDArray[np.float64],
    bounds: NDArray[np.float64],
    ticks: NDArray[np.float64],
    top: float,
) -> None:
    """한 초 전진. ``x``(:data:`INPUT_COLUMNS_V2`) + 체결 CSR → ``out``(:data:`STEP_COLUMNS_V2`).

    그 초(``t = state[0]``)의 체결은 ``ptr[t-ptr_t0]:ptr[t+1-ptr_t0]``, 최근 10초 창은
    ``ptr[max(0, t-9)-ptr_t0]`` 부터 읽는다. 배치는 하루치 CSR 에 ``ptr_t0=0``, 실시간은 최근
    10초만 남긴 버퍼에 ``ptr_t0=max(0, t-9)`` 를 넘긴다 — 읽는 체결은 같다.
    ``out`` 의 ``np.log1p`` 열은 **아직 원값**이다 — 호출부가 :data:`_LOG_IDX` 에 건다.
    """
    t = int(state[_S_T])
    for k in range(9):
        v = x[_BB + k]
        if v == v:
            state[_S_FF + k] = v

    # 83번 _per_second
    ntr = 0.0
    buy = 0.0
    sell = 0.0
    maxbuy = 0.0
    nbuy = 0
    bhi = np.nan
    blo = np.nan
    p0 = ptr[t - ptr_t0]
    p1 = ptr[t + 1 - ptr_t0]
    for i in range(p0, p1):
        pr = price[i]
        v = pr * vol[i]
        ntr += 1
        if side[i] > 0:
            buy += v
            nbuy += 1
            if v > maxbuy:
                maxbuy = v
            if not (bhi == bhi) or pr > bhi:
                bhi = pr
            if not (blo == blo) or pr < blo:
                blo = pr
        elif side[i] < 0:
            sell += v
    if p1 > p0:
        lp = price[p1 - 1]
        if lp == lp:
            state[_S_LAST] = lp

    # 전방채움 뒤 0 이하 가격은 nan(원본 순서)
    bb = state[_S_FF + _BB]
    ba = state[_S_FF + _BA]
    if bb <= 0.0:
        bb = np.nan
    if ba <= 0.0:
        ba = np.nan
    stg = state[_S_FF + _STG]
    bq1 = state[_S_FF + _BQ1]
    aq1 = state[_S_FF + _AQ1]
    bd3 = state[_S_FF + _BD3]
    ad3 = state[_S_FF + _AD3]
    bqr = state[_S_FF + _BQR]
    aqr = state[_S_FF + _AQR]
    mid = (bb + ba) / 2.0
    pt = ba if ba == ba else 1000.0  # _tick_np: nan_to_num(nan=1000) → tick_size(int(max(p, 1)))
    if pt < 1.0:
        pt = 1.0
    tick = tick_of(pt, bounds, ticks, top)

    # OFI(83번): bool×잔량 네 항 → nan_to_num. 첫 초는 lag 가 nan.
    pb = state[_S_PB]
    pa = state[_S_PA]
    t1 = (1.0 if bb >= pb else 0.0) * bq1
    t2 = (1.0 if bb <= pb else 0.0) * state[_S_PBQ]
    t3 = (1.0 if ba <= pa else 0.0) * aq1
    t4 = (1.0 if ba >= pa else 0.0) * state[_S_PAQ]
    e = _nan_to_num(t1 - t2 - t3 + t4)
    state[_S_PB] = bb
    state[_S_PA] = ba
    state[_S_PBQ] = bq1
    state[_S_PAQ] = aq1

    c_ntr = state[_S_CNTR] + ntr
    c_buy = state[_S_CBUY] + buy
    c_sell = state[_S_CSELL] + sell
    c_e = state[_S_CE] + e
    c_burst = state[_S_CBURST] + (1.0 if nbuy >= 5 else 0.0)
    c_val = state[_S_CVAL] + (buy + sell)
    slot = t % RING_V2
    hist[_H_CNTR, slot] = c_ntr
    hist[_H_CBUY, slot] = c_buy
    hist[_H_CSELL, slot] = c_sell
    hist[_H_CE, slot] = c_e
    hist[_H_CBURST, slot] = c_burst
    hist[_H_MID, slot] = mid
    state[_S_CNTR] = c_ntr
    state[_S_CBUY] = c_buy
    state[_S_CSELL] = c_sell
    state[_S_CE] = c_e
    state[_S_CBURST] = c_burst
    state[_S_CVAL] = c_val

    # 84번 ofiw 분모: pandas rolling(60, min_periods=1).mean(nan_to_num((bq1_raw + aq1_raw) / 2))
    dval = _nan_to_num((bqr + aqr) / 2.0)
    if t == 0:
        state[_S_RM_PREV] = dval
        state[_S_RM_NCONS] = 0.0
    elif t >= _DEPTH_WIN:
        _mean_remove(state, hist[_H_DEP, (t - _DEPTH_WIN) % RING_V2])
    _mean_add(state, dval)
    hist[_H_DEP, slot] = dval
    depth = _mean_value(state) + 1.0

    sw = _nan_to_num((bhi - blo) / tick)
    hist[_H_SW, slot] = sw
    hist[_H_MB, slot] = maxbuy

    hv = _extreme(hist, state, _H_DQHI, _S_HI_H, _S_HI_T, mid, t, True)
    lv = _extreme(hist, state, _H_DQLO, _S_LO_H, _S_LO_T, mid, t, False)

    out[_O_SEC] = float(t)
    out[_O_LOGCUM] = c_val
    out[_O_SPREAD] = div(ba - bb, mid) * 1e4
    out[_O_NTR10] = _rsum(hist, _H_CNTR, c_ntr, t, 10)
    out[_O_DLO300] = (div(mid, lv) - 1.0) * 1e4
    out[_O_RET300] = (div(mid, _lag(hist, _H_MID, t, 300)) - 1.0) * 1e4
    out[_O_SAME] = _same_size_val(ptr, ptr_t0, price, vol, side, t)
    out[_O_SW3] = _roll_max(hist, _H_SW, t, 3)
    out[_O_DHI300] = (div(mid, hv) - 1.0) * 1e4
    micro = div(bb * aqr + ba * bqr, bqr + aqr)
    out[_O_MICRO] = (div(micro, mid) - 1.0) * 1e4
    out[_O_IMB3] = div(bd3 - ad3, bd3 + ad3)
    out[_O_NTR60] = _rsum(hist, _H_CNTR, c_ntr, t, 60)
    out[_O_NTR5] = _rsum(hist, _H_CNTR, c_ntr, t, 5)
    out[_O_BV3] = _rsum(hist, _H_CBUY, c_buy, t, 3)
    out[_O_NTR1] = _rsum(hist, _H_CNTR, c_ntr, t, 1)
    out[_O_BURST10] = _rsum(hist, _H_CBURST, c_burst, t, 10)
    out[_O_BV1] = _rsum(hist, _H_CBUY, c_buy, t, 1)
    out[_O_SV20] = _rsum(hist, _H_CSELL, c_sell, t, 20)
    out[_O_BV30] = _rsum(hist, _H_CBUY, c_buy, t, 30)
    out[_O_BV10] = _rsum(hist, _H_CBUY, c_buy, t, 10)
    out[_O_SV60] = _rsum(hist, _H_CSELL, c_sell, t, 60)
    out[_O_SV5] = _rsum(hist, _H_CSELL, c_sell, t, 5)
    out[_O_SV3] = _rsum(hist, _H_CSELL, c_sell, t, 3)
    out[_O_MB5] = _roll_max(hist, _H_MB, t, 5)
    out[_O_SW10] = _roll_max(hist, _H_SW, t, 10)
    out[_O_SV1] = _rsum(hist, _H_CSELL, c_sell, t, 1)
    out[_O_SW1] = sw
    out[_O_DSTR3] = stg - _lag(hist, _H_STG, t, 3)
    out[_O_DSTR5] = stg - _lag(hist, _H_STG, t, 5)
    out[_O_DS2] = stg - _lag(hist, _H_STG, t, 2)
    out[_O_OFIW20] = _rsum(hist, _H_CE, c_e, t, 20) / depth
    out[_O_OFIW30] = _rsum(hist, _H_CE, c_e, t, 30) / depth
    out[_O_OFIW60] = _rsum(hist, _H_CE, c_e, t, 60) / depth
    out[_O_OFIW1] = _rsum(hist, _H_CE, c_e, t, 1) / depth
    out[_O_OFIW2] = _rsum(hist, _H_CE, c_e, t, 2) / depth
    out[_O_CUMVAL] = c_val
    out[_O_LAST] = state[_S_LAST]

    hist[_H_STG, slot] = stg  # 지연 조회가 끝난 뒤에 쓴다(창 ≤ 5 라 순서는 무관하지만 명시)
    state[_S_T] = t + 1


@njit
def _run_v2(
    X: NDArray[np.float64],
    ptr: NDArray[np.int64],
    price: NDArray[np.float64],
    vol: NDArray[np.float64],
    side: NDArray[np.float64],
    state: NDArray[np.float64],
    hist: NDArray[np.float64],
    bounds: NDArray[np.float64],
    ticks: NDArray[np.float64],
    top: float,
) -> NDArray[np.float64]:
    n = X.shape[0]
    out = np.empty((n, _N_OUT), dtype=np.float64)
    for i in range(n):
        step_v2(X[i], ptr, 0, price, vol, side, state, hist, out[i], bounds, ticks, top)
    return out


def _f64(a: ArrayLike) -> NDArray[np.float64]:
    return np.ascontiguousarray(a, dtype=np.float64)


def compute_features_v2(
    inputs: ArrayLike,
    trade_ptr: ArrayLike,
    trade_price: ArrayLike,
    trade_volume: ArrayLike,
    trade_side: ArrayLike,
    *,
    tick_table: TickTable | None = None,
) -> NDArray[np.float64]:
    """격자 전체 배치 → ``(n초, len(STEP_COLUMNS_V2))`` float64.

    Args:
        inputs: ``(n초, len(INPUT_COLUMNS_V2))`` — :func:`aggregate_seconds_v2` 의 ``X``.
        trade_ptr: ``(n초+1,)`` int — 초 ``s`` 의 체결은 ``trade_*[ptr[s]:ptr[s+1]]``(시각·seq 순).
        trade_price, trade_volume, trade_side: 체결가·수량·방향(+1 매수 주도/−1 매도/0 모름).

    횡단면 순위(:data:`FEATURE_SET_V2_EXCLUDED`)는 없다 — 필요하면 :func:`cross_section_ranks_v2`.
    """
    X = _f64(inputs)
    if X.ndim != 2 or X.shape[1] != len(INPUT_COLUMNS_V2):
        raise ValueError(f"inputs must be (n, {len(INPUT_COLUMNS_V2)})")
    ptr = np.ascontiguousarray(trade_ptr, dtype=np.int64)
    price, vol, side = _f64(trade_price), _f64(trade_volume), _f64(trade_side)
    if ptr.shape != (X.shape[0] + 1,):
        raise ValueError("trade_ptr must have length n + 1")
    if not (len(price) == len(vol) == len(side)) or (len(ptr) and ptr[-1] > len(price)):
        raise ValueError("trade arrays length mismatch with trade_ptr")
    tt = tick_table or stock_tick_table()
    state, hist = new_state_v2()
    F = _run_v2(X, ptr, price, vol, side, state, hist, tt.bounds, tt.ticks, tt.top)
    F[:, _LOG_IDX] = np.log1p(F[:, _LOG_IDX])
    return F  # type: ignore[no-any-return]


def cross_section_ranks_v2(
    cumval: ArrayLike, last: ArrayLike, prevclose: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """84번 ``build_day`` 의 ``cum_rank``·``dayret_rank`` 식 그대로(편의 함수, 커널 출력 아님).

    Args:
        cumval, last: 축 0 이 종목인 ``(종목, n초)``(배치) 또는 ``(종목,)``(실시간 한 초) —
            :data:`STEP_COLUMNS_V2` 의 보조 열. 종목 순서는 원본처럼 **종목코드 정렬**이어야
            동점 순위가 같다(``argsort`` 동점 처리가 입력 순서에 달려 있다).
        prevclose: 종목별 전일 종가(없으면 nan).

    Returns:
        ``(cum_rank, dayret_rank)`` float64, 1 이 최상위. 순위 대상 종목 집합은 호출부 몫이다 —
        원본은 "그날 틱 500건 이상"(하루가 끝나야 아는 조건) 종목끼리였다.
    """
    cum = np.asarray(cumval, dtype=np.float64)
    lst = np.asarray(last, dtype=np.float64)
    pc = np.asarray(prevclose, dtype=np.float64)
    if cum.shape != lst.shape or cum.ndim not in (1, 2) or pc.shape != (cum.shape[0],):
        raise ValueError("cumval/last must be (codes,) or (codes, n); prevclose (codes,)")
    rank = (-cum).argsort(axis=0).argsort(axis=0) + 1
    pc = pc.reshape((-1,) + (1,) * (cum.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        dayret = (lst / pc - 1) * 100
    dayret_rank = (-np.nan_to_num(dayret, nan=-1e9)).argsort(axis=0).argsort(axis=0) + 1
    return rank.astype(float), dayret_rank.astype(float)


@njit
def _push_trades(
    ptr: NDArray[np.int64],
    pbuf: NDArray[np.float64],
    vbuf: NDArray[np.float64],
    sbuf: NDArray[np.float64],
    t: int,
    price: NDArray[np.float64],
    vol: NDArray[np.float64],
    side: NDArray[np.float64],
) -> None:
    """실시간 체결 버퍼를 최근 10초로 유지하고 초 ``t`` 의 체결을 붙인다.

    호출 전: ``ptr[k]`` = 초 ``max(0, t-10)+k`` 의 시작(``ptr[0] = 0``), ``ptr[min(t, 10)]`` = 끝.
    호출 뒤: 초 ``max(0, t-9)`` 부터 ``t`` 까지, ``ptr[min(t, 9)+1]`` = 끝.
    용량은 호출부가 확인한다.
    """
    if t >= _SAME_WIN:
        base = ptr[1]
        end = ptr[_SAME_WIN]
        for i in range(base, end):
            pbuf[i - base] = pbuf[i]
            vbuf[i - base] = vbuf[i]
            sbuf[i - base] = sbuf[i]
        for k in range(_SAME_WIN):
            ptr[k] = ptr[k + 1] - base
        m = _SAME_WIN - 1
    else:
        m = t
    end = ptr[m]
    for i in range(price.shape[0]):
        pbuf[end + i] = price[i]
        vbuf[end + i] = vol[i]
        sbuf[end + i] = side[i]
    ptr[m + 1] = end + price.shape[0]


class SecondFeatureStreamV2:
    """실시간 누적 갱신 — 초가 닫힐 때 :meth:`update_array` 한 번. 배치와 같은 :func:`step_v2`.

    - 장중 체결·호가가 없는 초도 **빠짐없이** 넣는다(입력 nan, 체결 빈 배열). 창이 "초 개수"다.
    - 반환은 **매번 새 float64 배열**이다. 내부 버퍼를 돌려주면 모아 둔 행이 다음 호출에 조용히
      덮인다(v1 실데이터 대조에서 겪었다).
    - 메모리는 하루 내내 고정이다: 링버퍼(301초)와 **최근 10초** 체결만 든다(체결 버퍼는 256건에서
      시작해 10초 창 최대 체결 수까지만 는다). 종목 수천 개를 한꺼번에 띄우는 소비자 기준으로
      인스턴스당 수십 KB.
    - 장중 재시작: 커널 상태는 그날 09:00 부터의 누적(누적대금·Kahan 합·300초 덱)이라 중간부터
      시작하면 숫자가 달라진다. 그날 초 격자를 처음부터 다시 넣는다 — :meth:`from_history`.
    """

    _INIT_TRADES = 256

    def __init__(self, *, tick_table: TickTable | None = None) -> None:
        tt = tick_table or stock_tick_table()
        self._bounds, self._ticks, self._top = tt.bounds, tt.ticks, tt.top
        self._state, self._hist = new_state_v2()
        self._out = np.empty(_N_OUT, dtype=np.float64)
        self._ptr = np.zeros(_SAME_WIN + 1, dtype=np.int64)
        self._price = np.empty(self._INIT_TRADES)
        self._vol = np.empty(self._INIT_TRADES)
        self._side = np.empty(self._INIT_TRADES)

    @classmethod
    def from_history(
        cls,
        inputs: ArrayLike,
        trade_ptr: ArrayLike,
        trade_price: ArrayLike,
        trade_volume: ArrayLike,
        trade_side: ArrayLike,
        *,
        tick_table: TickTable | None = None,
    ) -> SecondFeatureStreamV2:
        """장중 재시작 복구 — 그날 09:00 부터 지난 초들(:func:`compute_features_v2` 와 같은 입력)을
        :meth:`update_array` 로 다시 넣은 스트림. 다음 호출은 초 ``len(inputs)`` 다."""
        X = _f64(inputs)
        if X.ndim != 2 or X.shape[1] != len(INPUT_COLUMNS_V2):
            raise ValueError(f"inputs must be (n, {len(INPUT_COLUMNS_V2)})")
        ptr = np.asarray(trade_ptr, dtype=np.int64)
        if ptr.shape != (X.shape[0] + 1,):
            raise ValueError("trade_ptr must have length n + 1")
        price, vol, side = _f64(trade_price), _f64(trade_volume), _f64(trade_side)
        s = cls(tick_table=tick_table)
        for t in range(X.shape[0]):
            a, b = int(ptr[t]), int(ptr[t + 1])
            s._advance(X[t], price[a:b], vol[a:b], side[a:b])
        return s

    @property
    def seconds(self) -> int:
        """지금까지 넣은 초 수(= 다음 초의 격자 인덱스)."""
        return int(self._state[_S_T])

    @property
    def nbytes(self) -> int:
        """내부 배열이 차지하는 바이트(메모리 상한 점검용)."""
        arrs = (self._state, self._hist, self._out, self._ptr, self._price, self._vol, self._side)
        return sum(a.nbytes for a in arrs)

    def _advance(
        self, x: NDArray[np.float64], price: NDArray[np.float64], vol: NDArray[np.float64],
        side: NDArray[np.float64],
    ) -> None:  # fmt: skip
        t = self.seconds
        ptr = self._ptr
        k = len(price)
        need = int(ptr[min(t, _SAME_WIN)]) - (int(ptr[1]) if t >= _SAME_WIN else 0) + k
        if need > len(self._price):
            cap = max(2 * len(self._price), need)
            used = int(ptr[min(t, _SAME_WIN)])
            for name in ("_price", "_vol", "_side"):
                buf = np.empty(cap)
                buf[:used] = getattr(self, name)[:used]
                setattr(self, name, buf)
        _push_trades(ptr, self._price, self._vol, self._side, t, price, vol, side)
        t0 = max(0, t - (_SAME_WIN - 1))
        step_v2(x, ptr, t0, self._price, self._vol, self._side, self._state, self._hist,
                self._out, self._bounds, self._ticks, self._top)  # fmt: skip

    def update_array(
        self,
        x: ArrayLike,
        price: ArrayLike = (),
        volume: ArrayLike = (),
        side: ArrayLike = (),
    ) -> NDArray[np.float64]:
        """한 초(``x``: :data:`INPUT_COLUMNS_V2`, 그 초 체결들)를 넣고 새 배열
        (:data:`STEP_COLUMNS_V2`)을 돌려준다."""
        xa = _f64(x)
        pa, va, sa = _f64(price).reshape(-1), _f64(volume).reshape(-1), _f64(side).reshape(-1)
        if xa.shape != (len(INPUT_COLUMNS_V2),):
            raise ValueError(f"x must have length {len(INPUT_COLUMNS_V2)}")
        if not (len(pa) == len(va) == len(sa)):
            raise ValueError("price, volume, side must have the same length")
        self._advance(xa, pa, va, sa)
        r = self._out.copy()
        r[_LOG_IDX] = np.log1p(r[_LOG_IDX])
        return r

    def update(
        self,
        *,
        bid: float = math.nan,
        ask: float = math.nan,
        strength: float = math.nan,
        bidqty1: float = math.nan,
        askqty1: float = math.nan,
        bid_depth3: float = math.nan,
        ask_depth3: float = math.nan,
        bidqty1_raw: float = math.nan,
        askqty1_raw: float = math.nan,
        price: ArrayLike = (),
        volume: ArrayLike = (),
        side: ArrayLike = (),
    ) -> dict[str, float]:
        """:meth:`update_array` 의 키워드판 → ``{열: 값}``."""
        x = (bid, ask, strength, bidqty1, askqty1, bid_depth3, ask_depth3, bidqty1_raw,
             askqty1_raw)  # fmt: skip
        r = self.update_array(x, price, volume, side)
        return dict(zip(STEP_COLUMNS_V2, r.tolist(), strict=True))


_TICK_COLS = ("ts", "price", "volume", "side", "strength", "best_bid", "best_ask")
_QUOTE_COLS = ("ts", "bid1", "ask1", "bidqty1", "bidqty2", "bidqty3", "askqty1", "askqty2",
               "askqty3")  # fmt: skip


def aggregate_seconds_v2(
    ticks: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    start_sec: int | None = None,
    end_sec: int | None = None,
) -> tuple[
    NDArray[np.float64],
    tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
]:
    """틱·호가 → ``(X, (ptr, price, volume, side))`` (:func:`compute_features_v2` 입력).

    ``ticks``(``ts, price, volume, side, strength, best_bid, best_ask``)는 ``(ts, seq)`` 순,
    ``quotes``(``ts, bid1, ask1, bidqty1..3, askqty1..3``)는 ``ts`` 순이어야 한다(같은 초는 마지막
    값). 장 밖(기본 09:00~15:20) 행은 버린다. 원본 83번 ``build_code``·84번 ``_extra`` 의 초 격자
    입력과 같은 값이다.
    """
    start = SESSION_START_SEC if start_sec is None else start_sec
    end = SESSION_END_SEC if end_sec is None else end_sec
    _need(ticks, _TICK_COLS, "ticks")
    n = end - start
    sec = _sec_of_day(ticks["ts"], start)
    m = (sec >= 0) & (sec < n)
    tk = ticks[m]
    sec = sec[m].astype(np.int64)
    if len(sec) > 1 and np.any(np.diff(sec) < 0):
        raise ValueError("ticks must be sorted by (ts, seq)")
    price = tk.price.to_numpy(np.float64)
    vol = tk.volume.to_numpy(np.float64)
    side = tk.side.to_numpy(np.int64).astype(np.float64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(sec, minlength=n), out=ptr[1:])

    X = np.full((n, len(INPUT_COLUMNS_V2)), np.nan)
    X[sec, _STG] = tk.strength.to_numpy(np.float64)
    X[sec, _BB] = tk.best_bid.to_numpy(np.float64)
    X[sec, _BA] = tk.best_ask.to_numpy(np.float64)
    if len(quotes):
        _need(quotes, _QUOTE_COLS, "quotes")
        qs = _sec_of_day(quotes["ts"], start)
        mq = (qs >= 0) & (qs < n)
        # 84번 _extra: 유효성 필터 없이 1호가 잔량
        X[qs[mq], _BQR] = quotes.bidqty1.to_numpy(float)[mq]
        X[qs[mq], _AQR] = quotes.askqty1.to_numpy(float)[mq]
        qt, qs = quotes[mq], qs[mq]
        b1 = qt.bid1.to_numpy(np.float64)
        a1 = qt.ask1.to_numpy(np.float64)
        ok = (b1 > 0) & (a1 > 0)
        X[qs[ok], _BB] = b1[ok]
        X[qs[ok], _BA] = a1[ok]
        X[qs[ok], _BQ1] = qt.bidqty1.to_numpy(np.float64)[ok]
        X[qs[ok], _AQ1] = qt.askqty1.to_numpy(np.float64)[ok]
        bsum = qt[["bidqty1", "bidqty2", "bidqty3"]].sum(axis=1).to_numpy(np.float64)
        asum = qt[["askqty1", "askqty2", "askqty3"]].sum(axis=1).to_numpy(np.float64)
        X[qs[ok], _BD3] = bsum[ok] * b1[ok]
        X[qs[ok], _AD3] = asum[ok] * a1[ok]
    return X, (ptr, price, vol, side)
