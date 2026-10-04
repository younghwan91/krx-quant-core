"""오라클 천장 — **완벽한 예지력** 으로 매도 시점을 골라도 비용을 넘는가 (scalp-it 95번 이식).

전략이 아니라 산술 검사다. 어떤 진입 방식(테이커/메이커, 롱/숏)과 보유 지평 H 에서, 사후에
최적 청산을 골라도 순수익 중앙값이 음수면 그 자리에는 **어떤 모형도** 우위를 만들 수 없다.
scalp-it 83~94 의 2,200회 시행이 −30bp 로 수렴한 이유가 이것이었다(95번: 10초 테이커 천장
−35.9bp). 더 좋은 모형이 아니라 다른 자리(지평·상품)가 처방이라는 걸 한 표로 보여 준다.

초 격자 ``bid``·``ask`` (:func:`~krx_quant_core.backtest.lob.build_second_grid` 의 path) 를 받는다.

- 테이커 롱: t+1 매도1호가 매수 → [t+2, t+1+H] 매수1호가 **최댓값** 매도 − cost
- 메이커 롱(낙관): t+1 매수1호가에 체결됐다 치고(대기열 무시) → 같은 구간 매도1호가 최댓값 − cost
- 메이커 롱(비관): 그 지정가가 ``fill_sec`` 안에 **뚫려 내려와야만** 체결로 본다 — 역선택이
  가장 심한 부분집합. 낙관·비관이 실전을 위아래로 감싼다.
- 숏: 대칭(먼저 팔고 창 안 최솟값에 되산다). 비용은 같다.

세 값 모두 실전 도달 불가능한 상·하한이다. ``cost`` 는 왕복 비용 비율(0.0041 = 41bp)이며
**필수 인자** 다 — 원본의 상수 0.0023 을 박지 않고 :func:`krx_quant_core.costs.round_trip_cost`
를 넘기게 했다. 결과는 원본 ``scripts/ceiling_95/ceiling.py`` 와 비트 단위로 같다(골든 테스트).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ._jit import njit

__all__ = [
    "DEFAULT_HORIZONS",
    "ceiling_table",
    "maker_pessimistic",
    "oracle_long",
    "oracle_short",
    "through_fill_second",
    "window_max",
]

#: 95번이 쓴 보유 지평(초).
DEFAULT_HORIZONS: tuple[int, ...] = (10, 30, 60, 300, 600, 1800, 3600, 7200)


def _future_extreme(x: np.ndarray, h: int, *, largest: bool) -> np.ndarray:
    """out[t] = max/min(x[t+2 : t+2+h]) — 원본과 같은 순서로 fmax/fmin 누적(NaN 무시)."""
    n = len(x)
    best = np.full(n, np.nan, np.float64)
    op = np.fmax if largest else np.fmin
    for j in range(2, h + 2):
        s = np.full(n, np.nan)
        s[: n - j] = x[j:]
        best = op(best, s)
    return best


def oracle_long(
    bid: np.ndarray, ask: np.ndarray, horizon: int, *, cost: float
) -> tuple[np.ndarray, np.ndarray]:
    """초 t 진입 결정 기준 **롱** 사후 최적 순수익 → ``(taker, maker)`` (비율, NaN=평가 불가)."""
    bid = np.asarray(bid, np.float64)
    ask = np.asarray(ask, np.float64)
    n = len(bid)
    best_bid = _future_extreme(bid, horizon, largest=True)
    best_ask = _future_extreme(ask, horizon, largest=True)
    entry_taker = np.full(n, np.nan)
    entry_maker = np.full(n, np.nan)
    entry_taker[: n - 1] = ask[1:]
    entry_maker[: n - 1] = bid[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        tk = best_bid / entry_taker - 1 - cost
        mk = best_ask / entry_maker - 1 - cost
    bad = ~(entry_taker > 0) | ~(entry_maker > 0) | ~(best_bid > 0) | ~(best_ask > 0)
    tk[bad] = np.nan
    mk[bad] = np.nan
    return tk, mk


def oracle_short(
    bid: np.ndarray, ask: np.ndarray, horizon: int, *, cost: float
) -> tuple[np.ndarray, np.ndarray]:
    """**숏** 천장 — 먼저 팔고 창 안 최저가에 되산다 → ``(taker, maker)``."""
    bid = np.asarray(bid, np.float64)
    ask = np.asarray(ask, np.float64)
    n = len(bid)
    lo_ask = _future_extreme(ask, horizon, largest=False)
    lo_bid = _future_extreme(bid, horizon, largest=False)
    e_taker = np.full(n, np.nan)
    e_maker = np.full(n, np.nan)
    e_taker[: n - 1] = bid[1:]
    e_maker[: n - 1] = ask[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        tk = e_taker / lo_ask - 1 - cost
        mk = e_maker / lo_bid - 1 - cost
    bad = ~(e_taker > 0) | ~(e_maker > 0) | ~(lo_ask > 0) | ~(lo_bid > 0)
    tk[bad] = np.nan
    mk[bad] = np.nan
    return tk, mk


@njit
def _through_fill_second(bid: np.ndarray, fill_sec: int) -> np.ndarray:
    n = len(bid)
    e = np.full(n, -1, np.int64)
    for t in range(n - fill_sec - 3):
        p = bid[t + 1]
        if not (p > 0):
            continue
        for s in range(t + 2, t + 2 + fill_sec):
            if bid[s] > 0 and bid[s] < p:
                e[t] = s
                break
    return e


def through_fill_second(bid: np.ndarray, fill_sec: int = 10) -> np.ndarray:
    """지정가 ``bid[t+1]`` 이 ``fill_sec`` 초 안에 **뚫려서**(더 낮은 bid 출현) 체결된 초. 없으면 −1

    t 당 한 번이고 보유 구간과 무관하다. numba 커널(없으면 같은 파이썬)이다.
    """
    return _through_fill_second(np.ascontiguousarray(bid, np.float64), int(fill_sec))


def window_max(x: np.ndarray, h: int) -> np.ndarray:
    """``w[i] = nanmax(x[i : i+h])`` — 뒤가 모자라면 NaN."""
    x = np.asarray(x, np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    if n > h:
        with np.errstate(all="ignore"):
            out[: n - h + 1] = np.nanmax(sliding_window_view(x, h), axis=1)
    return out


def maker_pessimistic(
    bid: np.ndarray, fill_idx: np.ndarray, max_ask: np.ndarray, *, cost: float
) -> np.ndarray:
    """뚫려서 체결된 초 다음부터 사후 최적 매도 — 역선택 최대 부분집합의 순수익. 미체결은 NaN.

    ``fill_idx`` 는 :func:`through_fill_second`, ``max_ask`` 는 :func:`window_max` ``(ask, H)``.
    """
    bid = np.asarray(bid, np.float64)
    n = len(bid)
    out = np.full(n, np.nan)
    idx = np.nonzero(fill_idx >= 0)[0]
    nxt = fill_idx[idx] + 1
    good = nxt < n
    idx, nxt = idx[good], nxt[good]
    p = bid[idx + 1]
    best = np.asarray(max_ask, np.float64)[nxt]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = best / p - 1 - cost
    r[~(p > 0) | ~(best > 0)] = np.nan
    out[idx] = r
    return out


def _stats(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        nan = np.nan
        return {"n": 0, "median_bp": nan, "mean_bp": nan, "p90_bp": nan, "frac_pos": nan}
    x = x * 1e4
    return {
        "n": int(len(x)),
        "median_bp": float(np.median(x)),
        "mean_bp": float(x.mean()),
        "p90_bp": float(np.quantile(x, 0.90)),
        "frac_pos": float((x > 0).mean()),
    }


def ceiling_table(
    paths: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    cost: float,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    lo: int = 300,
    hi: int = 22800 - 600,
    fill_sec: int = 10,
) -> pd.DataFrame:
    """여러 종목·일의 ``(bid, ask)`` 초 격자에서 지평×진입방식 천장 요약표.

    행 = (horizon, kind) with kind ∈ {taker, maker, maker_pessimistic, short_taker, short_maker},
    열 = n·median_bp·mean_bp·p90_bp·frac_pos, ``maker_pessimistic`` 에는 ``fill_rate`` 도.
    ``[lo, hi)`` 는 진입 결정을 허용하는 초 구간(기본 09:05~15:10, 09:00 기준 초). 원본처럼
    롱 테이커·메이커가 둘 다 유한한 초만 센다(모든 kind 가 같은 초 집합을 본다).
    """
    acc: dict[int, dict[str, list[np.ndarray]]] = {
        h: {k: [] for k in ("tk", "mk", "mp", "stk", "smk")} for h in horizons
    }
    sl = slice(lo, hi)
    for bid, ask in paths:
        bid = np.asarray(bid, np.float64)
        ask = np.asarray(ask, np.float64)
        e_fill = through_fill_second(bid, fill_sec)
        for h in horizons:
            tk, mk = oracle_long(bid, ask, h, cost=cost)
            ok = np.isfinite(tk[sl]) & np.isfinite(mk[sl])
            stk, smk = oracle_short(bid, ask, h, cost=cost)
            mp = maker_pessimistic(bid, e_fill, window_max(ask, h), cost=cost)
            for key, arr in (("tk", tk), ("mk", mk), ("mp", mp), ("stk", stk), ("smk", smk)):
                acc[h][key].append(arr[sl][ok])
    rows = []
    names = {
        "tk": "taker",
        "mk": "maker",
        "mp": "maker_pessimistic",
        "stk": "short_taker",
        "smk": "short_maker",
    }
    for h in horizons:
        for key, kind in names.items():
            x = np.concatenate(acc[h][key]) if acc[h][key] else np.empty(0)
            rec = {"horizon": int(h), "kind": kind, **_stats(x)}
            if key == "mp":
                rec["fill_rate"] = float(np.isfinite(x).sum() / max(len(x), 1))
            rows.append(rec)
    return pd.DataFrame(rows)
