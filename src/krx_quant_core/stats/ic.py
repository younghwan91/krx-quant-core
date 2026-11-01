"""알파 모니터링 — IC 감쇠·이동 ICIR·상위 분위 회전율.

:func:`ic_decay`·:func:`rolling_ic` 는 :func:`krx_quant_core.backtest.crosssectional.rank_ic` 를
**그대로 부르는** 얇은 층이다(다시 구현하지 않는다). 그래서 같은 인자면 ``rank_ic`` 와 같은 일별 IC,
같은 ``ic_mean``·``nw_t``·``n`` 이 나온다 — 적격 필터(ADV 하한·20종목 하한·``start_index``),
선행수익 정의(``t+1`` 진입 → ``t+1+h`` 청산), HAC lag = ``h`` 가 전부 ``rank_ic`` 의 것이다.

입력 패널 규약은 ``backtest.crosssectional`` 과 같다: ``code × date`` numpy 배열,
``dates`` 는 열 라벨.

``trade_value`` 를 안 주면 ADV 하한을 끈다(``adv_floor=0``). ``trade_value`` 없이
``adv_floor > 0`` 을 주면 ``ValueError`` — 하한을 조용히 무시하지 않는다.
``trade_value`` 를 주면 ``rank_ic`` 기본값(``adv_floor=5000`` 등)이 그대로 적용된다.

숫자만 낸다(판정 키 없음). ICIR 은 **일별 IC 기준**(``ic_mean / ic_sd``, 연환산 안 함)이다.

포팅 출처: portfolio-research ``factor/research/ic.py``·``quantiles.py`` 의 아이디어(감쇠·IR·교체율)
— 미국·Sharadar·DataFrame 가정은 가져오지 않았다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["ic_decay", "rolling_ic", "signal_turnover"]

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 40, 60)


def _ic_series(
    signal: np.ndarray,
    close: np.ndarray,
    dates: list,
    horizon: int,
    trade_value: np.ndarray | None,
    rank_ic_kwargs: dict,
) -> dict:
    # 지연 import — stats 를 import 할 때 backtest 패키지 전체를 끌어오지 않는다.
    from ..backtest.crosssectional import rank_ic

    for k in ("horizon", "return_series"):
        if k in rank_ic_kwargs:
            raise TypeError(f"{k!r} 는 이 함수가 정한다 — rank_ic_kwargs 로 넘기지 않는다")
    kw = dict(rank_ic_kwargs)
    if trade_value is None:
        if kw.get("adv_floor", 0) > 0:
            raise ValueError("adv_floor > 0 은 trade_value 가 있어야 한다")
        kw["adv_floor"] = 0.0
        trade_value = np.zeros_like(np.asarray(close, dtype=float))  # adv_floor=0 이라 안 쓰인다
    return rank_ic(close, trade_value, signal, dates, horizon=horizon, return_series=True, **kw)


def _sd(ic: np.ndarray) -> float:
    return float(np.std(ic, ddof=1)) if len(ic) >= 2 else float("nan")


def _ir(mean: float, sd: float) -> float:
    return mean / sd if np.isfinite(sd) and sd > 0 else float("nan")


def ic_decay(
    signal: np.ndarray,
    close: np.ndarray,
    dates: list,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    *,
    trade_value: np.ndarray | None = None,
    **rank_ic_kwargs,
) -> pd.DataFrame:
    """지평별 순위 IC — 신호가 며칠 동안 살아 있나.

    지평 ``h`` 마다 ``rank_ic(close, trade_value, signal, dates, horizon=h, **rank_ic_kwargs)``
    를 한 번 부른다. ``ic_mean``·``nw_t``·``n`` 은 그 반환의 ``ic_mean``·``ic_nw_t``·``n_days``
    그대로, ``ic_sd`` 는 같은 일별 IC 열의 표본 표준편차(ddof=1), ``icir = ic_mean / ic_sd``.

    Returns:
        ``DataFrame[h, ic_mean, ic_sd, icir, nw_t, n]`` — ``horizons`` 순서, 한 지평 한 행.
    """
    rows = []
    for h in horizons:
        res = _ic_series(signal, close, dates, int(h), trade_value, rank_ic_kwargs)
        sd = _sd(res["ic"])
        rows.append({"h": int(h), "ic_mean": res["ic_mean"], "ic_sd": sd,
                     "icir": _ir(res["ic_mean"], sd), "nw_t": res["ic_nw_t"],
                     "n": int(res["n_days"])})
    return pd.DataFrame(rows, columns=["h", "ic_mean", "ic_sd", "icir", "nw_t", "n"])


def rolling_ic(
    signal: np.ndarray,
    close: np.ndarray,
    dates: list,
    *,
    horizon: int,
    window: int = 12,
    trade_value: np.ndarray | None = None,
    **rank_ic_kwargs,
) -> pd.DataFrame:
    """겹치지 않는 표본의 이동 IC·ICIR — 모델 드리프트 모니터링.

    ``rank_ic`` 의 일별 IC 열에서 **선행수익 창이 겹치지 않는 날**만 고른다: 첫 IC 날부터,
    직전 표본보다 ``dates`` 위치가 ``horizon`` 이상 뒤인 첫 IC 날을 차례로 집는다(20종목 미만이라
    빠진 날은 건너뛴다). 그 표본열에 ``window`` 개 이동창으로 평균·표준편차(ddof=1)·IR 을 낸다.
    ``horizon=20, window=12`` 면 대략 월 IC 1년 창이다. 창이 덜 찬 행은 NaN.

    Returns:
        ``DataFrame[date, ic, ic_mean, ic_sd, icir, n]`` — ``n`` 은 창 안 표본 수(덜 차면 그만큼).
    """
    if window < 2:
        raise ValueError("window 는 2 이상이어야 한다(표준편차)")
    res = _ic_series(signal, close, dates, int(horizon), trade_value, rank_ic_kwargs)
    pos = {d: i for i, d in enumerate(dates)}
    keep_ic: list[float] = []
    keep_dates: list = []
    last = None
    for v, d in zip(res["ic"], res["ic_dates"], strict=True):
        p = pos[d]
        if last is None or p - last >= horizon:
            keep_ic.append(float(v))
            keep_dates.append(d)
            last = p
    s = pd.Series(keep_ic, dtype=float)
    roll = s.rolling(window, min_periods=window)
    mean = roll.mean()
    sd = roll.std(ddof=1)
    icir = (mean / sd).where(sd > 0)
    n = s.rolling(window, min_periods=1).count().astype(int)
    return pd.DataFrame({"date": keep_dates, "ic": s, "ic_mean": mean, "ic_sd": sd,
                         "icir": icir, "n": n},
                        columns=["date", "ic", "ic_mean", "ic_sd", "icir", "n"])


def _top_members(col: np.ndarray, top_q: float) -> np.ndarray:
    ok = np.flatnonzero(np.isfinite(col))
    if ok.size == 0:
        return ok
    k = max(1, int(np.floor(top_q * ok.size)))
    # 큰 값 우선, 동점은 행 순서(코드 순) — 결정적.
    order = ok[np.argsort(-col[ok], kind="stable")]
    return order[:k]


def signal_turnover(
    signal: np.ndarray,
    *,
    top_q: float = 0.2,
    dates: list | None = None,
) -> pd.Series:
    """상위 분위 구성의 기간 간 교체율 ``1 − |A_t ∩ A_{t−1}| / |A_{t−1}|``.

    ``signal`` 은 ``code × date`` 패널이고 **열 하나 = 한 기간**이다 — 리밸런스 주기로 보려면
    호출부가 열을 고른다(예: ``signal[:, ::20]``). ``A_t`` 는 열 ``t`` 의 유한값 중 상위
    ``floor(top_q·n)``(최소 1) 종목, 동점은 행 순서로 끊는다. 첫 열과 어느 한쪽 구성이 빈 열은 NaN.

    Returns:
        길이 ``T`` Series(``name="turnover"``, 인덱스 = ``dates`` 또는 열 위치).
    """
    if not 0 < top_q <= 1:
        raise ValueError("top_q 는 (0, 1] 이어야 한다")
    S = np.asarray(signal, dtype=float)
    if S.ndim != 2:
        raise ValueError("signal 은 code × date 2차원 패널이어야 한다")
    T = S.shape[1]
    if dates is not None and len(dates) != T:
        raise ValueError("dates 길이가 signal 열 수와 다르다")
    out = np.full(T, np.nan)
    prev: set[int] | None = None
    for t in range(T):
        cur = set(_top_members(S[:, t], top_q).tolist())
        if prev and cur:
            out[t] = 1.0 - len(cur & prev) / len(prev)
        prev = cur
    return pd.Series(out, index=list(dates) if dates is not None else range(T), name="turnover")
