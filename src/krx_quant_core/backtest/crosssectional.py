"""횡단면 순위 틸트 시뮬레이션 — 정기 리밸런스 + 측정 회전율 비용.

swing-it ``engine/sim_crosssectional.py`` 를 **수치 동일**로 옮겼다(루프·기본값 그대로).
원래 PEAD 전략에서 뽑아낸 회계(``t+1`` 진입, ADV 하한, 회전율 비용, 대차 비용, 롱온리
초과수익)라, 다른 횡단면 실험이 그걸 다시 유도하지 않고 재사용하게 하려는 것이다.

모든 입력은 ``code × date`` numpy 패널이고 ``dates`` 는 열 라벨 목록이다(반환 ``periods``
프레임의 ``date`` 열에 그대로 실린다). DataFrame → 패널 변환은 :mod:`.panels` 참고.

기본값(``start_index=130``·``adv_floor`` 등)은 swing-it 연구 수치에 고정돼 있다 —
바꾸면 swing-it 의 발표 수치가 조용히 달라지므로 그대로 둔다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats.metrics import newey_west_t, summarize_periods

__all__ = ["rank_ic", "rank_tilt_backtest", "staggered_tranche_backtest"]


def _trailing_adv(V: np.ndarray, nD: int, adv_window: int, like: np.ndarray) -> np.ndarray:
    adv = np.full_like(like, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)
    return adv


def rank_tilt_backtest(
    close: np.ndarray,
    trade_value: np.ndarray,
    signal: np.ndarray,
    dates: list,
    *,
    horizon: int = 40,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
    cost_one_way: float = 0.0023,
    min_names: int = 30,
    start_index: int = 130,
    fresh_days: int = 0,
    long_only: bool = False,
    borrow_cost_annual: float = 0.0,
    top_n: int = 0,
    age: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """순위 가중 달러중립(또는 롱온리 초과수익) 북, 측정 비용 차감.

    ``horizon`` 거래일마다 리밸런스한다. ``t`` 시점 신호로 ``t+1`` 종가 진입(같은 종가
    look-ahead 방지), ``t+1+horizon`` 종가 청산. ADV 는 ``t`` 직전 ``adv_window`` 일
    평균(당일 제외).

    Args:
        close: ``code × date`` 가격 패널(상류에서 abs).
        trade_value: ``code × date`` 거래대금 패널(``adv_floor`` 와 같은 단위).
        signal: ``code × date`` 신호 패널.
        dates: 패널 열에 맞춘 날짜 라벨.
        long_only: 참이면 상위 틸트(또는 ``top_n`` 동일가중) 롱북의 **적격 유니버스 대비
            초과수익**. 공매도가 막힌 경우의 구현 가능한 형태.
        borrow_cost_annual: 달러중립일 때 숏 총액에 물리는 연 대차비용.
        age: ``code × date`` 공시 경과일 패널; ``fresh_days>0`` 일 때만 본다.
            ``None`` → 전부 NaN(신선도 게이트 없음).

    Returns:
        ``(periods, summary)`` — periods 열은 ``date, gross, turnover, net``,
        summary 는 :func:`~krx_quant_core.stats.metrics.summarize_periods` 결과.
    """
    C = close
    V = trade_value
    yoy = signal
    if age is None:
        age = np.full_like(yoy, np.nan)
    nD = len(dates)

    adv = _trailing_adv(V, nD, adv_window, C)

    def fwd(t: int, h: int) -> np.ndarray:
        return C[:, t + h] / C[:, t] - 1.0 if t + h < nD else np.full(C.shape[0], np.nan)

    rows: list[dict] = []
    prev_w = np.zeros(C.shape[0])
    t = start_index
    while t < nD - horizon - 1:
        sig = yoy[:, t].copy()
        if fresh_days > 0:
            sig = np.where(age[:, t] <= fresh_days, sig, np.nan)
        ok = np.isfinite(sig)
        ret = fwd(t + 1, horizon)  # t+1 진입 — 같은 종가 look-ahead 방지
        ok &= np.isfinite(ret)
        if adv_floor > 0:
            ok &= adv[:, t] >= adv_floor
        if ok.sum() < min_names:
            t += horizon
            continue
        idx = np.where(ok)[0]
        pct = pd.Series(sig[ok]).rank(pct=True).to_numpy()
        w = np.zeros(C.shape[0])
        if long_only:
            if top_n > 0:
                # 집중 동일가중 top-N: 적고 큰 비대칭 베팅(낮은 승률·높은 손익비).
                sel = idx[np.argsort(-sig[ok])[:top_n]]
                w[sel] = 1.0 / len(sel)
            else:
                lw = np.clip(pct - 0.5, 0, None)  # 북 전체 순위 틸트
                w[idx] = lw / lw.sum() if lw.sum() > 0 else 0.0
            bench = float(np.nanmean(ret[idx]))
            gross = float(np.nansum(w * np.nan_to_num(ret))) - bench
            short_gross = 0.0
        else:
            w[idx] = (pct - 0.5) / np.abs(pct - 0.5).sum()  # 달러중립, 총노출=1
            gross = float(np.nansum(w * np.nan_to_num(ret)))
            short_gross = float(np.abs(w[w < 0]).sum())  # 중립북이면 ~0.5
        turnover = float(np.abs(w - prev_w).sum())
        borrow = borrow_cost_annual * (horizon / 252.0) * short_gross
        rows.append({"date": dates[t], "gross": gross, "turnover": turnover,
                     "net": gross - turnover * cost_one_way - borrow})
        prev_w = w
        t += horizon

    periods = pd.DataFrame(rows)
    return periods, summarize_periods(periods, horizon)


def staggered_tranche_backtest(
    close: np.ndarray,
    trade_value: np.ndarray,
    signal: np.ndarray,
    dates: list,
    *,
    horizon: int = 60,
    step: int = 20,
    top_n: int = 40,
    adv_floor: float = 20000.0,
    adv_window: int = 20,
    start_index: int = 130,
    min_names: int = 20,
    cap_array: np.ndarray | None = None,
    cap_rank: tuple[int, int] | None = None,
    delisting_exit: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """분할 진입 롱온리 초과수익 — ``horizon // step`` 개 트랜치를 겹쳐 보유.

    ``step`` 일마다 새 트랜치를 top-N 으로 짜고, 지난 트랜치들(``t - k·step``)의 북을 함께
    든다. 각 구간 수익은 트랜치별 북 평균 − 적격 유니버스 평균(벤치)의 평균이고, 회전율은
    ``1/n_tranches`` 로 고정 보고한다(이 함수는 비용을 떼지 않는다 — ``net == gross``).

    Args:
        cap_array: ``code × date`` 시가총액 패널 또는 ``None``.
        cap_rank: ``(lo, hi)`` 시총 순위 구간으로 유니버스 제한(``cap_array`` 와 함께).
        delisting_exit: 보유 중 상장폐지된 종목의 손실을 반영할지.

            기본 ``False`` 는 기존 동작이다: 폐지 후 가격이 NaN 이 되고 ``np.nanmean`` 이
            그 종목을 **조용히 빼버린다** — 북과 벤치마크 양쪽을 낙관 쪽으로 왜곡한다.
            ``True`` 면 마지막 관측 종가(정리매매 종료가)로 청산한 것으로 본다. 한국 시장은
            폐지 전 정리매매(보통 7거래일)가 있어 임의 상수(-30% 등)보다 데이터에 충실하다.
            다만 정리매매 자체를 못 판 경우는 반영되지 않으므로 이 역시 낙관 쪽 하한이다.

    Returns:
        ``(periods, summary)`` — periods 열은 ``date, gross, turnover, net, book, bench,
        n_universe``. book/bench 를 함께 남기는 건 초과수익이 전략 개선인지 벤치 악화인지
        가르기 위해서다(생존편향 같은 유니버스 변경에서 결론을 뒤집는다).
    """
    C = close
    V = trade_value
    sig_m = signal
    nD = len(dates)
    adv = _trailing_adv(V, nD, adv_window, C)
    n_tranches = max(1, horizon // step)
    capm = cap_array

    # 폐지 청산가: 각 시점까지의 마지막 관측 종가(행 방향 forward-fill). 꺼져 있으면 안 만든다.
    C_ff = pd.DataFrame(C).ffill(axis=1).to_numpy(float) if delisting_exit else None

    # eligible/book 은 t 의 순수함수라 메모이즈해도 숫자가 안 바뀐다. 스태거링이 지난
    # 리밸런스일을 다시 부르므로, 캐시가 없으면 같은 날 전유니버스 스캔이 트랜치 수만큼 반복된다.
    _elig_cache: dict[int, np.ndarray] = {}
    _book_cache: dict[int, np.ndarray | None] = {}

    def eligible(t: int) -> np.ndarray:
        cached = _elig_cache.get(t)
        if cached is not None:
            return cached
        ok = np.isfinite(sig_m[:, t]) & (adv[:, t] >= adv_floor)
        if capm is not None and cap_rank is not None:
            liq = np.where(ok & np.isfinite(capm[:, t]))[0]
            order = liq[np.argsort(-capm[liq, t])]  # 시총 내림차순
            tier = order[cap_rank[0]:cap_rank[1]]
            mask = np.zeros(C.shape[0], bool)
            mask[tier] = True
            ok = ok & mask
        _elig_cache[t] = ok
        return ok

    def book(t: int) -> np.ndarray | None:
        if t in _book_cache:
            return _book_cache[t]
        ok = eligible(t)
        if ok.sum() < min_names:
            _book_cache[t] = None
            return None
        idx = np.where(ok)[0]
        out = idx[np.argsort(-sig_m[idx, t])[:top_n]]
        _book_cache[t] = out
        return out

    rows: list[dict] = []
    for t in range(start_index, nD - step - 1, step):
        uni = np.where(eligible(t))[0]
        if uni.size < min_names:
            continue
        ret = C[:, t + step] / C[:, t] - 1.0
        if C_ff is not None:
            # 진입 시점엔 가격이 있었는데 청산 시점에 없는 종목 = 보유 중 상장폐지.
            gone = np.isfinite(C[:, t]) & ~np.isfinite(C[:, t + step])
            ret[gone] = C_ff[gone, t + step] / C[gone, t] - 1.0
        bench = float(np.nanmean(ret[uni]))
        tranche_excess, tranche_book = [], []
        for k in range(n_tranches):
            b = book(t - k * step)
            if b is not None:
                book_ret = float(np.nanmean(ret[b]))
                tranche_book.append(book_ret)
                tranche_excess.append(book_ret - bench)
        if tranche_excess:
            rows.append({"date": dates[t], "gross": float(np.mean(tranche_excess)),
                         "turnover": 1.0 / n_tranches, "net": float(np.mean(tranche_excess)),
                         "book": float(np.mean(tranche_book)), "bench": bench,
                         "n_universe": int(uni.size)})
    periods = pd.DataFrame(rows)
    return periods, summarize_periods(periods, step)


def rank_ic(
    close: np.ndarray,
    trade_value: np.ndarray,
    signal: np.ndarray,
    dates: list,
    *,
    horizon: int = 40,
    adv_floor: float = 5000.0,
    adv_window: int = 20,
    start_index: int = 130,
    fresh_days: int = 0,
    n_regimes: int = 4,
    age: np.ndarray | None = None,
) -> dict:
    """신호 vs 선행수익의 일별 횡단면 순위 IC + Newey-West t + 국면별 분해.

    적격 종목이 20개 미만인 날은 건너뛴다. HAC lag = ``horizon`` (겹치는 선행수익 보정).
    ``dates`` 라벨의 앞 7자(``YYYY-MM``)가 국면 start/end 로 실린다.

    Returns:
        ``{"ic_mean", "ic_nw_t", "n_days", "frac_positive", "regimes"}``.
    """
    C = close
    V = trade_value
    yoy = signal
    if age is None:
        age = np.full_like(yoy, np.nan)
    nD = len(dates)
    adv = _trailing_adv(V, nD, adv_window, C)

    ics: list[float] = []
    ic_dates: list[str] = []
    for t in range(start_index, nD - horizon - 1):
        sig = yoy[:, t].copy()
        if fresh_days > 0:
            sig = np.where(age[:, t] <= fresh_days, sig, np.nan)
        ok = np.isfinite(sig)
        if adv_floor > 0:
            ok &= adv[:, t] >= adv_floor
        ret = C[:, t + 1 + horizon] / C[:, t + 1] - 1.0
        ok &= np.isfinite(ret)
        if ok.sum() < 20:
            continue
        a = pd.Series(sig[ok]).rank().to_numpy()
        b = pd.Series(ret[ok]).rank().to_numpy()
        if a.std() > 0 and b.std() > 0:
            ics.append(float(np.corrcoef(a, b)[0, 1]))
            ic_dates.append(dates[t])

    ic = np.array(ics)
    mean_ic, nw_t = newey_west_t(ic, horizon)
    regimes: list[dict] = []
    if len(ic) >= n_regimes:
        b = len(ic) // n_regimes
        for k in range(n_regimes):
            s0 = k * b
            s1 = (k + 1) * b if k < n_regimes - 1 else len(ic)
            m, tt = newey_west_t(ic[s0:s1], horizon)
            regimes.append({"start": ic_dates[s0][:7], "end": ic_dates[s1 - 1][:7],
                            "ic_mean": m, "nw_t": tt})
    return {
        "ic_mean": mean_ic, "ic_nw_t": nw_t, "n_days": len(ic),
        "frac_positive": float((ic > 0).mean()) if len(ic) else float("nan"),
        "regimes": regimes,
    }
