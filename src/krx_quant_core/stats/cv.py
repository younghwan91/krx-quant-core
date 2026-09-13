"""워크포워드 검증 — 폴드·no-lookahead 슬라이스·purge/embargo·폴드 일관성.

swing-it ``validation/walkforward.py`` 에서 옮겼다(수치 동일). 원본이 이미 시그널
무관(numpy 만 의존, 수익·진입일 배열과 ``simulate(params)`` 콜러블만 받음)이라
시장·전략에 묶인 것이 없다.

옮기지 **않은** 것: swing-it 의 모듈 상수 ``FOLDS`` (= ``rolling_folds()``).
그건 "폴드는 실험마다 바꾸는 손잡이가 아니다" 라는 swing-it 연구 규율의 동결값이고,
동결값은 그 규율을 지키는 레포가 들고 있어야 한다. ``rolling_folds()`` 기본 인자는
원본과 같아서 swing-it 은 ``FOLDS = rolling_folds()`` 한 줄로 그대로 재현한다.

두 설계 규칙:

1. **폴드는 동결 기본값이지 실험별 손잡이가 아니다.** 창을 조금씩 넓히고 옮기며 OOS
   숫자를 고르는 fold-shopping 은 그 자체로 look-ahead 위반이다.
2. **TRAIN 에서만 fit, OOS 에서만 평가.** ``walk_forward`` 는 ``[train_lo, train_hi)``
   에서 ``fit`` 하고 ``[test_lo, test_hi)`` 에서 평가한다. ``train_hi <= test_lo``.

날짜는 ISO 문자열(사전식 비교 = 시간 비교)을 기본으로 한다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

__all__ = [
    "Fold",
    "FoldMask",
    "Simulate",
    "entry_mask",
    "fold_consistency",
    "fold_slices",
    "oos_fixed",
    "purge_embargo",
    "rdist",
    "resolve_exit_dates",
    "rolling_folds",
    "slice_by_entry",
    "walk_forward",
]

Simulate = Callable[[dict], "tuple[np.ndarray, np.ndarray]"]


class Fold(NamedTuple):
    """하나의 walk-forward fold. TRAIN[lo,hi)에서 학습, TEST[lo,hi)에서 평가.

    NamedTuple 이라 평범한 4-튜플과 == 비교된다(기존 튜플 폴드와 호환)."""

    train_lo: str
    train_hi: str
    test_lo: str
    test_hi: str


def rolling_folds(
    *,
    first_test_year: int = 2020,
    last_test_year: int = 2025,
    train_years: int = 3,
    test_years: int = 1,
    final_test_hi: str | None = "2027-01-01",
) -> tuple[Fold, ...]:
    """롤링 fold 생성기: train ``train_years``년 → test ``test_years``년, test 연도 이동.

    기본 인자는 swing-it 의 frozen 6-fold 와 정확히 일치한다(회귀 고정).
    마지막 fold 는 잔여(부분 연도) 데이터를 모두 담도록 test 상한을 ``final_test_hi`` 로
    연장한다 — 마지막 test 연도 이후 데이터가 dangling partial fold 로 새지 않게."""
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        folds.append(
            Fold(f"{y - train_years}-01-01", f"{y}-01-01",
                 f"{y}-01-01", f"{y + test_years}-01-01")
        )
    if final_test_hi is not None and folds:
        f = folds[-1]
        folds[-1] = Fold(f.train_lo, f.train_hi, f.test_lo, final_test_hi)
    return tuple(folds)


def _expectancy(rets: np.ndarray) -> float:
    """유한수익의 건당 기대값(=평균). 비어있으면 NaN. 기본 fold 통계."""
    r = rets[np.isfinite(rets)] if len(rets) else rets
    return float(r.mean()) if len(r) else float("nan")


def entry_mask(edates: np.ndarray, lo=None, hi=None) -> np.ndarray:
    """[lo, hi) 진입일 마스크. lo/hi 는 ISO 날짜 문자열(사전식 비교 = 시간 비교)."""
    m = np.ones(len(edates), bool)
    if lo is not None:
        m &= edates >= lo
    if hi is not None:
        m &= edates < hi
    return m


def slice_by_entry(rets: np.ndarray, edates: np.ndarray, lo=None, hi=None) -> np.ndarray:
    """진입일이 [lo, hi) 인 수익만 반환. no-lookahead fold 슬라이스의 기본 연산."""
    return rets[entry_mask(edates, lo, hi)]


class FoldMask(NamedTuple):
    """한 fold 의 최종 TRAIN/TEST 소속 마스크(트레이드 배열에 대한 bool).

    기본(purge/embargo 미적용) 시 각각 ``entry_mask(edates, train_lo, train_hi)`` /
    ``entry_mask(edates, test_lo, test_hi)`` 와 정확히 동일하다(회귀 고정)."""

    train: np.ndarray
    test: np.ndarray


def resolve_exit_dates(entry_dates, *, exit_dates=None, max_hold=None) -> np.ndarray:
    """트레이드의 실현(청산)일을 ``datetime64[D]`` 로 정한다 — purge 의 라벨 종료시점.

    실현시점을 두 방식으로 받는다(둘 중 하나만):
      - ``exit_dates``: 명시적 청산일 배열.
      - ``max_hold``: 보유 상한 → 청산 ≈ 진입 + ``max_hold`` **달력일**. 거래일 상한을
        쓰는 호출자는 넉넉히(주말 포함) 올린 값을 넣거나 명시적 ``exit_dates`` 를 쓴다.

    둘 다 없으면 청산=진입(보유 0 근사) → purge 가 퇴화(=미적용). 둘 다 주면 ``ValueError``."""
    if exit_dates is not None and max_hold is not None:
        raise ValueError("exit_dates 와 max_hold 는 동시 지정 불가 — 하나만.")
    if exit_dates is not None:
        return np.asarray(exit_dates, dtype="datetime64[D]")
    entry = np.asarray(entry_dates, dtype="datetime64[D]")
    if max_hold is not None:
        if int(max_hold) < 0:
            raise ValueError("max_hold 는 음수 불가.")
        return entry + np.timedelta64(int(max_hold), "D")
    return entry


def purge_embargo(
    entry_dates,
    fold,
    *,
    exit_dates=None,
    max_hold=None,
    embargo_days: int = 0,
) -> FoldMask:
    """Purged K-fold + Embargo (López de Prado, *AFML* §7.4) — 보유기간 라벨 인접-누출 차단.

    폴드는 ``train_hi == test_lo`` 로 인접한다. 경계 직전 진입해 며칠 보유한 트레이드는
    진입일로는 TRAIN 이지만 **실현은 TEST 창 안**이라 라벨이 겹친다. 그걸 TRAIN 에서 걷어낸다.

      - **PURGE**: ``exit_date >= test_lo`` 인 TRAIN 트레이드 제거.
      - **EMBARGO**: ``[test_lo - embargo_days, test_lo)`` 에 진입한 TRAIN 트레이드 제거
        (TRAIN-앞-TEST 기하에 맞춘 AFML embargo 의 대칭형).

    embargo 는 **TRAIN 에서만** 제거하고 TEST 평가창은 절대 줄이지 않는다(OOS 지표 불변).

    **기본값은 무개입**: ``exit_dates``/``max_hold`` 없고 ``embargo_days=0`` 이면 반환 마스크는
    ``entry_mask`` 소속과 **정확히 동일**하다. purge/embargo 는 opt-in."""
    if int(embargo_days) < 0:
        raise ValueError("embargo_days 는 음수 불가.")
    train = entry_mask(entry_dates, fold.train_lo, fold.train_hi)
    test = entry_mask(entry_dates, fold.test_lo, fold.test_hi)
    test_lo = np.datetime64(fold.test_lo, "D")

    # PURGE: 실현이 TEST 창으로 넘어가는(exit >= test_lo) TRAIN 트레이드 제거.
    exits = resolve_exit_dates(entry_dates, exit_dates=exit_dates, max_hold=max_hold)
    train = train & ~(exits >= test_lo)

    # EMBARGO: [test_lo - embargo_days, test_lo) 진입 TRAIN 제거(테스트셋 앞 완충).
    if int(embargo_days) > 0:
        entry_dt = np.asarray(entry_dates, dtype="datetime64[D]")
        embargo_lo = test_lo - np.timedelta64(int(embargo_days), "D")
        train = train & ~(entry_dt >= embargo_lo)

    return FoldMask(train, test)


def oos_fixed(
    folds,
    simulate: Simulate,
    params: dict,
    *,
    stat: Callable[[np.ndarray], float] = _expectancy,
) -> np.ndarray:
    """고정 params 를 각 fold 의 TEST 구간에서 평가(재최적화 없음) → OOS 지표 배열.

    params 가 고정이라 sim 은 한 번만 돌리고 fold 별로 슬라이스한다(결과 동일)."""
    rets, edates = simulate(params)
    return np.array([stat(slice_by_entry(rets, edates, f.test_lo, f.test_hi))
                     for f in folds])


def walk_forward(
    folds,
    simulate: Simulate,
    fit: Callable[[str, str], dict],
    *,
    stat: Callable[[np.ndarray], float] = _expectancy,
) -> list:
    """fold 마다 TRAIN 서 fit → TEST 서 평가. no-lookahead 롤링 검증.

    ``fit(train_lo, train_hi) -> params`` 는 TRAIN 경계만 본다. IS≫OOS 반복이면 과최적.

    반환: fold 별 ``{"fold", "params", "is", "oos"}`` 레코드 리스트."""
    records = []
    for f in folds:
        params = fit(f.train_lo, f.train_hi)  # TRAIN-only fit
        rets, edates = simulate(params)
        records.append({
            "fold": f,
            "params": params,
            "is": stat(slice_by_entry(rets, edates, f.train_lo, f.train_hi)),
            "oos": stat(slice_by_entry(rets, edates, f.test_lo, f.test_hi)),
        })
    return records


def rdist(R: np.ndarray) -> dict:
    """개별표본 R-분포 요약: n·기대값R·≥3R 빈도·손익비. 복리 없음 — 개별 트레이드가 표본."""
    if len(R) == 0:
        return dict(n=0, expR=float("nan"), tail3=float("nan"), payoff=float("nan"))
    wins, losses = R[R > 0], R[R < 0]
    payoff = (wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan")
    return dict(n=len(R), expR=float(R.mean()), tail3=float(np.mean(R >= 3.0)), payoff=payoff)


def fold_slices(
    entry: np.ndarray,
    feature: np.ndarray,
    value: np.ndarray,
    fold,
    frac: float,
    *,
    min_train: int = 40,
    exit_dates=None,
    max_hold=None,
    embargo_days: int = 0,
):
    """TRAIN 에서 feature 의 (1-frac) 분위 θ 학습 → TEST 에 그대로 적용. look-ahead 없음.

    θ 를 TRAIN 에서 *학습*하므로 여기가 purge 가 필요한 자리다. ``exit_dates`` 또는
    ``max_hold`` 를 주면 :func:`purge_embargo` 로 TEST 에서 실현되는 트레이드를 TRAIN 에서
    걷어낸다. 안 주면 기존 동작 그대로(발표 수치가 그 경로에 고정돼 있어 opt-in).
    TEST 슬라이스는 어떤 경우에도 줄이지 않는다.

    반환: ``(theta, base_values, selected_values)`` 또는 TRAIN 부족시 ``None``."""
    fin = np.isfinite(feature) & np.isfinite(value)
    mask = purge_embargo(entry, fold, exit_dates=exit_dates, max_hold=max_hold,
                         embargo_days=embargo_days)
    tr = mask.train & fin
    te = mask.test & fin
    if tr.sum() < min_train:
        return None
    theta = np.quantile(feature[tr], 1.0 - frac)  # TRAIN 분위 = 고정 임계값
    base = value[te]
    sel = value[te & (feature >= theta)]  # TEST 에 θ 적용(진입시점 결정 가능)
    return theta, base, sel


def fold_consistency(
    entry: np.ndarray,
    feature: np.ndarray,
    value: np.ndarray,
    folds,
    frac: float,
    *,
    min_train: int = 40,
    min_sel: int = 15,
    exit_dates=None,
    max_hold=None,
    embargo_days: int = 0,
) -> dict:
    """여러 fold 에서 "선별이 R-분포를 굽히나" 를 재현성으로 센다(결정론 판정 아님).

    fold 마다 base(전부) vs 선별(feature≥θ) 기대값R 비교. 선별표본<min_sel 이면 폴드 무효.

    반환: ``{"valid", "bent", "rows"}`` — rows 는 fold 별 상세 dict."""
    valid = bent = 0
    rows = []
    for f in folds:
        fs = fold_slices(entry, feature, value, f, frac, min_train=min_train,
                         exit_dates=exit_dates, max_hold=max_hold,
                         embargo_days=embargo_days)
        if fs is None:
            rows.append({"fold": f, "status": "train_short"})
            continue
        theta, base, sel = fs
        if len(sel) < min_sel:
            rows.append({"fold": f, "status": "sparse", "theta": float(theta),
                         "base": rdist(base)})
            continue
        valid += 1
        b, s = rdist(base), rdist(sel)
        up = s["expR"] > b["expR"]
        bent += int(up)
        rows.append({"fold": f, "status": "ok", "theta": float(theta),
                     "base": b, "sel": s, "bent": bool(up)})
    return {"valid": valid, "bent": bent, "rows": rows}
