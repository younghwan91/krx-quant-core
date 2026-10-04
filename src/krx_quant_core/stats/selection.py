"""후보 선택편향 — "검증 최대를 고른 값"이 실력 없이도 얼마나 올라가는가 (scalp-it 98번 이식).

체크포인트·시드·파라미터 후보 m 개 중 검증 점수 최대를 고르면, 후보들이 전부 동일한
잡음이어도 뽑힌 값은 풀 평균보다 높다. 그 몫을 **순서통계량으로 정확히** 센다 — 표본
시뮬레이션이 아니라 조합 가중치라 시드에 흔들리지 않는다. scalp-it rl92 는 후보 24 개·풀
sd 58bp 로 선택 프리미엄이 +166bp 였고, 풀링 평균은 −17.5bp 였다(98번).

여기 함수는 **기술(description)** 이지 판정이 아니다. "프리미엄이 Xbp 이상이면 기각" 같은
문턱은 각 레포의 사전등록 문서가 정한다. 입력 행의 열 이름은 scalp-it ``val_log.json``
(``val_n``·``val_daymean_bp``·``val_mean_bp``) 이 기본값이고 인자로 바꾼다.

원본 ``scripts/selection_98/selection_bias.py`` 와 같은 숫자를 내도록 필터·순회 순서·
동점 처리(strict ``>``, 먼저 만난 쪽이 이김)를 그대로 두었다. Spearman 은 scipy 대신
:func:`krx_quant_core.stats.metrics.spearman` (순위에 Pearson) — 동점 없는 자료에선 같다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .metrics import spearman

__all__ = [
    "SelectionBiasReport",
    "argmax_first",
    "eligible_rows",
    "expected_max_of_m",
    "selection_bias_report",
]

Row = Mapping[str, object]

#: 최댓값 기댓값 곡선을 찍는 후보 수(여기에 실제 후보 수가 더해진다).
DEFAULT_CURVE_M: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


def expected_max_of_m(values: Sequence[float] | np.ndarray, m: int) -> float:
    """후보 n 개에서 **비복원으로 m 개**를 뽑았을 때 최댓값의 기댓값.

    정렬한 값 v(0)≤…≤v(n−1) 에서 v(i) 가 최댓값일 확률은 C(i, m−1)/C(n, m) 이다.
    m=1 이면 평균, m≥n 이면 최댓값. 빈 입력이면 NaN.
    """
    v = np.sort(np.asarray(values, np.float64))
    n = len(v)
    if n == 0 or m < 1:
        return math.nan
    if m >= n:
        return float(v[-1])
    tot = math.comb(n, m)
    w = np.array([math.comb(i, m - 1) for i in range(n)], np.float64) / tot
    return float((v * w).sum())


def eligible_rows(rows: Sequence[Row], *, key: str, n_key: str, min_n: int) -> list[Row]:
    """``pick()`` 의 필터 — ``n_key ≥ min_n`` 이고 ``key`` 가 NaN/None 이 아닌 행."""
    out = []
    for r in rows:
        n = r.get(n_key, 0)
        v = r.get(key)
        if n is None or n < min_n or v is None or v != v:  # noqa: PLR0124 (NaN 검사)
            continue
        out.append(r)
    return out


def argmax_first(rows: Sequence[Row], key: str) -> Row | None:
    """strict ``>`` 로 최대를 고른다 — 동점이면 **먼저 만난** 행. NaN/None 은 건너뛴다."""
    best: Row | None = None
    for r in rows:
        v = r.get(key)
        if v is None or v != v:
            continue
        if best is None or v > best[key]:  # type: ignore[operator]
            best = r
    return best


@dataclass
class SelectionBiasReport:
    """:func:`selection_bias_report` 의 결과. ``to_dict()`` 가 JSON 직렬화용."""

    n_rows: int
    n_eligible: int
    min_n: int
    note: str | None = None
    #: 후보 풀 요약 — 평균·중앙·sd·최대, 풀링 평균·중앙, gap 중앙·p90.
    pool: dict[str, float] = field(default_factory=dict)
    #: 실제 규칙(일평균 최대)으로 뽑힌 행(식별 열 포함 원본 행 그대로).
    selected: dict[str, object] = field(default_factory=dict)
    #: 대안 규칙(풀링 평균 최대)으로 뽑힌 행.
    selected_by_pooled: dict[str, object] = field(default_factory=dict)
    same_selection: bool | None = None
    premium_vs_random: float = math.nan
    premium_vs_median: float = math.nan
    #: m → 비복원 m 개 중 최댓값의 기댓값.
    expected_max_curve: dict[int, float] = field(default_factory=dict)
    selected_gap: float = math.nan
    selected_gap_pctile: float = math.nan
    rho_n_vs_score: float = math.nan
    rho_n_vs_absdev: float = math.nan
    rho_n_vs_absgap: float = math.nan
    by_n_quartile: list[dict[str, float | int | None]] = field(default_factory=list)
    #: 대안 규칙 − 실제 규칙 (score, pooled, n).
    rule_delta: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expected_max_curve"] = {str(k): v for k, v in self.expected_max_curve.items()}
        return d


def selection_bias_report(
    rows: Sequence[Row],
    *,
    key: str = "val_daymean_bp",
    pooled_key: str = "val_mean_bp",
    n_key: str = "val_n",
    min_n: int = 30,
    curve_m: Sequence[int] = DEFAULT_CURVE_M,
) -> SelectionBiasReport:
    """후보 행들에서 선택 규칙 "``key`` 최대"가 얼마나 낙관적인지 잰다.

    세 가지를 낸다 — (1) 선택 프리미엄과 후보 수별 최댓값 기댓값 곡선, (2) 거래 수(``n_key``)와
    점수의 관계(적을수록 극단으로 가는가), (3) ``pooled_key`` 최대로 골랐다면 무엇이 뽑혔을까.
    ``key − pooled_key`` 를 gap 이라 부른다: 하루 거래 건수와 그날 성과의 공분산이다.
    """
    el = eligible_rows(rows, key=key, n_key=n_key, min_n=min_n)
    rep = SelectionBiasReport(n_rows=len(rows), n_eligible=len(el), min_n=min_n)
    if len(el) < 2:
        rep.note = "후보가 2개 미만이라 잴 게 없다"
        return rep

    score = np.array([float(r[key]) for r in el], np.float64)  # type: ignore[arg-type]
    pooled = np.array([float(r.get(pooled_key, math.nan)) for r in el], np.float64)  # type: ignore[arg-type]
    nn = np.array([float(r[n_key]) for r in el], np.float64)  # type: ignore[arg-type]
    gaps = score - pooled

    sel = argmax_first(el, key)
    sel_pool = argmax_first(el, pooled_key)
    assert sel is not None  # el 이 비어 있지 않고 key 가 NaN 아님
    rep.selected = dict(sel)
    rep.selected_by_pooled = dict(sel_pool) if sel_pool is not None else {}
    rep.same_selection = sel_pool is sel if sel_pool is not None else None

    med = float(np.median(score))
    rep.pool = {
        "score_mean": float(score.mean()),
        "score_median": med,
        "score_sd": float(score.std(ddof=1)),
        "score_max": float(score.max()),
        "pooled_mean": float(np.nanmean(pooled)) if np.isfinite(pooled).any() else math.nan,
        "pooled_median": float(np.nanmedian(pooled)) if np.isfinite(pooled).any() else math.nan,
        "gap_median": float(np.nanmedian(gaps)) if np.isfinite(gaps).any() else math.nan,
        "gap_p90": float(np.nanquantile(gaps, 0.9)) if np.isfinite(gaps).any() else math.nan,
    }
    sel_score = float(sel[key])  # type: ignore[arg-type]
    rep.premium_vs_random = sel_score - float(score.mean())
    rep.premium_vs_median = sel_score - med
    ms = sorted({*(int(m) for m in curve_m), len(el)})
    rep.expected_max_curve = {m: expected_max_of_m(score, m) for m in ms if 1 <= m <= len(el)}

    sel_gap = sel_score - float(sel.get(pooled_key, math.nan))  # type: ignore[arg-type]
    rep.selected_gap = sel_gap
    if np.isfinite(sel_gap) and np.isfinite(gaps).any():
        rep.selected_gap_pctile = float((gaps[np.isfinite(gaps)] <= sel_gap).mean())

    s_n = pd.Series(nn)
    with np.errstate(invalid="ignore"):  # 상수열(전부 같은 gap)이면 NaN — 경고 없이
        rep.rho_n_vs_score = spearman(s_n, pd.Series(score))
        rep.rho_n_vs_absdev = spearman(s_n, pd.Series(np.abs(score - med)))
        if np.isfinite(gaps).all():
            rep.rho_n_vs_absgap = spearman(s_n, pd.Series(np.abs(gaps)))

    q = np.quantile(nn, [0.25, 0.5, 0.75])
    edges = [nn.min() - 1, *q, nn.max() + 1]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (nn > lo) & (nn <= hi)
        if not m.any():
            continue
        rep.by_n_quartile.append(
            {
                "n_lo": float(nn[m].min()),
                "n_hi": float(nn[m].max()),
                "k": int(m.sum()),
                "score_mean": float(score[m].mean()),
                "score_sd": float(score[m].std(ddof=1)) if m.sum() > 1 else None,
                "score_max": float(score[m].max()),
                "absgap_median": float(np.nanmedian(np.abs(gaps[m])))
                if np.isfinite(gaps[m]).any()
                else None,
            }
        )
    if sel_pool is not None:
        rep.rule_delta = {
            "score": float(sel_pool[key]) - sel_score,  # type: ignore[arg-type]
            "pooled": float(sel_pool[pooled_key]) - float(sel[pooled_key]),  # type: ignore[arg-type]
            "n": float(sel_pool[n_key]) - float(sel[n_key]),  # type: ignore[arg-type]
        }
    return rep
