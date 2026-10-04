"""매치드 널 대조 — "신호 뒤의 수익이 같은 자리의 무작위 수익보다 정말 큰가" (scalp-it 101번 A).

딥 체결·급등주·고변동 순간처럼 신호가 **정의상** 특정 국면에서만 켜지면, 무작위 대조군과
비교할 때 이기는 게 당연할 수 있다 — 창 안 최댓값은 σ√H 로 커지므로 변동성이 큰 순간을 골랐다는
것만으로 성과가 난다. 그래서 대조군을 신호와 **같은 층**(날짜 × 분 × 변동성 분위 × 거래대금 분위
등)에서 뽑고, 층 안에서만 비교한다. 층 안에서 차이가 사라지면 국면이었고, 남으면 신호가 정보다.

- :func:`matched_control` — 각 트레이드에 같은 strata 키 조합의 풀에서 대조 ``n_per`` 개를 뽑는다.
- :func:`matched_alpha` — alpha = 트레이드 평균 − 대조 평균, **날짜 클러스터 부트스트랩** CI.
  같은 날의 트레이드와 대조를 통째로 복원 추출한다(날 안의 관측은 독립이 아니다).

판정하지 않는다. "CI 하한 > 0" 같은 통과 기준은 사전등록 문서가 쓴다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["MatchedAlpha", "cluster_bootstrap_ci", "matched_alpha", "matched_control"]


def matched_control(
    trades: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    strata: Sequence[str],
    n_per: int = 1,
    seed: int = 0,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """트레이드마다 같은 strata 의 풀 행을 ``n_per`` 개 무작위(복원) 추출.

    반환: ``trade_idx``(trades 의 index 값)·``control_idx``(pool 의 index 값)·strata 열.
    짝이 없는 strata 의 트레이드는 결과에 없고, ``attrs["unmatched"]`` 에 그 index 목록이 남는다.
    ``exclude_self`` 면 pool 과 trades 가 같은 index 공간일 때 자기 자신은 뽑지 않는다.
    strata 열이 없으면 ``KeyError`` 에 열 이름을 담는다.
    """
    strata = list(strata)
    for col in strata:
        if col not in trades.columns:
            raise KeyError(f"trades 에 strata 열이 없다: {col!r}")
        if col not in pool.columns:
            raise KeyError(f"pool 에 strata 열이 없다: {col!r}")
    rng = np.random.default_rng(seed)
    groups = {k: g.index.to_numpy() for k, g in pool.groupby(strata, sort=False, dropna=True)}
    rows: list[dict] = []
    unmatched: list = []
    for key, g in trades.groupby(strata, sort=False, dropna=True):
        cand_all = groups.get(key)
        for tidx in g.index:
            cand = cand_all
            if cand is not None and exclude_self:
                cand = cand[cand != tidx]
            if cand is None or len(cand) == 0:
                unmatched.append(tidx)
                continue
            pick = rng.choice(cand, size=n_per, replace=True)
            key_t = key if isinstance(key, tuple) else (key,)
            for c in pick:
                rows.append(
                    {"trade_idx": tidx, "control_idx": c, **dict(zip(strata, key_t, strict=True))}
                )
    nan_mask = trades[strata].isna().any(axis=1)
    unmatched.extend(trades.index[nan_mask].tolist())
    out = pd.DataFrame(rows, columns=["trade_idx", "control_idx", *strata])
    out.attrs["unmatched"] = unmatched
    return out


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """클러스터(날짜)를 복원 추출해 가중 평균의 분포를 만든다 → ``(lo, hi, boot_means)``.

    ``weights`` 는 관측별 가중치(대조군 평균을 트레이드 1 : 대조 −1 로 접을 때 쓴다). 클러스터가
    2 개 미만이면 NaN 쌍.
    """
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = len(uniq)
    if k < 2:
        return (np.nan, np.nan, np.empty(0))
    # 클러스터별 합과 가중치 합 → 재추출은 클러스터 단위로만 하면 된다.
    s = np.bincount(inv, weights=values * w, minlength=k)
    n = np.bincount(inv, weights=w, minlength=k)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, k, size=(n_boot, k))
    num = s[draw].sum(axis=1)
    den = n[draw].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = num / den
    lo = (1.0 - ci) / 2.0 * 100.0
    hi = (1.0 + ci) / 2.0 * 100.0
    return (float(np.nanpercentile(means, lo)), float(np.nanpercentile(means, hi)), means)


@dataclass
class MatchedAlpha:
    alpha: float
    ci_low: float
    ci_high: float
    #: 부트스트랩 표본 중 alpha > 0 비율.
    p_greater: float
    trade_mean: float
    control_mean: float
    n_trades: int
    n_controls: int
    n_clusters: int
    n_unmatched: int
    #: strata 조합별 alpha·n (층 안 비교 — 어느 층에서 남는지 본다).
    by_stratum: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


def matched_alpha(
    trades: pd.DataFrame,
    pool: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    value: str,
    cluster: str = "date",
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> MatchedAlpha:
    """alpha = 트레이드 ``value`` 평균 − 매치드 대조 ``value`` 평균, 클러스터 부트스트랩 CI.

    ``controls`` 는 :func:`matched_control` 의 결과. 대조는 트레이드 하나당 ``n_per`` 개이므로
    각 대조에 ``1/n_per`` 가중치를 줘 트레이드 1 건 : 대조 1 건으로 맞춘다. 클러스터 열은 트레이드
    쪽에서 읽고 대조는 짝지은 트레이드의 클러스터를 따른다(같은 날 묶음).
    """
    if len(controls) == 0:
        return MatchedAlpha(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0,
                            len(controls.attrs.get("unmatched", [])))
    strata = [c for c in controls.columns if c not in ("trade_idx", "control_idx")]
    matched_trades = controls["trade_idx"].unique()
    tv = trades.loc[matched_trades, value].astype(float)
    tc = trades.loc[matched_trades, cluster]
    cv = pool.loc[controls["control_idx"].to_numpy(), value].astype(float).to_numpy()
    per = controls.groupby("trade_idx").size()
    cw = (1.0 / per.loc[controls["trade_idx"]].to_numpy())
    cc = tc.loc[controls["trade_idx"]].to_numpy()

    vals = np.concatenate([tv.to_numpy(), cv])
    wts = np.concatenate([np.ones(len(tv)), cw])
    sign = np.concatenate([np.ones(len(tv)), -np.ones(len(cv))])
    clus = np.concatenate([tc.to_numpy(), cc])
    # 가중 평균 = Σ(sign·w·v)/Σ(w)/… 를 한 번에: 트레이드 쪽 가중 합과 대조 쪽 가중 합이 같으므로
    # (sign·v) 의 가중 평균 ×2 가 alpha 다 (Σw_trade = Σw_control = n_trades).
    lo, hi, means = cluster_bootstrap_ci(
        sign * vals, clus, n_boot=n_boot, seed=seed, ci=ci, weights=wts
    )
    scale = 2.0
    alpha = float(tv.mean() - np.average(cv, weights=cw))
    ok = np.isfinite(means)
    rows = []
    if strata:
        joined = controls.assign(
            tv=tv.loc[controls["trade_idx"]].to_numpy(), cv=cv, w=cw
        )
        for k, g in joined.groupby(strata, sort=True):
            t_mean = g.drop_duplicates("trade_idx")["tv"].mean()
            c_mean = np.average(g["cv"], weights=g["w"])
            rows.append(
                {**dict(zip(strata, k if isinstance(k, tuple) else (k,), strict=True)),
                 "alpha": float(t_mean - c_mean), "n_trades": int(g["trade_idx"].nunique()),
                 "n_controls": int(len(g))}
            )
    return MatchedAlpha(
        alpha=alpha,
        ci_low=lo * scale,
        ci_high=hi * scale,
        p_greater=float((means[ok] > 0).mean()) if ok.any() else np.nan,
        trade_mean=float(tv.mean()),
        control_mean=float(np.average(cv, weights=cw)),
        n_trades=int(len(tv)),
        n_controls=int(len(cv)),
        n_clusters=int(pd.unique(clus).size),
        n_unmatched=len(controls.attrs.get("unmatched", [])),
        by_stratum=pd.DataFrame(rows),
    )
