"""다중검정 보정 — Deflated Sharpe, PSR, Harvey-Liu t-haircut, 기대값 부트스트랩 CI.

swing-it ``diagnostics/gate_report.py`` 에서 **계산 부분만** 옮겼다. ``gate_report`` 자체
(괴물 의존도·폴드 일관성·엣지 사망 비용을 한 dict 로 묶는 리포터)는 swing-it 의 연구
규율에 묶여 있어 거기 남긴다. 여기 있는 건 어느 레포에서든 똑같이 계산돼야 하는
숫자들이다 — 같은 N 개 시행에 대해 레포마다 다른 SR0 가 나오면 비교가 무의미해진다.

이식 대응(수치 동일):
    _bootstrap_ci           -> bootstrap_mean_ci
    _expected_max_sharpe_h0 -> expected_max_sharpe_h0
    _prob_sharpe            -> probabilistic_sharpe
    _t_haircut              -> t_haircut
    deflated_sharpe         -> deflated_sharpe
    _deflation_block        -> deflated_sharpe_from_sample

시도한 config 수 N 을 안 세면 "N개 중 최고 Sharpe" 는 선택편향으로 부풀려진다.
아래는 그 부풀림을 **숫자로** 깎는다(판정 아님 — PASS/FAIL 없음).

  Deflated Sharpe Ratio — Bailey & López de Prado (2014), "The Deflated Sharpe
    Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
    Journal of Portfolio Management 40(5), 94–107.
  t-haircut — Harvey & Liu (2014), "…and the Cross-Section of Expected Returns":
    다중검정하에서 새 팩터는 t>2.0 이 아니라 t>3.0 수준을 요구한다.

scipy 를 쓰지 않는다 — 표준정규 분위/누적은 ``statistics.NormalDist`` 로 충분하다.
"""

from __future__ import annotations

from math import e
from statistics import NormalDist

import numpy as np

__all__ = [
    "bootstrap_mean_ci",
    "deflated_sharpe",
    "deflated_sharpe_from_sample",
    "expected_max_sharpe_h0",
    "probabilistic_sharpe",
    "t_haircut",
]

NAN = float("nan")

_EULER_MASCHERONI = 0.5772156649015329  # γ — E[max of N Gaussians] 근사에 사용
_NORM = NormalDist()  # 표준정규 (inv_cdf=ppf, cdf)


def bootstrap_mean_ci(
    R: np.ndarray, *, n_boot: int, seed: int, ci: float
) -> tuple[float, float]:
    """기대값(=평균)의 i.i.d. 부트스트랩 신뢰구간 ``(lo, hi)``. 유한 표본 2개 미만이면 NaN 쌍.

    분위는 ``(1-ci)/2·100`` 로 계산한다 — ``ci=0.95`` 여도 부동소수 때문에 정확히 2.5 가
    아니다. swing-it 원본과 비트 단위로 같게 하려고 그 식을 그대로 둔다.
    """
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    if len(R) < 2:
        return (NAN, NAN)
    rng = np.random.default_rng(seed)
    means = R[rng.integers(0, len(R), size=(n_boot, len(R)))].mean(axis=1)
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    return (float(np.percentile(means, lo_pct)), float(np.percentile(means, hi_pct)))


def expected_max_sharpe_h0(n_trials: int, sr_std: float) -> float:
    """N개 독립 시행 중 **최고 Sharpe 의 기대치** (H0: 참 Sharpe=0) = deflation 벤치마크 SR0.

    참 엣지가 전혀 없어도(모든 시행 SR=0) 표본잡음만으로 최고 시행의 Sharpe 는 0보다
    크다. 그 기대 최대값(Bailey & LdP):

        SR0 ≈ sr_std · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    ``sr_std`` 는 시행별 Sharpe 추정치의 표준편차, γ=Euler-Mascheroni. N≤1 이면 0.
    """
    if n_trials is None or n_trials <= 1 or not np.isfinite(sr_std):
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * e))
    return float(sr_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def probabilistic_sharpe(
    sharpe: float, benchmark: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """PSR — 참 Sharpe 가 ``benchmark`` 를 넘을 확률 (Bailey & LdP). ``kurtosis`` 는 비초과(정규=3).

        PSR = Φ( (SR − benchmark)·√(T−1) / √(1 − skew·SR + (kurt−1)/4·SR²) )

    비정규성(음의 왜도·두꺼운 꼬리)과 표본길이 T 를 반영해 Sharpe 유의성을 깎는다.
    표본부족·퇴화(분모≤0)면 NaN.
    """
    if n_obs < 2 or not np.isfinite(sharpe):
        return NAN
    denom = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denom <= 0:
        return NAN
    z = (sharpe - benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(_NORM.cdf(z))


def t_haircut(n_trials: int, *, alpha: float = 0.05) -> dict:
    """Harvey-Liu 스타일 t-haircut — 시행 N 이 늘면 유의 t 문턱이 오른다(Bonferroni).

    단일검정 양측 α 의 문턱 t₀(≈1.96, α=0.05)는 N개 다중검정에서 α/N 로 조여져 문턱이
    상승한다. ``haircut_multiple`` = 조정문턱/기본문턱 → raw t 는 그만큼 약해 보인다.
    (참고: Harvey-Liu 는 새 팩터에 t>3.0 권고 — N≈20 이면 문턱이 대략 3.0.)
    """
    base = _NORM.inv_cdf(1.0 - alpha / 2.0)
    n = max(int(n_trials), 1)
    adj = _NORM.inv_cdf(1.0 - alpha / (2.0 * n))
    return {
        "alpha": float(alpha),
        "method": "Bonferroni (two-sided)",
        "base_hurdle_t": float(base),
        "adjusted_hurdle_t": float(adj),
        "haircut_multiple": float(adj / base),
    }


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
    *,
    sr_std: float | None = None,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014) — 선택편향 보정 리포트.

    관측 Sharpe 를 **N개 시행 중 최고를 뽑았다는 선택편향** + 비정규성 + 표본길이로 깎는다.

    반환 dict:
        n_trials, observed_sharpe, expected_max_sharpe_h0 (=SR0, deflation 벤치마크),
        deflated_sharpe (=SR − SR0 — SR0 아래면 "N개 뽑기의 운"으로 설명 가능),
        prob_sharpe_gt0 (PSR vs 0), prob_deflated_sharpe (PSR vs SR0 = 정통 DSR 확률),
        t_haircut (Harvey-Liu 조정문턱).

    ``sr_std`` (시행별 Sharpe 추정치 표준편차)를 안 주면 H0 하의 Sharpe 추정량
    표준오차 ≈ 1/√(T−1) 로 근사한다. ``sharpe`` 는 표본(건당) Sharpe = mean/std.
    ``kurtosis`` 는 비초과(정규=3). 판정 아님 — 숫자만.
    """
    n_obs = int(n_obs)
    if sr_std is None:
        sr_std = 1.0 / np.sqrt(n_obs - 1) if n_obs > 1 else NAN
    sr0 = expected_max_sharpe_h0(n_trials, sr_std)
    return {
        "n_trials": int(n_trials),
        "observed_sharpe": float(sharpe),
        "expected_max_sharpe_h0": float(sr0),
        "deflated_sharpe": float(sharpe - sr0),
        "prob_sharpe_gt0": probabilistic_sharpe(sharpe, 0.0, n_obs, skew, kurtosis),
        "prob_deflated_sharpe": probabilistic_sharpe(sharpe, sr0, n_obs, skew, kurtosis),
        "t_haircut": t_haircut(n_trials),
    }


def deflated_sharpe_from_sample(R: np.ndarray, n_trials: int) -> dict:
    """표본(건당 수익·R-멀티플) → Sharpe·왜도·첨도를 계산해 :func:`deflated_sharpe` 로 리포트.

    Sharpe = mean / std(ddof=1). 왜도·첨도는 **모집단** 모멘트(ddof=0) 관례이고 첨도는
    비초과(정규=3). 표본<2 이거나 std=0 이면 NaN 으로 넘긴다(퇴화 방어).
    swing-it ``gate_report._deflation_block`` 과 같은 식이다.

    퇴화 판정은 ``std < 1e-12`` 다. swing-it 원본은 ``std == 0`` 정확 비교라 0.3 처럼
    이진 표현이 안 되는 상수열에서 반올림 잔차(std≈1e-17)로 Sharpe 가 1e15 대로
    폭주했다 — 이식하면서 고쳤다(``stats.metrics.ann_sharpe`` 와 같은 문턱).
    """
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    n = int(len(R))
    if n < 2 or R.std(ddof=1) < 1e-12:
        sr = skew = kurt = NAN
    else:
        mu = R.mean()
        sr = float(mu / R.std(ddof=1))
        sd = R.std()  # 모집단 std — 왜도 관례와 일치
        skew = float(((R - mu) ** 3).mean() / sd**3)
        kurt = float(((R - mu) ** 4).mean() / sd**4)
    return deflated_sharpe(sr, n_trials, n, skew, kurt)
