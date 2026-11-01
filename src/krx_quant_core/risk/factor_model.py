"""횡단면 팩터 위험모델 — 노출 표준화 · 제약 WLS 팩터 수익 · EWMA 공분산 · 특이위험 수축.

구조는 Barra 류다. 날짜 ``t`` 의 수익은 **전일(t-1) 마감 기준 노출**로 설명한다::

    r_t = X_{t-1} f_t + u_t,        Σ = X F Xᵀ + diag(D)

팩터 순서는 ``["market", <업종…>, <스타일…>]`` 로 고정이다. 시장(전 종목 노출 1)과 업종 더미는
공선이라, 업종 팩터 수익의 **시총가중 합 = 0** 제약으로 푼다(더미 하나를 빼는 방식이 아니다 —
그러면 빠진 업종이 시장 팩터에 섞인다). 제약 WLS 는 기준 업종 하나를 나머지로 치환해
제약 없는 WLS 로 **정확히** 바꿔 푼다(:func:`cross_section_regression`).

패널 규약은 :mod:`krx_quant_core.backtest.crosssectional` 과 같다 — ``code × date`` numpy 배열.
날짜 ``t`` 열에는 ``t`` 장 마감까지 알 수 있던 값만 있어야 한다(로더 책임). 이 모듈은
:func:`build_risk_model` 에서 ``asof_index`` 뒤 열을 **잘라낸 뒤** 계산을 시작하므로, 그 뒤 값이
무엇이든 결과는 비트 단위로 같다.

결측 규약: NaN = 모름. 표준화·회귀에서 결측은 0(평균)으로 채우지 않고 행째 뺀다.
위험모델은 숫자만 낸다(판정 키 없음). 공분산·분산은 **일 단위**다 — 연환산은 호출부가 한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FactorReturns",
    "RiskModel",
    "bias_statistic",
    "build_risk_model",
    "cross_section_regression",
    "ewma_cov",
    "factor_returns",
    "industry_dummies",
    "specific_var",
    "standardize",
]

MARKET = "market"
_MAD_TO_SD = 1.4826  # 정규분포에서 MAD → 표준편차 환산 계수


# ── 노출 ────────────────────────────────────────────────────────────────────────


def standardize(
    raw: np.ndarray,
    cap: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    clip: float = 3.0,
) -> np.ndarray:
    """한 날짜의 원시 스타일 값을 노출로 — 윈저 → 시총가중 평균 0 → 동일가중 표준편차 1.

    1. 유효 = ``mask`` 참 · ``raw`` 유한 · ``cap`` 유한 양수. 나머지는 결과에서 NaN.
    2. 유효값의 중앙값 ± ``clip``·σ̂ 로 윈저(σ̂ = 1.4826·MAD, 정규 일치 척도). MAD 가 0 이면 건너뛴다.
    3. 시총가중 평균을 빼고, 동일가중 표준편차(ddof=0)로 나눈다. 평균을 시총가중으로 빼므로
       시총가중 유니버스의 노출이 0 이다(시장 팩터와 분리).

    결측은 0(평균)으로 채우지 않는다 — 호출부가 마스크로 뺀다. 유효가 2개 미만이면 전부 NaN,
    유효값이 모두 같으면(표준편차 0) 유효 자리는 0.

    Args:
        raw: ``(N,)`` 원시 값.
        cap: ``(N,)`` 시가총액(가중용, 단위 무관).
        mask: ``(N,)`` bool 유니버스. ``None`` 이면 전부 참.
        clip: 윈저 폭(σ̂ 배수). ``0`` 이하면 윈저하지 않는다.
    """
    x = np.asarray(raw, dtype=np.float64)
    c = np.asarray(cap, dtype=np.float64)
    if x.ndim != 1 or c.shape != x.shape:
        raise ValueError("raw·cap 은 같은 길이의 1차원 배열이어야 한다")
    ok = np.isfinite(x) & np.isfinite(c) & (c > 0)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != x.shape:
            raise ValueError("mask 길이가 raw 와 다르다")
        ok &= m
    out = np.full(x.shape, np.nan)
    if ok.sum() < 2:
        return out
    v = x[ok]
    if clip > 0:
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad > 0:
            half = clip * _MAD_TO_SD * mad
            v = np.clip(v, med - half, med + half)
    cw = c[ok]
    v = v - (cw @ v) / cw.sum()
    sd = v.std()
    out[ok] = v / sd if sd > 0 else 0.0
    return out


def _label_ok(x: object) -> bool:
    return isinstance(x, str) and x != ""


def _industry_names(labels: np.ndarray) -> list[str]:
    return sorted({x for x in np.asarray(labels, dtype=object).ravel() if _label_ok(x)})


def _label_index(labels: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """라벨 배열 → 같은 모양의 업종 번호(``names`` 위치), 없는 라벨은 -1."""
    lab = np.asarray(labels, dtype=object)
    pos = {g: j for j, g in enumerate(names)}
    look = np.frompyfunc(lambda x: pos.get(x, -1) if _label_ok(x) else -1, 1, 1)
    return np.asarray(look(lab), dtype=np.int64).reshape(lab.shape)


def _dummies_from_index(idx: np.ndarray, G: int) -> np.ndarray:
    D = np.zeros((idx.shape[0], G))
    rows = np.flatnonzero(idx >= 0)
    D[rows, idx[rows]] = 1.0
    return D


def _dummies(labels: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return _dummies_from_index(_label_index(labels, names), len(names))


def industry_dummies(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """업종 라벨 → ``(N, G)`` 0/1 더미와 정렬된 업종 이름.

    라벨이 문자열이 아니거나 빈 문자열(``None``·NaN·``""``)이면 그 행은 전부 0 이다 —
    :func:`cross_section_regression` 이 업종 없는 행을 뺀다.
    """
    lab = np.asarray(labels, dtype=object)
    if lab.ndim != 1:
        raise ValueError("labels 는 1차원 배열이어야 한다")
    names = _industry_names(lab)
    return _dummies(lab, names), names


# ── 팩터 수익 ────────────────────────────────────────────────────────────────────


def _as_2d(a: np.ndarray | None, n: int) -> np.ndarray:
    if a is None:
        return np.zeros((n, 0))
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] != n:
        raise ValueError(f"행 수가 {n} 이 아니다: {a.shape}")
    return a


def cross_section_regression(
    r: np.ndarray,
    X_style: np.ndarray | None,
    X_ind: np.ndarray | None,
    w: np.ndarray,
    *,
    cap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """한 날짜의 제약 WLS — 시장 + 업종 더미 + 스타일.

    목적함수 ``Σ_i w_i u_i²`` (``w`` = 회귀 가중, Barra 관례로 √시총), 제약
    ``Σ_g s_g f_g = 0`` (``s_g`` = 업종 g 의 유효 종목 시총 합). 시총이 가장 큰 업종 ``g0`` 를
    ``f_g0 = −Σ_{g≠g0} (s_g/s_g0) f_g`` 로 치환하면 제약 없는 WLS 가 되고, 그 해는 원래 제약
    문제의 해와 정확히 같다(KKT 해와 대조하는 테스트가 있다).

    유효 행 = ``r``·``w``·``cap`` 유한(가중은 양수)·스타일 전부 유한·(업종이 있으면) 정확히 한 업종.
    유효 종목이 없는 업종의 팩터 수익은 NaN(정의 안 됨)이고 제약 합에서도 빠진다. 유효 행이
    추정할 계수 수 이하이면 ``f``·``u`` 전부 NaN. 스타일끼리 공선이면 ``lstsq`` 의 최소노름 해다.

    Args:
        r: ``(N,)`` 수익.
        X_style: ``(N, S)`` 스타일 노출(``None`` = 스타일 없음).
        X_ind: ``(N, G)`` 업종 더미(``None`` = 업종 없음).
        w: ``(N,)`` 회귀 가중(√시총).
        cap: ``(N,)`` 제약용 시총. ``None`` 이면 ``w²`` (w = √시총일 때 시총).

    Returns:
        ``(f, u)`` — ``f`` 는 ``(1+G+S,)`` ``[market, 업종…, 스타일…]``, ``u`` 는 ``(N,)`` 잔차
        (유효 행 밖은 NaN).
    """
    r = np.asarray(r, dtype=np.float64)
    n = r.shape[0]
    w = np.asarray(w, dtype=np.float64)
    Xs = _as_2d(X_style, n)
    Xi = _as_2d(X_ind, n)
    c = w * w if cap is None else np.asarray(cap, dtype=np.float64)
    if w.shape != (n,) or c.shape != (n,):
        raise ValueError("w·cap 길이가 r 과 다르다")
    G, S = Xi.shape[1], Xs.shape[1]
    f = np.full(1 + G + S, np.nan)
    u = np.full(n, np.nan)

    ok = np.isfinite(r) & np.isfinite(w) & (w > 0) & np.isfinite(c) & (c > 0)
    ok &= np.isfinite(Xs).all(axis=1)
    if G:
        ok &= np.isfinite(Xi).all(axis=1) & (Xi.sum(axis=1) == 1.0)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return f, u

    cols = [np.ones((idx.size, 1))]
    nf = 0
    if G:
        Xio = Xi[idx]
        s = c[idx] @ Xio  # 업종별 시총 합
        present = np.flatnonzero(Xio.sum(axis=0) > 0)
        g0 = present[np.argmax(s[present])]
        free = present[present != g0]
        ratio = s[free] / s[g0]
        cols.append(Xio[:, free] - Xio[:, [g0]] * ratio)
        nf = free.size
    cols.append(Xs[idx])
    Z = np.hstack(cols)
    if idx.size <= Z.shape[1]:
        return f, u

    sw = np.sqrt(w[idx])
    beta, *_ = np.linalg.lstsq(Z * sw[:, None], r[idx] * sw, rcond=None)

    f[0] = beta[0]
    if G:
        fi = beta[1 : 1 + nf]
        f[1 + free] = fi
        f[1 + g0] = -(ratio @ fi)
    f[1 + G :] = beta[1 + nf :]
    u[idx] = r[idx] - Z @ beta
    return f, u


@dataclass(frozen=True, eq=False)
class FactorReturns:
    """:func:`factor_returns` 결과.

    Attributes:
        f: ``(T, K)`` 팩터 수익. ``f[t]`` 는 ``t-1`` 노출로 설명한 ``t`` 수익 — ``f[0]`` 은 NaN.
        u: ``(N, T)`` 특이(잔차) 수익. 회귀에서 빠진 자리는 NaN.
        names: 길이 K, ``["market", <업종…>, <스타일…>]``.
    """

    f: np.ndarray
    u: np.ndarray
    names: list[str]


def _style_items(X_style: Mapping[str, np.ndarray] | None, shape: tuple[int, int]):
    if not X_style:
        return [], np.zeros((0,) + shape)
    names = [str(k) for k in X_style]
    if MARKET in names:
        raise ValueError(f"스타일 이름 {MARKET!r} 는 시장 팩터 이름과 겹친다")
    arr = np.stack([np.asarray(X_style[k], dtype=np.float64) for k in X_style])
    if arr.shape[1:] != shape:
        raise ValueError(f"스타일 패널 모양이 {shape} 이 아니다: {arr.shape[1:]}")
    return names, arr


def _labels_2d(ind_labels: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if ind_labels is None:
        return None
    lab = np.asarray(ind_labels, dtype=object)
    if lab.ndim == 1:
        if lab.shape[0] != shape[0]:
            raise ValueError("ind_labels 길이가 종목 수와 다르다")
        lab = np.repeat(lab[:, None], shape[1], axis=1)
    if lab.shape != shape:
        raise ValueError(f"ind_labels 모양이 {shape} 이 아니다: {lab.shape}")
    return lab


def factor_returns(
    R: np.ndarray,
    X_style: Mapping[str, np.ndarray] | None,
    ind_labels: np.ndarray | None,
    cap: np.ndarray,
    mask: np.ndarray | None = None,
) -> FactorReturns:
    """날짜마다 :func:`cross_section_regression` — ``R[:, t]`` 를 ``t-1`` 열의 노출로 회귀.

    업종·시총·마스크도 ``t-1`` 열을 쓴다.

    Args:
        R: ``(N, T)`` 수익 패널. ``R[:, t]`` 는 ``t-1`` 종가 → ``t`` 종가 수익.
        X_style: 스타일 이름 → ``(N, T)`` **이미 표준화된** 노출 패널(:func:`standardize`).
            dict 순서가 팩터 순서다. ``None``·빈 dict = 스타일 없음.
        ind_labels: ``(N,)`` 고정 또는 ``(N, T)`` 시점별 업종 라벨(``None`` = 업종 없음).
            업종 이름은 들어온 라벨 전체의 정렬된 합집합.
        cap: ``(N, T)`` 시총. 회귀 가중 = √cap, 업종 제약 가중 = cap.
        mask: ``(N, T)`` bool 유니버스(``None`` = 전부 참). ``t-1`` 열로 ``t`` 회귀 행을 고른다.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("R 은 (N, T) 패널이어야 한다")
    N, T = R.shape
    cap = np.asarray(cap, dtype=np.float64)
    if cap.shape != (N, T):
        raise ValueError("cap 모양이 R 과 다르다")
    m = np.ones((N, T), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != (N, T):
        raise ValueError("mask 모양이 R 과 다르다")
    style_names, Xs = _style_items(X_style, (N, T))
    lab = _labels_2d(ind_labels, (N, T))
    ind_names = [] if lab is None else _industry_names(lab)
    names = [MARKET, *ind_names, *style_names]

    gidx = None if lab is None else _label_index(lab, ind_names)

    f = np.full((T, len(names)), np.nan)
    u = np.full((N, T), np.nan)
    for t in range(1, T):
        p = t - 1
        r = np.where(m[:, p], R[:, t], np.nan)
        xi = None if gidx is None else _dummies_from_index(gidx[:, p], len(ind_names))
        xs = Xs[:, :, p].T
        cp = cap[:, p]
        with np.errstate(invalid="ignore"):
            w = np.sqrt(cp)
        f[t], u[:, t] = cross_section_regression(r, xs, xi, w, cap=cp)
    return FactorReturns(f=f, u=u, names=names)


# ── 공분산 · 특이위험 ────────────────────────────────────────────────────────────


def _ewma_weights(T: int, half_life: float) -> np.ndarray:
    if not half_life > 0:
        raise ValueError("half_life 는 양수여야 한다")
    lam = 0.5 ** (1.0 / half_life)
    return lam ** np.arange(T - 1, -1, -1, dtype=np.float64)  # 마지막 관측 가중 1


def ewma_cov(F: np.ndarray, *, half_life: float, nw_lags: int = 0) -> np.ndarray:
    """팩터 수익 ``(T, K)`` 의 지수가중 공분산 + Newey-West(Bartlett) 자기상관 보정. 일 단위.

    가중 ``w_t = 0.5^((T-1-t)/half_life)`` (최근이 무겁다). 팩터마다 지수가중 평균을 빼고::

        Γ_l = Σ_t w_t a_t a_{t-l}ᵀ / Σ_t w_t,      Σ = Γ_0 + Σ_{l=1..L} (1 − l/(L+1)) (Γ_l + Γ_lᵀ)

    결측(NaN — 예: 첫 행, 유효 종목이 없던 날의 업종)은 쌍별로 뺀다: 각 원소의 분모는 두 팩터가
    함께 관측된 쌍의 가중 합이다. 함께 관측된 적이 없는 쌍은 NaN.
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("F 는 (T, K) 이어야 한다")
    L = int(nw_lags)
    if L < 0:
        raise ValueError("nw_lags 는 0 이상이어야 한다")
    T = F.shape[0]
    wt = _ewma_weights(T, half_life)
    M = np.isfinite(F).astype(np.float64)
    Fz = np.where(M > 0, F, 0.0)
    sw = wt @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (wt @ Fz) / sw
    A = np.where(M > 0, F - mu, 0.0)

    def gamma(lag: int) -> np.ndarray:
        if lag >= T:
            return np.zeros((F.shape[1], F.shape[1]))
        wl = wt[lag:, None]
        num = (A[lag:] * wl).T @ A[: T - lag]
        den = (M[lag:] * wl).T @ M[: T - lag]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    S = gamma(0)
    for lag in range(1, L + 1):
        g = gamma(lag)
        S = S + (1.0 - lag / (L + 1.0)) * (g + g.T)
    return 0.5 * (S + S.T)


def specific_var(
    U: np.ndarray,
    *,
    half_life: float,
    shrink: float = 0.3,
    size: np.ndarray | None = None,
    n_groups: int = 10,
) -> np.ndarray:
    """특이수익 ``(N, T)`` → 종목별 특이분산 ``(N,)`` — EWMA 분산을 규모 그룹 평균 쪽으로 수축.

    - 종목별 ``σ²_i = Σ_t w_t u²_it / Σ_t w_t`` (유한 관측만, 평균 0 가정 — 잔차는 횡단면 평균이
      0 이고 시계열 평균을 빼면 짧은 표본에서 분산이 과소추정된다).
    - ``size`` (예: log 시총)의 분위로 ``n_groups`` 그룹, 그룹 사전값 = 관측 수가 ``half_life``
      이상인 종목의 ``σ²`` 단순평균(그룹에 그런 종목이 없으면 전체 평균). ``size`` 가 ``None`` 이면
      한 그룹, NaN 인 종목은 전체 평균을 사전값으로 쓴다.
    - 관측 수 ≥ ``half_life``: ``(1−shrink)·σ²_i + shrink·사전값``. 미만: 사전값.
    - 사전값을 만들 종목이 하나도 없으면 전부 NaN.
    """
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError("U 는 (N, T) 이어야 한다")
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink 는 [0, 1] 이어야 한다")
    N, T = U.shape
    wt = _ewma_weights(T, half_life)
    M = np.isfinite(U)
    nobs = M.sum(axis=1)
    U2 = np.where(M, U * U, 0.0)
    den = M.astype(np.float64) @ wt
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = (U2 @ wt) / den
    qual = (nobs >= half_life) & np.isfinite(raw)
    out = np.full(N, np.nan)
    if not qual.any():
        return out
    overall = raw[qual].mean()
    prior = np.full(N, overall)
    if size is not None:
        sz = np.asarray(size, dtype=np.float64)
        if sz.shape != (N,):
            raise ValueError("size 길이가 종목 수와 다르다")
        has = np.flatnonzero(np.isfinite(sz))
        if has.size:
            order = has[np.argsort(sz[has], kind="stable")]
            k = max(1, int(n_groups))
            grp = np.full(N, -1)
            grp[order] = (np.arange(order.size) * k) // order.size
            for g in range(k):
                members = grp == g
                q = members & qual
                if q.any():
                    prior[members] = raw[q].mean()
    out[:] = prior
    out[qual] = (1.0 - shrink) * raw[qual] + shrink * prior[qual]
    return out


# ── 위험모델 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class RiskModel:
    """``Σ = X F Xᵀ + diag(D)`` (일 단위). N×N 은 :meth:`cov` 에서만 만든다.

    Attributes:
        codes: 길이 N 종목 라벨(:func:`build_risk_model` 에서는 asof 에 유효한 종목만).
        names: 길이 K 팩터 이름 ``["market", <업종…>, <스타일…>]``.
        X: ``(N, K)`` 노출. F: ``(K, K)`` 팩터 공분산. D: ``(N,)`` 특이분산.
    """

    codes: np.ndarray
    names: list[str]
    X: np.ndarray
    F: np.ndarray
    D: np.ndarray

    def __post_init__(self) -> None:
        N, K = self.X.shape
        if self.F.shape != (K, K) or self.D.shape != (N,) or len(self.names) != K:
            raise ValueError("RiskModel 모양 불일치: X(N,K)·F(K,K)·D(N,)·names(K)")
        if len(self.codes) != N:
            raise ValueError("codes 길이가 X 행 수와 다르다")

    def cov(self) -> np.ndarray:
        """``(N, N)`` 공분산. 종목이 많으면 크다 — :meth:`var` 는 이걸 만들지 않는다."""
        return self.X @ self.F @ self.X.T + np.diag(self.D)

    def _w(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != self.D.shape:
            raise ValueError("w 길이가 종목 수와 다르다")
        return w

    def var(self, w: np.ndarray) -> float:
        """포트 분산 ``wᵀΣw = (Xᵀw)ᵀ F (Xᵀw) + Σ w_i² D_i``."""
        w = self._w(w)
        y = self.X.T @ w
        return float(y @ self.F @ y + (w * w) @ self.D)

    def decompose(self, w: np.ndarray) -> dict:
        """분산 분해 — ``factor_var + specific_var = var(w)``.

        ``by_factor[k] = y_k (F y)_k`` (``y = Xᵀw``), 합이 ``factor_var`` 다.
        팩터 간 상관 때문에 음수일 수 있다.
        """
        w = self._w(w)
        y = self.X.T @ w
        Fy = self.F @ y
        contrib = y * Fy
        return {
            "factor_var": float(y @ Fy),
            "specific_var": float((w * w) @ self.D),
            "by_factor": {n: float(v) for n, v in zip(self.names, contrib, strict=True)},
        }


def build_risk_model(
    R: np.ndarray,
    X_style: Mapping[str, np.ndarray] | None,
    ind_labels: np.ndarray | None,
    cap: np.ndarray,
    mask: np.ndarray | None,
    asof_index: int,
    *,
    codes: Sequence | np.ndarray | None = None,
    cov_half_life: float = 90,
    spec_half_life: float = 90,
    nw_lags: int = 2,
    spec_shrink: float = 0.3,
    n_groups: int = 10,
) -> RiskModel:
    """``asof_index`` 열까지만 써서 다음 날 수익의 위험모델을 만든다.

    입력 패널을 ``[:, : asof_index + 1]`` 로 **먼저 잘라** :func:`factor_returns` →
    :func:`ewma_cov`(``cov_half_life``·``nw_lags``) · :func:`specific_var`(``spec_half_life``·
    ``spec_shrink``, 규모 = asof 의 log 시총) 순서로 계산한다. 노출 ``X`` 는 asof 열의
    ``[1, 업종 더미, 스타일]`` — ``r_{asof+1}`` 을 설명하는 노출이다.

    반환 모델의 종목은 asof 에 **유효한 것만**(마스크 참·스타일 유한·업종 있음·``D`` 유한)이고
    ``codes`` 가 그 라벨이다(``codes=None`` 이면 행 번호). 그래서 ``X``·``F``·``D`` 에 NaN 이 없다.
    이력이 전혀 없는 팩터(예: asof 에 처음 생긴 업종)는 ``F`` 의 행·열을 0 으로 두고 그 팩터에
    노출된 종목을 뺀다(위험을 모르는 종목을 0 위험으로 넣지 않는다). 함께 관측된 적 없는
    팩터 쌍의 공분산도 0 으로 둔다.
    """
    R = np.asarray(R, dtype=np.float64)
    N, T = R.shape
    a = int(asof_index)
    if not 0 <= a < T:
        raise ValueError(f"asof_index {a} 가 [0, {T}) 밖이다")
    sl = slice(0, a + 1)
    Rs = R[:, sl]
    caps = np.asarray(cap, dtype=np.float64)[:, sl]
    ms = None if mask is None else np.asarray(mask, dtype=bool)[:, sl]
    Xs = None if not X_style else {k: np.asarray(v)[:, sl] for k, v in X_style.items()}
    lab = _labels_2d(ind_labels, (N, T))
    labs = None if lab is None else lab[:, sl]

    fr = factor_returns(Rs, Xs, labs, caps, ms)
    F = ewma_cov(fr.f, half_life=cov_half_life, nw_lags=nw_lags)
    with np.errstate(invalid="ignore", divide="ignore"):
        size = np.where(caps[:, a] > 0, np.log(caps[:, a]), np.nan)
    D = specific_var(fr.u, half_life=spec_half_life, shrink=spec_shrink, size=size,
                     n_groups=n_groups)

    names = fr.names
    G = 0 if labs is None else len(names) - 1 - (0 if Xs is None else len(Xs))
    parts = [np.ones((N, 1))]
    if labs is not None:
        parts.append(_dummies(labs[:, a], names[1 : 1 + G]))
    if Xs:
        parts.append(np.stack([v[:, a] for v in Xs.values()], axis=1).astype(np.float64))
    X = np.hstack(parts)

    ok = np.isfinite(X).all(axis=1) & np.isfinite(D)
    if ms is not None:
        ok &= ms[:, a]
    if G:
        ok &= X[:, 1 : 1 + G].sum(axis=1) == 1.0
    dead = ~np.isfinite(np.diag(F))
    if dead.any():
        ok &= ~(X[:, dead] != 0).any(axis=1)
    F = np.where(np.isfinite(F), F, 0.0)
    F[dead, :] = 0.0
    F[:, dead] = 0.0

    lab_codes = np.arange(N) if codes is None else np.asarray(codes)
    if lab_codes.shape[0] != N:
        raise ValueError("codes 길이가 종목 수와 다르다")
    return RiskModel(codes=lab_codes[ok], names=list(names), X=X[ok], F=F, D=D[ok])


# ── 검증 ────────────────────────────────────────────────────────────────────────


def bias_statistic(
    pred_sd: np.ndarray,
    realized: np.ndarray,
    *,
    window: int | None = None,
) -> dict:
    """바이어스 통계 — ``z = realized / pred_sd`` 의 표준편차(ddof=1).

    1 = 예측 적중, >1 = 위험 과소예측, <1 = 과대예측.

    ``pred_sd``·``realized`` 는 같은 모양 ``(T,)`` (포트 하나) 또는 ``(P, T)`` (포트 P 개 × 기간).
    ``window=None`` 이면 포트마다 전 기간 한 값, 아니면 포트마다 길이 ``window`` 이동 창
    (한 칸씩)마다 한 값. 창 안 유한 z 가 2개 미만이면 그 창은 뺀다. ``pred_sd`` 가 0 이하이면
    그 관측은 NaN.

    Returns:
        ``{mean, q05, q95, n}`` — 모든 (포트, 창) 바이어스 값의 평균·5%·95% 분위와 개수. 숫자만.
    """
    p = np.asarray(pred_sd, dtype=np.float64)
    r = np.asarray(realized, dtype=np.float64)
    if p.shape != r.shape or p.ndim not in (1, 2):
        raise ValueError("pred_sd·realized 는 같은 모양의 (T,) 또는 (P, T) 여야 한다")
    if p.ndim == 1:
        p, r = p[None, :], r[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(p > 0, r / p, np.nan)
    T = z.shape[1]
    if window is None:
        win = T
    else:
        win = int(window)
        if win < 2:
            raise ValueError("window 는 2 이상이어야 한다")
    vals: list[float] = []
    for row in z:
        for s in range(0, max(T - win, 0) + 1):
            seg = row[s : s + win]
            seg = seg[np.isfinite(seg)]
            if seg.size >= 2:
                vals.append(float(seg.std(ddof=1)))
    if not vals:
        nan = float("nan")
        return {"mean": nan, "q05": nan, "q95": nan, "n": 0}
    b = np.asarray(vals)
    return {
        "mean": float(b.mean()),
        "q05": float(np.quantile(b, 0.05)),
        "q95": float(np.quantile(b, 0.95)),
        "n": int(b.size),
    }
