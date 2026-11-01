"""risk.factor_model — 합성 복원·업종 제약·미래 누출·분산 분해·표준화·바이어스 통계.

합성 시장은 알려진 ``X``(전일 노출)·``f``·``D`` 로 ``r_t = X_{t-1} f_t + u_t`` 를 만든다.
업종 팩터 수익은 시총가중 평균을 빼 제약(``Σ s_g f_g = 0``)을 만족시킨 값을 정답으로 둔다 —
모델이 같은 제약으로 풀기 때문이다. 스펙 §4 완료 조건 1 의 바(상관 > 0.99, 공분산 원소 상대오차
중앙값 < 10%, 미래 누출 시 비트 동일)를 그대로 단언한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from krx_quant_core.risk import (
    FactorReturns,
    RiskModel,
    bias_statistic,
    build_risk_model,
    cross_section_regression,
    ewma_cov,
    factor_returns,
    industry_dummies,
    specific_var,
    standardize,
)

_IND = ["A", "B", "C", "D", "E"]


def _simulate(seed: int, N: int, T: int, n_ind: int, n_style: int, *, f_sd_ind=0.006,
              persist: float = 0.98):
    """알려진 위험모델에서 수익 패널을 만든다. 반환: (R, styles, labels, cap, f_true, F_true, D)."""
    rng = np.random.default_rng(seed)
    ind = np.array(_IND[:n_ind], dtype=object)
    labels = ind[rng.integers(0, n_ind, N)]
    cap1 = np.exp(rng.normal(25.0, 1.2, N))
    cap = np.repeat(cap1[:, None], T, axis=1)

    # 스타일: 지속성 있는 원시값을 날짜마다 standardize
    raw = rng.normal(size=(n_style, N))
    styles = {}
    for s in range(n_style):
        panel = np.empty((N, T))
        x = raw[s].copy()
        for t in range(T):
            x = persist * x + np.sqrt(1 - persist**2) * rng.normal(size=N)
            panel[:, t] = standardize(x, cap[:, t])
        styles[f"s{s}"] = panel

    K = 1 + n_ind + n_style
    sd = np.concatenate([[0.012], np.full(n_ind, f_sd_ind), np.full(n_style, 0.004)])
    # 상관을 원소마다 다르게 — 모두 같으면 제약 사영 뒤 정확히 0 이 되는 원소가 생긴다
    B = rng.normal(size=(K, 3))
    C = B @ B.T + np.eye(K)
    C /= np.sqrt(np.outer(np.diag(C), np.diag(C)))
    F_raw = C * np.outer(sd, sd)
    # 제약: 업종 수익에서 시총가중 평균을 뺀다(시총 고정 → 선형 사영 P)
    Xd, names_ind = industry_dummies(labels)
    s_g = cap1 @ Xd
    a = s_g / s_g.sum()
    P = np.eye(K)
    P[1 : 1 + n_ind, 1 : 1 + n_ind] -= np.outer(np.ones(n_ind), a)
    F_true = P @ F_raw @ P.T
    f_true = rng.multivariate_normal(np.zeros(K), F_raw, size=T) @ P.T
    f_true[0] = np.nan

    D = rng.uniform(0.002, 0.006, N) ** 2
    R = np.full((N, T), np.nan)
    for t in range(1, T):
        X = np.column_stack([np.ones(N), Xd, *[styles[k][:, t - 1] for k in styles]])
        R[:, t] = X @ f_true[t] + rng.normal(size=N) * np.sqrt(D)
    assert names_ind == list(ind[:n_ind])
    return R, styles, labels, cap, f_true, F_true, D


@pytest.fixture(scope="module")
def big_sim():
    return _simulate(7, N=600, T=2500, n_ind=5, n_style=3)


# ── 완료 조건 1: 합성 복원 ──────────────────────────────────────────────────────


def test_factor_returns_recovers_true_f(big_sim):
    R, styles, labels, cap, f_true, _, _ = big_sim
    fr = factor_returns(R, styles, labels, cap, None)
    assert isinstance(fr, FactorReturns)
    assert fr.names == ["market", *_IND, "s0", "s1", "s2"]
    assert fr.f.shape == (R.shape[1], 9) and fr.u.shape == R.shape
    assert np.isnan(fr.f[0]).all() and np.isnan(fr.u[:, 0]).all()
    for k, name in enumerate(fr.names):
        rho = np.corrcoef(fr.f[1:, k], f_true[1:, k])[0, 1]
        assert rho > 0.99, (name, rho)


def test_ewma_cov_recovers_true_F(big_sim):
    R, styles, labels, cap, _, F_true, _ = big_sim
    fr = factor_returns(R, styles, labels, cap, None)
    F_hat = ewma_cov(fr.f, half_life=1e6, nw_lags=0)
    rel = np.abs(F_hat - F_true) / np.abs(F_true)
    assert np.median(rel) < 0.10, np.median(rel)
    # NW 보정은 i.i.d. 에서 거의 바꾸지 않는다
    F_nw = ewma_cov(fr.f, half_life=1e6, nw_lags=2)
    assert np.median(np.abs(F_nw - F_true) / np.abs(F_true)) < 0.10


# ── 업종 제약 ────────────────────────────────────────────────────────────────


def _kkt_solution(r, Xfull, w, a):
    """min Σ w (r − X f)² s.t. aᵀf = 0 을 KKT 연립으로 직접."""
    XtW = Xfull.T * w
    K = Xfull.shape[1]
    A = np.zeros((K + 1, K + 1))
    A[:K, :K] = XtW @ Xfull
    A[:K, K] = a
    A[K, :K] = a
    b = np.concatenate([XtW @ r, [0.0]])
    return np.linalg.solve(A, b)[:K]


def test_cross_section_constraint_exact_and_matches_kkt():
    rng = np.random.default_rng(3)
    N, S = 400, 3
    labels = np.array(_IND, dtype=object)[rng.integers(0, 4, N)]  # "E" 는 비어 있다
    labels[:5] = None  # 업종 없음 → 제외
    Xd, names = industry_dummies(np.concatenate([labels, np.array(["E"], dtype=object)]))
    Xd = Xd[:N]
    assert names == _IND
    Xs = rng.normal(size=(N, S))
    Xs[5:8, 1] = np.nan  # 스타일 결측 → 제외
    cap = np.exp(rng.normal(25, 1.5, N))
    r = rng.normal(0, 0.02, N)
    r[8] = np.nan
    f, u = cross_section_regression(r, Xs, Xd, np.sqrt(cap), cap=cap)

    ok = np.zeros(N, bool)
    ok[9:] = True
    assert np.isnan(u[~ok]).all() and np.isfinite(u[ok]).all()
    assert np.isnan(f[1 + 4])  # 빈 업종 E
    s_g = cap[ok] @ Xd[ok]
    fi = f[1:5]
    share = s_g[:4] / s_g[:4].sum()
    assert abs(share @ fi) < 1e-10  # 시총 비중 가중 합(수익 단위) = 0

    # 정확해: KKT 해와 같다(E 열은 빼고)
    Xfull = np.column_stack([np.ones(ok.sum()), Xd[ok][:, :4], Xs[ok]])
    f_kkt = _kkt_solution(r[ok], Xfull, np.sqrt(cap[ok]), np.concatenate([[0], s_g[:4], [0] * S]))
    f_ours = np.concatenate([[f[0]], fi, f[6:]])
    np.testing.assert_allclose(f_ours, f_kkt, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(u[ok], r[ok] - Xfull @ f_kkt, rtol=1e-9, atol=1e-13)


def test_factor_returns_constraint_every_date(big_sim):
    R, styles, labels, cap, *_ = big_sim
    T = 300
    fr = factor_returns(R[:, :T], {k: v[:, :T] for k, v in styles.items()}, labels, cap[:, :T])
    Xd, _ = industry_dummies(labels)
    s_g = cap[:, 0] @ Xd
    resid = fr.f[1:, 1:6] @ s_g / s_g.sum()
    assert np.abs(resid).max() < 1e-10


def test_cross_section_without_industries_is_plain_wls():
    rng = np.random.default_rng(11)
    N = 200
    Xs = rng.normal(size=(N, 2))
    w = rng.uniform(1, 5, N)
    r = rng.normal(size=N)
    f, u = cross_section_regression(r, Xs, None, w)
    Z = np.column_stack([np.ones(N), Xs])
    beta = np.linalg.solve((Z.T * w) @ Z, (Z.T * w) @ r)
    np.testing.assert_allclose(f, beta, rtol=1e-10)
    np.testing.assert_allclose(u, r - Z @ beta, rtol=1e-9, atol=1e-13)


def test_cross_section_too_few_rows_is_nan():
    f, u = cross_section_regression(np.array([0.1, 0.2]), np.ones((2, 2)), None, np.ones(2))
    assert np.isnan(f).all() and np.isnan(u).all()


# ── 미래 누출 ────────────────────────────────────────────────────────────────


def _small_inputs(seed=5):
    R, styles, labels, cap, *_ = _simulate(seed, N=150, T=320, n_ind=4, n_style=2)
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(R.shape) > 0.05
    lab2 = np.repeat(labels[:, None], R.shape[1], axis=1)
    return R, styles, lab2, cap, mask


def _copy(R, styles, lab, cap, mask):
    return R.copy(), {k: v.copy() for k, v in styles.items()}, lab.copy(), cap.copy(), mask.copy()


def _same(a: RiskModel, b: RiskModel) -> bool:
    return (
        a.names == b.names
        and np.array_equal(a.codes, b.codes)
        and np.array_equal(a.X, b.X)
        and np.array_equal(a.F, b.F)
        and np.array_equal(a.D, b.D)
    )


def test_build_risk_model_no_lookahead():
    R, styles, lab, cap, mask = _small_inputs()
    asof = 250
    kw = dict(codes=[f"c{i:03d}" for i in range(R.shape[0])], cov_half_life=60,
              spec_half_life=60, nw_lags=2)
    base = build_risk_model(R, styles, lab, cap, mask, asof, **kw)

    R2, st2, lab2, cap2, m2 = _copy(R, styles, lab, cap, mask)
    rng = np.random.default_rng(99)
    after = slice(asof + 1, None)
    R2[:, after] = rng.normal(0, 5, R2[:, after].shape)
    R2[:3, after] = np.nan
    for v in st2.values():
        v[:, after] = rng.normal(0, 9, v[:, after].shape)
    lab2[:, after] = "ZZZ_NEW"
    cap2[:, after] *= 100.0
    m2[:, after] = ~m2[:, after]
    corrupted = build_risk_model(R2, st2, lab2, cap2, m2, asof, **kw)
    assert _same(base, corrupted)

    # 민감도: asof 당일 값을 바꾸면 결과가 바뀐다(위 테스트가 헛돌지 않는다)
    R3 = R.copy()
    R3[:, asof] += 0.05
    assert not _same(base, build_risk_model(R3, styles, lab, cap, mask, asof, **kw))


def test_build_risk_model_shapes_and_validity():
    R, styles, lab, cap, mask = _small_inputs()
    asof = 300
    m = build_risk_model(R, styles, lab, cap, mask, asof)
    assert m.names == ["market", "A", "B", "C", "D", "s0", "s1"]
    N = len(m.codes)
    assert m.X.shape == (N, 7) and m.F.shape == (7, 7) and m.D.shape == (N,)
    assert np.isfinite(m.X).all() and np.isfinite(m.F).all() and np.isfinite(m.D).all()
    assert (m.D > 0).all()
    # 마스크 거짓인 종목은 빠진다
    np.testing.assert_array_equal(m.codes, np.flatnonzero(mask[:, asof]))
    assert np.all(np.linalg.eigvalsh(m.F) > -1e-12)


def test_new_industry_without_history_is_excluded():
    R, styles, lab, cap, mask = _small_inputs()
    asof = 300
    lab = lab.copy()
    lab[:4, asof] = "NEW"
    m = build_risk_model(R, styles, lab, cap, None, asof)
    j = m.names.index("NEW")
    assert (m.F[j] == 0).all() and (m.F[:, j] == 0).all()
    assert not np.isin(np.arange(4), m.codes).any()
    assert (m.X[:, j] == 0).all()


# ── 분산 분해 ────────────────────────────────────────────────────────────────


def test_decompose_sums_to_var_and_matches_cov():
    rng = np.random.default_rng(21)
    N, K = 80, 6
    X = rng.normal(size=(N, K))
    A = rng.normal(size=(K, K))
    F = A @ A.T * 1e-4
    D = rng.uniform(1e-5, 4e-4, N)
    m = RiskModel(codes=np.arange(N), names=[f"k{i}" for i in range(K)], X=X, F=F, D=D)
    w = rng.normal(size=N)
    v = m.var(w)
    dec = m.decompose(w)
    assert set(dec) == {"factor_var", "specific_var", "by_factor"}
    assert abs(dec["factor_var"] + dec["specific_var"] - v) <= 1e-12 * v
    assert abs(sum(dec["by_factor"].values()) - dec["factor_var"]) <= 1e-12 * abs(
        dec["factor_var"])
    assert list(dec["by_factor"]) == m.names
    np.testing.assert_allclose(w @ m.cov() @ w, v, rtol=1e-12)
    with pytest.raises(ValueError):
        RiskModel(codes=np.arange(N), names=["x"], X=X, F=F, D=D)


# ── 표준화 ──────────────────────────────────────────────────────────────────


def test_standardize_properties():
    rng = np.random.default_rng(1)
    N = 500
    raw = rng.standard_t(3, N) * 10 + 4
    raw[[3, 50]] = np.nan
    cap = np.exp(rng.normal(25, 1.5, N))
    cap[7] = np.nan
    mask = np.ones(N, bool)
    mask[9] = False
    z = standardize(raw, cap, mask, clip=3.0)
    bad = np.zeros(N, bool)
    bad[[3, 50, 7, 9]] = True
    assert np.isnan(z[bad]).all() and np.isfinite(z[~bad]).all()
    zv, cv = z[~bad], cap[~bad]
    assert abs(cv @ zv / cv.sum()) < 1e-12
    assert abs(zv.std() - 1.0) < 1e-12
    # 윈저: 꼬리가 잘려 원시값의 극단 순서가 동점으로 묶인다
    assert np.sum(zv == zv.max()) > 1 and np.sum(zv == zv.min()) > 1


def test_standardize_degenerate():
    assert np.isnan(standardize(np.array([1.0, np.nan]), np.ones(2))).all()
    z = standardize(np.array([2.0, 2.0, 2.0, np.nan]), np.ones(4))
    np.testing.assert_array_equal(z[:3], 0.0)
    assert np.isnan(z[3])


def test_industry_dummies_missing_labels():
    D, names = industry_dummies(np.array(["b", None, "a", "", "b", np.nan], dtype=object))
    assert names == ["a", "b"]
    np.testing.assert_array_equal(D, [[0, 1], [0, 0], [1, 0], [0, 0], [0, 1], [0, 0]])


# ── 공분산·특이위험 단위 ───────────────────────────────────────────────────────


def test_ewma_cov_matches_manual_bartlett():
    rng = np.random.default_rng(4)
    T, K, hl, L = 60, 3, 15.0, 2
    F = rng.normal(size=(T, K))
    lam = 0.5 ** (1 / hl)
    w = lam ** np.arange(T - 1, -1, -1)
    mu = w @ F / w.sum()
    A = F - mu
    S = (A * w[:, None]).T @ A / w.sum()
    for lag in range(1, L + 1):
        g = (A[lag:] * w[lag:, None]).T @ A[:-lag] / w[lag:].sum()
        S += (1 - lag / (L + 1)) * (g + g.T)
    np.testing.assert_allclose(ewma_cov(F, half_life=hl, nw_lags=L), S, rtol=1e-12)


def test_ewma_cov_leading_nan_row_is_ignored():
    rng = np.random.default_rng(6)
    F = rng.normal(size=(40, 3))
    Fn = np.vstack([np.full((1, 3), np.nan), F])
    np.testing.assert_allclose(ewma_cov(Fn, half_life=10, nw_lags=2),
                               ewma_cov(F, half_life=10, nw_lags=2), rtol=1e-12)


def test_specific_var_shrink_and_short_history():
    rng = np.random.default_rng(8)
    N, T = 40, 200
    sd = rng.uniform(0.01, 0.03, N)
    U = rng.normal(size=(N, T)) * sd[:, None]
    U[0, :-10] = np.nan  # 관측 10개 < half_life
    hl = 50
    raw = specific_var(U, half_life=hl, shrink=0.0)
    lam = 0.5 ** (1 / hl)
    w = lam ** np.arange(T - 1, -1, -1)
    np.testing.assert_allclose(raw[1:], (U[1:] ** 2) @ w / w.sum(), rtol=1e-12)
    assert raw[0] == pytest.approx(raw[1:].mean())  # 짧은 이력 → (단일 그룹) 평균

    size = np.arange(N, dtype=float)
    sh = specific_var(U, half_life=hl, shrink=0.3, size=size, n_groups=2)
    lo, hi = np.arange(N) < N // 2, np.arange(N) >= N // 2
    prior_lo = raw[1 : N // 2].mean()
    prior_hi = raw[hi].mean()
    np.testing.assert_allclose(sh[hi], 0.7 * raw[hi] + 0.3 * prior_hi, rtol=1e-12)
    assert sh[0] == pytest.approx(prior_lo)
    np.testing.assert_allclose(sh[1 : N // 2], 0.7 * raw[1 : N // 2] + 0.3 * prior_lo,
                               rtol=1e-12)
    assert lo.sum() == N // 2


# ── 바이어스 통계 ────────────────────────────────────────────────────────────


def test_bias_statistic_known_scale():
    rng = np.random.default_rng(12)
    sd = rng.uniform(0.01, 0.03, (20, 400))
    real = rng.normal(size=sd.shape) * sd
    b = bias_statistic(sd, real)
    assert b["n"] == 20 and abs(b["mean"] - 1.0) < 0.05
    assert set(b) == {"mean", "q05", "q95", "n"}
    b2 = bias_statistic(sd, 2 * real, window=100)
    assert b2["n"] == 20 * 301 and abs(b2["mean"] - 2.0) < 0.1
    assert b2["q05"] < b2["mean"] < b2["q95"]
    assert bias_statistic(np.ones(1), np.ones(1))["n"] == 0


def test_bias_statistic_near_one_on_model_simulated_data():
    """모델에서 만든 데이터에 build_risk_model 예측을 대면 바이어스 ≈ 1."""
    R, styles, labels, cap, *_ = _simulate(17, N=300, T=700, n_ind=4, n_style=2, persist=0.995)
    rng = np.random.default_rng(18)
    P, H = 30, 20
    pred, real = [], []
    for asof in range(400, 700 - H, H):
        m = build_risk_model(R, styles, labels, cap, None, asof, cov_half_life=90,
                             spec_half_life=90, nw_lags=0)
        row_p, row_r = [], []
        for _ in range(P):
            pick = rng.choice(len(m.codes), 30, replace=False)
            w = np.zeros(len(m.codes))
            w[pick] = rng.dirichlet(np.ones(30))
            sd = np.sqrt(m.var(w))
            rows = m.codes[pick]
            r = w[pick] @ R[rows, asof + 1 : asof + 1 + H]
            row_p.append(np.full(H, sd))
            row_r.append(r)
        pred.append(np.array(row_p))
        real.append(np.array(row_r))
    b = bias_statistic(np.concatenate(pred, axis=1), np.concatenate(real, axis=1))
    assert b["n"] == P
    assert 0.85 < b["mean"] < 1.15, b
