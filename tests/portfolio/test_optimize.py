"""제약 평균-분산 최적화(portfolio/optimize.py) — 해석해·제약 활성·실패 상태·비용 일치·속도.

위험모델은 ``X``·``F``·``D``·``names`` 속성만 보므로 여기서는 작은 대역 dataclass 를 쓴다
(``risk.factor_model.RiskModel`` 에 묶이지 않는다).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pytest

from krx_quant_core.costs import sqrt_impact_bp, trade_cost_bp
from krx_quant_core.portfolio import Constraints, CostParams, OptResult, optimize

TOL = 1e-6


@pytest.fixture
def cvx():
    return pytest.importorskip("cvxpy")


@dataclass(frozen=True)
class StandInModel:
    X: np.ndarray
    F: np.ndarray
    D: np.ndarray
    names: list


def make_model(n: int, n_ind: int, n_style: int, seed: int = 0) -> tuple[StandInModel, np.ndarray]:
    """market(전 종목 1) + 업종 더미 + 스타일(표준정규) — §3.1 과 같은 열 순서. 일간 단위."""
    rng = np.random.default_rng(seed)
    labels = np.array([f"G{g:02d}" for g in rng.integers(0, n_ind, n)])
    inds = sorted(set(labels))
    X_ind = (labels[:, None] == np.array(inds)[None, :]).astype(float)
    X_sty = rng.standard_normal((n, n_style))
    X = np.hstack([np.ones((n, 1)), X_ind, X_sty])
    k = X.shape[1]
    A = rng.standard_normal((k, k)) * 0.004
    F = A @ A.T + np.diag(np.full(k, 1e-5))
    F[0, 0] += 1.2e-4  # 시장 분산이 가장 크다
    D = rng.uniform(1e-4, 9e-4, n)
    names = ["market", *inds, *[f"style{j}" for j in range(n_style)]]
    return StandInModel(X, F, D, names), labels


def full_cov(m: StandInModel) -> np.ndarray:
    return m.X @ m.F @ m.X.T + np.diag(m.D)


@pytest.fixture
def small():
    model, labels = make_model(60, 6, 3, seed=1)
    rng = np.random.default_rng(2)
    alpha = rng.normal(0, 4e-4, 60)
    cap = rng.lognormal(0, 1, 60)
    return model, labels, alpha, cap / cap.sum()


# ── 해석해 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lam", [1.0, 5.0, 40.0])
def test_unconstrained_matches_analytic(cvx, small, lam):
    model, _, alpha, _ = small
    want = np.linalg.solve(full_cov(model), alpha) / lam
    w_prev = np.random.default_rng(3).uniform(0, 0.05, 60)  # 비용이 없으면 w_prev 와 무관해야 한다
    res = optimize(
        alpha,
        model,
        w_prev,
        risk_aversion=lam,
        constraints=Constraints(budget=None, long_only=False),
    )
    assert res.ok and res.solver == "CLARABEL"
    assert np.max(np.abs(res.weights - want)) < TOL
    sigma = full_cov(model)
    assert res.expected_alpha == pytest.approx(alpha @ res.weights, abs=1e-15)
    assert res.expected_risk == pytest.approx(np.sqrt(res.weights @ sigma @ res.weights), rel=1e-10)
    for j, name in enumerate(model.names):
        assert res.exposures[name] == pytest.approx(model.X[:, j] @ res.weights, abs=1e-12)


def test_budget_only_matches_lagrange(cvx, small):
    model, _, alpha, _ = small
    lam = 10.0
    sigma = full_cov(model)
    si_a = np.linalg.solve(sigma, alpha)
    si_1 = np.linalg.solve(sigma, np.ones(60))
    mu = (si_1 @ alpha - lam) / si_1.sum()  # Σw = 1 의 라그랑주 승수
    want = (si_a - mu * si_1) / lam
    res = optimize(alpha, model, None, risk_aversion=lam, constraints=Constraints(long_only=False))
    assert res.ok
    assert np.max(np.abs(res.weights - want)) < TOL


# ── 제약 위반 측정(모든 제약 공통) ─────────────────────────────────────────


def violations(res: OptResult, con: Constraints, model, w_prev, *, aum=None, adv=None):
    """제약별 최대 위반량(양수면 위반). 켜진 제약만."""
    w = res.weights
    dw = w - w_prev
    out: dict[str, float] = {}
    if con.budget is not None:
        s = w.sum() - con.budget
        out["budget"] = s if con.allow_cash else abs(s)
    if con.long_only:
        out["long_only"] = -w.min()
    if con.max_weight is not None:
        out["max_weight"] = np.max(w - con.max_weight)
    if con.sector_band is not None:
        labs = np.asarray(con.sector)
        out["sector"] = max(
            abs(w[labs == g].sum() - con.benchmark[labs == g].sum()) - con.sector_band
            for g in np.unique(labs)
        )
    if con.beta_bounds is not None:
        if con.beta is not None:
            beta = con.beta
        else:
            sigma = full_cov(model)
            beta = sigma @ con.benchmark / (con.benchmark @ sigma @ con.benchmark)
        lo, hi = con.beta_bounds
        bw = beta @ w
        out["beta"] = max(
            (lo - bw) if lo is not None else -np.inf, (bw - hi) if hi is not None else -np.inf
        )
    for name, (lo, hi) in (con.style_bounds or {}).items():
        e = model.X[:, model.names.index(name)] @ w
        out[f"style:{name}"] = max(
            (lo - e) if lo is not None else -np.inf, (e - hi) if hi is not None else -np.inf
        )
    if con.max_turnover is not None:
        out["turnover"] = np.abs(dw).sum() - con.max_turnover
    if con.max_participation is not None:
        out["participation"] = np.max(np.abs(dw) * aum / adv - con.max_participation)
    if con.no_buy is not None:
        out["no_buy"] = np.max(np.where(con.no_buy, dw, -np.inf))
    if con.no_trade is not None:
        out["no_trade"] = np.max(np.where(con.no_trade, np.abs(dw), -np.inf))
    return out


AUM = 5e10
LAM = 4.0
#: 비용 사례의 α 배수 — 일간 α(4bp 대)로는 편도 비용(10bp+)을 못 넘어 아무것도 안 사고판다.
COST_ALPHA = 10.0


def case_inputs(small):
    model, labels, alpha, bench = small
    rng = np.random.default_rng(7)
    adv = rng.uniform(3e8, 3e10, 60)
    vol = rng.uniform(0.01, 0.04, 60)
    w_prev = bench.copy()
    return model, labels, alpha, bench, adv, vol, w_prev


def constraint_cases(model, labels, bench, w_prev):
    n = len(bench)
    no_buy = np.zeros(n, bool)
    no_buy[:10] = True
    no_trade = np.zeros(n, bool)
    no_trade[10:16] = True
    beta = np.linspace(0.5, 1.5, n)
    return {
        "allow_cash": Constraints(budget=0.3, allow_cash=True),
        "max_weight": Constraints(max_weight=0.03),
        "sector": Constraints(sector=labels, benchmark=bench, sector_band=0.02),
        "beta_explicit": Constraints(beta=beta, beta_bounds=(0.95, 1.0)),
        "beta_predicted": Constraints(benchmark=bench, beta_bounds=(0.97, 1.02)),
        "style": Constraints(style_bounds={"style0": (-0.05, 0.05), "style2": (None, 0.0)}),
        "industry_exposure": Constraints(style_bounds={model.names[1]: (None, 0.2)}),
        "turnover": Constraints(max_turnover=0.15),
        "participation": Constraints(max_participation=0.01),
        "no_buy": Constraints(no_buy=no_buy),
        "no_trade": Constraints(no_trade=no_trade),
    }


@pytest.mark.parametrize(
    "case",
    [
        "allow_cash", "max_weight", "sector", "beta_explicit", "beta_predicted", "style",
        "industry_exposure", "turnover", "participation", "no_buy", "no_trade",
    ],
)
def test_each_constraint_active_and_satisfied(cvx, small, case):
    model, labels, alpha, bench, adv, vol, w_prev = case_inputs(small)
    con = constraint_cases(model, labels, bench, w_prev)[case]
    kw = {"aum": AUM, "adv_value": adv, "daily_vol": vol, "risk_aversion": LAM}

    base = optimize(alpha, model, w_prev, constraints=Constraints(), **kw)
    assert base.ok
    v_base = violations(base, con, model, w_prev, aum=AUM, adv=adv)
    if case == "allow_cash":
        key = ["budget"]
    else:
        key = [k for k in v_base if k not in ("budget", "long_only")]
    assert max(v_base[k] for k in key) > 1e-3, f"{case}: 기본해가 이미 만족 — 활성 아님 {v_base}"

    res = optimize(alpha, model, w_prev, constraints=con, **kw)
    assert res.ok, res.status
    v = violations(res, con, model, w_prev, aum=AUM, adv=adv)
    assert max(v.values()) <= TOL, v
    # 활성: 새 제약 중 적어도 하나가 경계에 붙어 있다.
    assert max(v[k] for k in key) > -1e-5, v


def test_all_constraints_active_together(cvx, small):
    model, labels, alpha, bench, adv, vol, w_prev = case_inputs(small)
    n = len(alpha)
    # 거래 불가·매수 금지는 벤치 비중이 작은 종목에 — 큰 종목을 묶으면 상한과 충돌해 infeasible
    # 이다.
    order = np.argsort(w_prev)
    no_trade = np.zeros(n, bool)
    no_trade[order[:3]] = True
    no_buy = np.zeros(n, bool)
    no_buy[order[3:8]] = True
    # 상한은 최대 종목이 참여율 한도의 절반만 팔면 닿는 곳 — 활성이면서 닿을 수 있다.
    top = int(np.argmax(w_prev))
    max_w = w_prev[top] - 0.5 * 0.05 * adv[top] / AUM
    # 스타일 범위는 출발점(벤치) 노출 ±0.05 — 회전율·참여율 한도 안에서 닿을 수 있게.
    e0, e1 = (model.X[:, model.names.index(f"style{j}")] @ w_prev for j in (0, 1))
    alpha = alpha * COST_ALPHA
    con = Constraints(
        budget=1.0,
        max_weight=max_w,
        sector=labels,
        benchmark=bench,
        sector_band=0.03,
        beta_bounds=(0.98, 1.02),
        style_bounds={
            "style0": (e0 - 0.05, e0 + 0.05),
            "style1": (e1 - 0.05, e1 + 0.05),
        },
        max_turnover=0.4,
        max_participation=0.05,
        no_buy=no_buy,
        no_trade=no_trade,
    )
    costs = CostParams(trade_date=date(2024, 6, 3), market="KOSPI", half_spread_bp=5.0)
    res = optimize(
        alpha, model, w_prev, aum=AUM, adv_value=adv, daily_vol=vol, cost_params=costs,
        constraints=con, risk_aversion=LAM,
    )
    assert res.ok, res.status
    v = violations(res, con, model, w_prev, aum=AUM, adv=adv)
    assert max(v.values()) <= TOL, v
    assert v["max_weight"] > -1e-5, v  # 상한은 실제로 붙어 있다


# ── 실패 = 상태 + w_prev ──────────────────────────────────────────────


def test_infeasible_returns_status_and_w_prev(cvx, small):
    model, _, alpha, bench = small
    w_prev = bench.copy()
    # 60 종목 × 상한 1% = 60% < 예산 100%.
    res = optimize(alpha, model, w_prev, constraints=Constraints(max_weight=0.01))
    assert not res.ok
    assert "infeasible" in res.status
    np.testing.assert_array_equal(res.weights, w_prev)
    assert res.weights is not w_prev
    assert res.turnover == 0.0 and res.est_cost == 0.0
    assert res.expected_alpha == pytest.approx(alpha @ w_prev)


def test_infeasible_no_trade_vs_budget(cvx, small):
    model, _, alpha, bench = small
    w_prev = bench * 0.5  # 절반 현금, 전 종목 거래 불가인데 Σw = 1 요구
    res = optimize(alpha, model, w_prev, constraints=Constraints(no_trade=np.ones(60, bool)))
    assert not res.ok
    np.testing.assert_array_equal(res.weights, w_prev)


def test_input_errors_raise(cvx, small):
    model, labels, alpha, bench = small
    with pytest.raises(ValueError, match="benchmark"):
        optimize(alpha, model, None, constraints=Constraints(sector=labels, sector_band=0.02))
    with pytest.raises(ValueError, match="benchmark"):
        optimize(alpha, model, None, constraints=Constraints(beta_bounds=(0.9, 1.1)))
    with pytest.raises(ValueError, match="model.names"):
        optimize(alpha, model, None, constraints=Constraints(style_bounds={"nope": (0, 1)}))
    with pytest.raises(ValueError, match="aum"):
        optimize(alpha, model, None, constraints=Constraints(max_participation=0.01))
    with pytest.raises(ValueError, match="NaN"):
        optimize(np.where(np.arange(60) == 3, np.nan, alpha), model, None)
    costs = CostParams(trade_date=date(2024, 6, 3), market="KOSPI", half_spread_bp=5.0)
    with pytest.raises(ValueError, match="원 단위"):  # 백만원 단위 ADV
        optimize(alpha, model, None, aum=AUM, adv_value=np.full(60, 5000.0),
                 daily_vol=np.full(60, 0.02), cost_params=costs)


# ── 비용: §3.2 와 같은 숫자 ───────────────────────────────────────────


def test_est_cost_agrees_with_trade_cost_bp(cvx, small):
    model, _, alpha, bench, adv, vol, w_prev = case_inputs(small)
    alpha = alpha * COST_ALPHA
    on = date(2024, 6, 3)
    markets = np.where(np.arange(60) % 2 == 0, "KOSPI", "KOSDAQ")
    hs = np.linspace(2.0, 15.0, 60)
    costs = CostParams(trade_date=on, market=markets, half_spread_bp=hs, y=1.5)
    res = optimize(
        alpha, model, w_prev, aum=AUM, adv_value=adv, daily_vol=vol, cost_params=costs,
        risk_aversion=LAM,
    )
    assert res.ok
    dw = res.weights - w_prev
    traded = np.abs(dw) > 1e-9
    assert traded.sum() >= 5
    q = np.abs(dw) * AUM
    side = np.where(dw > 0, "BUY", "SELL")
    bp = trade_cost_bp(side, q, adv, vol, markets, on, half_spread_bp=hs, y=1.5)
    want = float(np.sum(bp / 1e4 * q) / AUM)
    assert res.est_cost == pytest.approx(want, rel=1e-10)
    assert res.turnover == pytest.approx(np.abs(dw).sum(), rel=1e-12)
    # 충격 부분만: k_i |Δw|^{3/2} = 충격 bp × 대금 / AUM.
    k = sqrt_impact_bp(AUM, adv, vol, y=1.5) / 1e4
    imp_bp = sqrt_impact_bp(q, adv, vol, y=1.5)
    np.testing.assert_allclose(
        k * np.abs(dw) ** 1.5, imp_bp / 1e4 * q / AUM, rtol=1e-12, atol=1e-18
    )


def test_costs_shrink_turnover_monotonically(cvx, small):
    model, _, alpha, bench, adv, vol, w_prev = case_inputs(small)
    alpha = alpha * COST_ALPHA
    kw = {"aum": AUM, "adv_value": adv, "daily_vol": vol, "risk_aversion": LAM}
    free = optimize(alpha, model, w_prev, **kw)
    turns = [free.turnover]
    for y in (0.5, 2.0, 8.0):
        c = CostParams(trade_date=date(2024, 6, 3), market="KOSPI", half_spread_bp=5.0, y=y)
        r = optimize(alpha, model, w_prev, cost_params=c, **kw)
        assert r.ok
        turns.append(r.turnover)
    assert all(a > b for a, b in zip(turns, turns[1:], strict=False)), turns


def _objective(res, alpha, model, lam):
    """목적함수 값 αᵀw − λ/2·wᵀΣw − 비용(est_cost 는 trade_cost_bp 와 같다 — 별도 테스트)."""
    w = res.weights
    q = model.X.T @ w
    return alpha @ w - 0.5 * lam * (q @ model.F @ q + model.D @ (w * w)) - res.est_cost


@pytest.mark.parametrize("attempt", [0, 1, 2])
def test_every_retry_attempt_solves_the_same_problem(cvx, small, monkeypatch, attempt):
    """재시도 형태(원뿔 눈금·스텝)마다 같은 해 — 눈금을 목적함수에 잘못 곱하면 여기서 걸린다.

    각 해는 같은 제약의 다른 feasible 점(충격 없는 해와 w_prev 의 볼록 결합)보다 목적값이
    나빠선 안 된다.
    """
    mod = sys.modules["krx_quant_core.portfolio.optimize"]
    model, labels, alpha, bench, adv, vol, w_prev = case_inputs(small)
    alpha = alpha * COST_ALPHA
    con = Constraints(max_turnover=0.5, sector=labels, benchmark=bench, sector_band=0.05)
    costs = CostParams(trade_date=date(2024, 6, 3), market="KOSPI", half_spread_bp=5.0, y=2.0)
    kw = {"aum": AUM, "adv_value": adv, "daily_vol": vol, "constraints": con, "risk_aversion": LAM}
    ref = optimize(alpha, model, w_prev, cost_params=costs, **kw)
    no_impact = optimize(alpha, model, w_prev, cost_params=replace(costs, y=0.0), **kw)
    assert ref.ok and no_impact.ok

    monkeypatch.setattr(mod, "_CLARABEL_ATTEMPTS", (mod._CLARABEL_ATTEMPTS[attempt],))
    res = optimize(alpha, model, w_prev, cost_params=costs, **kw)
    assert res.ok, res.status
    assert np.max(np.abs(res.weights - ref.weights)) < 1e-4
    best = _objective(res, alpha, model, LAM)
    for e in (0.01, 0.1, 0.5, 1.0):
        w_alt = w_prev + e * (no_impact.weights - w_prev)
        ad = np.abs(w_alt - w_prev)
        q = ad * AUM
        side = np.where(w_alt > w_prev, "BUY", "SELL")
        cost = float(np.sum(
            trade_cost_bp(side, q, adv, vol, "KOSPI", date(2024, 6, 3), half_spread_bp=5.0, y=2.0)
            / 1e4 * q
        ) / AUM)
        qa = model.X.T @ w_alt
        f_alt = alpha @ w_alt - 0.5 * LAM * (qa @ model.F @ qa + model.D @ (w_alt**2)) - cost
        assert best >= f_alt - 1e-9, (e, best, f_alt)


def test_sell_tax_only_on_sells(cvx, small):
    """매도세는 매도에만 — 빈 북에서 사기만 하는 해는 세율과 무관해야 한다."""
    model, _, alpha, bench, adv, vol, _ = case_inputs(small)
    alpha = alpha * COST_ALPHA
    kw = {"aum": AUM, "adv_value": adv, "daily_vol": vol, "risk_aversion": LAM}
    out = []
    for on in (date(2019, 1, 2), date(2025, 3, 4)):  # 매도세 0.30% vs 0.15%
        c = CostParams(trade_date=on, market="KOSPI", half_spread_bp=5.0)
        out.append(optimize(alpha, model, None, cost_params=c, **kw))
    assert out[0].ok and out[1].ok
    # 빈 북·롱온리라 매도가 없다 → 세율이 달라도 같은 해(솔버 정확도 안에서). 매수에 세율이 붙으면
    # 0.15%p 차이가 비중을 1e-3 대로 움직인다.
    np.testing.assert_allclose(out[0].weights, out[1].weights, atol=1e-5)
    assert out[0].est_cost == pytest.approx(out[1].est_cost, rel=1e-4)


# ── 선택 의존성 ──────────────────────────────────────────────────────


def test_missing_cvxpy_gives_install_hint(monkeypatch, small):
    model, _, alpha, _ = small
    monkeypatch.setitem(sys.modules, "cvxpy", None)
    with pytest.raises(ImportError, match=r"krx-quant-core\[portfolio\]"):
        optimize(alpha, model, None)


# ── 속도: 2,500 종목 · K=35 한 번 < 5초 (완료 조건 2) ─────────────────────────


@pytest.mark.skipif(
    bool(os.environ.get("KQC_SKIP_PERF")), reason="KQC_SKIP_PERF 설정 — 속도 테스트 생략"
)
def test_perf_2500_names_k35(cvx):
    n = 2500
    model, labels = make_model(n, 28, 6, seed=11)  # 1 + 28 + 6 = 35
    assert model.X.shape == (n, 35)
    rng = np.random.default_rng(12)
    alpha = rng.normal(0, 3e-4, n)
    cap = rng.lognormal(0, 1.5, n)
    bench = cap / cap.sum()
    adv = rng.uniform(5e8, 5e10, n)
    vol = rng.uniform(0.01, 0.05, n)
    con = Constraints(
        max_weight=0.02,
        sector=labels,
        benchmark=bench,
        sector_band=0.05,
        beta_bounds=(0.9, 1.1),
        style_bounds={"style0": (-0.3, 0.3)},
        max_turnover=0.5,
        max_participation=0.05,
    )
    costs = CostParams(trade_date=date(2024, 6, 3), market="KOSPI", half_spread_bp=10.0)
    kw = {
        "aum": 1e11, "adv_value": adv, "daily_vol": vol, "cost_params": costs,
        "constraints": con, "risk_aversion": LAM,
    }
    t0 = time.perf_counter()
    res = optimize(alpha, model, bench, **kw)
    elapsed = time.perf_counter() - t0
    print(f"\n[perf] N={n} K=35 optimize {elapsed:.2f}s status={res.status}")
    assert res.ok, res.status
    assert elapsed < 5.0
    v = violations(res, con, model, bench, aum=1e11, adv=adv)
    assert max(v.values()) <= TOL, v


def test_constraints_is_frozen_dataclass():
    c = Constraints()
    assert c.budget == 1.0 and c.long_only and not c.allow_cash
    assert replace(c, max_weight=0.1).max_weight == 0.1
