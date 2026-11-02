"""제약 평균-분산 최적화 — 팩터 위험모델 + 선형·3/2 제곱 거래비용 (cvxpy).

푸는 문제(분수 비중 ``w``, 직전 비중 ``w_prev``, ``Δw = w − w_prev``)::

    max_w  αᵀw − (λ/2)·wᵀΣw − Σ_i [c_i |Δw_i| + h_i Δw_i] − Σ_i k_i |Δw_i|^{3/2}

- ``Σ = X F Xᵀ + diag(D)`` 를 **팩터 형태 그대로** 쓴다. 보조변수 ``y = Xᵀw`` (K 개)를 두고
  ``wᵀΣw = ‖G y‖² + Σ D_i w_i²`` (``GᵀG = F``, 고유분해로 만든다 — 반정부호 F 도 받는다).
  N×N 행렬은 어디서도 만들지 않는다(2,500 종목이면 625만 원소).
- 선형 비용은 매수·매도가 다르다(매도세는 매도에만). 매수 단가 ``b_i``·매도 단가 ``s_i`` 를
  ``b·max(Δw,0) + s·max(−Δw,0) = ((b+s)/2)|Δw| + ((b−s)/2)Δw`` 로 **정확히** 바꿔 변수를
  늘리지 않는다(``c_i = (b+s)/2``, ``h_i = (b−s)/2``; ``b, s ≥ 0`` 이면 볼록).
  ``b_i``·``s_i`` 는 :func:`~krx_quant_core.costs.impact.trade_cost_bp` 에 대금 0 을 넣어 얻는다 —
  수수료(정가)+매도세(일자별)+반스프레드. 비용 정의를 여기서 다시 쓰지 않는다.
- 충격 ``k_i = sqrt_impact_bp(AUM, ADV_i, σ_i, y) / 1e4`` — 주문 ``|Δw_i|·AUM`` 의 충격 bp 가
  ``k_i·1e4·√|Δw_i|`` 이므로 원화 비용 / AUM 이 정확히 ``k_i |Δw_i|^{3/2}`` 다(§3.2 와 같은 식).

단위 약속: ``alpha`` 와 ``Σ`` 는 같은 지평(예: 둘 다 일간)이어야 하고, 비용은 이번 리밸런스 1회분
(AUM 대비 분수)이다 — 지평이 길면 호출부가 α 를 그 지평으로 맞춘다.

위험모델은 속성 ``X (N×K)``·``F (K×K)``·``D (N,)``·``names`` (``names[0] == "market"``)만 본다 —
:mod:`krx_quant_core.risk.factor_model` 의 ``RiskModel`` 에 묶이지 않는다.

cvxpy 는 선택 의존성이다(``pip install krx-quant-core[portfolio]``). 이 모듈 import 는 cvxpy 없이
되고, :func:`optimize` 를 부를 때 없으면 설치 안내 ``ImportError`` 를 낸다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from krx_quant_core.costs.impact import participation, sqrt_impact_bp, trade_cost_bp

__all__ = ["Constraints", "CostParams", "OptResult", "optimize"]

_BP = 1e4
_OK = "optimal"
#: 3차원 거듭제곱 원뿔을 직접 받는 솔버 — 충격 항을 원뿔로 바로 쓴다(그 밖은 cp.power 분해).
_POW_CONE_SOLVERS = frozenset({"CLARABEL", "SCS", "MOSEK"})
#: 충격 원뿔이 있을 때 Clarabel 시도 순서 ``(원뿔 눈금, 솔버 옵션)``.
#: 전부 같은 문제다(``optimize`` 끝 주석).
_CLARABEL_ATTEMPTS: tuple[tuple[float, dict[str, Any]], ...] = (
    (1.0, {"max_step_fraction": 0.8}),
    (1.0, {"max_step_fraction": 0.7}),
    (0.01, {"max_step_fraction": 0.8}),
)

Bounds = tuple[float | None, float | None]


@dataclass(frozen=True)
class CostParams:
    """거래비용 입력(:mod:`krx_quant_core.costs.impact` 로 그대로 넘어간다).

    Attributes:
        trade_date: 체결일 — 매도세율을 이 날짜로 고른다
            (:func:`~krx_quant_core.costs.tax.sell_tax_rate`).
        market: ``Market.parse`` 가 받는 값. 스칼라 또는 종목별 ``(N,)`` 배열.
        half_spread_bp: 반스프레드(bp). 스칼라 또는 ``(N,)``. 기본값 없음 — 잊으면 낙관 편향이다.
        y: 제곱근 충격 계수. 기본 1.0 은 **보정 전 자리표시값**(``costs/impact.py`` 설명) —
            1·2배로 돌린다. 0 이면 충격 없이 선형 비용만.
    """

    trade_date: date
    market: Any
    half_spread_bp: Any
    y: float = 1.0


@dataclass(frozen=True)
class Constraints:
    """최적화 제약. ``None`` 인 필드는 꺼진 제약이다.

    기본값 = 완전 투자 롱온리(``Σw = 1``, ``w ≥ 0``). 배열 필드는 전부 길이 N(``alpha`` 와 같은
    종목 순서).

    Attributes:
        budget: ``Σw = budget``. ``None`` 이면 예산 제약 없음.
        allow_cash: 참이면 ``Σw ≤ budget``(나머지는 현금).
        long_only: 참이면 ``w ≥ 0``. 거짓이면 하한 없음(분석·검증용 — 실매매 북은 롱온리).
        max_weight: 종목 상한 ``w ≤ max_weight`` (스칼라 또는 ``(N,)``).
        sector: 업종 라벨 ``(N,)``. ``sector_band`` 와 같이 줘야 켜진다.
        benchmark: 벤치 비중 ``(N,)``. 업종 밴드의 기준(``b_g = Σ_{i∈g} benchmark_i``)이자
            ``beta`` 를 안 줬을 때 예측 베타의 기준. 시총가중 유니버스면 호출부가
            ``cap / cap.sum()`` 을 넘긴다(최적화엔 시총이 없다).
        sector_band: ``|Σ_{i∈g} w_i − b_g| ≤ sector_band`` (모든 업종 g).
        beta: 종목 베타 ``(N,)``. ``None`` 이면 위험모델의 예측 베타 ``Σb / (bᵀΣb)``
            (``benchmark`` 필요). 주의: §3.1 위험모델의 ``market`` 열은 전 종목 1 이라 그 노출은
            ``Σw`` 와 같다 — 베타가 아니다.
        beta_bounds: 포트 베타 ``βᵀw`` 의 ``(하한, 상한)``. 한쪽 ``None`` 은 열린 쪽.
        style_bounds: ``{팩터 이름: (하한, 상한)}`` — 위험모델 노출 ``(Xᵀw)_k`` 의 범위. 이름은
            ``model.names`` 안의 무엇이든(스타일·업종·market). 모르는 이름은 ``ValueError``.
        max_turnover: 편도 회전율 ``Σ|Δw| ≤ max_turnover``.
        max_participation: 참여율 ``|Δw_i|·AUM ≤ max_participation·ADV_i``
            (``aum``·``adv_value`` 필요).
        no_buy: 매수 금지 마스크 ``(N,)`` bool — ``Δw_i ≤ 0`` (관리종목·상한가 등, 호출부가 준다).
        no_trade: 거래 불가 마스크 ``(N,)`` bool — ``Δw_i = 0`` (거래정지는 매도도 못 한다).
    """

    budget: float | None = 1.0
    allow_cash: bool = False
    long_only: bool = True
    max_weight: Any = None
    sector: Any = None
    benchmark: Any = None
    sector_band: float | None = None
    beta: Any = None
    beta_bounds: Bounds | None = None
    style_bounds: Mapping[str, Bounds] | None = None
    max_turnover: float | None = None
    max_participation: float | None = None
    no_buy: Any = None
    no_trade: Any = None


@dataclass(frozen=True)
class OptResult:
    """최적화 결과. 실패 때도 같은 모양 — ``weights = w_prev`` 이고 나머지는 그 비중에서 잰 값.

    Attributes:
        weights: 목표 비중 ``(N,)``. ``status != "optimal"`` 이면 ``w_prev`` 사본(거래 안 함).
        expected_alpha: ``αᵀw``.
        expected_risk: 예측 표준편차 ``√(wᵀΣw)`` (``Σ`` 의 지평 단위).
        est_cost: 추정 거래비용(AUM 대비 분수) — 선형(수수료·매도세·반스프레드) + 충격.
            비용을 안 줬으면 0.
        turnover: 편도 회전율 ``Σ|Δw|``.
        exposures: 팩터 노출 ``{name: (Xᵀw)_k}``.
        status: cvxpy 상태(``"optimal"``·``"infeasible"``·``"optimal_inaccurate"``…) 또는
            ``"solver_error"``. ``"optimal"`` 만 성공으로 본다 — ``optimal_inaccurate`` 는 제약
            위반이 있을 수 있어 거래하지 않는다.
        solver: 쓴 솔버 이름.
    """

    weights: np.ndarray
    expected_alpha: float
    expected_risk: float
    est_cost: float
    turnover: float
    exposures: dict[str, float] = field(default_factory=dict)
    status: str = _OK
    solver: str = ""

    @property
    def ok(self) -> bool:
        """``status == "optimal"``."""
        return self.status == _OK


def _vec(x: Any, n: int, name: str, *, dtype: Any = np.float64) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full(n, arr.item(), dtype=dtype)
    if arr.shape != (n,):
        raise ValueError(f"{name} 의 모양이 ({n},) 이 아니다: {arr.shape}")
    if dtype is np.float64 and not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"{name} 에 NaN/inf 가 {bad}개 — 빼거나 채워서 넘긴다")
    return arr


def _factor_root(F: np.ndarray) -> np.ndarray:
    """``GᵀG = F`` 인 ``G`` (고유분해, 음의 고유값은 0 으로 자른다)."""
    sym = 0.5 * (F + F.T)
    evals, evecs = np.linalg.eigh(sym)
    scale = max(float(np.max(np.abs(evals))), 1.0)
    if evals.min() < -1e-10 * scale:
        raise ValueError(f"F 가 반정부호가 아니다(최소 고유값 {evals.min():.3g})")
    return np.sqrt(np.clip(evals, 0.0, None))[:, None] * evecs.T


def _bounded(expr: Any, bounds: Bounds, name: str) -> list:
    lo, hi = bounds
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"{name} 범위의 하한 > 상한: {bounds}")
    out = []
    if lo is not None:
        out.append(expr >= lo)
    if hi is not None:
        out.append(expr <= hi)
    return out


def _import_cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - 설치 환경에 따라
        raise ImportError(
            "portfolio.optimize 는 cvxpy 가 필요하다 — pip install krx-quant-core[portfolio]"
        ) from exc
    return cp


def optimize(
    alpha: Any,
    model: Any,
    w_prev: Any,
    *,
    aum: float | None = None,
    adv_value: Any = None,
    daily_vol: Any = None,
    cost_params: CostParams | None = None,
    constraints: Constraints | None = None,
    risk_aversion: float = 1.0,
    solver: str = "CLARABEL",
    solver_opts: Mapping[str, Any] | None = None,
) -> OptResult:
    """롱온리 제약 평균-분산 리밸런스 한 번(모듈 설명의 목적함수).

    Args:
        alpha: 기대수익 ``(N,)`` (``Σ`` 와 같은 지평).
        model: ``X``·``F``·``D``·``names`` 속성을 가진 위험모델(``RiskModel`` 또는 같은 모양).
        w_prev: 직전 비중 ``(N,)``. ``None`` 이면 빈 북(전부 0).
        aum: 운용자산(원). 비용·참여율 제약에 필요.
        adv_value: 평균 일 거래대금 ``(N,)`` (**원**, 1e6 미만이면 ``ValueError``). 비용·참여율
            제약에 필요.
        daily_vol: 일간 변동성 ``(N,)`` (분수). 비용에 필요.
        cost_params: ``None`` 이면 비용 항 없음.
        constraints: ``None`` 이면 :class:`Constraints` 기본값(완전 투자 롱온리).
        risk_aversion: ``λ`` (> 0).
        solver: cvxpy 솔버 이름(기본 Clarabel).
        solver_opts: 솔버 옵션(허용오차 등)을 그대로 넘긴다.

    Returns:
        :class:`OptResult`. infeasible·솔버 실패는 예외가 아니라 ``status`` 로 돌려주고
        ``weights = w_prev``.

    Raises:
        ImportError: cvxpy 가 없을 때(설치 안내 포함).
        ValueError: 입력 모양·단위·NaN 오류, 필요한 입력(``aum``·``adv_value``·``benchmark`` 등)
            누락. 입력 오류는 infeasible 과 달리 조용히 넘기지 않는다.
    """
    cp = _import_cvxpy()
    con = constraints if constraints is not None else Constraints()
    if not risk_aversion > 0:
        raise ValueError("risk_aversion 은 양수여야 한다")

    X = np.asarray(model.X, dtype=np.float64)
    F = np.asarray(model.F, dtype=np.float64)
    names = list(model.names)
    if X.ndim != 2:
        raise ValueError(f"model.X 는 2차원이어야 한다: {X.shape}")
    n, k = X.shape
    if F.shape != (k, k) or len(names) != k:
        raise ValueError(f"model.F {F.shape}·names {len(names)} 가 X 의 K={k} 와 안 맞는다")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(F))):
        raise ValueError("model.X/F 에 NaN/inf 가 있다")
    D = _vec(model.D, n, "model.D")
    if np.any(D < 0):
        raise ValueError("model.D(특이분산)에 음수가 있다")
    a = _vec(alpha, n, "alpha")
    w0 = np.zeros(n) if w_prev is None else _vec(w_prev, n, "w_prev")
    G = _factor_root(F)

    def cov_times(v: np.ndarray) -> np.ndarray:
        return X @ (F @ (X.T @ v)) + D * v

    def need(value: Any, what: str, why: str) -> Any:
        if value is None:
            raise ValueError(f"{why} 에는 {what} 가 필요하다")
        return value

    # ── 비용 계수(§3.2 함수에서) ────────────────────────────────────────────
    c_abs = h_lin = k_imp = None
    if cost_params is not None:
        aum_v = float(need(aum, "aum", "비용"))
        adv = _vec(need(adv_value, "adv_value", "비용"), n, "adv_value")
        vol = _vec(need(daily_vol, "daily_vol", "비용"), n, "daily_vol")
        args = (adv, vol, cost_params.market, cost_params.trade_date)
        kw = {"half_spread_bp": cost_params.half_spread_bp, "y": cost_params.y}
        buy = trade_cost_bp("BUY", 0.0, *args, **kw) / _BP
        sell = trade_cost_bp("SELL", 0.0, *args, **kw) / _BP
        buy = np.broadcast_to(buy, (n,)).astype(np.float64)
        sell = np.broadcast_to(sell, (n,)).astype(np.float64)
        c_abs, h_lin = 0.5 * (buy + sell), 0.5 * (buy - sell)
        k_imp = sqrt_impact_bp(aum_v, adv, vol, y=cost_params.y) / _BP

    def realized_cost(dw: np.ndarray) -> float:
        if c_abs is None:
            return 0.0
        ad = np.abs(dw)
        return float(c_abs @ ad + h_lin @ dw + k_imp @ ad**1.5)

    def result(w: np.ndarray, status: str, solver_name: str) -> OptResult:
        dw = w - w0
        q = X.T @ w
        var = float(q @ F @ q + D @ (w * w))
        return OptResult(
            weights=w,
            expected_alpha=float(a @ w),
            expected_risk=float(np.sqrt(max(var, 0.0))),
            est_cost=realized_cost(dw),
            turnover=float(np.abs(dw).sum()),
            exposures=dict(zip(names, (float(v) for v in q), strict=True)),
            status=status,
            solver=solver_name,
        )

    # ── 상수 준비(문제를 다시 만들 때도 같은 숫자) ─────────────────────────────
    from scipy import sparse  # cvxpy 의존성 — cvxpy 가 있으면 있다

    # 결정변수는 거래 가능한 종목의 Δw 뿐이다(w = w_prev + P·d, P 는 선택 행렬).
    # - w 를 변수로 두고 Δw = w − w_prev 를 식으로 쓰면 충격 원뿔 입력이 아핀식이 되어, 비용·다중
    #   제약 60 종목 사례에서 Clarabel 이 InsufficientProgress(solver_error)로 멈췄다 — Δw 변수로는
    #   optimal(실측).
    # - 거래 불가 종목을 Δw_i = 0 등식으로 묶으면 그 종목의 충격 원뿔이 꼭짓점에 고정돼, 무작위
    #   120 문제 중 15개가 feasible 인데 solver_error 였다(14개가 no_trade) — 변수에서 빼서 상수로.
    frozen = np.zeros(n, bool)
    if con.no_trade is not None:
        frozen = _vec(con.no_trade, n, "no_trade", dtype=bool)
    free = np.flatnonzero(~frozen)
    m = free.size
    pos = np.full(n, -1)
    pos[free] = np.arange(m)
    P = sparse.csr_matrix((np.ones(m), (free, np.arange(m))), shape=(n, max(m, 1)))

    max_w = None if con.max_weight is None else _vec(con.max_weight, n, "max_weight")
    bench = None if con.benchmark is None else _vec(con.benchmark, n, "benchmark")
    sector = None
    if con.sector_band is not None:
        labels = np.asarray(need(con.sector, "sector", "sector_band"), dtype=object)
        if labels.shape != (n,):
            raise ValueError(f"sector 의 모양이 ({n},) 이 아니다: {labels.shape}")
        b = need(bench, "benchmark", "sector_band")
        groups, inv = np.unique(labels.astype(str), return_inverse=True)
        S = sparse.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(len(groups), n))
        sector = (S, S @ b)
    beta = None
    if con.beta_bounds is not None:
        if con.beta is not None:
            beta = _vec(con.beta, n, "beta")
        else:
            b = need(bench, "benchmark", "beta_bounds(beta 미지정 — 예측 베타)")
            sb = cov_times(b)
            beta = sb / float(b @ sb)
    style_idx = {}
    for fname in con.style_bounds or {}:
        if fname not in names:
            raise ValueError(f"style_bounds 의 {fname!r} 가 model.names 에 없다")
        style_idx[fname] = names.index(fname)
    part_cap = None
    if con.max_participation is not None:
        aum_p = float(need(aum, "aum", "max_participation"))
        adv_p = _vec(need(adv_value, "adv_value", "max_participation"), n, "adv_value")
        # |Δw_i|·AUM/ADV_i ≤ mp ⇔ |Δw_i| ≤ mp / participation(AUM, ADV_i) — 같은 함수·단위 가드.
        part_cap = (con.max_participation / participation(aum_p, adv_p))[free]
    nb_pos = np.empty(0, dtype=np.intp)
    if con.no_buy is not None:
        nb_pos = pos[np.flatnonzero(_vec(con.no_buy, n, "no_buy", dtype=bool) & ~frozen)]
    kf = None if k_imp is None else k_imp[free]
    has_impact = kf is not None and m > 0 and bool(np.any(kf > 0))
    use_cone = has_impact and solver.upper() in _POW_CONE_SOLVERS

    # 목적함수 크기 맞춤: 일간 α·Σ 는 1e-4 대라 솔버의 절대 허용오차(1e-8)가 비중 오차 1e-3 대로
    # 번진다(실측: 스케일 없이 2,500 종목이 optimal_inaccurate, 60 종목 베타 경계에서 1.6e-4 이탈).
    # λ·평균 총분산으로 나눠 이차항 곡률을 O(1)로 둔다 — argmax 는 그대로다.
    avg_var = float(np.mean(np.sum((X @ G.T) ** 2, axis=1) + D))
    scale = 1.0 / (risk_aversion * avg_var) if avg_var > 0 else 1.0

    def build(cone_scale: float) -> tuple[Any, Any]:
        y = cp.Variable(k)
        if m:
            d = cp.Variable(m)
            w = w0 + P @ d
        else:
            d = None
            w = cp.Constant(w0)
        cons: list = [y == X.T @ w]
        risk = cp.sum_squares(G @ y) + cp.sum_squares(cp.multiply(np.sqrt(D), w))
        objective = a @ w - 0.5 * risk_aversion * risk
        if c_abs is not None and m:
            objective = objective - (c_abs[free] @ cp.abs(d) + h_lin[free] @ d)
            if use_cone:
                # 에피그래프: t_i^{2/3}·s^{1/3} ≥ |Δw_i| ⇔ √s·t_i ≥ |Δw_i|^{3/2} (3차원 거듭제곱
                # 원뿔, 정확). cp.power(cp.abs(Δw), 1.5) 의 SOC 분해는 Δw=0(거래 안 함) 꼭짓점에서
                # Clarabel 이 수렴을 못 끝낸다 — 실측 60·2,500 종목, α 세 크기 모두
                # optimal_inaccurate, 이 형태는 전부 optimal. s 는 조건수만 바꾸는 재시도용 눈금.
                t = cp.Variable(m)
                cons.append(cp.constraints.PowCone3D(t, np.full(m, cone_scale), d, 2.0 / 3.0))
                objective = objective - (kf * np.sqrt(cone_scale)) @ t
            elif has_impact:
                objective = objective - kf @ cp.power(cp.abs(d), 1.5)

        if con.budget is not None:
            cons.append(cp.sum(w) <= con.budget if con.allow_cash else cp.sum(w) == con.budget)
        if con.long_only:
            cons.append(w >= 0)
        if max_w is not None:
            cons.append(w <= max_w)
        if sector is not None:
            cons.append(cp.abs(sector[0] @ w - sector[1]) <= con.sector_band)
        if beta is not None:
            cons += _bounded(beta @ w, con.beta_bounds, "beta_bounds")
        for fname, j in style_idx.items():
            cons += _bounded(y[j], con.style_bounds[fname], f"style_bounds[{fname}]")
        if m:
            if con.max_turnover is not None:
                cons.append(cp.sum(cp.abs(d)) <= con.max_turnover)
            if part_cap is not None:
                cons += [d <= part_cap, d >= -part_cap]
            if nb_pos.size:
                cons.append(d[nb_pos] <= 0)
        return cp.Problem(cp.Maximize(scale * objective), cons), d

    # 충격 원뿔 + Clarabel 기본 설정(max_step_fraction 0.99)은 feasible 문제에서
    # InsufficientProgress 로 자주 멈춘다. 실측(무작위 685 문제, 60~800 종목, 제약·비용 무작위,
    # 비용 없이는 전부 풀리는 것만): 기본 121개 중 8개 solver_error, 0.9 는 251개 중 1개,
    # 0.8·0.7 은 685개 중 0개. 그래서 0.8 로 풀고, 안 되면 0.7 → 원뿔 눈금 0.01 로 같은 문제를
    # 다시 푼다(시도 사이 비중 차이 ≤ 2e-5).
    # infeasible 은 솔버의 증명서라 재시도하지 않는다. 호출부 solver_opts 가 이 기본값을 덮는다.
    if use_cone and solver.upper() == "CLARABEL":
        attempts = _CLARABEL_ATTEMPTS
    else:
        attempts = ((1.0, {}),)
    status = "solver_error"
    for cone_scale, default_opts in attempts:
        prob, d = build(cone_scale)
        try:
            prob.solve(solver=solver, **{**default_opts, **dict(solver_opts or {})})
        except cp.error.SolverError:
            status = "solver_error"
            continue
        status = str(prob.status)
        if status in (_OK, "infeasible"):
            break
    if status != _OK:
        return result(w0.copy(), status, solver)
    w_opt = w0.copy()
    if m:
        if d.value is None:
            return result(w0.copy(), "solver_error", solver)
        w_opt[free] += np.asarray(d.value, dtype=np.float64)
    return result(w_opt, status, solver)
