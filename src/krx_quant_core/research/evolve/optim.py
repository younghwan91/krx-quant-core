"""BIPOP 구동기 + 최적화기 백엔드(CMAwM 가속판·CatCMAwM·순정 CMAwM).

구동기 :func:`bipop` 은 swing-it ``pattern_ga_run.cma_bipop_x0`` 과 **같은 재시작 규칙·같은 난수
순서**다 (Hansen 2009 BIPOP, 정체 재시작, 첫 런 평균 = ``x0``, 재시작 평균 = ``x_restart`` 풀에서).
다른 점은 평가가 **배치**라는 것뿐이다 — 세대마다 λ개를 한 번에 묻고, 키가 memo 에 없는 것만 처음
나온 순서대로 소비자 ``evaluate`` 에 넘긴다(순차 평가와 같은 순서). 그래서 같은 시드면 이력·memo
순서·재시작 기록이 같다.

백엔드:

- ``"cmawm"`` — :class:`FastCMAwM`. ``cmaes.CMAwM`` 상속, 표본마다 다시 하던 ``B·diag(D)`` 를 세대당
  한 번, 이산 부호화 일괄, scipy 없을 때 ``chi2_ppf``·``norm_cdf`` 를 라이브러리 스칼라 함수
  그대로(+ 값 memo) 루프로. **라이브러리와 비트 동일 궤적**(시험). cmaes 가 정확히 0.13.1 이
  아니면 라이브러리 경로로 폴백.
- ``"catcmawm"`` — ``cmaes.CatCMAwM``(Hamano et al. 2025). Real → 연속, Ordinal → 정수(수준 번호),
  Choice → 범주형. 범위 밖 Real 은 재표본이 아니라 잘림(끔 끝에 질량이 쌓인다 — 백엔드 비교 때
  공개).
- ``"cmawm_lib"`` — 순정 ``cmaes.CMAwM``(대조용).

cmaes 는 선택 의존성(``pip install "krx-quant-core[evo]"``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from krx_quant_core._jit import HAVE_NUMBA, njit
from krx_quant_core.research.evolve.space import Space, layout

__all__ = ["BACKENDS", "BipopResult", "FastCMAwM", "bipop", "make_backend"]

BACKENDS = ("cmawm", "catcmawm", "cmawm_lib")
_FAST_VERSIONS = ("0.13.1",)          # FastCMAwM 이 가정하는 cmaes 내부 — 정확히 이 판


def _cmaes():
    try:
        import cmaes  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - 설치 안내
        raise ImportError('research.evolve 최적화기는 cmaes 가 필요하다: '
                          'pip install "krx-quant-core[evo]"') from e
    return cmaes


# ---------------------------------------------------------------------------
# FastCMAwM
# ---------------------------------------------------------------------------

def _chi2_ppf_bisect(q):
    """``cmaes._stats.chi2_ppf`` 스칼라 원본과 같은 연산 순서(df=1, 100회 이분법) — numba 로
    컴파일한다."""
    if q == 0:
        return 0.0
    if q == 1:
        return math.inf
    a, b = 0.0, 100.0
    m = 0.0
    if q < 0.9:
        for _ in range(100):
            m = (a + b) / 2
            if math.erf(math.sqrt(m / 2)) < q:
                a = m
            else:
                b = m
    else:
        for _ in range(100):
            m = (a + b) / 2
            if math.erfc(math.sqrt(m / 2)) > 1.0 - q:
                a = m
            else:
                b = m
    return m


def _jit_chi2(chi_py):
    """numba 판을 만들고 원본과 비트 대조 — 하나라도 다르면 ``None``(원본 사용)."""
    if not HAVE_NUMBA:
        return None
    f = njit(_chi2_ppf_bisect)
    rng = np.random.default_rng(20261011)
    qs = np.concatenate([rng.random(400), 1.0 - 10 ** rng.uniform(-10, -0.5, 200),
                         10 ** rng.uniform(-10, -0.5, 200),
                         [0.0, 1.0, 0.9, 0.9 - 1e-16, 1e-10, 1 - 1e-10]])
    for q in qs.tolist():
        if f(q) != chi_py(q):
            return None
    return f


_STATS: tuple | None = None


def _fast_stats():
    """scipy 없는 라이브러리 경로의 ``chi2_ppf``·``norm_cdf`` 와 같은 값을 내는 대체(스칼라 원본 +
    memo). ``chi2_ppf`` 는 같은 이분법을 numba 로 돌린다(쓰기 전에 원본과 비트 대조, 다르면
    원본)."""
    global _STATS
    if _STATS is not None:
        return _STATS
    from cmaes import _stats  # noqa: PLC0415

    chi_py = _jit_chi2(_stats.chi2_ppf.pyfunc) or _stats.chi2_ppf.pyfunc
    cdf_py = _stats.norm_cdf.pyfunc
    memo: dict[float, float] = {}

    def chi2_ppf(q):
        a = np.asarray(q, dtype=float)
        out = np.empty(a.shape)
        flat_out = out.reshape(-1)
        if len(memo) > 4096:                         # 되풀이되는 값은 q = 1 − 2·margin 정도뿐
            memo.clear()
        for i, v in enumerate(a.ravel().tolist()):
            r = memo.get(v)
            if r is None:
                r = memo[v] = chi_py(v)
            flat_out[i] = r
        return out if out.ndim else out[()]

    def norm_cdf(x, loc=0.0, scale=1.0):
        bx, bl, bs = np.broadcast_arrays(np.asarray(x, float), np.asarray(loc, float),
                                         np.asarray(scale, float))
        out = np.empty(bx.shape)
        o = out.reshape(-1)
        for i, (xv, lv, sv) in enumerate(zip(bx.ravel().tolist(), bl.ravel().tolist(),
                                             bs.ravel().tolist(), strict=True)):
            o[i] = cdf_py(xv, lv, sv)
        return out if out.ndim else out[()]

    _STATS = (chi2_ppf, norm_cdf)
    return _STATS


def _ask_k(opt: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """라이브러리 ``ask`` 를 k 번 — (평가용 x, tell 용 원 x)."""
    pairs = [opt.ask() for _ in range(k)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _make_fast_cls():
    cmaes = _cmaes()
    from cmaes import _cmawm, _stats  # noqa: PLC0415

    fast_ok = cmaes.__version__ in _FAST_VERSIONS
    no_scipy = _cmawm.chi2_ppf is _stats.chi2_ppf

    class _FastCMAwM(cmaes.CMAwM):
        """``cmaes.CMAwM`` 와 비트 동일한 궤적을 내는 일괄 ``ask``·가속 ``tell``."""

        FAST = fast_ok

        def ask_batch(self, k: int) -> tuple[np.ndarray, np.ndarray]:
            """(평가용 x, tell 용 원 x) 를 ``k`` 개 — 라이브러리 ``ask`` 를 k 번 부른 것과 같다.

            라이브러리는 표본마다 ``randn(n)`` → ``B·diag(D)·z`` → 연속 축이 범위 안이면 받고,
            아니면 다시(최대 ``n_max_resampling`` 번, 그 뒤 한 번 더 뽑아 잘라 고친다).

            여기서는 같은 난수열을 덩어리로 뽑아 행렬곱 한 번으로 **확실히 범위 밖인** 행만 걸러
            낸다. 받을 행·경계 근처 행은 라이브러리와 같은 행렬-벡터 곱으로 다시 계산해 판정한다.
            끝에 난수 상태를 되감아 쓴 행 수만큼만 전진시킨다(라이브러리와 같은 상태)."""
            if not self.FAST:
                return _ask_k(self, k)
            cma = self._cma
            B, D = cma._eigen_decomposition()
            BD = B.dot(np.diag(D))
            mean, sigma, rng, n = cma._mean, cma._sigma, cma._rng, cma._n_dim
            ci = self._continuous_idx
            lo, hi = self._continuous_space[:, 0], self._continuous_space[:, 1]
            nmax = self._n_max_resampling
            block = max(64, 8 * k)
            state = rng.get_state()
            XT = np.empty((k, n))
            used = 0
            Z = np.empty((0, n))
            bad = np.empty(0, bool)
            pos = 0
            for s in range(k):
                tries = 0
                while True:
                    if pos == len(Z):
                        Z = rng.randn(block, n)
                        C = (mean + sigma * (Z @ BD.T))[:, ci]        # 근사(합 순서만 다르다)
                        tol = 1e-9 * (1.0 + np.abs(C))
                        bad = np.any((C < lo - tol) | (C > hi + tol), axis=1)
                        pos = 0
                    z, skip = Z[pos], bad[pos]
                    pos += 1
                    used += 1
                    if tries == nmax:                    # 라이브러리: 실패 뒤 한 번 더 → 고침
                        x = mean + sigma * BD.dot(z)
                        x[ci] = self._repair_continuous_params(x[ci])
                        break
                    tries += 1
                    if skip:
                        continue
                    x = mean + sigma * BD.dot(z)
                    c = x[ci]
                    if np.all(c >= lo) and np.all(c <= hi):
                        break
                XT[s] = x
            rng.set_state(state)
            rng.randn(used, n)                                         # 쓴 만큼만 전진
            XE = XT.copy()
            if self._n_zdim > 0:
                di = self._discrete_idx
                XE[:, di] = self._encode_discrete_params(XT[:, di])
            return XE, XT

        def tell(self, solutions) -> None:
            """``cmaes.CMAwM.tell`` (0.13.1) 과 같은 코드.

            scipy 가 없으면 ``chi2_ppf``·``norm_cdf`` 만 같은 값을 내는 빠른 판
            (:func:`_fast_stats`)으로 부른다."""
            if not self.FAST:
                return super().tell(solutions)
            chi2_ppf, norm_cdf = _fast_stats() if no_scipy else (_cmawm.chi2_ppf, _cmawm.norm_cdf)
            self._cma.tell(solutions)
            mean = self._cma._mean
            sigma = self._cma._sigma
            C = self._cma._C
            if self._n_zdim == 0:
                return
            di = self._discrete_idx
            updated_m_integer = mean[di]
            self.m_z_lim_low, self.m_z_lim_up = self._get_discrete_param_limits(updated_m_integer)
            z_scale = sigma * self._A[di] * np.sqrt(np.diag(C)[di])
            low_cdf = norm_cdf(self.m_z_lim_low, loc=updated_m_integer, scale=z_scale)
            up_cdf = 1.0 - norm_cdf(self.m_z_lim_up, loc=updated_m_integer, scale=z_scale)
            mid_cdf = 1.0 - (low_cdf + up_cdf)
            edge_mask = np.maximum(low_cdf, up_cdf) > 0.5
            side_mask = np.maximum(low_cdf, up_cdf) <= 0.5
            if np.any(edge_mask):
                modify_mask = np.minimum(low_cdf, up_cdf) < self.margin
                modify_sign = np.sign(mean[di] - self.m_z_lim_up)
                dist = (sigma * self._A[di]
                        * np.sqrt(chi2_ppf(q=1.0 - 2.0 * self.margin) * np.diag(C)[di]))
                mean[di] = mean[di] + modify_mask * edge_mask * (
                    self.m_z_lim_up + modify_sign * dist - mean[di])
            low_cdf = np.maximum(low_cdf, self.margin / 2.0)
            up_cdf = np.maximum(up_cdf, self.margin / 2.0)
            modified_low_cdf = low_cdf + (1.0 - low_cdf - up_cdf - mid_cdf) * (
                low_cdf - self.margin / 2) / (low_cdf + mid_cdf + up_cdf - 3.0 * self.margin / 2)
            modified_up_cdf = up_cdf + (1.0 - low_cdf - up_cdf - mid_cdf) * (
                up_cdf - self.margin / 2) / (low_cdf + mid_cdf + up_cdf - 3.0 * self.margin / 2)
            modified_low_cdf = np.clip(modified_low_cdf, 1e-10, 0.5 - 1e-10)
            modified_up_cdf = np.clip(modified_up_cdf, 1e-10, 0.5 - 1e-10)
            chi_low_sq = np.sqrt(chi2_ppf(q=1.0 - 2 * modified_low_cdf))
            chi_up_sq = np.sqrt(chi2_ppf(q=1.0 - 2 * modified_up_cdf))
            C_diag_sq = np.sqrt(np.diag(C))[di]
            self._A[di] = self._A[di] + side_mask * (
                (self.m_z_lim_up - self.m_z_lim_low)
                / ((chi_low_sq + chi_up_sq) * sigma * C_diag_sq) - self._A[di])
            mean[di] = mean[di] + side_mask * (
                (self.m_z_lim_low * chi_up_sq + self.m_z_lim_up * chi_low_sq)
                / (chi_low_sq + chi_up_sq) - mean[di])

    if not fast_ok:
        warnings.warn(f"cmaes {cmaes.__version__}: FastCMAwM 은 0.13.1 내부를 가정한다 — "
                      "라이브러리 경로로 돈다", RuntimeWarning, stacklevel=3)
    return _FastCMAwM


_FAST_CLS = None


def FastCMAwM(*args, **kwargs):  # noqa: N802 — 클래스처럼 쓰는 공장
    """``cmaes.CMAwM`` 과 같은 인자. ``ask_batch(k)`` 를 더 가진 가속판."""
    global _FAST_CLS
    if _FAST_CLS is None:
        _FAST_CLS = _make_fast_cls()
    return _FAST_CLS(*args, **kwargs)


# ---------------------------------------------------------------------------
# 백엔드 어댑터 — ask_batch(k) -> (배치 벡터, tell 토큰), tell(토큰, 값[최대화]) -> 세대 최고,
# should_stop()
# ---------------------------------------------------------------------------

class _Backend:
    opt: Any
    population_size: int

    def tell(self, tokens, values) -> float:
        """라이브러리 ``tell`` 에 (토큰, −값) 목록을 넘기고 세대 최고값을 돌려준다.

        swing-it ``cma_bipop_x0`` 과 같은 식 — 라이브러리가 **제자리 정렬한** 목록 위 파이썬 ``min``
        (NaN 이 섞이면 위치 의존까지 같게)."""
        sols = [(t, -float(v)) for t, v in zip(tokens, values, strict=True)]
        self.opt.tell(sols)
        return -min(v for _, v in sols)

    def should_stop(self) -> bool:
        return bool(self.opt.should_stop())


class _CMAwMBackend(_Backend):
    def __init__(self, opt: Any):
        self.opt = opt
        self.population_size = opt.population_size

    def ask_batch(self, k: int):
        XE, XT = self.opt.ask_batch(k) if hasattr(self.opt, "ask_batch") else _ask_k(self.opt, k)
        return XE, list(XT)


class _CatBackend(_Backend):
    """CatCMAwM ↔ 배치 벡터. 범주형은 원핫을 로짓 자리에 넣는다(argmax = 고른 범주)."""

    def __init__(self, space: Space, mean: np.ndarray, sigma: float, lam: int, seed: int):
        cmaes = _cmaes()
        nr, nc = len(space.reals), space.n_cont
        self.space = space
        self.k_choice = [len(g.levels) for g in space.choices]
        x_space = [[0.0, 1.0]] * nr if nr else None
        z_space = [list(range(k)) for k in space.disc_sizes] or None
        c_space = [max(2, k) for k in self.k_choice] or None
        m = np.concatenate([np.asarray(mean[:nr], float), np.asarray(mean[nc:], float)]).copy()
        scale = np.array([1.0] * nr + [float(k - 1) for k in space.disc_sizes])
        cov = np.diag(scale ** 2) if len(m) else None
        self.opt = cmaes.CatCMAwM(x_space=x_space, z_space=z_space, c_space=c_space,
                                  population_size=lam, mean=m if len(m) else None, sigma=sigma,
                                  cov=cov, seed=seed)
        self.population_size = self.opt.population_size

    def ask_batch(self, k: int):
        sp = self.space
        X = np.empty((k, sp.dim))
        sols = []
        nr = len(sp.reals)
        for s in range(k):
            sol = self.opt.ask()
            sols.append(sol)
            if nr:
                X[s, :nr] = sol.x
            j = nr
            for i, kc in enumerate(self.k_choice):
                X[s, j:j + kc] = sol.c[i, :kc] if kc >= 2 else 1.0
                j += kc
            if sp.disc_sizes:
                X[s, sp.n_cont:] = sol.z
        return X, sols


def make_backend(backend: str, *, space: Space | None, mean: np.ndarray, sigma: float, lam: int,
                 seed: int, cov0: np.ndarray, bounds: np.ndarray, steps: np.ndarray) -> _Backend:
    if backend in ("cmawm", "cmawm_lib"):
        cls = FastCMAwM if backend == "cmawm" else _cmaes().CMAwM
        return _CMAwMBackend(cls(mean=mean, sigma=sigma, bounds=bounds, steps=steps, seed=seed,
                                 population_size=lam, cov=cov0.copy()))
    if backend == "catcmawm":
        if space is None:
            raise ValueError("catcmawm 는 Space 가 필요하다(범주형 묶음을 알아야 한다)")
        return _CatBackend(space, mean, sigma, lam, seed)
    raise ValueError(f"backend 는 {BACKENDS} 중 하나")


# ---------------------------------------------------------------------------
# BIPOP 구동기
# ---------------------------------------------------------------------------

@dataclass
class BipopResult:
    """이력(평가 순서)·memo(처음 본 순서)·재시작 기록."""

    hist_x: np.ndarray
    hist_values: np.ndarray
    hist_keys: list
    memo: dict = field(repr=False)            # 키 → (값, 진입 수, payload)
    restarts: list[dict]

    @property
    def best_index(self) -> int:
        """이력 순서 첫 최대."""
        return int(np.argmax(self.hist_values))

    @property
    def best_key(self) -> Any:
        return self.hist_keys[self.best_index]

    @property
    def best_value(self) -> float:
        return float(self.hist_values[self.best_index])

    @property
    def best_payload(self) -> Any:
        return self.memo[self.best_key][2]

    @property
    def uniq(self) -> int:
        return len(self.memo)

    def curve(self, marks: Sequence[int] = (100, 1000, 3000, 10000)) -> dict[int, float]:
        c = np.maximum.accumulate(self.hist_values)
        ms = [m for m in (*marks, len(c)) if m <= len(c)]
        return {m: float(c[m - 1]) for m in ms}

    def top(self, k: int) -> list[tuple[float, Any]]:
        """memo 값 상위 k(안정 정렬 — 같은 값이면 처음 본 순서)."""
        vals = sorted(self.memo.values(), key=lambda v: -v[0])[:k]
        return [(v[0], v[2]) for v in vals]


def _default_keyfn(space: Space, canonical: Callable[[np.ndarray], np.ndarray] | None):
    def keyfn(X: np.ndarray):
        G = space.decode(X)
        if canonical is not None:
            G = canonical(G)
        return space.keys(G).tolist(), list(G)
    return keyfn


def bipop(evaluate: Callable, space: Space | None = None, *, budget: int, seed: int,
          x0: np.ndarray | None = None, x_restart: np.ndarray | None = None, sigma0: float = 0.3,
          stall: bool = True, backend: str = "cmawm",
          canonical: Callable[[np.ndarray], np.ndarray] | None = None,
          keyfn: Callable[[np.ndarray], tuple[list, list]] | None = None,
          n_cont: int | None = None, disc_sizes: Sequence[int] | None = None) -> BipopResult:
    """BIPOP-CMA(-wM) 한 런 — **최대화**.

    ``evaluate(payloads) -> values`` 또는 ``(values, counts)`` — memo 에 없는 표현형만, 처음 나온
    순서로 온다. ``counts`` 가 있으면 재시작별 '진입 0건 표본 비율'(``zero_frac``, memo 적중 포함)을
    센다.

    키·payload 는 ``keyfn(X) -> (keys, payloads)`` 가 정한다. 없으면 ``space.decode`` →
    ``canonical(G)`` → ``space.keys`` (payload = G 행). ``keyfn`` 을 주면 배치는 ``n_cont``·
    ``disc_sizes`` 로 알려 준다(``space`` 가 있으면 거기서).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend 는 {BACKENDS} 중 하나")
    if space is not None:
        n_cont, disc_sizes = space.n_cont, space.disc_sizes
    if n_cont is None or disc_sizes is None:
        raise ValueError("space 또는 (n_cont, disc_sizes) 가 필요하다")
    if keyfn is None:
        if space is None:
            raise ValueError("keyfn 이 없으면 space 가 필요하다")
        keyfn = _default_keyfn(space, canonical)
    bounds, steps = layout(n_cont, disc_sizes)
    dim = len(bounds)
    scale = bounds[:, 1] - bounds[:, 0]
    cov0 = np.diag(scale ** 2)
    if backend == "catcmawm" and space is None:
        raise ValueError("catcmawm 는 Space 가 필요하다(범주형 묶음을 알아야 한다)")
    dim_eff = len(space.genes) if backend == "catcmawm" else dim
    rng = np.random.default_rng(seed)
    lam0 = 4 + int(3 * math.log(dim_eff))
    memo: dict = {}
    hx: list[np.ndarray] = []
    hv: list[float] = []
    hk: list = []
    restarts: list[dict] = []
    n_eval, large_used, small_used, large_pop = 0, 0, 0, lam0

    def run_batch(XE: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys, payloads = keyfn(XE)
        first_at: dict = {}                              # 배치 안 새 키 → 처음 나온 자리(순서 유지)
        for i, kk in enumerate(keys):
            if kk not in memo:
                first_at.setdefault(kk, i)
        new = list(first_at.values())
        if new:
            out = evaluate([payloads[i] for i in new])
            vals, cnts = out if isinstance(out, tuple) else (out, None)
            vals = np.asarray(vals, float)
            cnts = np.full(len(new), -1) if cnts is None else np.asarray(cnts)
            if len(vals) != len(new):
                raise ValueError(f"evaluate 가 {len(new)}개 대신 {len(vals)}개를 돌려줬다")
            for i, v, c in zip(new, vals.tolist(), cnts.tolist(), strict=True):
                memo[keys[i]] = (float(v), c, payloads[i])
        v = np.array([memo[kk][0] for kk in keys])
        c = np.array([memo[kk][1] for kk in keys])
        hk.extend(keys)
        return v, c

    while n_eval < budget:
        mean = bounds[:, 0] + rng.random(dim) * scale
        if x_restart is not None and len(x_restart):
            mean = np.clip(np.asarray(x_restart[int(rng.integers(len(x_restart)))], float),
                           bounds[:, 0], bounds[:, 1])
        if not restarts:
            regime, lam, sig = "first", lam0, sigma0
            if x0 is not None:
                mean = np.clip(np.asarray(x0, float), bounds[:, 0], bounds[:, 1])
        elif small_used < large_used:
            u = rng.random()
            regime = "small"
            lam = max(2, int(lam0 * (0.5 * large_pop / lam0) ** (u ** 2)))
            sig = sigma0 * 10 ** (-2 * rng.random())
        else:
            large_pop *= 2
            regime, lam, sig = "large", large_pop, sigma0
        opt = make_backend(backend, space=space, mean=mean, sigma=sig, lam=lam,
                           seed=int(rng.integers(2 ** 31 - 1)), cov0=cov0, bounds=bounds,
                           steps=steps)
        used, gens, run_best, since, zero = 0, 0, -np.inf, 0, 0
        stall_n = 10 + math.ceil(30 * dim_eff / lam)
        why = "budget"
        while n_eval < budget:
            if opt.should_stop():
                why = "should_stop"
                break
            if stall and since >= stall_n:
                why = "stall"
                break
            k = min(opt.population_size, budget - n_eval)
            XE, tokens = opt.ask_batch(k)
            vals, cnts = run_batch(XE)
            hx.extend(XE)
            hv.extend(vals.tolist())
            n_eval += k
            used += k
            zero += int(np.sum(cnts == 0))
            if k < opt.population_size:
                break
            g_best = opt.tell(tokens, vals)
            gens += 1
            if g_best > run_best + 1e-12:
                run_best, since = g_best, 0
            else:
                since += 1
        restarts.append({"regime": regime, "popsize": lam, "sigma": sig, "evals": used,
                         "gens": gens, "why": why, "zero_frac": zero / used if used else np.nan})
        if regime == "small":
            small_used += used
        else:
            large_used += used
    return BipopResult(np.array(hx).reshape(-1, dim), np.array(hv), hk, memo, restarts)
