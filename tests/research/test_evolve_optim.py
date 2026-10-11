"""research.evolve.optim — FastCMAwM ↔ cmaes.CMAwM 비트 동일, bipop ↔ swing-it cma_bipop_x0 동일."""

from __future__ import annotations

import math

import numpy as np
import pytest

cmaes = pytest.importorskip("cmaes")

from krx_quant_core.research.evolve.optim import FastCMAwM, bipop  # noqa: E402
from krx_quant_core.research.evolve.space import Choice, Ordinal, Real, Space  # noqa: E402


def _space() -> Space:
    return Space([
        Real("a", 0.0, 90.0, off="lo"), Real("b", -1.5, 1.0), Real("c", 0.05, 1.5, off="hi"),
        Real("d", 0.0, 1.0), Real("e", 2.0, 5.0),
        Choice("entry", ("bottom", "horiz", "trend")), Choice("pyr", ("off", "1r")),
        Ordinal("theta", (0.03, 0.06, 0.08, 0.1)), Ordinal("be", (0.5, 1.0, math.inf)),
        Ordinal("flag", (0, 1)), Ordinal("dup", (2, 2)), Ordinal("win", (1, 2, 3, 4, 5)),
    ])


def _fit_rows(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """잡음 없는 혼합 시험 함수(행렬) — 끔(NaN)·코드 섞임, 진입 수 흉내(0 이 꽤 나오게)."""
    G = np.nan_to_num(np.atleast_2d(G), nan=-1.0)
    v = -((G[:, 0] - 40.0) / 30) ** 2 - (G[:, 1] - 0.2) ** 2 - (G[:, 3] - 0.7) ** 2
    v += 0.3 * (G[:, 5] == 2)
    v += 0.1 * G[:, 7] - 0.05 * (G[:, 11] - 3) ** 2 + 0.01 * G[:, 4]
    n = np.where(G[:, 0] > 60, 0, 5).astype(np.int64)
    return np.round(v, 1), n                     # 평평한 칸 → 정체 재시작이 일어난다


def _x0(sp: Space) -> np.ndarray:
    x = np.full(sp.dim, 0.5)
    x[0] = 0.0
    x[2] = 1.0
    x[sp.n_cont:] = 0
    return x


# --- swing-it pattern_ga_run.cma_bipop_x0 원본 사본 ---
def ref_cma_bipop_x0(fitness, n_cont, disc_sizes, *, budget, seed, x0=None, sigma0=0.3, stall=True,
                     opt_cls=None, x_restart=None):
    if opt_cls is None:
        from cmaes import CMAwM as opt_cls  # noqa: N813
    dim = n_cont + len(disc_sizes)
    bounds = np.array([[0.0, 1.0]] * n_cont + [[0.0, float(k - 1)] for k in disc_sizes])
    steps = np.array([0.0] * n_cont + [1.0] * len(disc_sizes))
    scale = bounds[:, 1] - bounds[:, 0]
    cov0 = np.diag(scale ** 2)
    rng = np.random.default_rng(seed)
    lam0 = 4 + int(3 * math.log(dim))
    hist, restarts = [], []
    n_eval, large_used, small_used, large_pop = 0, 0, 0, lam0
    first = True
    while n_eval < budget:
        mean = bounds[:, 0] + rng.random(dim) * scale
        if x_restart is not None and len(x_restart):
            mean = np.clip(np.asarray(x_restart[int(rng.integers(len(x_restart)))], float),
                           bounds[:, 0], bounds[:, 1])
        if first:
            regime, lam, sig = "first", lam0, sigma0
            if x0 is not None:
                mean = np.clip(np.asarray(x0, float), bounds[:, 0], bounds[:, 1])
            first = False
        elif small_used < large_used:
            u = rng.random()
            regime = "small"
            lam = max(2, int(lam0 * (0.5 * large_pop / lam0) ** (u ** 2)))
            sig = sigma0 * 10 ** (-2 * rng.random())
        else:
            large_pop *= 2
            regime, lam, sig = "large", large_pop, sigma0
        opt = opt_cls(mean=mean, sigma=sig, bounds=bounds, steps=steps,
                      seed=int(rng.integers(2 ** 31 - 1)), population_size=lam, cov=cov0.copy())
        used, gens, run_best, since, zero = 0, 0, -np.inf, 0, 0
        stall_n = 10 + math.ceil(30 * dim / lam)
        why = "budget"
        while n_eval < budget:
            if opt.should_stop():
                why = "should_stop"
                break
            if stall and since >= stall_n:
                why = "stall"
                break
            sols = []
            for _ in range(opt.population_size):
                if n_eval >= budget:
                    break
                x_eval, x_tell = opt.ask()
                fv = fitness(x_eval)
                if isinstance(fv, tuple):
                    v = float(fv[0])
                    zero += int(fv[1] == 0)
                else:
                    v = float(fv)
                hist.append((n_eval, x_eval.copy(), v))
                n_eval += 1
                used += 1
                sols.append((x_tell, -v))
            if len(sols) < opt.population_size:
                break
            opt.tell(sols)
            gens += 1
            g_best = -min(v for _, v in sols)
            if g_best > run_best + 1e-12:
                run_best, since = g_best, 0
            else:
                since += 1
        restarts.append({"regime": regime, "popsize": lam, "sigma": sig, "evals": used,
                         "gens": gens, "why": why,
                         "zero_frac": zero / used if used else np.nan})
        if regime == "small":
            small_used += used
        else:
            large_used += used
    return hist, restarts


def test_fast_cmawm_bit_identical_to_library():
    sp = _space()
    for seed in (0, 1, 7):
        cov = np.diag((sp.bounds[:, 1] - sp.bounds[:, 0]) ** 2)
        kw = dict(mean=np.clip(_x0(sp), sp.bounds[:, 0], sp.bounds[:, 1]), sigma=0.3,
                  bounds=sp.bounds, steps=sp.steps, seed=seed, population_size=12, cov=cov)
        lib = cmaes.CMAwM(**{**kw, "cov": kw["cov"].copy()})
        fast = FastCMAwM(**{**kw, "cov": kw["cov"].copy()})
        assert fast.FAST
        for _ in range(150):
            pairs = [lib.ask() for _ in range(lib.population_size)]
            XE, XT = fast.ask_batch(fast.population_size)
            assert np.array_equal(np.array([p[0] for p in pairs]), XE)
            assert np.array_equal(np.array([p[1] for p in pairs]), XT)
            v, _ = _fit_rows(sp.decode(XE))
            lib.tell([(p[1], -float(x)) for p, x in zip(pairs, v, strict=True)])
            fast.tell([(t, -float(x)) for t, x in zip(XT, v, strict=True)])
            assert np.array_equal(lib._cma._mean, fast._cma._mean)
            assert np.array_equal(lib._A, fast._A)
            assert lib._cma._sigma == fast._cma._sigma
            if lib.should_stop():
                assert fast.should_stop()
                break


def test_patch_restores_library_globals():
    from cmaes import _cmawm, _stats
    sp = _space()
    opt = FastCMAwM(mean=np.full(sp.dim, 0.5).clip(sp.bounds[:, 0], sp.bounds[:, 1]), sigma=0.3,
                    bounds=sp.bounds, steps=sp.steps, seed=0)
    XE, XT = opt.ask_batch(opt.population_size)
    opt.tell([(t, 0.0) for t in XT])
    assert _cmawm.chi2_ppf is _stats.chi2_ppf


@pytest.mark.parametrize("backend", ["cmawm", "cmawm_lib"])
@pytest.mark.parametrize("seed", [0, 3])
def test_bipop_matches_swing_reference(backend, seed):
    sp = _space()
    budget = 3000
    pool = sp.random(np.random.default_rng(99), 40)

    memo: dict = {}

    def fn(x):
        G = sp.decode(x[None, :])
        k = int(sp.keys(G)[0])
        if k not in memo:
            v, n = _fit_rows(G)
            memo[k] = (float(v[0]), int(n[0]), G[0])
        return memo[k][0], memo[k][1]

    hist, rst = ref_cma_bipop_x0(fn, sp.n_cont, sp.disc_sizes, budget=budget, seed=seed, x0=_x0(sp),
                                 x_restart=pool)
    calls = []

    def ev(payloads):
        calls.append(len(payloads))
        return _fit_rows(np.array(payloads))

    r = bipop(ev, sp, budget=budget, seed=seed, x0=_x0(sp), x_restart=pool, backend=backend)
    assert np.array_equal(r.hist_x, np.array([h[1] for h in hist]))
    assert np.array_equal(r.hist_values, np.array([h[2] for h in hist]))
    assert list(r.memo) == list(memo)                                     # memo 처음 본 순서
    assert r.uniq == len(memo) == sum(calls)
    assert len(rst) == len(r.restarts) and len(rst) >= 2
    for a, b in zip(rst, r.restarts, strict=True):
        assert {k: a[k] for k in a if k != "zero_frac"} == {k: b[k] for k in b if k != "zero_frac"}
        za, zb = a["zero_frac"], b["zero_frac"]
        assert za == zb or (np.isnan(za) and np.isnan(zb))
    bi = int(np.argmax([h[2] for h in hist]))
    assert r.best_index == bi
    assert np.array_equal(r.best_payload, memo[r.best_key][2])
    top_ref = sorted(memo.values(), key=lambda v: -v[0])[:5]
    assert [t[0] for t in r.top(5)] == [t[0] for t in top_ref]


def test_bipop_budget_exact_and_custom_keyfn():
    sp = _space()

    def keyfn(X):
        G = sp.decode(X)
        return [tuple(np.round(g, 6).tolist()) for g in G], [sp.to_dict(g) for g in G]

    seen = []

    def ev(payloads):
        seen.extend(payloads)
        return [float(p["d"] or 0.0) for p in payloads]

    r = bipop(ev, budget=777, seed=1, keyfn=keyfn, n_cont=sp.n_cont, disc_sizes=sp.disc_sizes)
    assert len(r.hist_values) == 777
    assert r.uniq == len(seen)
    assert all(np.isnan(x["zero_frac"]) or x["zero_frac"] == 0 for x in r.restarts)


def test_catcmawm_smoke_improves():
    sp = _space()

    def ev(payloads):
        return _fit_rows(np.array(payloads))

    r = bipop(ev, sp, budget=1500, seed=0, x0=_x0(sp), backend="catcmawm")
    assert len(r.hist_values) == 1500
    c = r.curve((100, 1500))
    assert c[1500] >= c[100]
    G = np.array([r.memo[k][2] for k in r.memo])
    assert set(np.unique(G[:, sp.col["entry"]])) <= {0.0, 1.0, 2.0}
    assert set(np.unique(G[:, sp.col["dup"]])) == {0.0}


def test_catcmawm_single_level_choice():
    sp = Space([Real("a", 0, 1), Choice("one", ("x",)), Ordinal("o", (1, 2, 3))])
    r = bipop(lambda P: [float(p[0]) for p in P], sp, budget=200, seed=0, backend="catcmawm")
    assert len(r.hist_values) == 200
