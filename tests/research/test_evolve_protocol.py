"""research.evolve.protocol·pool — 원장·잡음 규칙·짝 레이싱·봉인·digest·풀 순서."""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.research.evolve.pool import THREAD_ENV, EvoPool
from krx_quant_core.research.evolve.protocol import (
    EvoLedger,
    Seal,
    SealBreach,
    digest,
    distinguish,
    paired_diff,
    provenance,
    race,
    summarize,
)


def test_ledger_pre_result_pending_and_n(tmp_path):
    L = EvoLedger("pattern_ga_evo", logs_dir=tmp_path)
    L.register("G1A", menu={}, seeds=range(10), rationale="기준선", git_head="abc")
    L.register("G1B", menu={"nom": {"entry": ["bottom"]}}, seeds=range(10))
    # 같은 줄 — 중복 안 셈
    L.register("G1B", menu={"nom": {"entry": ["bottom"]}}, seeds=range(10))
    L.result("G1A", seeds=range(10), metrics={"ann": -0.0378, "n": 106})
    assert L.n_trials() == 2
    assert [p["candidate"] for p in L.pending()] == ["G1B"]
    L.result("G1B", seeds=range(10), metrics={"ann": -0.0129})
    assert L.pending() == [] and L.n_trials() == 2            # 결과 줄은 N 에 안 센다


def test_summarize_and_distinguish_reproduce_gen2_table():
    """META_EVOLUTION 2세대 표: G1C −1.48 ± 1.03, 어느 후보도 구분 안 됨."""
    g1c = [-0.0220, -0.0194, -0.0030]
    s = summarize(g1c)
    assert round(s["mean"], 4) == -0.0148 and round(s["sd"], 4) == 0.0103
    g2a = [0.0019, -0.0396, -0.0113]
    g2b = [-0.0236, -0.0211, -0.0282]
    g2c = [-0.0263, -0.0057, -0.0105]
    for a in (g1c, g2a, g2b, g2c):
        for b in (g1c, g2a, g2b, g2c):
            if a is not b:
                assert distinguish(a, b) is None
    assert distinguish([1.0, 1.1, 0.9], [0.0, 0.1, -0.1]) == "a"
    assert distinguish([0.0, 0.1, -0.1], [1.0, 1.1, 0.9]) == "b"


def test_paired_diff_common_keys():
    d = paired_diff({0: 1.0, 100: 2.0, 200: 4.0}, {0: 0.5, 100: 1.0, 300: 9.0})
    assert d["n"] == 2 and d["mean"] == 0.75
    assert math.isclose(d["se"], np.std([0.5, 1.0], ddof=1) / math.sqrt(2))


def test_race_eliminates_clearly_worse_and_keeps_noise_twins():
    rng = np.random.default_rng(0)
    # 시드 집합 공통 잡음(짝이 지운다)
    seed_fx = {o: rng.normal(0, 1.0) for o in range(0, 2000, 100)}
    true = {"A": 0.0, "B": -0.3, "C": 0.0, "D": -3.0}

    def run(c, o):
        return true[c] + seed_fx[o] + np.random.default_rng([o, ord(c)]).normal(0, 0.1)

    r = race(list(true), list(seed_fx), run=run, min_sets=3, k=2.0)
    assert "D" in r.eliminated and r.eliminated["D"] == 3
    assert "B" in r.eliminated
    assert set(r.alive) == {"A", "C"}
    n = {c: len(v) for c, v in r.scores.items()}
    assert n["A"] == n["C"] == 20 and n["D"] == 3


def test_race_run_round_batches_and_validates():
    calls = []

    def run_round(pairs):
        calls.append(pairs)
        return [1.0 if c == "A" else 0.0 for c, _ in pairs]

    r = race(["A", "B"], [0, 1, 2, 3], run_round=run_round, min_sets=3)
    assert r.alive == ["A"] and len(calls) == 3                # 짝 차 −1, SE 0 → 3번째에 탈락
    with pytest.raises(ValueError):
        race(["A"], [0], run=lambda c, o: 0.0, min_sets=1)
    with pytest.raises(ValueError):
        race(["A", "A"], [0, 1, 2], run=lambda c, o: 0.0)


def test_seal_clip_and_check():
    days = pd.bdate_range("2023-12-20", "2024-01-10")
    s = Seal(days, "2024-01-01")
    assert days[s.row] == pd.Timestamp("2024-01-01") and s.end_row == s.row - 1
    assert list(s.clip(np.array([0, s.row + 3, len(days)]))) == [0, s.row, s.row]
    s.check([0, s.row - 1])
    with pytest.raises(SealBreach):
        s.check([s.row])


def test_digest_stable_and_typed():
    a = {"b": np.array([1.0, np.inf]), "a": np.int64(3), "s": {2, 1}}
    assert digest(a) == digest({"s": [1, 2], "a": 3, "b": [1.0, math.inf]})
    with pytest.raises(TypeError):
        digest({"x": object()})


def test_provenance_fields():
    import krx_quant_core.research.evolve.space as sp_mod
    p = provenance(None, modules=[sp_mod])
    assert p["numpy"] and len(p["code_hash"]) == 16


def test_pool_order_and_single_thread_env():
    before = {k: os.environ.get(k) for k in THREAD_ENV}
    with EvoPool(3) as pool:
        out = pool.map(math.factorial, [9, 1, 7, 3, 12, 0])
        env = pool.map(os.getenv, list(THREAD_ENV))
    assert out == [math.factorial(x) for x in [9, 1, 7, 3, 12, 0]]
    assert env == ["1"] * len(THREAD_ENV)
    assert {k: os.environ.get(k) for k in THREAD_ENV} == before
    with pytest.raises(RuntimeError):
        EvoPool(2).map(abs, [1])
