"""stats.selection — scalp-it 98번 selection_bias.py 와 같은 숫자를 내는지(골든) + 성질."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from krx_quant_core.stats import selection as S

# ---- 원본(scalp-it scripts/selection_98/selection_bias.py) 복사본 — 대조 기준 ---------------


def _orig_expected_max(values, m):
    v = np.sort(np.asarray(values, np.float64))
    n = len(v)
    if m >= n:
        return float(v[-1])
    tot = math.comb(n, m)
    w = np.array([math.comb(i, m - 1) for i in range(n)], np.float64) / tot
    return float((v * w).sum())


def _orig_eligible(rows, min_n=30):
    return [r for r in rows
            if r.get("val_n", 0) >= min_n and r.get("val_daymean_bp") == r.get("val_daymean_bp")]


def _orig_argmax(rows, key):
    best = None
    for r in rows:
        v = r.get(key)
        if v is None or v != v:
            continue
        if best is None or v > best[key]:
            best = r
    return best


def _rows(seed: int, n: int = 40) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        val_n = int(rng.integers(5, 400))
        pooled = float(rng.normal(-30, 10))
        day = pooled + float(rng.normal(0, 60 / math.sqrt(max(val_n, 1))))
        if i % 13 == 0:
            day = math.nan
        rows.append({"seed": i % 3, "steps": 1000 * i, "val_n": val_n,
                     "val_daymean_bp": day, "val_mean_bp": pooled})
    rows[5]["val_daymean_bp"] = rows[7]["val_daymean_bp"] = 999.0  # 동점 — 먼저 만난 5 가 이겨야
    rows[5]["val_n"] = rows[7]["val_n"] = 100
    return rows


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_expected_max_matches_original_and_brute_force(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=9)
    for m in range(1, 12):
        assert S.expected_max_of_m(v, m) == _orig_expected_max(v, m)
    for m in range(1, 10):  # 조합 전수 평균과 같은가
        brute = float(np.mean([max(c) for c in itertools.combinations(v, m)]))
        assert S.expected_max_of_m(v, m) == pytest.approx(brute, rel=1e-12)
    assert S.expected_max_of_m(v, 1) == pytest.approx(v.mean())
    assert S.expected_max_of_m(v, 99) == v.max()
    assert math.isnan(S.expected_max_of_m([], 1))


def test_eligible_and_argmax_match_original_including_ties():
    rows = _rows(7)
    el = S.eligible_rows(rows, key="val_daymean_bp", n_key="val_n", min_n=30)
    assert el == _orig_eligible(rows)
    sel = S.argmax_first(el, "val_daymean_bp")
    assert sel is _orig_argmax(el, "val_daymean_bp")
    assert sel["steps"] == 5000  # 동점 999.0 중 먼저 만난 행
    assert S.argmax_first([{"x": math.nan}, {"x": None}], "x") is None


@pytest.mark.parametrize("seed", [11, 12])
def test_report_reproduces_original_numbers(seed):
    rows = _rows(seed, n=60)
    rep = S.selection_bias_report(rows)
    el = _orig_eligible(rows)
    day = np.array([r["val_daymean_bp"] for r in el])
    pooled = np.array([r["val_mean_bp"] for r in el])
    sel = _orig_argmax(el, "val_daymean_bp")
    assert rep.n_eligible == len(el)
    assert rep.selected == sel
    assert rep.premium_vs_random == sel["val_daymean_bp"] - day.mean()
    assert rep.premium_vs_median == sel["val_daymean_bp"] - np.median(day)
    assert rep.pool["score_sd"] == day.std(ddof=1)
    assert rep.pool["gap_median"] == np.median(day - pooled)
    for m, v in rep.expected_max_curve.items():
        assert v == _orig_expected_max(day, m)
    assert set(rep.expected_max_curve) == {m for m in (1, 2, 3, 5, 10, 20, len(el)) if m <= len(el)}
    assert rep.selected_gap == sel["val_daymean_bp"] - sel["val_mean_bp"]
    assert sum(b["k"] for b in rep.by_n_quartile) == len(el)
    assert -1 <= rep.rho_n_vs_score <= 1
    d = json.loads(json.dumps(rep.to_dict()))  # JSON 직렬화 가능
    assert d["expected_max_curve"]["1"] == pytest.approx(day.mean())


def test_report_too_few_candidates_has_note():
    rep = S.selection_bias_report([{"val_n": 50, "val_daymean_bp": 1.0, "val_mean_bp": 0.5}])
    assert rep.n_eligible == 1 and rep.note
    assert rep.pool == {} and math.isnan(rep.premium_vs_random)


def test_pure_noise_premium_equals_curve_endpoint():
    """후보가 전부 같은 분포의 잡음이면 '선택된 값' = 풀 최댓값 = 곡선 끝. 프리미엄은 양수."""
    rng = np.random.default_rng(0)
    rows = [{"val_n": 100, "val_daymean_bp": float(x), "val_mean_bp": float(x)}
            for x in rng.normal(0, 10, 30)]
    rep = S.selection_bias_report(rows)
    assert rep.expected_max_curve[30] == rep.selected["val_daymean_bp"]
    assert rep.premium_vs_random > 0
    assert rep.selected_gap == 0.0
