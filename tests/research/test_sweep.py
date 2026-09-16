"""research.sweep · research.optuna — grid·캐시·병렬·시행 원장·optuna 통합."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from krx_quant_core.research.sweep import grid, run_sweep
from krx_quant_core.runtime import DataSpec
from krx_quant_core.runtime.gitstate import label_dir
from krx_quant_core.stats.trials import count_trials

DATA = DataSpec("2026-08-24", "2026-09-07", "train")


def _host(monkeypatch, name):
    monkeypatch.setattr("krx_quant_core.runtime.host.socket.gethostname", lambda: name)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("KQC_RUNS_ROOT", raising=False)
    monkeypatch.delenv("KR_QUANT_DB", raising=False)
    monkeypatch.setenv("KQC_CACHE", str(tmp_path / "cache"))
    _host(monkeypatch, "simnode")
    g = ["git", "-C", str(tmp_path), "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*g, "init", "-q"], check=True)
    subprocess.run([*g, "config", "user.email", "t@example.com"], check=True)
    subprocess.run([*g, "config", "user.name", "t"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run([*g, "add", "a.py"], check=True)
    subprocess.run([*g, "commit", "-qm", "init"], check=True)
    return tmp_path


# --- 모듈 최상위 objective — ProcessPoolExecutor 로 자식 프로세스에 피클된다. ---


def objective_ok(cfg: dict) -> dict:
    log = os.environ.get("KQC_CALL_LOG")
    if log:
        p = Path(log)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write("1\n")
    return {"score": cfg["thr"] * 10}


def objective_with_error(cfg: dict) -> dict:
    if cfg["thr"] == 0.9:
        raise ValueError("boom")
    return {"score": cfg["thr"]}


def objective_returns(cfg: dict) -> dict:
    rng = np.random.default_rng(0)
    R = rng.normal(cfg["thr"], 1.0, size=50)
    return {"score": float(R.mean()), "returns": R.tolist()}


def test_grid_cartesian_product_and_key_order():
    configs = grid(thr=[0.6, 0.7], hold=[300, 600])
    assert configs == [
        {"thr": 0.6, "hold": 300},
        {"thr": 0.6, "hold": 600},
        {"thr": 0.7, "hold": 300},
        {"thr": 0.7, "hold": 600},
    ]


def test_run_sweep_n_jobs_1_and_2_produce_same_frame(repo):
    configs = grid(thr=[0.1, 0.2, 0.3])
    r1 = run_sweep(objective_ok, configs, label="s1", repo_root=repo, data=DATA, n_jobs=1)
    r2 = run_sweep(objective_ok, configs, label="s2", repo_root=repo, data=DATA, n_jobs=2)
    f1 = r1.frame.sort_values("thr").reset_index(drop=True)
    f2 = r2.frame.sort_values("thr").reset_index(drop=True)
    assert f1[["thr", "score"]].equals(f2[["thr", "score"]])


def test_second_run_hits_cache(repo, monkeypatch, tmp_path):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("KQC_CALL_LOG", str(log))
    configs = grid(thr=[0.1, 0.2, 0.3])
    run_sweep(objective_ok, configs, label="cache1", repo_root=repo, data=DATA, n_jobs=1)
    assert log.read_text().count("1") == 3
    run_sweep(objective_ok, configs, label="cache1", repo_root=repo, data=DATA, n_jobs=1)
    assert log.read_text().count("1") == 3  # 두 번째는 캐시 적중 — objective 다시 안 불림


def test_trials_ledger_counts_distinct_configs(repo):
    # start_run 자신도 스윕 요약 config({"sweep_n":..., "configs_fp":..., "seed":...})를
    # 같은 원장에 한 행 적는다 — 그래서 distinct 개별 config 2개 + 요약 1개 = 3.
    configs = [{"thr": 0.1}, {"thr": 0.1}, {"thr": 0.2}]
    result = run_sweep(objective_ok, configs, label="dedup", repo_root=repo, data=DATA, n_jobs=1)
    assert result.n_trials == 3
    assert count_trials("dedup", logs_dir=label_dir(repo, "dedup").parent) == 3


def test_objective_exception_isolated_as_error_row(repo):
    configs = grid(thr=[0.5, 0.9])
    result = run_sweep(
        objective_with_error, configs, label="err", repo_root=repo, data=DATA, n_jobs=1
    )
    frame = result.frame
    ok_row = frame[frame["thr"] == 0.5].iloc[0]
    bad_row = frame[frame["thr"] == 0.9].iloc[0]
    assert ok_row["score"] == 0.5
    assert "boom" in bad_row["error"]


def test_best_with_returns_key_includes_dsr(repo):
    configs = grid(thr=[0.0, 0.5, 1.0])
    result = run_sweep(
        objective_returns, configs, label="dsr", repo_root=repo, data=DATA, n_jobs=1
    )
    best = result.best("score", returns_key="returns")
    assert best["config"]["thr"] == 1.0
    assert best["score"] == pytest.approx(best["config"]["thr"], abs=0.5)
    assert best["dsr"] is not None
    assert best["dsr"]["n_trials"] == 4  # 개별 config 3개 + start_run 자신의 요약 config 1개
    assert "deflated_sharpe" in best["dsr"]


def test_best_without_returns_key_has_no_dsr(repo):
    configs = grid(thr=[0.1, 0.2])
    result = run_sweep(objective_ok, configs, label="nodsr", repo_root=repo, data=DATA, n_jobs=1)
    best = result.best("score")
    assert best["dsr"] is None
    assert best["config"]["thr"] == 0.2


def test_optuna_search_skips_without_optuna(repo):
    pytest.importorskip("optuna")
    from krx_quant_core.research.optuna import optuna_search

    def space(trial):
        return {"thr": trial.suggest_float("thr", 0.1, 0.9)}

    def objective(cfg):
        return {"score": cfg["thr"]}

    result = optuna_search(
        objective,
        space,
        label="opt1",
        repo_root=repo,
        data=DATA,
        n_trials=5,
        seed=0,
    )
    # start_run 자신의 요약 config 1행 + optuna trial 5개(서로 다른 연속값이라 중복 제거 없음).
    assert result.n_trials == 6
    assert len(result.frame) == 5
    best = result.best("score")
    assert 0.1 <= best["config"]["thr"] <= 0.9
