"""runtime.runs — start_run 게이트·기록 모양·실패 기록·DB 색인 실패 허용."""

from __future__ import annotations

import json
import subprocess

import pytest

from krx_quant_core.runtime import oos, runindex, runs
from krx_quant_core.runtime.gitstate import label_dir


def _host(monkeypatch, name):
    monkeypatch.setattr("krx_quant_core.runtime.host.socket.gethostname", lambda: name)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("KQC_RUNS_ROOT", raising=False)
    monkeypatch.delenv("KR_QUANT_DB", raising=False)
    _host(monkeypatch, "simnode")
    g = ["git", "-C", str(tmp_path), "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*g, "init", "-q"], check=True)
    subprocess.run([*g, "config", "user.email", "t@example.com"], check=True)
    subprocess.run([*g, "config", "user.name", "t"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run([*g, "add", "a.py"], check=True)
    subprocess.run([*g, "commit", "-qm", "init"], check=True)
    return tmp_path


def _rows(repo, label):
    p = label_dir(repo, label) / "RUNS.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()]


CFG = {"thr": 0.7, "hold": 300}


def test_refused_off_simnode_writes_nothing(repo, monkeypatch):
    _host(monkeypatch, "trader")
    with pytest.raises(runs.RunRefused, match="simnode"):
        with runs.start_run("a", CFG, repo_root=repo):
            pass
    assert not label_dir(repo, "a").exists()


def test_dirty_tree_refused_unless_allowed(repo):
    (repo / "a.py").write_text("x = 2\n")
    with pytest.raises(runs.RunRefused, match="uncommitted"):
        with runs.start_run("a", CFG, repo_root=repo):
            pass
    with runs.start_run("a", CFG, repo_root=repo, allow_dirty=True):
        pass
    assert _rows(repo, "a")[0]["dirty"] is True


def test_untracked_files_do_not_make_dirty(repo):
    (repo / "scratch.txt").write_text("tmp")
    with runs.start_run("a", CFG, repo_root=repo):
        pass


def test_committed_run_records_do_not_block_next_run(repo):
    with runs.start_run("a", CFG, repo_root=repo):
        pass
    g = ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*g, "add", "research"], check=True)
    subprocess.run([*g, "commit", "-qm", "runs"], check=True)
    with runs.start_run("a", CFG, repo_root=repo):  # 기록 파일이 append 돼도 다음 실행은 된다
        pass
    with runs.start_run("a", CFG, repo_root=repo):
        pass
    assert len(_rows(repo, "a")) == 6


def test_start_and_end_rows(repo):
    with runs.start_run(
        "a", CFG, repo_root=repo, data=runs.DataSpec("2026-08-24", "2026-09-07", "train"), seed=7
    ) as run:
        assert run.n_trials == 1
        run.log_result({"mean_bp": -3.1, "n": 812, "obj": object()})
    start, end = _rows(repo, "a")
    assert start["event"] == "start" and end["event"] == "end"
    assert start["run_id"] == end["run_id"] == run.run_id and len(run.run_id) == 12
    assert start["host"] == "simnode" and len(start["git_sha"]) == 40 and start["dirty"] is False
    assert start["data"] == {"start": "2026-08-24", "end": "2026-09-07", "split": "train"}
    assert start["seed"] == 7 and start["config"] == CFG and len(start["fingerprint"]) == 16
    assert "krx_quant_core" in start["versions"] and "python" in start["versions"]
    assert end["status"] == "ok" and end["result"]["mean_bp"] == -3.1
    assert isinstance(end["result"]["obj"], str) and end["duration_s"] >= 0


def test_trials_ledger_counts_distinct_configs(repo):
    for cfg in (CFG, CFG, {**CFG, "thr": 0.8}):
        with runs.start_run("a", cfg, repo_root=repo) as run:
            pass
    assert run.n_trials == 2
    assert (label_dir(repo, "a") / "TRIALS.jsonl").exists()


def test_exception_recorded_and_reraised(repo):
    with pytest.raises(ZeroDivisionError):
        with runs.start_run("a", CFG, repo_root=repo):
            1 / 0  # noqa: B018
    end = _rows(repo, "a")[1]
    assert end["status"] == "failed" and "ZeroDivisionError" in end["error"]


def test_final_goes_through_oos_check(repo):
    oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    oos_range = runs.DataSpec("2026-09-08", "2026-09-15", "oos")
    with pytest.raises(oos.OOSLocked):
        with runs.start_run("a", CFG, repo_root=repo, data=oos_range):
            pass
    doc = repo / "prereg.md"
    doc.write_text("rules")
    oos.lock_prereg("a", doc, repo_root=repo)
    with runs.start_run("a", CFG, repo_root=repo, data=oos_range, final=True) as run:
        pass
    assert json.loads((label_dir(repo, "a") / "FINAL.lock").read_text())["run_id"] == run.run_id
    with pytest.raises(oos.RunRefused, match="FINAL"):
        with runs.start_run("a", CFG, repo_root=repo, data=oos_range, final=True):
            pass


def test_index_failure_does_not_stop_run(repo, monkeypatch):
    calls = []
    monkeypatch.setattr(runindex, "upsert", lambda row: calls.append(row["event"]) or False)
    with runs.start_run("a", CFG, repo_root=repo) as run:
        run.log_result({"x": 1})
    assert calls == ["start", "end"]
    assert len(_rows(repo, "a")) == 2


def test_upsert_without_dsn_is_false(monkeypatch):
    monkeypatch.delenv("KR_QUANT_DB", raising=False)
    assert runindex.upsert({"run_id": "x", "event": "start"}) is False


def test_read_runs_merges(repo):
    with runs.start_run("a", CFG, repo_root=repo) as r1:
        r1.log_result({"k": 1})
    with pytest.raises(ValueError):
        with runs.start_run("a", CFG, repo_root=repo):
            raise ValueError("boom")
    merged = runs.read_runs("a", repo_root=repo)
    assert [m["status"] for m in merged] == ["ok", "failed"]
    assert merged[0]["result"] == {"k": 1} and merged[0]["git_sha"]


def test_trials_dir_keeps_existing_ledger(repo):
    from krx_quant_core.stats.trials import record_trial

    record_trial("a", {"old": 1}, logs_dir=repo / "research" / "logs")
    with runs.start_run("a", CFG, repo_root=repo, trials_dir="research/logs") as run:
        pass
    assert run.n_trials == 2
    assert not (label_dir(repo, "a") / "TRIALS.jsonl").exists()
    assert _rows(repo, "a")[0]["n_trials"] == 2
