"""runtime.nightly — research/nightly.toml job 격리 실행."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from krx_quant_core.runtime import RunRefused
from krx_quant_core.runtime import nightly as ni

MONDAY = date(2026, 9, 14)
SATURDAY = date(2026, 9, 19)

TOML = """
[[job]]
name = "ok-job"
cmd = ["true"]

[[job]]
name = "fail-job"
cmd = ["false"]

[[job]]
name = "slow-job"
cmd = ["sleep", "5"]
timeout_min = 0.01
"""


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "research").mkdir(parents=True)
    (repo / "research" / "nightly.toml").write_text(TOML)
    return repo


def _sim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("krx_quant_core.runtime.host.socket.gethostname", lambda: "simnode")


def test_load_jobs_parses_toml(tmp_path):
    repo = _repo(tmp_path)
    jobs = ni.load_jobs(repo)
    assert [j.name for j in jobs] == ["ok-job", "fail-job", "slow-job"]
    assert jobs[0].cmd == ["true"]
    assert jobs[0].timeout_min == 60
    assert jobs[0].weekdays_only is True
    assert jobs[2].timeout_min == 0.01


def test_load_jobs_missing_file_is_empty(tmp_path):
    assert ni.load_jobs(tmp_path) == []


def test_run_nightly_isolates_success_failure_timeout(tmp_path, monkeypatch):
    _sim(monkeypatch)
    repo = _repo(tmp_path)
    out = tmp_path / "out"
    results = ni.run_nightly(repo, out_root=out, today=MONDAY)
    by_name = {r.name: r for r in results}

    assert by_name["ok-job"].rc == 0
    assert by_name["ok-job"].timed_out is False
    assert by_name["ok-job"].skipped is None

    assert by_name["fail-job"].rc == 1
    assert by_name["fail-job"].timed_out is False

    assert by_name["slow-job"].timed_out is True
    assert by_name["slow-job"].secs < 4  # 0.01min=0.6s 타임아웃, sleep 5 까지 안 기다림

    summary = json.loads((out / "2026-09-14" / "repo.json").read_text())
    assert {row["name"] for row in summary} == {"ok-job", "fail-job", "slow-job"}
    for log_name in ("repo-ok-job.log", "repo-fail-job.log", "repo-slow-job.log"):
        assert (out / "2026-09-14" / log_name).exists()


def test_run_nightly_weekend_skips_weekdays_only(tmp_path, monkeypatch):
    _sim(monkeypatch)
    repo = _repo(tmp_path)
    out = tmp_path / "out"
    results = ni.run_nightly(repo, out_root=out, today=SATURDAY)
    assert all(r.skipped == "weekend" for r in results)
    assert all(r.rc is None and r.timed_out is False for r in results)


def test_run_nightly_only_filters_to_single_job(tmp_path, monkeypatch):
    _sim(monkeypatch)
    repo = _repo(tmp_path)
    out = tmp_path / "out"
    results = ni.run_nightly(repo, out_root=out, today=MONDAY, only="ok-job")
    assert [r.name for r in results] == ["ok-job"]
    assert results[0].rc == 0


def test_run_nightly_dry_run_does_not_execute(tmp_path, monkeypatch):
    _sim(monkeypatch)
    repo = _repo(tmp_path)
    out = tmp_path / "out"
    results = ni.run_nightly(repo, out_root=out, today=MONDAY, dry_run=True)
    assert all(r.skipped == "dry-run" for r in results)
    assert all(r.rc is None for r in results)
    assert not (out / "2026-09-14" / "repo-ok-job.log").exists()


def test_run_nightly_refuses_off_simnode(tmp_path, monkeypatch):
    monkeypatch.setattr("krx_quant_core.runtime.host.socket.gethostname", lambda: "trader")
    repo = _repo(tmp_path)
    with pytest.raises(RunRefused, match="simnode"):
        ni.run_nightly(repo, out_root=tmp_path / "out", today=MONDAY)


def test_cli_nightly_exit_code(tmp_path, monkeypatch, capsys):
    """CLI 는 ``nightly.run_nightly`` 를 그대로 호출한다 — 실패 유무만 종료코드로 옮긴다."""
    from krx_quant_core.runtime import cli

    ok_results = [ni.JobResult(name="ok-job", rc=0, secs=0.1, timed_out=False)]
    bad_results = [
        ni.JobResult(name="ok-job", rc=0, secs=0.1, timed_out=False),
        ni.JobResult(name="fail-job", rc=1, secs=0.1, timed_out=False),
        ni.JobResult(name="slow-job", rc=None, secs=1.0, timed_out=True),
        ni.JobResult(name="weekend-job", rc=None, secs=0.0, timed_out=False, skipped="weekend"),
    ]

    calls = []
    monkeypatch.setattr(
        cli, "run_nightly", lambda repo_root, **kw: calls.append(kw) or ok_results
    )
    assert cli.main(["nightly", "repo", "--only", "ok-job"]) == 0
    assert calls[-1]["only"] == "ok-job" and calls[-1]["dry_run"] is False

    monkeypatch.setattr(cli, "run_nightly", lambda repo_root, **kw: bad_results)
    assert cli.main(["nightly", "repo"]) == 1  # 실패 2건이어도 최대 1
    out = capsys.readouterr().out
    assert "ok-job" in out and "fail-job" in out and "weekend" in out
