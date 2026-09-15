"""runtime.cli — kqc 원격 스크립트·로컬 ref 검사·파일 서브커맨드."""

from __future__ import annotations

import json
import subprocess

import pytest

from krx_quant_core.runtime import cli
from krx_quant_core.runtime.gitstate import label_dir
from krx_quant_core.runtime.oos import RunRefused

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_remote_script_shape():
    s = cli.remote_script("scalp-it", SHA, ["uv", "run", "python", "x.py", "--q", "a b; rm -rf /"])
    assert 'WT="$HOME/.kqc/wt/scalp-it-0123456789ab"' in s
    assert f"git worktree add -q --detach \"$WT\" {SHA}" in s
    assert "uv sync -q --frozen" in s
    assert "grep -E '^(KR_QUANT_DB|DART_API_KEY)='" in s and "KIWOOM" not in s
    assert 'export KQC_RUNS_ROOT="$REPO"' in s
    assert "uv run python x.py --q 'a b; rm -rf /'" in s  # 인자는 셸 인용
    assert "exit $rc" in s
    # bash 문법 검사
    subprocess.run(["bash", "-n"], input=s, text=True, check=True)


@pytest.mark.parametrize("repo", ["", "../x", "a/b", ".hidden"])
def test_remote_script_rejects_bad_repo(repo):
    with pytest.raises(ValueError):
        cli.remote_script(repo, SHA, ["true"])


def test_remote_script_rejects_bad_sha_and_empty_cmd():
    with pytest.raises(ValueError):
        cli.remote_script("scalp-it", "main; echo", ["true"])
    with pytest.raises(ValueError):
        cli.remote_script("scalp-it", SHA, [])


class _Fake:
    def __init__(self, dirty="", rev=SHA, remote="  origin/main\n"):
        self.out = {"status": dirty, "rev-parse": rev, "fetch": "", "branch": remote}
        self.calls = []

    def __call__(self, args, **kw):
        self.calls.append(args)
        key = args[3]
        code = 0
        if key == "rev-parse" and not self.out[key]:
            code = 1
        return subprocess.CompletedProcess(args, code, stdout=self.out[key] + "\n", stderr="")


def test_local_ref_check_ok(tmp_path):
    assert cli.local_ref_check(tmp_path, "HEAD", _Fake()) == SHA


def test_local_ref_check_refuses_dirty_and_unpushed(tmp_path):
    with pytest.raises(RunRefused, match="uncommitted"):
        cli.local_ref_check(tmp_path, "HEAD", _Fake(dirty=" M a.py"))
    with pytest.raises(RunRefused, match="origin"):
        cli.local_ref_check(tmp_path, "HEAD", _Fake(remote=""))
    with pytest.raises(RunRefused, match="unknown ref"):
        cli.local_ref_check(tmp_path, "nope", _Fake(rev=""))


def test_oos_prereg_and_runs_subcommands(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("KQC_RUNS_ROOT", raising=False)
    root = str(tmp_path)
    assert cli.main(["oos", "define", "lab", "--start", "2026-09-08", "--end", "2026-09-15",
                     "--repo-root", root]) == 0  # fmt: skip
    assert cli.main(["oos", "define", "lab", "--start", "2026-09-08", "--end", "2026-09-15",
                     "--repo-root", root]) == 2  # fmt: skip
    doc = tmp_path / "p.md"
    doc.write_text("x")
    assert cli.main(["prereg", "lock", "lab", str(doc), "--repo-root", root]) == 0
    assert json.loads((label_dir(tmp_path, "lab") / "PREREG.lock").read_text())["doc"] == "p.md"
    rows = [
        {"event": "start", "run_id": "r1", "ts": "2026-09-15T23:00:00+09:00", "git_sha": SHA,
         "n_trials": 3},
        {"event": "end", "run_id": "r1", "status": "ok", "result": {"mean_bp": 1.5}},
    ]  # fmt: skip
    (label_dir(tmp_path, "lab") / "RUNS.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    capsys.readouterr()
    assert cli.main(["runs", "ls", "lab", "--repo-root", root]) == 0
    out = capsys.readouterr().out
    assert "r1" in out and "ok" in out and "trials=3" in out and "mean_bp" in out


def test_run_splits_command_at_double_dash(monkeypatch):
    seen = {}

    def fake_run(a):
        seen["repo"], seen["ref"], seen["cmd"] = a.repo, a.ref, a.cmd
        return 0

    monkeypatch.setattr(cli, "_cmd_run", fake_run)  # _parser() 가 호출 시점에 전역을 읽는다
    argv = ["run", "scalp-it", "--ref", SHA, "--", "uv", "run", "pytest", "--ref", "x"]
    assert cli.main(argv) == 0
    assert seen == {"repo": "scalp-it", "ref": SHA, "cmd": ["uv", "run", "pytest", "--ref", "x"]}


def test_run_without_command_is_usage_error(capsys):
    assert cli.main(["run", "scalp-it"]) == 2
    assert "usage" in capsys.readouterr().err
