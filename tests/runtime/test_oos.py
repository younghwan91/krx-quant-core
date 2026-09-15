"""runtime.oos — OOS 하드 잠금 규칙 표."""

from __future__ import annotations

import json

import pytest

from krx_quant_core.runtime import oos
from krx_quant_core.runtime.gitstate import label_dir, runs_root


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("KQC_RUNS_ROOT", raising=False)
    return tmp_path


def _prereg(repo, text="# 사전등록\n임계값 0.7\n"):
    doc = repo / "docs" / "prereg.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(text, encoding="utf-8")
    return doc


def test_no_definition_passes(repo):
    oos.check("a", "2026-09-08", "2026-09-15", repo_root=repo)


def test_define_writes_file_and_refuses_redefine(repo):
    p = oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    assert p == label_dir(repo, "a") / "OOS.json"
    assert json.loads(p.read_text())["start"] == "2026-09-08"
    with pytest.raises(oos.RunRefused, match="already"):
        oos.define_oos("a", "2026-09-01", "2026-09-15", repo_root=repo)
    with pytest.raises(ValueError):
        oos.define_oos("b", "2026-09-15", "2026-09-08", repo_root=repo)


def test_non_overlapping_passes_and_boundary_counts_as_overlap(repo):
    oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    oos.check("a", "2026-08-24", "2026-09-07", repo_root=repo)
    with pytest.raises(oos.OOSLocked):
        oos.check("a", "2026-09-01", "2026-09-08", repo_root=repo)
    with pytest.raises(oos.OOSLocked):
        oos.check("a", "2026-09-15", "2026-09-20", repo_root=repo)


def test_final_requires_prereg_lock(repo):
    oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    with pytest.raises(oos.RunRefused, match="PREREG"):
        oos.check("a", "2026-09-08", "2026-09-15", repo_root=repo, final=True)
    assert not (label_dir(repo, "a") / "FINAL.lock").exists()


def test_final_refuses_edited_prereg(repo):
    oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    doc = _prereg(repo)
    oos.lock_prereg("a", doc, repo_root=repo)
    doc.write_text("# 사전등록\n임계값 0.6\n", encoding="utf-8")
    with pytest.raises(oos.RunRefused, match="changed"):
        oos.check("a", "2026-09-08", "2026-09-15", repo_root=repo, final=True)


def test_final_once(repo):
    oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    doc = _prereg(repo)
    lock = oos.lock_prereg("a", doc, repo_root=repo)
    assert json.loads(lock.read_text())["doc"] == "docs/prereg.md"
    with pytest.raises(oos.RunRefused, match="already"):
        oos.lock_prereg("a", doc, repo_root=repo)
    oos.check("a", "2026-09-08", "2026-09-15", repo_root=repo, final=True, run_id="r1")
    final = json.loads((label_dir(repo, "a") / "FINAL.lock").read_text())
    assert final["run_id"] == "r1"
    with pytest.raises(oos.RunRefused, match="FINAL"):
        oos.check("a", "2026-09-08", "2026-09-15", repo_root=repo, final=True)
    # 비 OOS 구간은 FINAL 뒤에도 계속 된다
    oos.check("a", "2026-08-24", "2026-09-07", repo_root=repo)


def test_final_on_non_oos_range_does_not_consume_lock(repo):
    oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    oos.check("a", "2026-08-24", "2026-09-07", repo_root=repo, final=True)
    assert not (label_dir(repo, "a") / "FINAL.lock").exists()


def test_runs_root_env_override(repo, tmp_path_factory, monkeypatch):
    other = tmp_path_factory.mktemp("checkout")
    monkeypatch.setenv("KQC_RUNS_ROOT", str(other))
    assert runs_root(repo) == other / "research" / "runs"
    oos.define_oos("a", "2026-09-08", "2026-09-15", repo_root=repo)
    assert (other / "research" / "runs" / "a" / "OOS.json").exists()


def test_bad_label_rejected(repo):
    with pytest.raises(ValueError):
        oos.define_oos("../x", "2026-09-08", "2026-09-15", repo_root=repo)
