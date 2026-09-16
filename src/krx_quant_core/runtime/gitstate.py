"""실행 기록이 쓰는 git 상태·기록 경로 헬퍼.

기록 루트는 ``KQC_RUNS_ROOT`` 환경변수가 있으면 그 경로, 없으면 ``repo_root`` 다. ``kqc run`` 은
코드를 simnode 의 일회용 worktree 에서 돌리면서 기록은 원래 체크아웃에 남기려고 이 변수를 넘긴다.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

__all__ = [
    "git_dirty",
    "git_head",
    "label_dir",
    "resolve_trials_dir",
    "runs_root",
    "validate_label",
]

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def validate_label(label: str) -> str:
    """label 은 디렉터리 이름이 된다 — 경로 조작 문자를 막는다."""
    if not _LABEL.match(label) or ".." in label:
        raise ValueError(f"invalid run label: {label!r} (letters, digits, . _ - only)")
    return label


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout


def git_head(repo: Path | str) -> str:
    """HEAD 커밋 sha(40자)."""
    return _git(Path(repo), "rev-parse", "HEAD").strip()


def git_dirty(repo: Path | str) -> bool:
    """추적 중인 파일에 커밋 안 된 변경이 있나.

    추적 안 되는 새 파일과 ``research/runs``(실행 기록 자신 — 실행할 때마다 append 된다)는 무시한다.
    """
    out = _git(
        Path(repo),
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        ".",
        ":(exclude)research/runs",
    )
    return bool(out.strip())


def runs_root(repo_root: Path | str) -> Path:
    base = os.environ.get("KQC_RUNS_ROOT") or str(repo_root)
    return Path(base) / "research" / "runs"


def label_dir(repo_root: Path | str, label: str) -> Path:
    return runs_root(repo_root) / validate_label(label)


def resolve_trials_dir(repo_root: Path | str, trials_dir: Path | str | None) -> Path:
    """시행 원장 폴더 경로 — :func:`~.runs.start_run` 이 쓰는 계산을 그대로 재사용한다.

    ``run_sweep``·``optuna_search`` 도 개별 config 를 같은 폴더의 ``TRIALS.jsonl`` 에
    적어야 ``count_trials`` 가 스윕 config 를 본다. 계산이 두 군데서 갈라지면 N 이
    조용히 틀어진다 — 그래서 한 곳에만 둔다.
    """
    if trials_dir is None:
        return runs_root(repo_root)
    d = Path(trials_dir)
    if not d.is_absolute():
        d = Path(repo_root) / d
    return d
