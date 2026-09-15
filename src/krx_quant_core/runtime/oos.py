"""OOS(표본 외) 구간 하드 잠금.

사전등록 규율을 부탁이 아니라 게이트로 만든다. label 마다 파일 셋이 git 에 남는다:

- ``OOS.json`` — 잠근 구간. 한 번 정하면 못 옮긴다(옮기려면 새 label).
- ``PREREG.lock`` — 사전등록 문서의 sha256. 잠근 뒤 문서를 고치면 최종 평가가 거부된다.
- ``FINAL.lock`` — 최종 평가를 **시작한** 기록. 평가가 중간에 죽어도 다시 못 돌린다(재시도가 곧
  OOS 를 두 번 보는 것이라서). 되돌리려면 사람이 git 에서 지우고, 그 커밋이 흔적으로 남는다.

규칙(:func:`check`): 정의 없음·안 겹침 → 통과. 겹침 + 비final → :class:`OOSLocked`.
겹침 + final → PREREG 잠금 있고 문서 해시 그대로이고 FINAL 없음일 때만 통과, 통과 즉시 FINAL 작성.
구간 경계일은 겹침으로 친다(양끝 포함).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from krx_quant_core.market.session import now_kst

from .gitstate import git_head, label_dir

__all__ = ["OOSLocked", "RunRefused", "check", "define_oos", "lock_prereg", "read_oos"]


class RunRefused(RuntimeError):
    """실행 기반 게이트가 거부했다(호스트·git 상태·OOS 규칙)."""


class OOSLocked(RunRefused):
    """잠긴 OOS 구간을 최종 평가가 아닌 실행이 읽으려 했다."""


def _write_new(path: Path, payload: dict[str, Any], what: str) -> Path:
    if path.exists():
        raise RunRefused(f"{what} already exists: {path} — it cannot be redefined")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head_or_none(repo_root: Path | str) -> str | None:
    try:
        return git_head(repo_root)
    except Exception:
        return None


def define_oos(label: str, start: str, end: str, *, repo_root: Path | str) -> Path:
    """label 의 OOS 구간(양끝 포함)을 정한다."""
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError(f"OOS start {start} is after end {end}")
    return _write_new(
        label_dir(repo_root, label) / "OOS.json",
        {"label": label, "start": start, "end": end, "defined_at": now_kst().isoformat()},
        "OOS definition",
    )


def read_oos(label: str, *, repo_root: Path | str) -> dict[str, Any] | None:
    p = label_dir(repo_root, label) / "OOS.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def lock_prereg(label: str, doc: Path | str, *, repo_root: Path | str) -> Path:
    """사전등록 문서 해시를 잠근다. ``doc`` 경로는 ``repo_root`` 기준 상대경로로 기록한다."""
    doc = Path(doc)
    root = Path(repo_root)
    full = doc if doc.is_absolute() else root / doc
    try:
        rel = str(full.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(full)
    return _write_new(
        label_dir(repo_root, label) / "PREREG.lock",
        {
            "doc": rel,
            "sha256": _sha256(full),
            "locked_at": now_kst().isoformat(),
            "git_sha": _head_or_none(repo_root),
        },
        "PREREG lock",
    )


def check(
    label: str,
    start: str,
    end: str,
    *,
    repo_root: Path | str,
    final: bool = False,
    run_id: str | None = None,
) -> None:
    """``[start, end]`` 데이터를 읽어도 되나. 안 되면 :class:`RunRefused`/:class:`OOSLocked`."""
    spec = read_oos(label, repo_root=repo_root)
    if spec is None:
        return
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    os_, oe = date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"])
    if not (s <= oe and e >= os_):
        return
    if not final:
        raise OOSLocked(
            f"{label}: {start}~{end} overlaps locked OOS {spec['start']}~{spec['end']} — "
            "only the single preregistered final evaluation may read it (final=True)"
        )
    d = label_dir(repo_root, label)
    lock_p = d / "PREREG.lock"
    if not lock_p.exists():
        raise RunRefused(f"{label}: final evaluation needs PREREG.lock (kqc prereg lock)")
    lock = json.loads(lock_p.read_text(encoding="utf-8"))
    doc = Path(lock["doc"])
    doc_full = doc if doc.is_absolute() else Path(repo_root) / doc
    if not doc_full.exists() or _sha256(doc_full) != lock["sha256"]:
        raise RunRefused(
            f"{label}: preregistration doc {lock['doc']} changed or missing since lock"
        )
    _write_new(
        d / "FINAL.lock",
        {"run_id": run_id, "ts": now_kst().isoformat(), "git_sha": _head_or_none(repo_root)},
        "FINAL lock (OOS already evaluated once)",
    )
