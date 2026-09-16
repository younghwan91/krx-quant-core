"""백테스트 실행 기록 — 어디서·어떤 코드로·어떤 데이터로·몇 번째 시도로 나온 숫자인가.

.. code-block:: python

    with start_run("scalp84-flow", config, repo_root=ROOT,
                   data=DataSpec("2026-08-24", "2026-09-07", "train"), seed=84) as run:
        ...
        run.log_result({"mean_bp": -3.1, "n": 812})

진입 게이트(하나라도 걸리면 :class:`~.oos.RunRefused`, 파일은 안 쓴다):

1. simnode 가 아니다(:func:`~.host.require_backtest_host`).
2. ``repo_root`` 의 추적 파일에 커밋 안 된 변경이 있다
   (``allow_dirty=True`` 면 통과하되 기록에 남는다).
3. ``data`` 구간이 OOS 잠금에 걸린다(:func:`~.oos.check`, ``final`` 전달).

기록: ``<runs_root>/<label>/RUNS.jsonl`` 에 ``start``·``end`` 두 줄(append-only). 시행 수 원장
(:mod:`krx_quant_core.stats.trials`)도 같은 폴더의 ``TRIALS.jsonl`` 에 자동으로 적는다 — DSR 의 N 은
``run.n_trials``. 파일이 정본이고 Postgres 색인(:mod:`.runindex`)은 실패해도 실행을 막지 않는다.
"""

from __future__ import annotations

import json
import platform
import socket
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from krx_quant_core.market.session import now_kst
from krx_quant_core.stats.trials import config_fingerprint, count_trials, record_trial

from . import oos, runindex
from .gitstate import git_dirty, git_head, label_dir, resolve_trials_dir
from .host import require_backtest_host
from .oos import RunRefused

__all__ = ["DataSpec", "Run", "RunRefused", "read_runs", "start_run"]


@dataclass(frozen=True)
class DataSpec:
    """이 실행이 읽는 데이터 구간(ISO 날짜, 양끝 포함)과 용도 표시."""

    start: str
    end: str
    split: str | None = None


@dataclass
class Run:
    label: str
    run_id: str
    dir: Path
    n_trials: int
    _result: dict[str, Any] = field(default_factory=dict)

    def log_result(self, result: dict[str, Any]) -> None:
        """결과 지표를 기록에 붙인다(여러 번 부르면 합쳐진다). 종료 행에 쓰인다."""
        self._result.update(result)


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for dist in ("krx-quant-core", "numpy", "pandas", "numba"):
        try:
            out[dist.replace("-", "_")] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return out


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str, ensure_ascii=False))


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    runindex.upsert(row)


@contextmanager
def start_run(
    label: str,
    config: dict[str, Any],
    *,
    repo_root: Path | str,
    data: DataSpec | None = None,
    seed: int | None = None,
    final: bool = False,
    allow_dirty: bool = False,
    trials_dir: Path | str | None = None,
) -> Iterator[Run]:
    """게이트를 통과하면 기록을 시작하고 :class:`Run` 을 준다. 예외는 기록한 뒤 다시 던진다.

    ``trials_dir`` 는 시행 원장 폴더(기본 ``research/runs``). 이미 다른 곳(예 swing-it
    ``research/logs``)에 원장을 쌓아 온 레포는 그 경로를 넘겨야 DSR 의 N 이 이어진다.
    상대경로는 ``repo_root`` 기준.
    """
    repo_root = Path(repo_root)
    d = label_dir(repo_root, label)
    try:
        require_backtest_host()
    except RuntimeError as exc:
        raise RunRefused(str(exc)) from exc
    try:
        sha = git_head(repo_root)
        dirty = git_dirty(repo_root)
    except Exception as exc:
        raise RunRefused(f"{repo_root} is not a readable git repository: {exc}") from exc
    if dirty and not allow_dirty:
        raise RunRefused(
            f"{repo_root} has uncommitted changes to tracked files — commit first "
            "(results must map to a git sha), or pass allow_dirty=True for exploration"
        )
    run_id = uuid.uuid4().hex[:12]
    if data is not None:
        oos.check(label, data.start, data.end, repo_root=repo_root, final=final, run_id=run_id)

    logs_dir = resolve_trials_dir(repo_root, trials_dir)
    record_trial(label, config, logs_dir=logs_dir)
    run = Run(label=label, run_id=run_id, dir=d, n_trials=count_trials(label, logs_dir=logs_dir))
    path = d / "RUNS.jsonl"
    start_row = {
        "event": "start",
        "run_id": run_id,
        "label": label,
        "ts": now_kst().isoformat(),
        "host": socket.gethostname(),
        "repo": repo_root.resolve().name,
        "git_sha": sha,
        "dirty": dirty,
        "config": _jsonable(config),
        "fingerprint": config_fingerprint(config),
        "data": asdict(data) if data is not None else None,
        "seed": seed,
        "final": final,
        "n_trials": run.n_trials,
        "versions": _versions(),
        "argv": list(sys.argv),
    }
    _append(path, start_row)
    t0 = time.monotonic()
    status, error = "ok", None
    try:
        yield run
    except BaseException as exc:
        status = "failed"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise
    finally:
        _append(
            path,
            {
                "event": "end",
                "run_id": run_id,
                "label": label,
                "ts": now_kst().isoformat(),
                "status": status,
                "error": error,
                "duration_s": round(time.monotonic() - t0, 3),
                "result": _jsonable(run._result),
            },
        )


def read_runs(label: str, *, repo_root: Path | str) -> list[dict[str, Any]]:
    """``start``·``end`` 행을 ``run_id`` 로 합친 실행 목록(시작 순, 끝 없으면 status None)."""
    path = label_dir(repo_root, label) / "RUNS.jsonl"
    if not path.exists():
        return []
    merged: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = row.get("run_id")
        if not rid:
            continue
        m = merged.setdefault(rid, {"status": None})
        if row.get("event") == "start":
            m.update({k: v for k, v in row.items() if k != "event"})
            m["started_at"] = row.get("ts")
        elif row.get("event") == "end":
            m.update(
                {
                    "status": row.get("status"),
                    "error": row.get("error"),
                    "duration_s": row.get("duration_s"),
                    "result": row.get("result"),
                    "ended_at": row.get("ts"),
                }
            )
    return list(merged.values())
