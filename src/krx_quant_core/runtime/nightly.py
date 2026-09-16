"""``kqc nightly`` — 소비 레포의 ``research/nightly.toml`` job 을 simnode 에서 격리 실행.

장 마감 뒤 cron 이 매 평일 돌리는 시뮬레이션 job(pair sweep, 파라미터 스윕 등)을 위한
자리다. job 하나가 실패하거나 멈춰도 다음 job 을 막지 않아야 한다 — 그래서 각 job 을
별도 프로세스 그룹(``start_new_session=True``)으로 띄우고 timeout 을 각자 건다. 타임아웃난
자식이 자손 프로세스를 남겨도 ``os.killpg`` 로 그룹째 죽인다.

``research/nightly.toml``::

    [[job]]
    name = "pair-sweep"
    cmd = ["uv", "run", "python", "scripts/pair_sweep.py", "--days", "20"]
    timeout_min = 60
    weekdays_only = true

로그는 ``<out>/<date>/<repo>-<job>.log``, 요약은 ``<out>/<date>/<repo>.json``
(``name, rc, secs, timed_out, skipped``) 에 남는다. ``out`` 기본값은 ``~/.kqc/nightly``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from subprocess import STDOUT

from krx_quant_core.market.session import now_kst

from .host import require_backtest_host
from .oos import RunRefused

__all__ = ["Job", "JobResult", "load_jobs", "run_nightly"]


@dataclass(frozen=True)
class Job:
    name: str
    cmd: list[str]
    timeout_min: float = 60
    weekdays_only: bool = True


@dataclass(frozen=True)
class JobResult:
    name: str
    rc: int | None
    secs: float
    timed_out: bool
    skipped: str | None = None


def load_jobs(repo_root: Path | str) -> list[Job]:
    """``<repo_root>/research/nightly.toml`` 을 읽는다. 파일이 없으면 빈 리스트."""
    path = Path(repo_root) / "research" / "nightly.toml"
    if not path.is_file():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    jobs = []
    for raw in data.get("job", []):
        jobs.append(
            Job(
                name=raw["name"],
                cmd=list(raw["cmd"]),
                timeout_min=float(raw.get("timeout_min", 60)),
                weekdays_only=bool(raw.get("weekdays_only", True)),
            )
        )
    return jobs


def _run_one(job: Job, *, repo_root: Path, log_path: Path) -> JobResult:
    start = time.monotonic()
    with log_path.open("wb") as log:
        try:
            proc = subprocess.Popen(
                ["nice", "-n", "10", *job.cmd],
                cwd=repo_root,
                stdout=log,
                stderr=STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            # 실행 파일이 없거나 권한이 없으면 — job 하나만 실패로 남기고 다음 job 은 돈다.
            log.write(f"kqc nightly: failed to start {job.cmd!r}: {exc}\n".encode())
            secs = time.monotonic() - start
            return JobResult(name=job.name, rc=127, secs=secs, timed_out=False)
        timed_out = False
        try:
            proc.wait(timeout=job.timeout_min * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # 타임아웃 판정과 실제 종료 사이에 이미 죽었을 수 있다
            proc.wait()
    secs = time.monotonic() - start
    return JobResult(name=job.name, rc=proc.returncode, secs=secs, timed_out=timed_out)


def run_nightly(
    repo_root: Path | str,
    *,
    only: str | None = None,
    dry_run: bool = False,
    out_root: Path | str | None = None,
    today: date | None = None,
) -> list[JobResult]:
    """레포의 nightly job 을 순서대로, 서로 격리해서 돌린다. simnode 전용."""
    try:
        require_backtest_host()
    except RuntimeError as exc:
        raise RunRefused(str(exc)) from exc
    repo_root = Path(repo_root)
    all_jobs = load_jobs(repo_root)
    jobs = [j for j in all_jobs if only is None or j.name == only]
    if only is not None and not jobs:
        raise ValueError(f"nightly job 없음: {only}")

    repo_name = repo_root.resolve().name
    day = today or now_kst().date()
    out = Path(out_root) if out_root is not None else Path.home() / ".kqc" / "nightly"
    date_dir = out / day.isoformat()
    date_dir.mkdir(parents=True, exist_ok=True)

    is_weekend = day.weekday() >= 5  # 5=토, 6=일

    results: list[JobResult] = []
    for job in jobs:
        if job.weekdays_only and is_weekend:
            results.append(
                JobResult(name=job.name, rc=None, secs=0.0, timed_out=False, skipped="weekend")
            )
            continue
        if dry_run:
            results.append(
                JobResult(name=job.name, rc=None, secs=0.0, timed_out=False, skipped="dry-run")
            )
            continue
        log_path = date_dir / f"{repo_name}-{job.name}.log"
        results.append(_run_one(job, repo_root=repo_root, log_path=log_path))

    summary_path = date_dir / f"{repo_name}.json"
    summary_path.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2) + "\n"
    )
    return results
