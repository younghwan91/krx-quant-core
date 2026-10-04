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

**실행 파일 해석.** cron 의 PATH 는 ``/usr/bin:/bin`` 뿐이라 ``uv`` 처럼 ``~/.local/bin`` 에
있는 명령은 못 찾는다 — daytrade-it 의 job 이 2026-09-23 부터 매일 rc=127 로 조용히 죽어
있었다. 그래서 ``cmd[0]`` 을 :data:`EXTRA_PATH_DIRS` 를 보탠 PATH 에서 찾고, 그 PATH 를
자식에게도 넘긴다(``uv run`` 이 안에서 또 python 을 찾아야 하므로). 못 찾으면 어디를
봤는지 로그에 적는다.

**실패를 보이게.** :func:`read_status` 가 최근 N 일 요약을 레포×job 표로 돌려주고,
``kqc nightly status`` 가 그걸 찍고 가장 최근 날짜에 실패가 있으면 1 로 끝난다 —
"요약 JSON 에 rc=127 이 8 일째" 같은 것을 아무도 안 보는 일을 막기 위해서다.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from subprocess import STDOUT

from krx_quant_core.market.session import now_kst

from .host import require_backtest_host
from .oos import RunRefused

__all__ = [
    "EXTRA_PATH_DIRS",
    "Job",
    "JobResult",
    "JobStatus",
    "augmented_path",
    "load_jobs",
    "read_status",
    "resolve_executable",
    "run_nightly",
]

#: cron 의 빈약한 PATH 에 보태는 디렉터리. ``~`` 는 실행 시점에 푼다.
EXTRA_PATH_DIRS: tuple[str, ...] = ("~/.local/bin", "~/.cargo/bin", "/usr/local/bin")


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
    """``<repo_root>/research/nightly.toml`` 을 읽는다. 파일이 없으면 빈 리스트.

    ``name``·``cmd`` 가 빠진 job 은 몇 번째(0부터)인지 담아 ``ValueError``.
    """
    path = Path(repo_root) / "research" / "nightly.toml"
    if not path.is_file():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    jobs = []
    for i, raw in enumerate(data.get("job", [])):
        for key in ("name", "cmd"):
            if key not in raw:
                # KeyError('name') 로는 toml 의 몇 번째 job 이 틀렸는지 모른다.
                raise ValueError(f"{path}: job[{i}] 에 {key!r} 가 없다")
        jobs.append(
            Job(
                name=raw["name"],
                cmd=list(raw["cmd"]),
                timeout_min=float(raw.get("timeout_min", 60)),
                weekdays_only=bool(raw.get("weekdays_only", True)),
            )
        )
    return jobs


def augmented_path(base: str | None = None) -> str:
    """현재 PATH 뒤에 :data:`EXTRA_PATH_DIRS` 를 붙인 PATH 문자열(중복 제거, 순서 유지).

    앞이 아니라 **뒤** 에 붙인다 — 사용자가 PATH 에 둔 것이 먼저다. cron 처럼 PATH 가
    빈약할 때만 보탠 디렉터리가 효력을 낸다.
    """
    cur = os.environ.get("PATH", "") if base is None else base
    seen: list[str] = []
    for d in [*cur.split(os.pathsep), *(os.path.expanduser(x) for x in EXTRA_PATH_DIRS)]:
        if d and d not in seen:
            seen.append(d)
    return os.pathsep.join(seen)


def resolve_executable(name: str, path: str | None = None) -> str | None:
    """``name`` 을 :func:`augmented_path` 에서 찾는다. 절대·상대 경로면 그대로 존재 여부만 본다."""
    return shutil.which(name, path=augmented_path() if path is None else path)


def _run_one(job: Job, *, repo_root: Path, log_path: Path) -> JobResult:
    start = time.monotonic()
    path = augmented_path()
    env = {**os.environ, "PATH": path}
    with log_path.open("wb") as log:
        exe = resolve_executable(job.cmd[0], path) if job.cmd else None
        if exe is None:
            name = job.cmd[0] if job.cmd else "(empty cmd)"
            log.write(
                f"kqc nightly: executable not found: {name}\n  searched PATH={path}\n".encode()
            )
            return JobResult(name=job.name, rc=127, secs=time.monotonic() - start, timed_out=False)
        try:
            proc = subprocess.Popen(
                ["nice", "-n", "10", exe, *job.cmd[1:]],
                cwd=repo_root,
                stdout=log,
                stderr=STDOUT,
                start_new_session=True,
                env=env,
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
    if not repo_root.is_dir():
        # 없는 경로면 load_jobs 가 조용히 [] 를 돌려 cron 이 "job 0개 성공"으로 끝난다.
        raise ValueError(f"repo_root 가 디렉터리가 아니다: {repo_root}")
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
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


@dataclass
class JobStatus:
    """한 레포의 한 job 이 최근 날짜들에 어떻게 끝났는가(:func:`read_status`)."""

    repo: str
    name: str
    #: 날짜(ISO) → 상태 문자열(``ok``·``rc=N``·``timeout``·``skipped:<why>``). 날짜 오름차순.
    by_date: dict[str, str] = field(default_factory=dict)

    @property
    def latest(self) -> str | None:
        return next(reversed(self.by_date.values()), None) if self.by_date else None

    @property
    def latest_failed(self) -> bool:
        """가장 최근 **실행된**(skipped 아닌) 날이 실패였나. 실행 기록이 없으면 False."""
        for status in reversed(self.by_date.values()):
            if status.startswith("skipped"):
                continue
            return status != "ok"
        return False

    @property
    def fail_streak(self) -> int:
        """가장 최근부터 거꾸로 센 연속 실패 일수(skipped 는 건너뛴다)."""
        n = 0
        for status in reversed(self.by_date.values()):
            if status.startswith("skipped"):
                continue
            if status == "ok":
                break
            n += 1
        return n


def _status_of(row: dict) -> str:
    if row.get("skipped"):
        return f"skipped:{row['skipped']}"
    if row.get("timed_out"):
        return "timeout"
    return "ok" if row.get("rc") == 0 else f"rc={row.get('rc')}"


def read_status(
    out_root: Path | str | None = None,
    *,
    days: int = 7,
    today: date | None = None,
    warn=None,
) -> list[JobStatus]:
    """``<out>/<date>/<repo>.json`` 을 최근 ``days`` 개 날짜 디렉터리에서 읽어 레포×job 으로 묶는다.

    날짜 디렉터리는 이름이 ISO 날짜인 것만 보고, ``today`` 이후는 무시한다. 깨진 JSON 은
    ``warn(msg)`` 로 알리고 건너뛴다(한 파일 때문에 전체가 안 보이면 안 된다). 호스트 가드는
    없다 — 읽기만 하므로 trader 에서도 본다.
    """
    out = Path(out_root) if out_root is not None else Path.home() / ".kqc" / "nightly"
    if not out.is_dir():
        return []
    limit = (today or now_kst().date()).isoformat()
    dirs = sorted(
        d for d in out.iterdir() if d.is_dir() and _is_iso_date(d.name) and d.name <= limit
    )
    dirs = dirs[-days:] if days > 0 else dirs
    table: dict[tuple[str, str], JobStatus] = {}
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            try:
                rows = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                if warn is not None:
                    warn(f"kqc nightly status: {f} 를 읽지 못했다 — {exc}")
                continue
            if not isinstance(rows, list):
                continue
            repo = f.stem
            for row in rows:
                if not isinstance(row, dict) or "name" not in row:
                    continue
                key = (repo, str(row["name"]))
                js = table.setdefault(key, JobStatus(repo=repo, name=str(row["name"])))
                js.by_date[d.name] = _status_of(row)
    return [table[k] for k in sorted(table)]


def _is_iso_date(name: str) -> bool:
    try:
        date.fromisoformat(name)
    except ValueError:
        return False
    return True
