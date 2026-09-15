"""``kqc`` — simnode 백테스트 실행 기반 CLI.

::

    kqc run scalp-it -- uv run python scripts/research_84/select.py     # trader 에서
    kqc runs ls scalp84-flow --repo-root ~/git/scalp-it
    kqc runs sync ~/git/scalp-it
    kqc oos define scalp84-flow --start 2026-09-08 --end 2026-09-15 --repo-root .
    kqc prereg lock scalp84-flow docs/research/manju/84-...md --repo-root .

``kqc run`` 은 커밋·푸시된 sha 만 돌린다. simnode 에 ``~/.kqc/wt/<repo>-<sha12>`` 일회용
worktree 를
만들어(원래 체크아웃은 건드리지 않음) ``uv sync --frozen`` 뒤 명령을 실행하고, 기록은
``KQC_RUNS_ROOT=~/git/<repo>`` 로 원래 체크아웃의 ``research/runs`` 에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import oos, runindex
from .gitstate import validate_label
from .host import BACKTEST_HOST
from .oos import RunRefused
from .runs import read_runs

__all__ = ["SSH_TARGET", "local_ref_check", "main", "remote_script"]

#: trader → simnode ssh 별칭(``~/.ssh/config``).
SSH_TARGET = "simnode-local"
#: DB·API 키 중 원격 실행에 넘기는 것만. 주문 키(KIWOOM_*)는 백테스트에 필요 없어 넘기지 않는다.
_ENV_KEYS = ("KR_QUANT_DB", "DART_API_KEY")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _default_runner(args: Sequence[str], **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), capture_output=True, text=True, check=False, **kw)  # type: ignore[call-overload, no-any-return]


def _validate_repo(repo: str) -> str:
    if not repo or "/" in repo or repo.startswith(".") or not validate_label(repo):
        raise ValueError(f"invalid repo name: {repo!r}")
    return repo


def local_ref_check(repo_dir: Path, ref: str, runner: Runner = _default_runner) -> str:
    """로컬 체크아웃 기준으로 ``ref`` 를 sha 로 풀고, 깨끗하고 origin 에 있는지 확인한다."""
    g = ["git", "-C", str(repo_dir)]
    st = runner([*g, "status", "--porcelain", "--untracked-files=no"])
    if st.returncode != 0:
        raise RunRefused(f"{repo_dir}: not a git repository ({st.stderr.strip()})")
    if st.stdout.strip() and ref == "HEAD":
        raise RunRefused(f"{repo_dir} has uncommitted changes — commit and push before kqc run")
    rev = runner([*g, "rev-parse", "--verify", f"{ref}^{{commit}}"])
    if rev.returncode != 0:
        raise RunRefused(f"{repo_dir}: unknown ref {ref!r}")
    sha = rev.stdout.strip()
    runner([*g, "fetch", "-q", "origin"])
    remote = runner([*g, "branch", "-r", "--contains", sha])
    if remote.returncode != 0 or not remote.stdout.strip():
        raise RunRefused(f"{sha[:12]} is not on origin — push it first (simnode fetches origin)")
    return sha


def remote_script(repo: str, sha: str, cmd: Sequence[str]) -> str:
    """simnode 에서 ``bash -s`` 로 돌릴 스크립트."""
    _validate_repo(repo)
    if len(sha) < 7 or not all(c in "0123456789abcdef" for c in sha.lower()):
        raise ValueError(f"invalid sha: {sha!r}")
    if not cmd:
        raise ValueError("empty command")
    short = sha[:12]
    keys = "|".join(_ENV_KEYS)
    quoted = " ".join(shlex.quote(c) for c in cmd)
    return f"""set -euo pipefail
REPO="$HOME/git/{repo}"
WT="$HOME/.kqc/wt/{repo}-{short}"
LOGDIR="$HOME/.kqc/logs"
mkdir -p "$HOME/.kqc/wt" "$LOGDIR"
cd "$REPO"
git fetch -q origin
if [ ! -d "$WT" ]; then git worktree add -q --detach "$WT" {sha}; fi
cd "$WT"
if [ "$(git rev-parse HEAD)" != "$(git -C "$REPO" rev-parse {sha}^{{commit}})" ]; then
  echo "kqc: worktree $WT is not at {short}" >&2; exit 2
fi
uv sync -q --frozen --all-extras 2>/dev/null || uv sync -q --frozen
ENVF="$HOME/git/quant-airflow/.env"
if [ -r "$ENVF" ]; then
  set -a; eval "$(grep -E '^({keys})=' "$ENVF")"; set +a
fi
export KQC_RUNS_ROOT="$REPO"
LOG="$LOGDIR/{repo}-{short}-$(date +%Y%m%d-%H%M%S).log"
echo "kqc: {repo}@{short} on $(hostname) → $LOG" >&2
set +e
{quoted} 2>&1 | tee "$LOG"
rc=${{PIPESTATUS[0]}}
echo "kqc: exit $rc" >&2
exit $rc
"""


def _cmd_run(a: argparse.Namespace) -> int:
    repo = _validate_repo(a.repo)
    cmd = list(a.cmd)
    if not cmd:
        raise ValueError("usage: kqc run <repo> [--ref REF] -- <command...>")
    local = Path.home() / "git" / repo
    if local.is_dir():
        sha = local_ref_check(local, a.ref)
    elif len(a.ref) >= 7 and a.ref != "HEAD":
        sha = a.ref
    else:
        raise RunRefused(f"{local} not found here — pass --ref <full sha> to run {repo} on simnode")
    script = remote_script(repo, sha, cmd)
    if socket.gethostname() == BACKTEST_HOST:
        argv = ["bash", "-s"]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", SSH_TARGET, "bash", "-s"]
    return subprocess.run(argv, input=script, text=True, check=False).returncode


def _cmd_runs_ls(a: argparse.Namespace) -> int:
    for r in read_runs(a.label, repo_root=a.repo_root):
        res = json.dumps(r.get("result"), ensure_ascii=False, default=str)
        print(
            f"{r.get('started_at', '?')[:19]}  {r.get('run_id')}  {str(r.get('status')):6}  "
            f"{str(r.get('git_sha', ''))[:10]}  trials={r.get('n_trials')}  {res}"
        )
    return 0


def _cmd_runs_sync(a: argparse.Namespace) -> int:
    print(f"synced {runindex.sync(a.repo_root)} rows")
    return 0


def _cmd_oos_define(a: argparse.Namespace) -> int:
    print(oos.define_oos(a.label, a.start, a.end, repo_root=a.repo_root))
    return 0


def _cmd_prereg_lock(a: argparse.Namespace) -> int:
    print(oos.lock_prereg(a.label, a.doc, repo_root=a.repo_root))
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kqc", description="simnode backtest run infrastructure")
    sub = p.add_subparsers(dest="cmd_name", required=True)

    r = sub.add_parser("run", help="run a command on simnode at a pushed git sha")
    r.add_argument("repo")
    r.add_argument("--ref", default="HEAD")
    r.set_defaults(func=_cmd_run, cmd=[])

    runs_p = sub.add_parser("runs").add_subparsers(dest="runs_cmd", required=True)
    ls = runs_p.add_parser("ls")
    ls.add_argument("label")
    ls.add_argument("--repo-root", default=".")
    ls.set_defaults(func=_cmd_runs_ls)
    sy = runs_p.add_parser("sync")
    sy.add_argument("repo_root")
    sy.set_defaults(func=_cmd_runs_sync)

    oos_p = sub.add_parser("oos").add_subparsers(dest="oos_cmd", required=True)
    de = oos_p.add_parser("define")
    de.add_argument("label")
    de.add_argument("--start", required=True)
    de.add_argument("--end", required=True)
    de.add_argument("--repo-root", default=".")
    de.set_defaults(func=_cmd_oos_define)

    pr = sub.add_parser("prereg").add_subparsers(dest="prereg_cmd", required=True)
    lk = pr.add_parser("lock")
    lk.add_argument("label")
    lk.add_argument("doc")
    lk.add_argument("--repo-root", default=".")
    lk.set_defaults(func=_cmd_prereg_lock)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # ``--`` 뒤는 원격 명령이다. argparse 에 넘기면 옵션(--ref)과 섞이므로 먼저 떼어 낸다.
    cmd: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1 :]
    a = _parser().parse_args(argv)
    if a.cmd_name == "run":
        a.cmd = cmd
    try:
        return int(a.func(a))
    except (RunRefused, ValueError) as exc:
        print(f"kqc: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
