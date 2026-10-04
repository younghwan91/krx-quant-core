"""``kqc pins`` — 소비 레포들이 이 패키지를 몇 버전에 고정했는지 한 표로.

2026-10-04 감사에서 scalp-it 0.5.1 · daytrade-it 0.5.0(워크트리 0.4.1) · swing-it 0.4.1 로
세 레포가 세 버전이었다. 핀이 갈라지면 "같은 비용·같은 판정"이라는 이 패키지의 전제가
깨진다. 고치는 도구가 아니라 **보는** 도구다 — 핀 범프는 각 레포의 PR 로 한다.

``pyproject.toml`` 의 요구사항 문자열(``krx-quant-core==0.5.1``)과 ``uv.lock`` 의 잠긴
버전을 둘 다 읽는다. 둘이 다르면 그것도 신호다(lock 을 안 갱신했다).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from krx_quant_core import __version__

__all__ = ["DEFAULT_REPOS", "PACKAGE", "Pin", "parse_version", "read_pin", "read_pins"]

PACKAGE = "krx-quant-core"
#: 기본 점검 대상. 없는 디렉터리는 조용히 건너뛴다.
DEFAULT_REPOS: tuple[str, ...] = ("scalp-it", "daytrade-it", "swing-it")

_REQ = re.compile(r"^\s*krx[-_]quant[-_]core\s*(\[[^\]]*\])?\s*(?P<spec>[^;#]*)")


@dataclass(frozen=True)
class Pin:
    repo: str
    #: pyproject 의 명세 문자열(``==0.5.1``·``>=0.4``). 의존성에 없으면 None.
    requirement: str | None
    #: uv.lock 에 잠긴 버전. lock 이 없거나 패키지가 없으면 None.
    locked: str | None

    def behind(self, current: str) -> bool:
        """잠긴 버전(없으면 ``==`` 핀)이 ``current`` 보다 낮은가. 알 수 없으면 False."""
        v = self.locked or _pinned_exact(self.requirement)
        return v is not None and parse_version(v) < parse_version(current)


def parse_version(v: str) -> tuple[int, ...]:
    """``0.5.1`` → ``(0, 5, 1)``. 숫자 아닌 꼬리(``rc1``)는 버린다 — 비교용이다."""
    parts: list[int] = []
    for piece in v.strip().split("."):
        m = re.match(r"\d+", piece)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts)


def _pinned_exact(req: str | None) -> str | None:
    if not req:
        return None
    m = re.match(r"\s*==\s*([0-9][0-9A-Za-z.]*)", req)
    return m.group(1) if m else None


def _requirement_from_pyproject(path: Path) -> str | None:
    with path.open("rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    deps: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        deps.extend(group)
    for d in deps:
        m = _REQ.match(d)
        if m:
            return m.group("spec").strip() or ""
    return None


def _locked_from_uv_lock(path: Path) -> str | None:
    with path.open("rb") as f:
        data = tomllib.load(f)
    for pkg in data.get("package", []):
        if str(pkg.get("name", "")).replace("_", "-") == PACKAGE:
            return str(pkg.get("version")) if pkg.get("version") is not None else None
    return None


def read_pin(repo_dir: Path | str) -> Pin:
    repo_dir = Path(repo_dir)
    pyproject = repo_dir / "pyproject.toml"
    lock = repo_dir / "uv.lock"
    req = _requirement_from_pyproject(pyproject) if pyproject.is_file() else None
    locked = _locked_from_uv_lock(lock) if lock.is_file() else None
    return Pin(repo=repo_dir.name, requirement=req, locked=locked)


def read_pins(root: Path | str, repos: list[str] | None = None) -> list[Pin]:
    """``root/<repo>`` 중 존재하는 것만 읽는다. ``repos`` 가 None 이면 :data:`DEFAULT_REPOS`."""
    root = Path(root)
    names = list(repos) if repos else list(DEFAULT_REPOS)
    return [read_pin(root / n) for n in names if (root / n).is_dir()]


def format_table(pins: list[Pin], current: str = __version__) -> tuple[str, int]:
    """표 문자열과 뒤처진 레포 수. ``kqc pins`` 가 그대로 찍는다."""
    lines = [f"{PACKAGE} here: {current}", f"{'repo':14} {'pyproject':14} {'uv.lock':10} note"]
    behind = 0
    for p in pins:
        notes = []
        if p.requirement is None:
            notes.append("not a dependency")
        elif p.locked is None:
            notes.append("no uv.lock")
        elif _pinned_exact(p.requirement) not in (None, p.locked):
            notes.append("lock != pin")
        if p.behind(current):
            notes.append(f"BEHIND {current}")
            behind += 1
        lines.append(
            f"{p.repo:14} {(p.requirement if p.requirement is not None else '-') or '(any)':14} "
            f"{p.locked or '-':10} {' '.join(notes)}"
        )
    return "\n".join(lines), behind
