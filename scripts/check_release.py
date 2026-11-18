"""릴리스 일관성 검사 — 버전 네 곳·태그·sdist 내용. 표준 라이브러리만 쓴다(설치 전에도 돈다).

    python scripts/check_release.py                          # 버전 네 곳이 같은가
    python scripts/check_release.py --tag v0.9.0             # + 태그가 그 버전인가 (publish.yml)
    python scripts/check_release.py --sdist dist/x.tar.gz    # + sdist 에 허용된 것만 들었나

왜 있나: 버전은 pyproject·``__init__.__version__``·README 설치 줄·uv.lock 네 곳에 손으로 쓴다.
v0.2.0~v0.4.2 다섯 릴리스가 ``__version__ = "0.1.1"`` 인 채로 PyPI 에 나갔고, 그동안 ``kqc pins`` 의
비교 기준도 0.1.1 이었다. sdist 는 include 목록이 없어 ``.claude/``(에이전트 스킬)·``CLAUDE.md``
까지 실었다.
문제가 있으면 한 줄씩 찍고 exit 1.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
import tomllib
from pathlib import Path

PACKAGE = "krx-quant-core"
ROOT = Path(__file__).resolve().parents[1]

#: sdist 최상위에 허용하는 이름. 바꾸면 pyproject 의 ``[tool.hatch.build.targets.sdist]`` 도 같이.
SDIST_ALLOWED = frozenset(
    {"src", "tests", "README.md", "LICENSE", "pyproject.toml", "PKG-INFO", ".gitignore"}
)


def versions(root: Path = ROOT) -> dict[str, list[str]]:
    """자리별로 적힌 버전 목록. README 는 설치 줄이 두 개라 목록이다."""
    out: dict[str, list[str]] = {}
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    out["pyproject.toml"] = [pyproject["project"]["version"]]
    init = (root / "src/krx_quant_core/__init__.py").read_text()
    out["__init__.__version__"] = re.findall(r'^__version__\s*=\s*"([^"]+)"', init, re.M)
    readme = (root / "README.md").read_text()
    out["README.md"] = re.findall(rf"{PACKAGE}==([0-9][0-9A-Za-z.+-]*)", readme) + re.findall(
        rf"{PACKAGE}@v([0-9][0-9A-Za-z.+-]*)", readme
    )
    lock = tomllib.loads((root / "uv.lock").read_text())
    out["uv.lock"] = [p["version"] for p in lock.get("package", []) if p.get("name") == PACKAGE]
    return out


def check_versions(root: Path = ROOT) -> list[str]:
    found = versions(root)
    expected = found["pyproject.toml"][0]
    errors = []
    for where, vs in found.items():
        if not vs:
            errors.append(f"{where}: 버전을 못 찾음 (기대 {expected})")
        errors += [f"{where}: {v} != pyproject {expected}" for v in vs if v != expected]
    return errors


def check_tag(tag: str, root: Path = ROOT) -> list[str]:
    expected = versions(root)["pyproject.toml"][0]
    return [] if tag == f"v{expected}" else [f"태그 {tag} != v{expected} (pyproject)"]


def check_sdist(path: Path) -> list[str]:
    with tarfile.open(path) as tf:
        names = tf.getnames()
    tops = {n.split("/", 2)[1] for n in names if n.count("/") >= 1 and n.split("/", 2)[1]}
    return [f"sdist 에 허용 밖 항목: {t}" for t in sorted(tops - SDIST_ALLOWED)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", help="푸시된 태그 이름 (예: v0.9.0)")
    ap.add_argument("--sdist", type=Path, help="빌드된 sdist .tar.gz")
    a = ap.parse_args(argv)
    errors = check_versions()
    if a.tag:
        errors += check_tag(a.tag)
    if a.sdist:
        errors += check_sdist(a.sdist)
    for e in errors:
        print(f"::error::{e}" if "GITHUB_ACTIONS" in os.environ else e)
    if not errors:
        print(f"release check ok: {versions()['pyproject.toml'][0]}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
