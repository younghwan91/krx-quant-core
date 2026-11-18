"""릴리스 일관성 — 버전 네 곳이 같아야 하고, sdist 는 허용 목록만 싣는다.

v0.2.0~v0.4.2 가 ``__version__ = "0.1.1"`` 인 채로 배포됐다(``kqc pins`` 기준도 따라 틀렸다).
버전 올림 커밋에서 한 곳이라도 빠지면 여기서 막는다. 검사 본체는 ``scripts/check_release.py``
(publish.yml 이 태그·sdist 검사에 같은 것을 쓴다).
"""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release.py"

# sdist 에서 테스트를 돌리면 scripts/·uv.lock 이 없다 — 그건 레포 검사라 건너뛴다.
pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="레포 체크아웃에서만")


@pytest.fixture(scope="module")
def cr():
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repo_versions_agree(cr):
    assert cr.check_versions() == []


def test_init_matches_installed_dunder(cr):
    import krx_quant_core

    assert krx_quant_core.__version__ == cr.versions()["pyproject.toml"][0]


def _fake_repo(
    tmp_path: Path, *, py="1.2.3", init="1.2.3", readme=("1.2.3", "1.2.3"), lock="1.2.3"
):
    (tmp_path / "src/krx_quant_core").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "krx-quant-core"\nversion = "{py}"\n'
    )
    (tmp_path / "src/krx_quant_core/__init__.py").write_text(f'__version__ = "{init}"\n')
    (tmp_path / "README.md").write_text(
        f"pip install krx-quant-core=={readme[0]}\n"
        f'pip install "krx-quant-core @ git+https://github.com/x/krx-quant-core@v{readme[1]}"\n'
    )
    (tmp_path / "uv.lock").write_text(
        f'version = 1\n\n[[package]]\nname = "krx-quant-core"\nversion = "{lock}"\n\n'
        '[[package]]\nname = "numpy"\nversion = "2.0.0"\n'
    )
    return tmp_path


def test_clean_fake_repo_passes(cr, tmp_path):
    assert cr.check_versions(_fake_repo(tmp_path)) == []


@pytest.mark.parametrize(
    "kw, where",
    [
        ({"init": "0.1.1"}, "__init__"),  # v0.2.0~v0.4.2 사고 그대로
        ({"readme": ("1.2.3", "1.2.2")}, "README"),
        ({"readme": ("1.2.2", "1.2.3")}, "README"),
        ({"lock": "1.2.2"}, "uv.lock"),
    ],
)
def test_any_stale_place_is_reported(cr, tmp_path, kw, where):
    errors = cr.check_versions(_fake_repo(tmp_path, **kw))
    assert len(errors) == 1 and errors[0].startswith(where)


def test_missing_place_is_reported(cr, tmp_path):
    root = _fake_repo(tmp_path)
    (root / "README.md").write_text("no install line\n")
    assert cr.check_versions(root) == ["README.md: 버전을 못 찾음 (기대 1.2.3)"]


def test_tag_must_match(cr, tmp_path):
    root = _fake_repo(tmp_path)
    assert cr.check_tag("v1.2.3", root) == []
    assert cr.check_tag("v1.2.4", root) and cr.check_tag("1.2.3", root)


def _sdist(tmp_path: Path, members: list[str]) -> Path:
    path = tmp_path / "pkg-1.2.3.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for m in members:
            info = tarfile.TarInfo(f"pkg-1.2.3/{m}")
            info.size = 0
            tf.addfile(info, io.BytesIO(b""))
    return path


def test_sdist_allowlist(cr, tmp_path):
    ok = ["PKG-INFO", "pyproject.toml", "README.md", "LICENSE", "src/krx_quant_core/__init__.py"]
    assert cr.check_sdist(_sdist(tmp_path, ok)) == []
    bad = ok + [".claude/skills/archify/SKILL.md", "CLAUDE.md"]
    assert cr.check_sdist(_sdist(tmp_path, bad)) == [
        "sdist 에 허용 밖 항목: .claude",
        "sdist 에 허용 밖 항목: CLAUDE.md",
    ]
