"""README Quick start 예제가 실제로 그 값을 내는지 — 첫 화면 숫자가 코드와 어긋나지 않게.

README(한·영)의 첫 ``python`` 블록에서 ``식  # 기대값[ — 설명]`` 줄마다 식을 평가해
``repr`` 이 기대값과 같은지 본다. 기대값을 고치지 않고 함수 출력이 바뀌면 여기서 걸린다.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READMES = [p for p in (ROOT / "README.md", ROOT / "README.en.md") if p.is_file()]


def _cases(path: Path) -> list[tuple[str, str, str]]:
    block = re.search(r"```python\n(.*?)```", path.read_text(), re.S)
    assert block, f"{path.name}: python 블록 없음"
    setup, cases = [], []
    for line in block.group(1).splitlines():
        if "#" not in line:
            setup.append(line)
            continue
        expr, expected = (s.strip() for s in line.split("#", 1))
        expected = expected.split(" — ")[0].strip()
        cases.append(("\n".join(setup), expr, expected))
    return cases


PARAMS = [pytest.param(*case, id=f"{p.name}:{case[1]}") for p in READMES for case in _cases(p)]


def test_readmes_have_cases():
    assert READMES and PARAMS


@pytest.mark.parametrize("setup, expr, expected", PARAMS)
def test_readme_quickstart_value(setup, expr, expected):
    ns: dict = {}
    exec(setup, ns)
    assert repr(eval(expr, ns)) == expected
