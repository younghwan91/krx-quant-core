"""runtime.pins — 소비 레포의 krx-quant-core 핀 읽기."""

from __future__ import annotations

from pathlib import Path

from krx_quant_core.runtime import cli, pins


def _repo(root: Path, name: str, req: str | None, locked: str | None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    deps = [f'"{req}"'] if req else []
    d.joinpath("pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = [' + ", ".join(deps) + "]\n"
    )
    if locked:
        d.joinpath("uv.lock").write_text(
            'version = 1\n\n[[package]]\nname = "numpy"\nversion = "2.0.0"\n\n'
            f'[[package]]\nname = "krx-quant-core"\nversion = "{locked}"\n'
        )
    return d


def test_parse_version_numeric_prefix():
    assert pins.parse_version("0.5.1") == (0, 5, 1)
    assert pins.parse_version("0.6.0rc1") == (0, 6, 0)
    assert pins.parse_version("0.5.1") < pins.parse_version("0.6.0")


def test_read_pin_reads_requirement_and_lock(tmp_path):
    d = _repo(tmp_path, "scalp-it", "krx-quant-core==0.5.1", "0.5.1")
    p = pins.read_pin(d)
    assert p == pins.Pin(repo="scalp-it", requirement="==0.5.1", locked="0.5.1")
    assert p.behind("0.6.0") is True
    assert p.behind("0.5.1") is False


def test_read_pin_handles_extras_and_missing(tmp_path):
    d = _repo(tmp_path, "swing-it", "krx-quant-core[fast]>=0.4.1", None)
    p = pins.read_pin(d)
    assert p.requirement == ">=0.4.1"
    assert p.locked is None
    assert p.behind("0.6.0") is False  # 범위 핀 + lock 없음 → 알 수 없음

    e = _repo(tmp_path, "quantbox", None, None)
    assert pins.read_pin(e) == pins.Pin(repo="quantbox", requirement=None, locked=None)


def test_read_pins_skips_missing_dirs_and_cli_exit_code(tmp_path, capsys):
    _repo(tmp_path, "scalp-it", "krx-quant-core==0.5.1", "0.5.1")
    _repo(tmp_path, "daytrade-it", "krx-quant-core==0.5.0", "0.4.1")  # lock 이 핀과 다름
    got = pins.read_pins(tmp_path)  # swing-it 없음 → 둘만
    assert [p.repo for p in got] == ["scalp-it", "daytrade-it"]

    text, behind = pins.format_table(got, current="0.5.1")
    assert behind == 1
    assert "lock != pin" in text and "BEHIND 0.5.1" in text

    rc = cli.main(["pins", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "scalp-it" in out and "daytrade-it" in out
    assert rc == 1  # 현재 버전(0.5.1 이상)보다 뒤처진 레포가 있다

    rc = cli.main(["pins", "--root", str(tmp_path), "scalp-it"])
    assert rc == (1 if pins.parse_version(pins.__version__) > (0, 5, 1) else 0)
