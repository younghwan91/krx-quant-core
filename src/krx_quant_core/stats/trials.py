"""다중검정 원장 — 한 알파에 **실제로 시도한 config 수**를 세는 장치.

swing-it ``diagnostics/trials.py`` 에서 옮겼다. 원본은 원장 위치를 레포 루트
기준 ``research/logs`` 로 추정했는데(``_repo_root()``), 설치된 패키지 안에서는 그
추정이 site-packages 를 가리키게 된다. 그래서 **``logs_dir`` 을 필수 인자로 바꿨다** —
원장이 어느 레포의 어느 폴더에 쌓이는지는 소비자가 정한다. 나머지 동작(지문·중복
제거·손상 줄 무시·JSONL 형식)은 한 글자도 바꾸지 않았다.

Deflated Sharpe / t-haircut 은 시행 수 ``N`` 을 입력으로 받는다. 그 N 을 사람이 손으로
적으면 규율이 아니라 부탁이 된다 — 첫 config 가 죽고 조용히 두 번째를 돌려도 아무도
못 잡는다. 그래서 게이트를 통과시킬 때마다 config 를 원장에 적고, N 은 원장에서 읽는다.

- **config 지문으로 중복 제거.** 같은 config 를 재실행해도 N 이 늘지 않는다 —
  시행은 "다르게 시도한 횟수"지 "실행 횟수"가 아니다.
- **append-only JSONL.** 사람이 읽을 수 있고, git 이력이 증인이 된다.

원장 파일: ``<logs_dir>/<label>/TRIALS.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "LEDGER_NAME",
    "config_fingerprint",
    "count_trials",
    "ledger_path",
    "read_trials",
    "record_trial",
]

LEDGER_NAME = "TRIALS.jsonl"


def ledger_path(label: str, *, logs_dir: Path | str) -> Path:
    """``<logs_dir>/<label>/TRIALS.jsonl`` 경로 (파일이 없어도 경로만 반환)."""
    return Path(logs_dir) / label / LEDGER_NAME


def config_fingerprint(config: dict) -> str:
    """config 의 안정적 지문(sha1 앞 16자). 키 순서에 흔들리지 않게 정규화한다."""
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def record_trial(label: str, config: dict, *, logs_dir: Path | str,
                 note: str | None = None) -> str:
    """config 를 원장에 append 하고 지문을 반환. 같은 지문이 이미 있으면 다시 안 적는다."""
    fp = config_fingerprint(config)
    path = ledger_path(label, logs_dir=logs_dir)
    if fp in _fingerprints(path):
        return fp
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {"fingerprint": fp, "config": config}
    if note:
        row["note"] = note
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return fp


def _fingerprints(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.add(json.loads(line)["fingerprint"])
        except (json.JSONDecodeError, KeyError):
            continue  # 손상된 줄은 세지 않는다(원장을 못 읽는 게 더 나쁘다)
    return out


def count_trials(label: str, *, logs_dir: Path | str) -> int:
    """이 알파에 기록된 **서로 다른** config 수. 원장이 없으면 0."""
    return len(_fingerprints(ledger_path(label, logs_dir=logs_dir)))


def read_trials(label: str, *, logs_dir: Path | str) -> list[dict[str, Any]]:
    """원장 전체를 순서대로 반환(사람이 읽거나 문서에 싣기 위한 것)."""
    path = ledger_path(label, logs_dir=logs_dir)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
