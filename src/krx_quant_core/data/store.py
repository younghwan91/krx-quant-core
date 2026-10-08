"""열 단위 디스크 캐시 — ``.npy`` 파일 하나가 열 하나, 읽기는 ``mmap``.

scalp-it 은 날짜별 피클(``~/of80`` 17GB)을 통째로 언피클한 뒤 몇 열만 쓴다. 여기서는

- 열마다 ``{col}.npy`` 를 따로 둔다 → 필요한 열만 디스크에서 올라온다.
- ``np.load(mmap_mode="r")`` 로 연다 → 같은 날을 16 워커가 읽어도 OS 페이지 캐시 한 벌을 공유하고,
  프로세스 힙에 복사본이 생기지 않는다. 배열은 **읽기 전용**이다(공유 캐시를 실수로 고치지 못한다).
- 쓰기는 임시 파일 → ``os.replace`` 로 원자적이다. 같은 키에 열을 덧붙이는 동시 쓰기는 ``flock``
  으로 줄 세운다(``meta.json`` 갱신 경합 방지).

의존성은 numpy 뿐이다(소비 레포 둘에 pyarrow 가 없다). 위치는 ``$KQC_DATA`` 또는
``~/.kqc/data``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["ColumnStore", "default_root"]

ROOT_ENV = "KQC_DATA"


def default_root() -> Path:
    """``$KQC_DATA`` 또는 ``~/.kqc/data``."""
    return Path(os.environ.get(ROOT_ENV, "~/.kqc/data")).expanduser()


def _safe(part: str) -> str:
    if not part or "/" in part or part.startswith(".") or "\0" in part:
        raise ValueError(f"bad store path component: {part!r}")
    return part


class ColumnStore:
    """``{root}/{dataset}/v{version}/{key}/`` 아래 열 파일과 ``meta.json``.

    ``version`` 은 스키마(열 정의·정렬·자료형)를 바꿀 때 올린다 — 옛 캐시를 지우지 않아도 새
    디렉터리를 쓴다.
    """

    def __init__(self, dataset: str, *, version: int = 1, root: Path | str | None = None) -> None:
        self.dataset = _safe(dataset)
        self.version = int(version)
        self.root = Path(root).expanduser() if root is not None else default_root()

    def path(self, key: str) -> Path:
        return self.root / self.dataset / f"v{self.version}" / _safe(key)

    def meta(self, key: str) -> dict[str, Any] | None:
        """키의 메타데이터(없으면 ``None``)."""
        try:
            return json.loads((self.path(key) / "meta.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError):
            return None

    def columns(self, key: str) -> set[str]:
        """키에 저장된 열 이름."""
        m = self.meta(key)
        return set(m.get("columns", {})) | set(m.get("aux", {})) if m else set()

    def has(self, key: str, columns: Iterable[str]) -> bool:
        return set(columns) <= self.columns(key)

    def read(
        self, key: str, columns: Iterable[str], *, mmap: bool = True
    ) -> dict[str, NDArray[Any]]:
        """열 배열들. ``mmap`` 이면 읽기 전용 메모리 맵(복사 없음), 아니면 힙에 읽는다."""
        d = self.path(key)
        mode = "r" if mmap else None
        return {
            c: np.load(d / f"{_safe(c)}.npy", mmap_mode=mode, allow_pickle=False) for c in columns
        }

    @contextlib.contextmanager
    def _locked(self, key: str) -> Iterator[Path]:
        d = self.path(key)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / ".lock", "a+b") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield d
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def write(
        self,
        key: str,
        arrays: Mapping[str, NDArray[Any]],
        *,
        aux: Mapping[str, NDArray[Any]] | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """열들을 저장(이미 있는 열은 덮어쓴다).

        ``arrays`` 는 행 열 — 길이가 모두 같아야 하고 키에 이미 행 열이 있으면 그 행 수와도 같아야
        한다(행 정렬이 같다는 최소 확인). ``aux`` 는 길이가 다른 보조 배열(종목 색인 등).
        ``info`` 는 ``meta.json`` 의 ``info`` 에 합쳐진다.
        """
        lens = {len(a) for a in arrays.values()}
        if len(lens) > 1:
            raise ValueError(
                f"columns differ in length: { {k: len(v) for k, v in arrays.items()} }"
            )
        with self._locked(key) as d:
            meta = self.meta(key) or {}
            meta.setdefault("columns", {})
            meta.setdefault("aux", {})
            meta.setdefault("info", {})
            n_old = meta.get("rows")
            n_new = lens.pop() if lens else n_old
            if n_old is not None and n_new is not None and n_old != n_new:
                raise ValueError(f"{self.dataset}/{key}: {n_new} rows != stored {n_old}")
            for kind, group in (("columns", arrays), ("aux", aux or {})):
                for name, arr in group.items():
                    meta[kind][name] = self._save(d, name, arr)
            meta["rows"] = n_new
            meta["info"].update(info or {})
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".meta.", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, default=str)
            os.replace(tmp, d / "meta.json")

    @staticmethod
    def _save(d: Path, name: str, arr: NDArray[Any]) -> str:
        a = np.ascontiguousarray(arr)
        if a.dtype == object:
            raise TypeError(f"column {name!r} is object dtype — store fixed-width types")
        fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{name}.", suffix=".npy")
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, a, allow_pickle=False)
        os.replace(tmp, d / f"{_safe(name)}.npy")
        return str(a.dtype.str)
