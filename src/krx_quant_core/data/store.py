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
import shutil
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
    """``{root}/{dataset}/v{version}/{key}/`` — ``meta.json`` + 세대 디렉터리 ``g{n}/`` 의 열 파일.

    ``version`` 은 스키마(열 정의·정렬·자료형)를 바꿀 때 올린다 — 옛 캐시를 지우지 않아도 새
    디렉터리를 쓴다. ``write(replace=True)`` 는 새 세대에 전부 쓴 뒤 ``meta.json`` 을 원자적으로
    바꿔 가리킨다 — 다른 프로세스가 읽는 중이어도 옛 열과 새 열이 섞이지 않는다(직전 세대 하나는
    남겨 둬 이미 메타를 읽은 독자가 파일을 연다).
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
        meta = self.meta(key)
        if meta is None:
            raise KeyError(f"{self.dataset}/{key} is not cached")
        d = self.path(key) / f"g{meta.get('gen', 0)}"
        mode = "r" if mmap else None
        return {
            c: np.load(d / f"{_safe(c)}.npy", mmap_mode=mode, allow_pickle=False) for c in columns
        }

    @contextlib.contextmanager
    def lock(self, key: str) -> Iterator[None]:
        """키 단위 배타 잠금(프로세스 간, ``flock``). 캐시 미스 때 "다시 확인 → 받기 → 쓰기" 를
        감싸면 16 워커가 같은 날을 동시에 DB 에서 받지 않는다. 안에서는 ``write(locked=True)``.
        """
        d = self.path(key)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / ".lock", "a+b") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def write(
        self,
        key: str,
        arrays: Mapping[str, NDArray[Any]],
        *,
        aux: Mapping[str, NDArray[Any]] | None = None,
        info: Mapping[str, Any] | None = None,
        replace: bool = False,
        locked: bool = False,
    ) -> None:
        """열들을 저장.

        ``arrays`` 는 행 열 — 길이가 모두 같아야 한다. ``aux`` 는 길이가 다른 보조 배열(종목 색인
        등), ``info`` 는 ``meta.json`` 의 ``info`` 에 합쳐진다.

        - ``replace=False``: 지금 세대에 열을 **덧붙인다**. 이미 있는 행 수와 다르면 ``ValueError``
          (행 정렬이 같다는 최소 확인).
        - ``replace=True``: 새 세대에 전부 쓰고 메타를 바꿔 가리킨다(행 수가 달라도 된다 — 새 날짜가
          붙은 연도 캐시). 옛 열·info 는 따라오지 않는다.
        - ``locked=True``: 호출부가 이미 :meth:`lock` 을 잡고 있다(같은 프로세스에서 ``flock`` 을
          다시 잡으면 교착).
        """
        lens = {len(a) for a in arrays.values()}
        if len(lens) > 1:
            raise ValueError(
                f"columns differ in length: { {k: len(v) for k, v in arrays.items()} }"
            )
        ctx = contextlib.nullcontext() if locked else self.lock(key)
        with ctx:
            self._write(key, arrays, aux or {}, info or {}, lens.pop() if lens else None, replace)

    def _write(
        self,
        key: str,
        arrays: Mapping[str, NDArray[Any]],
        aux: Mapping[str, NDArray[Any]],
        info: Mapping[str, Any],
        n_new: int | None,
        replace: bool,
    ) -> None:
        base = self.path(key)
        old = self.meta(key)
        if replace or old is None:
            gen = (old.get("gen", 0) + 1) if old else 0
            meta: dict[str, Any] = {"gen": gen, "columns": {}, "aux": {}, "info": {}, "rows": None}
        else:
            meta = old
            gen = meta.get("gen", 0)
            if meta.get("rows") is not None and n_new is not None and meta["rows"] != n_new:
                raise ValueError(f"{self.dataset}/{key}: {n_new} rows != stored {meta['rows']}")
        d = base / f"g{gen}"
        d.mkdir(parents=True, exist_ok=True)
        for kind, group in (("columns", arrays), ("aux", aux)):
            for name, arr in group.items():
                meta[kind][name] = self._save(d, name, arr)
        if n_new is not None:
            meta["rows"] = n_new
        meta["info"].update(info)
        fd, tmp = tempfile.mkstemp(dir=base, prefix=".meta.", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, default=str)
        os.replace(tmp, base / "meta.json")
        # 두 세대 전 것은 지운다(직전 세대는 메타를 막 읽은 독자를 위해 남긴다).
        for p in base.glob("g*"):
            if p.is_dir() and p.name[1:].isdigit() and int(p.name[1:]) < gen - 1:
                shutil.rmtree(p, ignore_errors=True)

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
