"""작업자 풀 — spawn, 자식 스레드 1개, 과제 순서 집계.

swing-it 진화 실행의 교훈 셋을 기본값으로 박았다.

- **과제 순서로 집계한다**(끝난 순서 금지). 재시작 풀을 작업자가 끝난 순서로 쌓아 같은 후보가
  실행마다 달랐다(검증 MUST-FIX 2). :meth:`EvoPool.map` 은 늘 과제 순서로 돌려준다.
- **자식은 스레드 1개.** 12 작업자 × numba/BLAS 스레드 과구독으로 FW 실행을 재시작했다. 풀이 열려
  있는 동안 ``NUMBA_NUM_THREADS``·``OMP_NUM_THREADS``·``MKL_NUM_THREADS``·``OPENBLAS_NUM_THREADS``
  를 1 로 두고(spawn 자식이 import 전에 물려받는다) 닫을 때 되돌린다.
- **한 풀에 여러 후보 × 시드 집합.** 초기화(memmap 캐시·JIT)는 작업자마다 한 번. numba 커널은
  ``cache=True`` 로 컴파일 결과를 디스크에 두면 다음 실행부터 JIT 가 사라진다.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = ["THREAD_ENV", "EvoPool"]

THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


class EvoPool:
    """``with EvoPool(12, init, (cache,)) as pool: results = pool.map(task, tasks)``."""

    def __init__(self, workers: int, initializer: Callable[..., Any] | None = None,
                 initargs: Sequence[Any] = (), *, single_thread: bool = True):
        if workers < 1:
            raise ValueError("workers ≥ 1")
        self.workers = int(workers)
        self.initializer = initializer
        self.initargs = tuple(initargs)
        self.single_thread = single_thread
        self._ex: cf.ProcessPoolExecutor | None = None
        self._saved: dict[str, str | None] = {}

    def __enter__(self) -> EvoPool:
        if self.single_thread:
            self._saved = {k: os.environ.get(k) for k in THREAD_ENV}
            for k in THREAD_ENV:
                os.environ[k] = "1"
        self._ex = cf.ProcessPoolExecutor(max_workers=self.workers,
                                          mp_context=mp.get_context("spawn"),
                                          initializer=self.initializer, initargs=self.initargs)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._ex is not None:
                self._ex.shutdown(wait=True, cancel_futures=exc[0] is not None)
        finally:
            self._ex = None
            for k, v in self._saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            self._saved = {}

    def map(self, fn: Callable[[Any], Any], tasks: Iterable[Any],
            progress: Callable[[int, int], None] | None = None) -> list[Any]:
        """``fn(task)`` 를 병렬로 — 결과는 **과제 순서**. ``progress(끝난 수, 전체)`` 는 끝날
        때마다."""
        if self._ex is None:
            raise RuntimeError("with EvoPool(...) as pool: 안에서 부른다")
        tasks = list(tasks)
        futs = {self._ex.submit(fn, t): i for i, t in enumerate(tasks)}
        out: list[Any] = [None] * len(tasks)
        for done, f in enumerate(cf.as_completed(futs), 1):
            out[futs[f]] = f.result()
            if progress is not None:
                progress(done, len(tasks))
        return out
