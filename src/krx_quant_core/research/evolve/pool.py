"""작업자 풀 — spawn, 자식 스레드 1개, 과제 순서 집계.

swing-it 진화 실행의 교훈 셋을 기본값으로 박았다.

- **과제 순서로 집계한다**(끝난 순서 금지). 재시작 풀을 작업자가 끝난 순서로 쌓아 같은 후보가
  실행마다 달랐다(검증 MUST-FIX 2). :meth:`EvoPool.map` 은 늘 과제 순서로 돌려준다.
- **자식은 스레드 1개.** 12 작업자 × numba/BLAS 스레드 과구독으로 FW 실행을 재시작했다. 열 때
  :data:`THREAD_ENV` 를 1 로 두고 **작업자를 모두 띄운 뒤**(장벽으로 강제 — spawn 자식은 시작 때
  환경을 물려받는다) 바로 되돌린다. 부모 환경은 그 짧은 순간만 바뀐다(부모 numba 가 스레드 풀을
  띄운 뒤 환경이 바뀐 채 컴파일하면 numba 가 죽는다).
- **한 풀에 여러 후보 × 시드 집합.** 초기화(memmap 캐시·JIT)는 작업자마다 한 번. numba 커널은
  ``cache=True`` 로 컴파일 결과를 디스크에 두면 다음 실행부터 JIT 가 사라진다.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from krx_quant_core.research.sweep import _default_n_jobs

__all__ = ["THREAD_ENV", "EvoPool"]

THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


_BARRIER: Any = None


def _worker_init(barrier: Any, init: Callable[..., Any] | None, initargs: tuple) -> None:
    global _BARRIER
    _BARRIER = barrier
    if init is not None:
        init(*initargs)


def _hold(timeout: float) -> int:
    """작업자 모두가 동시에 과제를 잡아야 풀리는 장벽 — 풀이 작업자를 끝까지 띄우게 한다."""
    _BARRIER.wait(timeout)
    return os.getpid()


class EvoPool:
    """``with EvoPool(12, init, (cache,)) as pool: results = pool.map(task, tasks)``."""

    def __init__(self, workers: int | None = None, initializer: Callable[..., Any] | None = None,
                 initargs: Sequence[Any] = (), *, single_thread: bool = True,
                 spawn_timeout: float = 300.0):
        """``workers`` 기본 = ``run_sweep`` 과 같은 상한: min(CPU − 2, ``KQC_MAX_JOBS``(기본 14)).

        simnode 는 실매매 데몬이 같이 돈다."""
        if workers is None:
            workers = _default_n_jobs()
        if workers < 1:
            raise ValueError("workers ≥ 1")
        self.workers = int(workers)
        self.initializer = initializer
        self.initargs = tuple(initargs)
        self.single_thread = single_thread
        self.spawn_timeout = float(spawn_timeout)
        self._ex: cf.ProcessPoolExecutor | None = None
        self.pids: list[int] = []

    def __enter__(self) -> EvoPool:
        ctx = mp.get_context("spawn")
        saved = {k: os.environ.get(k) for k in THREAD_ENV}
        if self.single_thread:
            for k in THREAD_ENV:
                os.environ[k] = "1"
        try:
            barrier = ctx.Barrier(self.workers)
            self._ex = cf.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=ctx, initializer=_worker_init,
                initargs=(barrier, self.initializer, self.initargs))
            futs = [self._ex.submit(_hold, self.spawn_timeout) for _ in range(self.workers)]
            self.pids = sorted(f.result() for f in futs)
        except BaseException:
            if self._ex is not None:
                self._ex.shutdown(wait=False, cancel_futures=True)
                self._ex = None
            raise
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        if len(set(self.pids)) != self.workers:
            raise RuntimeError(f"작업자 {self.workers} 중 {len(set(self.pids))} 만 떴다")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=exc[0] is not None)
        self._ex = None

    def map(self, fn: Callable[[Any], Any], tasks: Iterable[Any],
            progress: Callable[[int, int], None] | None = None) -> list[Any]:
        """``fn(task)`` 를 병렬로 — 결과는 **과제 순서**.

        ``progress(끝난 수, 전체)`` 는 과제가 끝날 때마다 부른다."""
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
