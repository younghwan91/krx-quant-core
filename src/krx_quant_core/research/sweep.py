"""research.run_sweep — config 리스트를 병렬로 돌리고 모두 시행 원장에 적는다.

.. code-block:: python

    configs = grid(thr=[0.6, 0.7, 0.8], hold=[300, 600])
    result = run_sweep(objective, configs, label="scalp84-flow", repo_root=ROOT,
                        data=DataSpec("2026-08-24", "2026-09-07", "train"))
    result.best("score", returns_key="returns")

``objective`` 는 **모듈 최상위 함수**여야 한다 — ``ProcessPoolExecutor`` 로 자식 프로세스에
피클해서 보낸다(중첩 함수·람다는 피클 안 됨). 스윕 전체를 :func:`~.runtime.start_run`
한 번으로 감싸고, 그 안에서 **모든** config 를 원장에 적는다 — 몇 개를 실제로 돌렸든
DSR 의 N 은 "시도한 서로 다른 config 수"를 반영해야 하기 때문이다(사람이 세면 부탁이지
규율이 아니다).

캐시: 같은 git sha·같은 데이터 구간·같은 objective(``모듈.qualname``)·같은 config 면 이전
결과를 그대로 쓴다(재실행 방지). 코드나 데이터가 바뀌면 키가 바뀌어 다시 계산한다.

- **dirty 트리면 캐시를 통째로 끈다**(읽기·쓰기 모두). 스윕 시작 때 ``git_dirty`` 를 본다.
  커밋 안 된 변경은 sha 에 안 잡혀, 읽으면 옛 코드 결과를 새 코드 결과로 믿고 쓰면 다음
  깨끗한 실행이 그 결과를 믿는다.
- 지표는 JSON 으로 저장한다. numpy 스칼라는 ``.item()``, 배열은 ``list`` 로 재귀 변환하고
  그 밖에 JSON 으로 못 쓰는 값은 ``TypeError`` — 예전 ``default=str`` 은 배열을 문자열로
  바꿔 캐시 적중 때 조용히 다른 타입을 돌려줬다. 캐시가 켜져 있으면 첫 실행도 변환된
  지표를 돌려준다(적중·미적중 결과 모양이 같게).
- 쓰기는 임시 파일 + ``os.replace`` 로 원자적이다. 읽을 수 없는(손상된) 캐시 파일은 miss 로
  보고 다시 계산해 덮어쓴다.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from krx_quant_core.runtime.gitstate import git_dirty, git_head, resolve_trials_dir
from krx_quant_core.runtime.runs import DataSpec, start_run
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
from krx_quant_core.stats.trials import config_fingerprint, count_trials, record_trial

__all__ = ["SweepResult", "grid", "run_sweep"]

_DEFAULT_MAX_JOBS = 14


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """``axes`` 의 데카르트 곱을 config 리스트로. 키 순서는 넘긴 순서 그대로 고정."""
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def _default_n_jobs() -> int:
    cap = int(os.environ.get("KQC_MAX_JOBS", _DEFAULT_MAX_JOBS))
    return max(1, min((os.cpu_count() or 2) - 2, cap))


def _objective_id(objective: Callable[[dict], dict]) -> str:
    return f"{objective.__module__}.{objective.__qualname__}"


def _cache_key(git_sha: str, data: DataSpec, cfg: dict[str, Any], objective_id: str) -> str:
    payload = json.dumps(
        [git_sha, asdict(data), config_fingerprint(cfg), objective_id],
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any, path: str = "metrics") -> Any:
    """지표를 JSON 기본형으로 재귀 변환한다. 못 바꾸는 타입은 ``TypeError``(경로 포함)."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), path)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), path)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{path}: JSON 캐시 키는 str 이어야 한다: {k!r}")
            out[k] = _jsonable(v, f"{path}[{k!r}]")
        return out
    if isinstance(value, list | tuple):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(value)]
    raise TypeError(f"{path}: JSON 캐시에 쓸 수 없는 타입 {type(value).__name__}")


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None  # 없음·손상 → miss
    return metrics if isinstance(metrics, dict) else None


def _write_cache(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _run_one(
    objective: Callable[[dict], dict],
    cfg: dict[str, Any],
    *,
    git_sha: str,
    data: DataSpec,
    cache_dir: Path | None,
) -> dict[str, Any]:
    """한 config 를 돈다(캐시 조회 → 없으면 objective 실행 → 캐시 기록). 행 하나를 반환.

    ``ProcessPoolExecutor`` 자식 프로세스에서 그대로 실행되므로 예외를 여기서 잡아
    ``dict`` 로 바꿔 돌려준다 — 임의 예외 객체는 피클이 안 될 수 있어 부모로 못 넘긴다.
    """
    cache_path = None
    if cache_dir is not None:
        key = _cache_key(git_sha, data, cfg, _objective_id(objective))
        cache_path = cache_dir / f"{key}.json"
        cached = _read_cache(cache_path)
        if cached is not None:
            return {**cfg, **cached}
    try:
        metrics = objective(cfg)
    except Exception as exc:  # noqa: BLE001 — 행으로 격리하고 스윕은 계속
        return {**cfg, "error": repr(exc)}
    if cache_path is not None:
        # 변환 실패(TypeError)는 행으로 삼키지 않는다 — objective 반환 모양의 버그다.
        metrics = _jsonable(metrics)
        _write_cache(cache_path, metrics)
    return {**cfg, **metrics}


@dataclass
class SweepResult:
    """스윕 결과 — config 열 + 지표 열을 가진 ``frame``, 원장에 쌓인 distinct config 수.

    ``n_trials`` = 서로 다른 개별 config 수 **+ 1**. ``start_run`` 이 스윕을 감쌀 때 자기
    자신의 요약 config(``{"sweep_n":..., "configs_fp":..., "seed":...}``)를 같은 label 의
    원장에 한 행으로 적기 때문이다(``runs.py`` 는 항상 그렇게 동작 — 바꾸지 않았다). 같은
    스윕(같은 configs·seed)을 다시 돌리면 그 요약 행은 지문이 같아 중복 집계되지 않는다.
    DSR 관점에서는 "이 config 조합을 시도해보기로 한 결정" 자체도 한 번의 시행으로 세는
    셈이라 보수적이다(N 을 부풀리는 쪽) — 깎는 쪽보다 안전하다.
    """

    frame: pd.DataFrame
    n_trials: int
    label: str
    config_keys: tuple[str, ...] = ()
    """``best()`` 가 행에서 ``"config"`` 서브딕트를 재구성할 때 쓰는 키 목록. 스윕의 첫
    config(``configs[0]``)의 키를 그대로 쓴다 — 모든 config 가 같은 키 집합이라는 가정이다
    (:func:`grid` 나 통상적 ``optuna`` ``space()`` 는 이 가정을 지킨다). config 마다 키가
    다른 이형(heterogeneous) 스윕을 직접 만드는 소비자는 이 가정이 깨질 수 있다."""

    def best(
        self, metric: str, *, returns_key: str | None = None, maximize: bool = True
    ) -> dict[str, Any]:
        """``metric`` 이 가장 좋은(기본 최대) 행. 실패 행(``error``·지표 NaN)은 후보에서 뺀다.

        ``returns_key`` 를 주면 그 열(수익 배열)로 :func:`deflated_sharpe_from_sample` 을
        ``n_trials=self.n_trials`` 로 같이 돌려준다 — 판정은 안 한다, 숫자만 붙인다.
        """
        df = self.frame
        if metric not in df.columns:
            raise KeyError(f"'{metric}' not in sweep frame columns: {list(df.columns)}")
        candidates = df[df[metric].notna()]
        if candidates.empty:
            raise ValueError(f"no successful trial has a value for metric '{metric}'")
        idx = candidates[metric].idxmax() if maximize else candidates[metric].idxmin()
        row = candidates.loc[idx]
        cfg = {k: row[k] for k in self.config_keys if k in row}
        out: dict[str, Any] = {"config": cfg, metric: row[metric], "dsr": None}
        if returns_key is not None:
            out["dsr"] = deflated_sharpe_from_sample(np.asarray(row[returns_key]), self.n_trials)
        return out


def run_sweep(
    objective: Callable[[dict], dict],
    configs: Sequence[dict[str, Any]],
    *,
    label: str,
    repo_root: Path | str,
    data: DataSpec,
    n_jobs: int | None = None,
    cache: bool = True,
    seed: int = 0,
    allow_dirty: bool = False,
    trials_dir: Path | str | None = None,
) -> SweepResult:
    """``configs`` 를 (병렬로) 돌리고 결과를 ``SweepResult`` 로 묶는다.

    ``trials_dir`` 는 :func:`~.runs.start_run` 이 쓰는 것과 **같은 경로 계산**을 쓴다
    (:func:`~.gitstate.resolve_trials_dir`) — 아니면 개별 config 원장 기록이 ``start_run``
    이 세는 폴더와 어긋나 ``count_trials`` 가 스윕 config 를 놓친다.

    반환된 ``SweepResult.n_trials`` = ``len(configs)`` 의 distinct 개수 **+ 1** —
    ``start_run`` 자신이 스윕 요약 config 를 같은 label 원장에 한 행 적기 때문이다
    (자세한 설명은 :class:`SweepResult` 참고). ``SweepResult.config_keys`` 는
    ``configs[0]`` 의 키를 그대로 쓰므로, ``best()`` 가 재구성하는 ``"config"`` 는
    모든 config 가 같은 키 집합이라는 가정 위에 있다.
    """
    repo_root = Path(repo_root)
    configs = list(configs)
    if n_jobs is None:
        n_jobs = _default_n_jobs()
    logs_dir = resolve_trials_dir(repo_root, trials_dir)
    cache_dir = None
    # dirty 트리면 캐시를 읽지도 쓰지도 않는다 — 모듈 docstring 참고.
    if cache and not git_dirty(repo_root):
        cache_dir = Path(os.environ.get("KQC_CACHE", "~/.kqc/cache")).expanduser() / label

    sweep_config = {
        "sweep_n": len(configs),
        "configs_fp": config_fingerprint({"c": configs}),
        "seed": seed,
    }
    with start_run(
        label,
        sweep_config,
        repo_root=repo_root,
        data=data,
        seed=seed,
        allow_dirty=allow_dirty,
        trials_dir=trials_dir,
    ) as run:
        for cfg in configs:
            record_trial(label, cfg, logs_dir=logs_dir)

        git_sha = git_head(repo_root)
        if n_jobs == 1 or len(configs) <= 1:
            rows = [
                _run_one(objective, cfg, git_sha=git_sha, data=data, cache_dir=cache_dir)
                for cfg in configs
            ]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = [
                    pool.submit(
                        _run_one, objective, cfg, git_sha=git_sha, data=data, cache_dir=cache_dir
                    )
                    for cfg in configs
                ]
                rows = [fut.result() for fut in futures]

        n_trials = count_trials(label, logs_dir=logs_dir)
        errors = sum(1 for row in rows if "error" in row)
        run.log_result({"n": len(rows), "errors": errors, "n_trials": n_trials})

    frame = pd.DataFrame(rows)
    config_keys = tuple(configs[0].keys()) if configs else ()
    return SweepResult(frame=frame, n_trials=n_trials, label=label, config_keys=config_keys)
