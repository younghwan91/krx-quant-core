"""research.optuna_search — optuna TPE 로 config 공간을 탐색하고 모두 시행 원장에 적는다.

``optuna`` 는 선택 의존성이다(``pip install "krx-quant-core[opt]"``) — 설치 안 해도
나머지 패키지는 그대로 쓸 수 있어야 해서 이 모듈 최상위에서 import 하지 않는다.

.. code-block:: python

    def space(trial):
        return {"thr": trial.suggest_float("thr", 0.5, 0.9)}

    result = optuna_search(objective, space, label="scalp84-flow", repo_root=ROOT,
                            data=DataSpec("2026-08-24", "2026-09-07", "train"), n_trials=50)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from krx_quant_core.research.sweep import SweepResult
from krx_quant_core.runtime.gitstate import resolve_trials_dir
from krx_quant_core.runtime.runs import DataSpec, start_run
from krx_quant_core.stats.trials import count_trials, record_trial

__all__ = ["optuna_search"]


def optuna_search(
    objective: Callable[[dict], dict],
    space: Callable[[Any], dict],
    *,
    label: str,
    repo_root: Path | str,
    data: DataSpec,
    n_trials: int,
    n_jobs: int = 1,
    direction: str = "maximize",
    metric: str = "score",
    seed: int = 0,
    allow_dirty: bool = False,
) -> SweepResult:
    """``space(trial) -> config`` 로 TPE 탐색. 매 trial 을 시행 원장에 적는다(중복 제거는 원장 몫).

    ``space`` 가 뽑은 config 로 ``objective(cfg)`` 를 돌리고 ``objective(cfg)[metric]`` 을
    optuna 가 최적화할 값으로 준다. ``TPESampler(seed=seed)`` 로 재현성을 고정한다.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError('pip install "krx-quant-core[opt]"') from exc

    repo_root = Path(repo_root)
    logs_dir = resolve_trials_dir(repo_root, None)
    rows: list[dict[str, Any]] = []
    config_keys: list[str] = []

    def _objective(trial: Any) -> float:
        cfg = space(trial)
        if not config_keys:
            config_keys.extend(cfg.keys())
        record_trial(label, cfg, logs_dir=logs_dir)
        metrics = objective(cfg)
        rows.append({**cfg, **metrics})
        return metrics[metric]

    sweep_config = {
        "n_trials": n_trials,
        "direction": direction,
        "metric": metric,
        "seed": seed,
    }
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    with start_run(
        label,
        sweep_config,
        repo_root=repo_root,
        data=data,
        seed=seed,
        allow_dirty=allow_dirty,
    ) as run:
        study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs)
        n_trials_recorded = count_trials(label, logs_dir=logs_dir)
        run.log_result({"n": len(rows), "n_trials": n_trials_recorded})

    frame = pd.DataFrame(rows)
    return SweepResult(
        frame=frame, n_trials=n_trials_recorded, label=label, config_keys=tuple(config_keys)
    )
