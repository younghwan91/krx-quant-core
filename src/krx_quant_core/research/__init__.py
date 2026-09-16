"""파라미터 스윕 엔진 — simnode 에서 여러 config 를 병렬로 돌리고 시행 원장에 모두 적는다.

:mod:`.sweep` 은 격자·임의 config 리스트를, :mod:`.optuna` 는 optuna TPE 탐색을 돈다.
둘 다 :func:`krx_quant_core.runtime.start_run` 한 번으로 스윕 전체를 감싸고, 시도한
**모든** config 를 :mod:`krx_quant_core.stats.trials` 원장에 적는다 — DSR 의 N 이
사람이 아니라 실행 자체에서 나온다.
"""

from krx_quant_core.research.optuna import optuna_search
from krx_quant_core.research.sweep import SweepResult, grid, run_sweep

__all__ = ["SweepResult", "grid", "optuna_search", "run_sweep"]
