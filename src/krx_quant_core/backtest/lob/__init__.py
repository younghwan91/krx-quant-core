"""초 격자 호가 리플레이 — 틱·호가 → 초 격자 특징·경로, 에피소드 시뮬레이터, 무작위 대조군.

scalp-it 80·81·82번 연구 스크립트에 흩어져 있던 엔진을 모았다. 커널은 numba 로 가속한다
(``pip install "krx-quant-core[fast]"``). numba 가 없으면 같은 커널을 파이썬으로 돌린다 —
느리지만 숫자는 같다. 지금 어느 쪽인지는 :data:`HAVE_NUMBA`.

``krx_quant_core.backtest`` 를 import 해도 이 패키지는 따라 올라오지 않는다(numba import
비용) — ``from krx_quant_core.backtest.lob import ...`` 로 직접 부른다.
"""

from ._jit import HAVE_NUMBA
from .book import BookGrid, LevelTrades, build_book_grid, build_level_trades
from .control import DecisionBook, random_entry_control
from .features import (
    INPUT_COLUMNS,
    STEP_COLUMNS,
    SecondFeatureStream,
    compute_features,
    forward_labels,
)
from .features_v2 import (
    FEATURE_SET_V2,
    FEATURE_SET_V2_EXCLUDED,
    INPUT_COLUMNS_V2,
    STEP_COLUMNS_V2,
    SecondFeatureStreamV2,
    aggregate_seconds_v2,
    compute_features_v2,
    cross_section_ranks_v2,
    new_state_v2,
    step_v2,
)
from .grid import (
    FEATURE_COLUMNS,
    PATH_KEYS,
    SESSION_END_SEC,
    SESSION_START_SEC,
    aggregate_seconds,
    build_second_grid,
)
from .queue import (
    QUEUE_MODELS,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_NOT_PLACED,
    STATUS_PARTIAL,
    LimitFillResult,
    QueueModel,
    simulate_limit_orders,
)
from .sim import (
    EXIT_NONE,
    EXIT_STOP,
    EXIT_STRENGTH,
    EXIT_TAKE,
    EXIT_TIME,
    SimResult,
    simulate_exits,
)
from .ticks import TickTable, stock_tick_table

__all__ = [
    "BookGrid",
    "DecisionBook",
    "EXIT_NONE",
    "EXIT_STOP",
    "EXIT_STRENGTH",
    "EXIT_TAKE",
    "EXIT_TIME",
    "FEATURE_COLUMNS",
    "FEATURE_SET_V2",
    "FEATURE_SET_V2_EXCLUDED",
    "HAVE_NUMBA",
    "INPUT_COLUMNS",
    "INPUT_COLUMNS_V2",
    "LevelTrades",
    "LimitFillResult",
    "PATH_KEYS",
    "QUEUE_MODELS",
    "QueueModel",
    "SESSION_END_SEC",
    "SESSION_START_SEC",
    "STATUS_CANCELED",
    "STATUS_FILLED",
    "STATUS_NOT_PLACED",
    "STATUS_PARTIAL",
    "STEP_COLUMNS",
    "STEP_COLUMNS_V2",
    "SecondFeatureStream",
    "SecondFeatureStreamV2",
    "SimResult",
    "TickTable",
    "aggregate_seconds",
    "aggregate_seconds_v2",
    "build_book_grid",
    "build_level_trades",
    "build_second_grid",
    "compute_features",
    "compute_features_v2",
    "cross_section_ranks_v2",
    "forward_labels",
    "new_state_v2",
    "random_entry_control",
    "simulate_exits",
    "simulate_limit_orders",
    "step_v2",
    "stock_tick_table",
]
