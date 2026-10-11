"""진화 엔진 — 탐색 공간·BIPOP 최적화기·세대 프로토콜·작업자 풀·일반 커널.

swing-it 패턴 GA 메타 진화(``research/logs/sepa_explore/META_EVOLUTION.md``)에서 일반화했다.
전략(표현형 canonical·적합도)은 소비 레포 콜백이고, 코어는 계산 기계와 절차를 낸다.

.. code-block:: python

    from krx_quant_core.research.evolve import Space, Real, Choice, Ordinal, bipop

    space = Space([Real("rs_min", 0, 90, off="lo"), Choice("entry", ("bottom", "horiz")),
                   Ordinal("theta", (0.06, 0.08, 0.10))]).override(menu)
    res = bipop(evaluate_batch, space, budget=10_000, seed=0, x0=x0, canonical=canon)

최적화기 백엔드는 cmaes 가 필요하다(``pip install "krx-quant-core[evo]"``). ``kernels`` 는 numba
선택.
"""

from krx_quant_core.research.evolve.kernels import (
    block_boot,
    block_boot_lower,
    h32,
    monthly_block_lower,
    monthly_sums,
    percentile_linear,
)
from krx_quant_core.research.evolve.optim import BACKENDS, BipopResult, FastCMAwM, bipop
from krx_quant_core.research.evolve.pool import EvoPool
from krx_quant_core.research.evolve.protocol import (
    EvoLedger,
    RaceResult,
    Seal,
    SealBreach,
    digest,
    distinguish,
    paired_diff,
    provenance,
    race,
    summarize,
)
from krx_quant_core.research.evolve.space import Choice, Ordinal, Real, Space

__all__ = [
    "BACKENDS",
    "BipopResult",
    "Choice",
    "EvoLedger",
    "EvoPool",
    "FastCMAwM",
    "Ordinal",
    "RaceResult",
    "Real",
    "Seal",
    "SealBreach",
    "Space",
    "bipop",
    "block_boot",
    "block_boot_lower",
    "digest",
    "distinguish",
    "h32",
    "monthly_block_lower",
    "monthly_sums",
    "paired_diff",
    "percentile_linear",
    "provenance",
    "race",
    "summarize",
]
