"""backtest.lob — numba 경로와 순수 파이썬 폴백이 같은 숫자를 내는지.

폴백은 import 시점에 정해지므로(``KRX_QUANT_CORE_DISABLE_NUMBA``) 별도 프로세스에서 돌려
결과를 파일로 받아 비교한다. numba 가 없는 환경에서는 건너뛴다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from krx_quant_core.backtest import lob

_SCRIPT = r"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[2])
from _lob_golden import synthetic_day
from krx_quant_core.backtest.lob import (
    HAVE_NUMBA, aggregate_seconds, compute_features, forward_labels, simulate_exits,
)
ticks, quotes = synthetic_day(11, n_ticks=3000, n_quotes=3000)
X, _ = aggregate_seconds(ticks, quotes)
F = compute_features(X)
lab = forward_labels(F[:, 15], F[:, 16], (5, 60), cost=0.0023)
secs = np.arange(300, 22000, 7)
hi = np.where(np.isfinite(F[:, 16]), F[:, 16] + 3, np.nan)
r = simulate_exits(secs, F[:, 15], F[:, 16], F[:, 10], strength_drop=2.0, max_hold=15,
                   stop_ticks=3, take_ticks=6, hi=hi, latency=2, cost=0.0023)
np.savez(sys.argv[1], have=HAVE_NUMBA, F=F, net5=lab["net5"], mfe60=lab["mfe60"],
         net=r.net, hold=r.hold, reason=r.reason)
"""


def _run(out: Path, *, disable: bool) -> dict[str, np.ndarray]:
    env = dict(os.environ)
    env.pop("KRX_QUANT_CORE_DISABLE_NUMBA", None)
    if disable:
        env["KRX_QUANT_CORE_DISABLE_NUMBA"] = "1"
    here = str(Path(__file__).parent)
    subprocess.run([sys.executable, "-c", _SCRIPT, str(out), here], check=True, env=env)
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.skipif(not lob.HAVE_NUMBA, reason="numba not installed ([fast] extra)")
def test_numba_and_fallback_identical(tmp_path):
    fast = _run(tmp_path / "fast.npz", disable=False)
    slow = _run(tmp_path / "slow.npz", disable=True)
    assert bool(fast["have"]) and not bool(slow["have"])
    for k in ("F", "net5", "mfe60", "net", "hold", "reason"):
        np.testing.assert_array_equal(fast[k], slow[k], err_msg=k)
    assert set(np.unique(fast["reason"])) >= {1, 2, 3, 4}
