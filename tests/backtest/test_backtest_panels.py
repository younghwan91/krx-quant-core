"""backtest.panels — swing-it ``tests/test_engine_panels.py`` 중 옮긴 함수 부분 이식."""

from __future__ import annotations

import numpy as np
import pandas as pd

from krx_quant_core.backtest.panels import adv_panel, forward_returns, lookup_panel, panel_pivot


def _prices() -> pd.DataFrame:
    dates = [f"2020-01-{d:02d}" for d in range(1, 8)]
    rows = []
    for code, base in (("A", 100.0), ("B", 200.0)):
        for k, d in enumerate(dates):
            # 부호 붙은 종가(키움 관례) — panel_pivot 이 abs 해야 한다.
            rows.append({"code": code, "date": d, "close": -(base + k),
                         "trade_value": (k + 1) * 1000.0})
    return pd.DataFrame(rows)


def test_panel_pivot_abs_and_shape():
    panel = panel_pivot(_prices(), "close")
    assert panel.shape == (2, 7)
    assert (panel.to_numpy() >= 0).all()
    assert panel.at["A", "2020-01-01"] == 100.0


def test_lookup_panel_reindexes():
    prices = _prices()
    piv = prices.assign(close=prices["close"].abs())
    codes, dates = ["A", "B", "Z"], ["2020-01-01", "2020-01-03", "2020-01-99"]
    arr = lookup_panel(piv, "close", codes, dates)
    assert arr.shape == (3, 3)
    assert arr[0, 0] == 100.0
    assert np.isnan(arr[2, 0])
    assert np.isnan(arr[0, 2])


def test_adv_panel_trailing_mean():
    out = adv_panel(_prices(), window=3)
    assert list(out.columns) == ["code", "date", "adv"]
    a = out[(out["code"] == "A") & (out["date"] == "2020-01-03")]["adv"].iloc[0]
    assert abs(a - 2000.0) < 1e-9
    assert len(out) == 2 * (7 - 2)  # 코드마다 첫 2행 드롭


def test_forward_returns_uses_abs_close():
    fr = forward_returns(_prices(), "2020-01-01", "2020-01-03")
    assert fr.name == "fwd_ret"
    assert abs(fr["A"] - (102.0 / 100.0 - 1)) < 1e-12
    assert abs(fr["B"] - (202.0 / 200.0 - 1)) < 1e-12
