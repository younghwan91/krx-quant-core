"""긴(long) 가격 프레임 → ``code × date`` 패널 변환 헬퍼.

swing-it ``engine/panels.py`` 에서 시장 공통인 것만 옮겼다(수치 동일):
``panel_pivot``·``forward_returns``·``lookup_panel``·``adv_panel``.

키움은 종가를 **부호 붙여** 저장한다(부호 = 그날 등락 방향). 그래서 피벗할 때 ``abs()``
로 가격 수준을 되찾는다 — 이 관례를 레포마다 따로 기억하게 두면 한 곳이 빠뜨린다.

옮기지 않은 것: ``yoy_panels``·``resolve_signal`` (PEAD 실적 패널 전용),
``PanelCache``·``price_arrays`` (세션 전역 캐시 상태 — 소비자 프로세스가 소유해야 한다).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["adv_panel", "forward_returns", "lookup_panel", "panel_pivot"]


def panel_pivot(prices: pd.DataFrame, value: str) -> pd.DataFrame:
    """긴 가격 프레임을 code × date 패널로 피벗(abs — 키움 종가는 부호가 있다)."""
    return prices.pivot_table(index="code", columns="date", values=value, aggfunc="first").abs()


def forward_returns(df: pd.DataFrame, base_date: str, eval_date: str) -> pd.Series:
    """종목별 ``base_date`` 종가 → ``eval_date`` 종가 수익률 (이름 ``fwd_ret``).

    키움 종가의 부호를 떼고(abs) 가격 수준으로 계산한다.
    """
    piv = df.pivot_table(index="code", columns="date", values="close", aggfunc="first").abs()
    return (piv[eval_date] / piv[base_date] - 1.0).rename("fwd_ret")


def lookup_panel(panel: pd.DataFrame, value: str, codes, dates) -> np.ndarray:
    """긴 code/date/value 프레임 → ``codes × dates`` numpy 배열(없는 칸 NaN, abs 안 함)."""
    return (
        panel.pivot_table(index="code", columns="date", values=value, aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )


def adv_panel(prices: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """직전 ``window`` 일 평균 거래대금 → 긴 code/date/adv 프레임(as-of, 당일 포함).

    ``min_periods=window`` 라 첫 ``window-1`` 행은 버린다.
    """
    tv = prices[["code", "date", "trade_value"]].copy()
    tv["trade_value"] = tv["trade_value"].abs()
    tv = tv.sort_values(["code", "date"])
    tv["adv"] = tv.groupby("code")["trade_value"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    return tv.dropna(subset=["adv"])[["code", "date", "adv"]].reset_index(drop=True)
