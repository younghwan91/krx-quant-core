"""일중·야간 수익 분해 — "이 유니버스의 일중은 구조적으로 음수인가" (scalp-it 101번 §1).

한국 주식의 수익은 대부분 **야간**(전일 종가→시가)에 나고 **일중**(시가→종가)은 잃는 구간이며,
손실은 거래대금 순위에 비례한다 — 전일 거래대금 상위 300 의 일중은 121개월 중 120개월
음수(−28bp/일), 거기에 전일 급등 상위를 더하면 −104bp/일 이었다. 롱 일중 전략은 이 역풍
위에 서 있으므로, 전략을 짜기 전에 그 자리의 드리프트부터 재야 한다.

순수 함수다. 입력은 일봉 프레임(``code``·``date``·``open``·``close``·``prev_close``), 버킷
열은 호출부가 **전일 정보**(전일 거래대금 순위 등)로 만들어 넘긴다 — 당일 값으로 버킷을
만들면 look-ahead 다. :func:`rank_buckets` 가 날짜별 분위 버킷을 만드는 도우미다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from krx_quant_core.stats.metrics import newey_west_t

__all__ = ["drift_summary", "intraday_overnight", "rank_buckets"]


def intraday_overnight(
    frame: pd.DataFrame,
    *,
    open_col: str = "open",
    close_col: str = "close",
    prev_close_col: str = "prev_close",
    code_col: str = "code",
    date_col: str = "date",
) -> pd.DataFrame:
    """일봉 → ``intraday_bp``(시가→종가)·``overnight_bp``(전일 종가→시가)·``close_to_close_bp``.

    ``prev_close`` 열이 없으면 ``code`` 별로 날짜순 ``close.shift(1)`` 을 만든다(종목 첫 행 NaN).
    시가·전일 종가가 0 이하이거나 NaN 인 행은 NaN. 원본 프레임은 건드리지 않는다.
    """
    out = frame.copy()
    if prev_close_col not in out.columns:
        out = out.sort_values([code_col, date_col], kind="stable")
        out[prev_close_col] = out.groupby(code_col, sort=False)[close_col].shift(1)
    o = out[open_col].astype(float).to_numpy()
    c = out[close_col].astype(float).to_numpy()
    pc = out[prev_close_col].astype(float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        intra = (c / o - 1.0) * 1e4
        over = (o / pc - 1.0) * 1e4
        cc = (c / pc - 1.0) * 1e4
    bad_o = ~(o > 0)
    bad_pc = ~(pc > 0)
    intra[bad_o] = np.nan
    over[bad_o | bad_pc] = np.nan
    cc[bad_pc] = np.nan
    out["intraday_bp"] = intra
    out["overnight_bp"] = over
    out["close_to_close_bp"] = cc
    return out


def _one(group: pd.DataFrame, value: str, date_col: str, hac_lag: int) -> dict[str, float]:
    x = group[value].astype(float)
    ok = x.notna()
    x = x[ok]
    if len(x) == 0:
        return {"mean_bp": np.nan, "median_bp": np.nan, "n": 0, "n_days": 0, "n_months": 0,
                "neg_month_share": np.nan, "t_hac": np.nan}
    d = pd.to_datetime(group.loc[ok, date_col])
    daily = x.groupby(d.dt.normalize().to_numpy()).mean().sort_index()
    monthly = x.groupby(d.dt.to_period("M").to_numpy()).mean()
    t = np.nan
    if len(daily) >= 3:
        _, t = newey_west_t(daily.to_numpy(), lag=min(hac_lag, len(daily) - 1))
    return {
        "mean_bp": float(x.mean()),
        "median_bp": float(x.median()),
        "n": int(len(x)),
        "n_days": int(len(daily)),
        "n_months": int(len(monthly)),
        "neg_month_share": float((monthly < 0).mean()),
        "t_hac": float(t),
    }


def drift_summary(
    frame: pd.DataFrame,
    value: str = "intraday_bp",
    *,
    by: str | Sequence[str] | None = None,
    date_col: str = "date",
    hac_lag: int = 5,
) -> pd.DataFrame:
    """``value`` 의 평균·중앙·관측 수·일수·월수·**음수 월 비율**·일별 평균열의 HAC t.

    ``by`` 를 주면 그 열(들)별 한 행, 없으면 전체 한 행(``index=["all"]``). "음수 월 비율"은
    달마다 평균을 내어 음수인 달의 비율이다 — 121개월 중 120개월 같은 표현이 그대로 나온다.
    t 는 날짜별 평균열(종목 간 공통 변동을 한 관측으로 접은 것)에 Newey-West(``hac_lag``)다.
    """
    if by is None:
        return pd.DataFrame([_one(frame, value, date_col, hac_lag)], index=pd.Index(["all"]))
    keys = [by] if isinstance(by, str) else list(by)
    rows = {}
    for k, g in frame.groupby(keys, sort=True, observed=True, dropna=True):
        rows[k if len(keys) > 1 else k[0] if isinstance(k, tuple) else k] = _one(
            g, value, date_col, hac_lag
        )
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = by if isinstance(by, str) else None
    return out


def rank_buckets(
    frame: pd.DataFrame,
    value: str,
    *,
    date_col: str = "date",
    n: int = 4,
    labels: Sequence[str] | None = None,
    ascending: bool = False,
) -> pd.Series:
    """날짜별 ``value`` 분위 버킷(0 = 가장 큰 쪽, ``ascending=False`` 기본).

    ``value`` 는 **전일** 정보(예: 전일 거래대금)여야 한다 — 당일 값이면 look-ahead 다. 그 책임은
    호출부에 있고 여기선 열 이름을 받을 뿐이다. 값이 NaN 인 행은 NaN 버킷.
    """
    def _bucket(s: pd.Series) -> pd.Series:
        r = s.rank(method="first", ascending=ascending)
        k = s.notna().sum()
        if k == 0:
            return pd.Series(np.nan, index=s.index)
        b = np.floor((r - 1) * n / k)  # r=1 → 0(상위), r=k → n−1(하위); k=1 이면 0
        return b.clip(lower=0, upper=n - 1)

    b = frame.groupby(date_col, sort=False, group_keys=False)[value].apply(_bucket)
    b = b.reindex(frame.index)
    if labels is not None:
        if len(labels) != n:
            raise ValueError(f"labels 길이 {len(labels)} ≠ n {n}")
        return b.map(dict(enumerate(labels)))
    return b
