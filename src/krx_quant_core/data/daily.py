"""일봉(원본·수정주가)과 거래일 달력 — 지문 검증 디스크 캐시, 방향이 명시된 패널.

swing-it 은 ``daily_bars_adjusted`` 를 매 실행 ``SELECT *`` 로 통째 읽고(580만 행) 같은 표를
5~6번 피벗한다. daytrade-it·scalp-it 은 전 거래일을 005930 일봉이나
``select distinct ts::date from ticks``(9.8GB 스캔)로 찾는다.

- :func:`load_daily_bars` — 범위를 한 번 ``COPY`` 로 받아 :class:`~.store.ColumnStore` 에 둔다.
  수정주가는 swing-it ``price_adjust`` 가 매일(주말 포함) 다시 계산하므로 **지문**
  ``(count, max(date), sum(close), sum(volume))`` 이 저장 때와 같을 때만 캐시를 쓴다(0.7s 쿼리 vs
  전체 재로딩).
- :func:`daily_panel` — 긴 프레임을 한 번만 피벗해 :class:`DailyPanel` 로. 배열은 **code × date**
  (``backtest.crosssectional`` 이 받는 방향), DataFrame 으로 꺼낼 때는 ``orient`` 를
  **반드시** 적는다(swing-it 에서 ``.T`` 를 빠뜨린 실제 버그가 있었다).
- :func:`trading_calendar` — ``daily_bars`` 의 날짜 집합 →
  :class:`~..market.calendar.TradingCalendar`. KST 하루에 한 번만 DB 에 묻는다.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..market.calendar import TradingCalendar
from .db import connect, fetch_frame, sql_literal
from .intraday import TS_UNIT, _today_kst
from .store import ColumnStore

__all__ = ["DAILY_COLUMNS", "DailyPanel", "daily_panel", "load_daily_bars", "trading_calendar"]

#: 일봉 값 열(두 테이블 공통). 원본은 정수, 수정주가는 float.
DAILY_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "trade_value")
_VERSION = 1


def _as_day(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _with_conn(conn: Any, fn: Any) -> Any:
    own = conn is None
    c = connect() if own else conn
    try:
        return fn(c)
    finally:
        if own:
            c.close()


def _fingerprint(conn: Any, table: str, lo: date, hi: date) -> list[Any]:
    """``(count, max(date), Σround(close×100), Σvolume)`` — 정수 합이라 병렬 집계 순서와 무관하다
    (float 합은 PG 병렬 집계에서 실행마다 마지막 자리가 흔들려 캐시가 매번 무효가 됐다)."""
    sql = (
        f"SELECT count(*), max(date), sum(CAST(round(close * 100) AS BIGINT)), "
        f"sum(CAST(volume AS BIGINT)) FROM {table} "
        f"WHERE date >= {sql_literal(lo)} AND date <= {sql_literal(hi)}"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql)
        n, mx, s_close, s_vol = cur.fetchone()
    finally:
        cur.close()
    return [int(n or 0), str(mx) if mx is not None else None, int(s_close or 0), int(s_vol or 0)]


def _fetch(conn: Any, table: str, lo: date, hi: date) -> dict[str, Any]:
    sql = (
        f"SELECT code, date, {', '.join(DAILY_COLUMNS)} FROM {table} "
        f"WHERE date >= {sql_literal(lo)} AND date <= {sql_literal(hi)} ORDER BY code, date"
    )
    dtypes = {c: np.float64 for c in DAILY_COLUMNS} | {"code": str}
    df = fetch_frame(
        conn, sql, ("code", "date", *DAILY_COLUMNS), dtypes=dtypes, timestamps=("date",)
    )
    uniq, inv = np.unique(df["code"].to_numpy(dtype="<U6"), return_inverse=True)
    parts: dict[str, Any] = {
        "codes": uniq.astype("<U6"),
        "code_idx": inv.astype(np.int32),
        "date": df["date"].to_numpy("datetime64[D]"),
    }
    for col in DAILY_COLUMNS:
        parts[col] = df[col].to_numpy(np.float64)
    return parts


def load_daily_bars(
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    adjusted: bool = True,
    codes: Sequence[str] | None = None,
    conn: Any = None,
    store: ColumnStore | bool | None = None,
) -> pd.DataFrame:
    """일봉 긴 프레임 ``code``(범주형)·``date``·:data:`DAILY_COLUMNS`, ``(code, date)`` 정렬.

    캐시는 **연도별**이다. 매 호출 연도마다 지문 쿼리(전 기간 합 0.7s)를 하고, 지문이 바뀐 연도만
    다시 받는다 — 수정주가 재계산이 새 날짜를 붙여도 올해 한 해만 다시 읽는다.

    Args:
        adjusted: ``daily_bars_adjusted``(기본 — 분할 반영) 또는 ``daily_bars``.
        codes: 이 종목만(캐시는 연도 전체로 만들고 잘라낸다).
        store: ``None`` → 기본 캐시, ``False`` → 캐시 안 씀.
    """
    if isinstance(codes, str):
        raise TypeError("codes must be a sequence of codes, not a str (wrap it: [code])")
    a, b = _as_day(start), _as_day(end)
    if a > b:
        raise ValueError("start must be <= end")
    table = "daily_bars_adjusted" if adjusted else "daily_bars"
    cache: ColumnStore | None
    if store is False:
        cache = None
    elif store is None or store is True:
        cache = ColumnStore(table, version=_VERSION)
    else:
        cache = store
    want = ["code_idx", "date", *DAILY_COLUMNS]

    def run(c: Any) -> list[dict[str, Any]]:
        out = []
        for y in range(a.year, b.year + 1):
            lo, hi = max(a, date(y, 1, 1)), min(b, date(y, 12, 31))
            if cache is None:
                out.append(_fetch(c, table, lo, hi))
                continue
            ylo, yhi = date(y, 1, 1), date(y, 12, 31)
            key = str(y)
            fp = _fingerprint(c, table, ylo, yhi)
            meta = cache.meta(key) or {}
            if not (
                cache.has(key, (*want, "codes")) and meta.get("info", {}).get("fingerprint") == fp
            ):
                with cache.lock(key):  # 워커 여럿이 같은 해를 동시에 받지 않게 잠근 채 다시 확인
                    meta = cache.meta(key) or {}
                    if meta.get("info", {}).get("fingerprint") != fp:
                        parts = _fetch(c, table, ylo, yhi)
                        rows = {k: v for k, v in parts.items() if k != "codes"}
                        # 새 날짜가 붙으면 행 수가 바뀐다 — 새 세대로 통째 교체.
                        cache.write(
                            key, rows, aux={"codes": parts["codes"]}, info={"fingerprint": fp},
                            replace=True, locked=True,
                        )  # fmt: skip
            parts = cache.read(key, want, mmap=True)
            parts.update(cache.read(key, ["codes"], mmap=False))
            out.append(parts)
        return out

    years = _with_conn(conn, run)
    allcodes = np.unique(np.concatenate([p["codes"] for p in years])) if years else np.array([])
    gidx = np.concatenate(
        [np.searchsorted(allcodes, p["codes"])[np.asarray(p["code_idx"])] for p in years]
    ).astype(np.int64)
    dt = np.concatenate([np.asarray(p["date"]) for p in years])
    keep = (dt >= np.datetime64(a)) & (dt <= np.datetime64(b))
    if codes is not None:
        sel = np.isin(allcodes, np.asarray(list(codes), dtype="<U6"))
        keep &= sel[gidx]
    # 연도 순으로 이어 붙였으니 (code, date) 로 다시 정렬 — 합성 정수 키 하나로.
    days = dt.astype("datetime64[D]").astype(np.int64)
    order = np.argsort((gidx << 20) | (days - days.min() if len(days) else days), kind="stable")
    order = order[keep[order]]
    used, inv = np.unique(gidx[order], return_inverse=True)
    cat = pd.Categorical.from_codes(
        inv.astype(np.int32), categories=pd.Index(allcodes[used].astype(str))
    )
    out = pd.DataFrame({"code": cat, "date": dt[order].astype(f"datetime64[{TS_UNIT}]")})
    for col in DAILY_COLUMNS:
        out[col] = np.concatenate([np.asarray(p[col]) for p in years])[order]
    return out


@dataclass(frozen=True)
class DailyPanel:
    """``values[field]`` 은 ``(len(codes), len(dates))`` — **code × date**. 없는 칸 nan."""

    codes: NDArray[np.str_]
    dates: pd.DatetimeIndex
    values: dict[str, NDArray[np.floating]]

    def array(self, field: str) -> NDArray[np.floating]:
        """code × date 배열(``backtest.crosssectional`` 입력 방향)."""
        return self.values[field]

    def frame(self, field: str, *, orient: Literal["date_x_code", "code_x_date"]) -> pd.DataFrame:
        """DataFrame — 방향을 반드시 적는다. ``date_x_code`` 는 행이 날짜(시계열 관례)."""
        a = self.values[field]
        codes = pd.Index(self.codes.astype(str), name="code")
        if orient == "code_x_date":
            return pd.DataFrame(a, index=codes, columns=self.dates)
        if orient == "date_x_code":
            return pd.DataFrame(a.T, index=self.dates, columns=codes)
        raise ValueError("orient must be 'date_x_code' or 'code_x_date'")

    @property
    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in self.values.values()))


def daily_panel(
    bars: pd.DataFrame,
    fields: Sequence[str] = ("close",),
    *,
    dtype: Any = np.float64,
    abs_prices: bool = True,
) -> DailyPanel:
    """긴 일봉 프레임(:func:`load_daily_bars` 또는 같은 열의 프레임) → :class:`DailyPanel`.

    ``(code, date)`` 가 중복이면 첫 행. ``abs_prices`` 면 가격·거래대금에 ``abs``
    (키움 종가 부호 관례 —
    :func:`krx_quant_core.backtest.panels.panel_pivot` 과 같다). ``dtype=np.float32`` 로 메모리를
    반으로 줄일 수 있다(정수 가격은 1,677만 원까지 정확).
    """
    if isinstance(bars["code"].dtype, pd.CategoricalDtype):
        codes_all = np.asarray(bars["code"].cat.categories, dtype="<U6")
        ci = bars["code"].cat.codes.to_numpy(np.int64)
        if (ci < 0).any():
            raise ValueError("bars has missing codes")
    else:
        if bars["code"].isna().any():
            raise ValueError("bars has missing codes")
        codes_all, ci = np.unique(bars["code"].astype(str).to_numpy(), return_inverse=True)
    day = bars["date"].to_numpy("datetime64[D]").astype(np.int64)
    key = (ci << 20) | (day - (day.min() if len(day) else 0))
    if len(key) > 1 and not (np.diff(key) > 0).all():
        # 정렬 안 됐거나 (code, date) 중복 — 첫 행을 남기고 정렬한다.
        _, first = np.unique(key, return_index=True)
        ci, day = ci[first], day[first]
        rows = first
    else:
        rows = slice(None)
    used, ci = np.unique(ci, return_inverse=True)
    codes = codes_all[used]
    uday, di = np.unique(day, return_inverse=True)
    dates = pd.DatetimeIndex(uday.astype("datetime64[D]").astype(f"datetime64[{TS_UNIT}]"))
    vals: dict[str, NDArray[np.floating]] = {}
    for f in fields:
        a = np.full((len(codes), len(dates)), np.nan, dtype=dtype)
        v = bars[f].to_numpy(np.float64)[rows]
        if abs_prices and f in ("open", "high", "low", "close", "trade_value"):
            v = np.abs(v)
        a[ci, di] = v
        vals[f] = a
    return DailyPanel(codes.astype("<U6"), dates, vals)


_CAL_MEMO: dict[tuple[date, date], tuple[TradingCalendar, float]] = {}
#: 직전 평일이 아직 없는(수집 실패·재시도 전) 달력을 다시 묻기까지의 초.
_STALE_TTL = 600.0


def _previous_weekday(d: date) -> date:
    p = d - timedelta(days=1)
    while p.weekday() >= 5:
        p -= timedelta(days=1)
    return p


def trading_calendar(
    *,
    start: date | str = date(2010, 1, 1),
    conn: Any = None,
    store: ColumnStore | bool | None = None,
    today: date | None = None,
) -> TradingCalendar:
    """``daily_bars`` 에 관측된 거래일 → :class:`TradingCalendar`. KST 하루 한 번 DB, 나머지는 캐시.

    오늘 장이 끝나 일봉이 들어오기 전에는 오늘이 달력에 없다 — ``previous_session(오늘)`` 은 그래도
    맞다(어제 이전 마지막 거래일). 단, 달력의 마지막 날이 **직전 평일보다 이르면**(전일 일봉 수집이
    실패해 다음 날 10:05 재시도를 기다리는 중이거나, 직전 평일이 휴장일) 디스크에 캐시하지 않고 10분
    뒤 다시 묻는다 — 하루 종일 하루 늦은 달력을 쓰는 사고를 막는다
    (휴장 다음 날엔 10분마다 0.7s 쿼리).
    """
    s = _as_day(start)
    t = today or _today_kst()
    memo = _CAL_MEMO.get((s, t))
    if memo is not None and time.monotonic() < memo[1]:
        return memo[0]
    cache: ColumnStore | None
    if store is False:
        cache = None
    elif store is None or store is True:
        cache = ColumnStore("calendar", version=_VERSION)
    else:
        cache = store
    key = f"{s.isoformat()}@{t.isoformat()}"
    if cache is not None and cache.has(key, ("date",)):
        days = cache.read(key, ["date"], mmap=False)["date"]
        fresh = True
    else:

        def run(c: Any) -> NDArray[Any]:
            sql = f"SELECT DISTINCT date FROM daily_bars WHERE date >= {sql_literal(s)} ORDER BY 1"
            df = fetch_frame(c, sql, ("date",), timestamps=("date",))
            return df["date"].to_numpy("datetime64[D]")

        days = _with_conn(conn, run)
        fresh = bool(len(days)) and days[-1] >= np.datetime64(_previous_weekday(t))
        if cache is not None and fresh:
            cache.write(key, {"date": days})
    cal = TradingCalendar(pd.DatetimeIndex(days).date)
    _CAL_MEMO[(s, t)] = (cal, math.inf if fresh else time.monotonic() + _STALE_TTL)
    return cal
