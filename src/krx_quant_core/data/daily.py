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

KRX 데이터 함정 방어(옵트인 — 기존 기본 출력은 그대로):

- **애프터마켓 오염(2026-09-14~, :data:`AFTERMARKET_START`).** 16:00~20:00 연속매매가 생겨 키움
  일봉·분봉·수급 종가에 정규장 뒤 체결이 섞였다. ``daily_bars``·``daily_bars_adjusted`` 의
  고가·저가·종가·거래량·거래대금이 정규장 값이 아니다(시가는 정상). :func:`load_daily_bars`
  ``session="regular"`` 가 분봉 정규장(09:00~15:32) 집계로 바꾼다.
- **가격제한 기준가 = 전일 정규장 종가.** 일봉 종가로 ``limit_up_price`` 를 계산하면 2026-09-14~
  대부분의 종목-일에서 틀린다. :func:`base_price` 를 쓴다.
- **``daily_bars.close`` 는 원 종가가 아니다** — 이후 분할·병합이 소급 반영돼 있다(005930 2017-03-31
  41,200 vs 실제 2,060,000). 원 종가는 ``daily_bars_raw``(백필 중) 또는 ``supply_demand.close``.
- ``trade_value`` 단위는 **백만원**(:data:`TRADE_VALUE_UNIT`).
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
from ..market.session import REGULAR_CLOSE
from .db import code_list, connect, fetch_frame, sql_literal
from .intraday import REGULAR_FIRST_BAR, REGULAR_LAST_BAR, TS_UNIT, _today_kst
from .store import ColumnStore

__all__ = [
    "AFTERMARKET_START",
    "BASE_CLOSE_FROM",
    "DAILY_COLUMNS",
    "REGULAR_FLAGS",
    "TRADE_VALUE_UNIT",
    "DailyPanel",
    "base_from_change",
    "base_price",
    "daily_panel",
    "load_daily_bars",
    "regular_session_bars",
    "trading_calendar",
]

#: 일봉 값 열(두 테이블 공통). 원본은 정수, 수정주가는 float. 가격은 원, ``volume`` 은 주,
#: ``trade_value`` 는 **백만원**(:data:`TRADE_VALUE_UNIT` 을 곱하면 원). 2026-09-14~
#: 고가·저가·종가·거래량·거래대금은 애프터마켓이 섞인 값이다(:data:`AFTERMARKET_START`).
DAILY_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "trade_value")
#: ``daily_bars``·``daily_bars_adjusted``·분봉 집계(:func:`regular_session_bars`) ``trade_value`` 1
#: 단위의 원 — 백만원. ADV 10억 원 = ``trade_value`` 1,000. (분봉 ``minute_bars.trade_value`` 는 원
#: 그대로다.)
TRADE_VALUE_UNIT = 1_000_000
#: KRX 애프터마켓(16:00~20:00 연속매매, ETF·ETN 제외) 첫 거래일. 이날부터 키움 일봉·분봉·수급 종가에
#: 정규장 뒤 체결이 섞였다. 이전 행은 정규장 값 그대로다.
AFTERMARKET_START = date(2026, 9, 14)
#: :func:`load_daily_bars` ``session="regular"`` 가 덧붙이는 표시 열. ``regular_ok`` =
#: 고가·저가·종가·거래량·거래대금이 모두 정규장 값, ``close_ok`` = 종가만이라도 정규장 값(수급
#: 항등식으로 복원한 행 포함).
REGULAR_FLAGS: tuple[str, ...] = ("regular_ok", "close_ok")
#: :func:`base_price` 의 ``src`` 우선순위(앞이 이긴다). 뒤 셋은 전일이 :data:`AFTERMARKET_START`
#: 이전일 때만.
BASE_CLOSE_FROM: tuple[str, ...] = (
    "supply_demand_identity", "minute_bars", "supply_demand", "daily_bars_raw", "daily_bars",
)  # fmt: skip
#: 수급 항등식 기준가를 맞출 호가 격자(굵은 것부터). 호가단위 **표**가 아니다 — 등락률을 재현하는
#: 가장 굵은 격자를 고를 뿐이라 2023-01-25 호가 개편 전후·KOSPI/KOSDAQ 구분 없이 같은 규칙이 된다(표
#: 정본은 kiwoom-client).
_IDENTITY_GRIDS = (1000, 500, 100, 50, 10, 5, 1)
#: ``flu_rt`` 는 등락률(%)×100 을 반올림한 값 — 전일 종가 기준 재계산과의 차이가 2019·2025 표본 100%
#: 에서 0.5bp 이내.
_IDENTITY_TOL_BP = 0.5 + 1e-4
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
    session: Literal["all", "regular"] = "all",
) -> pd.DataFrame:
    """일봉 긴 프레임 ``code``(범주형)·``date``·:data:`DAILY_COLUMNS`, ``(code, date)`` 정렬.

    캐시는 **연도별**이다. 매 호출 연도마다 지문 쿼리(전 기간 합 0.7s)를 하고, 지문이 바뀐 연도만
    다시 받는다 — 수정주가 재계산이 새 날짜를 붙여도 올해 한 해만 다시 읽는다.

    Args:
        adjusted: ``daily_bars_adjusted``(기본 — 분할 반영) 또는 ``daily_bars``.
            **``False`` 도 원 종가가 아니다** — ``daily_bars`` 에는 이후 액면분할·병합이 소급 반영돼
            있다(005930 2017-03-31 41,200 vs 실제 2,060,000). 시총·가격제한·호가처럼 그 시점 원
            가격이 필요하면 ``daily_bars_raw``·``supply_demand.close`` 또는 :func:`base_price`.
        codes: 이 종목만(캐시는 연도 전체로 만들고 잘라낸다).
        store: ``None`` → 기본 캐시, ``False`` → 캐시 안 씀.
        session: ``"all"``(기본 — 테이블 그대로) 또는 ``"regular"``. ``"regular"`` 이면
            :data:`AFTERMARKET_START` 이후 행의 고가·저가·종가·거래량·거래대금을 분봉
            정규장(09:00~15:32) 집계로 바꾸고 :data:`REGULAR_FLAGS` 열을 붙인다(아래). 캐시는 테이블
            값 그대로이고 보정은 매 호출 DB 집계다.

    ``session="regular"`` 보정 규칙:

    - 분봉이 있는 행: 가격 = 분봉 정규장 값 × (행 시가 / 분봉 첫 봉 시가) — 수정주가 테이블이면 이
      비율이 그날 조정계수다(시가는 오염되지 않았다). 거래량 = 분봉 합 / 그 비율, 거래대금 = 분봉
      합(백만원). ``regular_ok=True``.
    - 분봉이 없는 행: ``regular_ok=False``(고가·저가·거래량·거래대금은 테이블 값). 종가만 다음
      거래일 수급 항등식(:func:`base_from_change`)으로 복원할 수 있으면 복원하고 ``close_ok=True``
      — 조정계수는 행 종가 / 같은 날 ``supply_demand.close``(둘 다 같은 통합 종가). 다음 거래일
      수급이 아직 없으면 ``close_ok=False``.
    - 이전 행: 둘 다 ``True``.
    """
    if isinstance(codes, str):
        raise TypeError("codes must be a sequence of codes, not a str (wrap it: [code])")
    if session not in ("all", "regular"):
        raise ValueError("session must be 'all' or 'regular'")
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
    if session == "all":
        return out
    return _with_conn(conn, lambda c: _to_regular(c, out, a, b, codes is not None))


# ── 정규장 보정·가격제한 기준가 ──────────────────────────────────────────────────────────────


def base_from_change(close: Any, flu_rt: Any) -> NDArray[np.float64]:
    """수급 표의 (종가, ``flu_rt``) → **그날** 가격제한 기준가(보통 전일 정규장 종가).

    ``supply_demand.close`` 는 2026-09-14~ 애프터마켓이 섞인 통합 종가지만 ``flu_rt``(등락률 % ×
    100)는 KRX 기준가 대비라 ``close / (1 + flu_rt/1e4)`` 가 기준가다(항등식 — 전일 정규장 종가와
    99.98% 일치). 권리락·배당락처럼 기준가가 조정된 날은 조정 기준가가 나온다(가격제한에는 그게
    맞다).

    ``flu_rt`` 해상도(1bp, 반올림) 때문에 몫은 ±0.5bp 로만 맞으므로
    호가 격자(1000·500·100·50·10·5·1원)를 굵은 것부터 반올림해 등락률을 0.5bp 안에서 재현하는 첫
    값을 고른다 — 호가단위 표 없이 2023-01-25 개편 전후·시장 구분에 같은 규칙이다. 종가는
    절댓값(키움 부호 관례). 종가 ≤ 0·결측이거나 재현 격자가 없으면 NaN.

    한계: 호가단위/가격이 1bp 미만인 종목(10만 원대 ETF·ETN 5원 호가, 개편 전 KOSDAQ 100만 원대
    100원 호가)은 한 틱 어긋날 수 있다.
    """
    c, r = np.broadcast_arrays(
        np.abs(np.asarray(close, dtype=np.float64)), np.asarray(flu_rt, dtype=np.float64)
    )
    out = np.full(c.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = c / (1.0 + r / 1e4)
        todo = np.isfinite(b0) & (b0 > 0) & (c > 0)
        for g in _IDENTITY_GRIDS:
            v = np.round(b0 / g) * g
            ok = todo & (v > 0) & (np.abs((c / v - 1.0) * 1e4 - r) <= _IDENTITY_TOL_BP)
            out[ok] = v[ok]
            todo &= ~ok
    return out


#: 하루 분봉 정규장 집계 — 시가 = 첫 봉 시가, 종가 = 15:32 이하 마지막 봉 종가, 고저 = max/min, 양 =
#: 합. 이식성(Postgres·sqlite)을 위해 ``array_agg`` 대신 첫·끝 봉을 (code, ts) 로 다시 붙인다(PK
#: 조회).
_SESSION_SQL = (
    "SELECT a.code, f.open, a.high, a.low, l.close, a.volume, a.trade_value, a.last_ts FROM "
    "(SELECT code, min(ts) AS first_ts, max(ts) AS last_ts, max(high) AS high, min(low) AS low, "
    "sum(volume) AS volume, sum(trade_value) AS trade_value FROM minute_bars WHERE {where} "
    "GROUP BY code) a "
    "JOIN minute_bars f ON f.code = a.code AND f.ts = a.first_ts AND {f_span} "
    "JOIN minute_bars l ON l.code = a.code AND l.ts = a.last_ts AND {l_span} ORDER BY a.code"
)
_SESSION_COLUMNS = ("code", "open", "high", "low", "close", "volume", "trade_value", "last_ts")


def _session_day(conn: Any, day: date, codes: Sequence[str] | None) -> pd.DataFrame:
    lo = datetime.combine(day, REGULAR_FIRST_BAR)
    hi = datetime.combine(day, REGULAR_LAST_BAR)
    span = "{t}.ts >= " + sql_literal(lo) + " AND {t}.ts <= " + sql_literal(hi)
    where = span.format(t="minute_bars")
    if codes is not None:
        where += f" AND code IN {code_list(codes)}"
    # 되붙이는 조인에도 같은 시각 범위를 건다 — 없으면 TimescaleDB 가 청크를 못 걸러 전 테이블을
    # 훑는다(전 종목 하루 50초 넘게 → 범위를 걸면 0.3초).
    sql = _SESSION_SQL.format(where=where, f_span=span.format(t="f"), l_span=span.format(t="l"))
    dtypes = {c: np.float64 for c in _SESSION_COLUMNS[1:-1]} | {"code": str}
    df = fetch_frame(
        conn, sql, _SESSION_COLUMNS, dtypes=dtypes,
        timestamps=("last_ts",),
    )  # fmt: skip
    df.insert(1, "date", pd.Series(np.datetime64(day, "D"), index=df.index))
    return df


def regular_session_bars(
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    codes: Sequence[str] | None = None,
    conn: Any = None,
) -> pd.DataFrame:
    """``minute_bars`` 정규장(09:00~15:32 봉) → 일봉
    ``code``·``date``·:data:`DAILY_COLUMNS`·``last_ts``.

    하루 한 쿼리로 DB 에서 집계한다(분봉을 끌어오지 않는다). 시가 = 첫 봉 시가, 종가 = 15:32 이하
    마지막 봉 종가(연장된 종가 단일가 포함 — swing-it ``krx_session.session_bars_from_minutes`` 와
    같은 규칙), 고저 = max/min, ``volume`` = 합(주), ``trade_value`` = 합 ÷
    :data:`TRADE_VALUE_UNIT`(**백만원**, 일봉과 같은 단위). ``last_ts`` = 마지막 봉 시작 시각 —
    15:30 미만이면 종가 단일가 체결이 없었던(또는 분봉이 끊긴) 종목이다. 원 가격(분할 소급
    없음)이다. 분봉이 있는 종목만 나온다(하루 ~1,400~1,800 종목, 유동 종목 위주).
    """
    if isinstance(codes, str):
        raise TypeError("codes must be a sequence of codes, not a str (wrap it: [code])")
    a, b = _as_day(start), _as_day(end)
    if a > b:
        raise ValueError("start must be <= end")
    sel = sorted(set(codes)) if codes is not None else None
    days = [a + timedelta(days=i) for i in range((b - a).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    if sel == [] or not days:
        frames: list[pd.DataFrame] = []
    else:
        frames = _with_conn(conn, lambda c: [_session_day(c, d, sel) for d in days])
    cols = ("code", "date", *DAILY_COLUMNS, "last_ts")
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame({c: pd.Series(dtype=np.float64) for c in cols})
        out["code"] = out["code"].astype(str)
        out["last_ts"] = pd.Series(dtype=f"datetime64[{TS_UNIT}]")
    out["date"] = pd.to_datetime(out["date"]).astype(f"datetime64[{TS_UNIT}]")
    out["trade_value"] = out["trade_value"].astype(np.float64) / TRADE_VALUE_UNIT
    return out.loc[:, list(cols)].reset_index(drop=True)


def _rows_on(
    conn: Any, table: str, day: date, cols: Sequence[str], codes: Sequence[str] | None
) -> pd.DataFrame:
    """``table`` 의 ``day`` 하루 ``code``·``cols``(모두 NOT NULL). ``table`` 은 이 모듈 상수만."""
    where = f"date = {sql_literal(day)}" + "".join(f" AND {c} IS NOT NULL" for c in cols)
    if codes is not None:
        where += f" AND code IN {code_list(codes)}"
    sql = f"SELECT code, {', '.join(cols)} FROM {table} WHERE {where} ORDER BY code"
    dtypes = {c: np.float64 for c in cols} | {"code": str}
    return fetch_frame(conn, sql, ("code", *cols), dtypes=dtypes)


def _scalar_day(conn: Any, sql: str) -> date | None:
    cur = conn.cursor()
    try:
        cur.execute(sql)
        row = cur.fetchone()
    finally:
        cur.close()
    return None if row is None or row[0] is None else _as_day(str(row[0]))


def base_price(
    day: date | datetime | str,
    codes: Sequence[str] | None = None,
    *,
    conn: Any = None,
) -> pd.DataFrame:
    """``day`` 의 가격제한 기준가(= 전 거래일 **정규장** 종가) 표.

    열: ``code``·``base``·``src``·``prev_date``.

    ``market.limits.limit_up_price(base)`` 에 넣을 값이다. 2026-09-14~ 일봉·수급 종가는 애프터마켓이
    섞여 기준가가
    아니다(일봉 종가로 계산한 상한가가 84% 종목-일에서 틀렸다). 원천
    우선순위(:data:`BASE_CLOSE_FROM`, 앞이 이긴다):

    1. ``supply_demand_identity`` — ``day`` 의 수급 행 :func:`base_from_change`(호가 격자 반올림).
       권리락 등 기준가 조정일은 조정 기준가.
    2. ``minute_bars`` — 전 거래일 분봉의 15:32 이하 마지막 봉 종가, 그 봉이 15:30~15:32 일
       때만(종가 단일가).
    3. 전 거래일이 :data:`AFTERMARKET_START` 이전일 때만: ``supply_demand``(그날 종가 절댓값) →
       ``daily_bars_raw`` → ``daily_bars``. **``daily_bars`` 는 이후 분할·병합이 소급 반영된
       값**이라 그 사이 분할이 있으면 틀린다 — ``src`` 로 걸러 쓸 것.

    전 거래일은 ``daily_bars`` 의 ``day`` 직전 날짜. ``codes`` 를 주면 그 순서대로 행을 내고 못 찾은
    종목은 ``base`` NaN·``src`` None, ``None`` 이면 찾은 종목만(코드순).
    """
    if isinstance(codes, str):
        raise TypeError("codes must be a sequence of codes, not a str (wrap it: [code])")
    d = _as_day(day)
    sel = sorted(set(codes)) if codes is not None else None

    def run(c: Any) -> tuple[date | None, list[pd.DataFrame]]:
        if sel == []:
            return None, []
        prev = _scalar_day(c, f"SELECT max(date) FROM daily_bars WHERE date < {sql_literal(d)}")
        sd = _rows_on(c, "supply_demand", d, ("close", "flu_rt"), sel)
        parts = [
            pd.DataFrame({
                "code": sd["code"], "base": base_from_change(sd["close"], sd["flu_rt"]),
                "src": BASE_CLOSE_FROM[0],
            })
        ]  # fmt: skip
        if prev is None:
            return None, parts
        ms = _session_day(c, prev, sel)
        tod = ms["last_ts"] - ms["last_ts"].dt.normalize()
        cut = pd.Timedelta(hours=REGULAR_CLOSE.hour, minutes=REGULAR_CLOSE.minute)
        auction = (tod >= cut).to_numpy()
        parts.append(
            pd.DataFrame({"code": ms["code"][auction], "base": ms["close"][auction],
                          "src": BASE_CLOSE_FROM[1]})
        )  # fmt: skip
        if prev < AFTERMARKET_START:
            for table in BASE_CLOSE_FROM[2:]:
                rows = _rows_on(c, table, prev, ("close",), sel)
                parts.append(
                    pd.DataFrame({"code": rows["code"], "base": rows["close"].abs(), "src": table})
                )
        return prev, parts

    prev, parts = _with_conn(conn, run)
    if parts:
        allp = pd.concat(parts, ignore_index=True)
        allp["code"] = allp["code"].astype(str)
        allp = allp[np.isfinite(allp["base"].to_numpy(np.float64)) & (allp["base"] > 0)]
        out = allp.drop_duplicates("code", keep="first").sort_values("code", kind="stable")
    else:
        out = pd.DataFrame({"code": pd.Series(dtype=str), "base": pd.Series(dtype=np.float64),
                            "src": pd.Series(dtype=object)})  # fmt: skip
    if codes is not None:
        out = out.set_index("code").reindex(pd.Index(list(codes), name="code")).reset_index()
    out["base"] = out["base"].astype(np.float64)
    out["src"] = out["src"].astype(object).where(out["src"].notna(), None)
    out["prev_date"] = pd.Timestamp(prev) if prev is not None else pd.NaT
    return out.reset_index(drop=True)


def _restore_close(
    conn: Any, codes: NDArray[Any], days: NDArray[Any], closes: NDArray[np.float64], end: date
) -> NDArray[np.float64]:
    """분봉 없는 행의 정규장 종가 = 다음 거래일 수급 항등식 × (행 종가 / 같은 날 수급 종가).

    못 하면 NaN.
    """
    out = np.full(len(codes), np.nan)
    lo = pd.Timestamp(days.min()).date()
    hi = end + timedelta(days=20)  # 다음 거래일(연휴 포함)이 들어올 만큼
    span = f"date >= {sql_literal(lo)} AND date <= {sql_literal(hi)}"
    sessions = fetch_frame(
        conn, f"SELECT DISTINCT date FROM supply_demand WHERE {span} ORDER BY 1", ("date",),
        timestamps=("date",),
    )["date"].to_numpy("datetime64[D]")  # fmt: skip
    sel = sorted(set(map(str, codes)))
    sd = fetch_frame(
        conn,
        f"SELECT code, date, close, flu_rt FROM supply_demand WHERE {span} "
        f"AND code IN {code_list(sel)} AND close IS NOT NULL AND flu_rt IS NOT NULL",
        ("code", "date", "close", "flu_rt"),
        dtypes={"code": str, "close": np.float64, "flu_rt": np.float64},
        timestamps=("date",),
    )
    if not len(sessions) or not len(sd):
        return out
    sd["date"] = sd["date"].to_numpy("datetime64[D]")
    sd = sd.set_index(["code", "date"])
    d = np.asarray(days).astype("datetime64[D]")
    pos = np.searchsorted(sessions, d, side="right")
    has_next = pos < len(sessions)
    nxt = np.where(has_next, sessions[np.minimum(pos, len(sessions) - 1)], d)
    c = np.asarray(codes).astype(str)
    same = sd.reindex(pd.MultiIndex.from_arrays([c, d]))
    after = sd.reindex(pd.MultiIndex.from_arrays([c, nxt]))
    base = base_from_change(after["close"].to_numpy(), after["flu_rt"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(closes) / np.abs(same["close"].to_numpy(np.float64))
    val = base * ratio
    ok = has_next & np.isfinite(val) & (val > 0)
    out[ok] = val[ok]
    return out


def _to_regular(conn: Any, out: pd.DataFrame, a: date, b: date, restrict: bool) -> pd.DataFrame:
    """:func:`load_daily_bars` ``session="regular"`` 의 보정(규칙은 그 독스트링)."""
    n = len(out)
    regular_ok = np.ones(n, dtype=bool)
    close_ok = np.ones(n, dtype=bool)
    idx = np.flatnonzero(out["date"].to_numpy() >= np.datetime64(AFTERMARKET_START))
    if len(idx):
        vals = {col: out[col].to_numpy(np.float64).copy() for col in DAILY_COLUMNS}
        codes = np.asarray(out["code"].to_numpy()[idx]).astype(str)
        days = out["date"].to_numpy()[idx]
        lo = max(a, AFTERMARKET_START)
        want = sorted(set(codes)) if restrict else None
        sess = regular_session_bars(lo, b, codes=want, conn=conn)
        m = pd.DataFrame({"code": codes, "date": days}).merge(sess, on=["code", "date"], how="left")
        ro = np.abs(vals["open"][idx])
        so = m["open"].to_numpy(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            fac = np.where((so > 0) & (ro > 0), ro / so, np.nan)
        hit = np.isfinite(fac)
        j, f = idx[hit], fac[hit]
        for col in ("high", "low", "close"):
            vals[col][j] = m[col].to_numpy(np.float64)[hit] * f
        vals["volume"][j] = m["volume"].to_numpy(np.float64)[hit] / f
        vals["trade_value"][j] = m["trade_value"].to_numpy(np.float64)[hit]
        miss = idx[~hit]
        regular_ok[miss] = False
        close_ok[miss] = False
        if len(miss):
            rc = _restore_close(
                conn, codes[~hit], days[~hit], vals["close"][miss], b
            )
            got = np.isfinite(rc)
            vals["close"][miss[got]] = rc[got]
            close_ok[miss[got]] = True
        for col in ("high", "low", "close", "volume", "trade_value"):
            out[col] = vals[col]
    out["regular_ok"] = regular_ok
    out["close_ok"] = close_ok
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
