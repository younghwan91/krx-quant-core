"""하루치 장중 데이터(틱·호가·분봉) — 종목별 CSR 배열 + 닫힌 날 디스크 캐시.

scalp-it 에 같은 로더가 9벌 있다: ``pd.read_sql`` 로 하루치를 읽고
``dict(tuple(df.groupby("code")))`` 로 종목별 프레임을 복사한다(피크 메모리 2배, object 열).
여기서는

- 하루를 **한 번** ``COPY`` 로 읽어 ``(code, ts[, seq])`` 정렬 그대로 열 배열로 둔다.
  종목 ``i`` 의 행은 ``ptr[i]:ptr[i+1]`` — 복사 없는 슬라이스다.
- 시각은 ``sec``(그날 자정 기준 초, int32), 가격·잔량은 DB 정수형 그대로(int32), 수량 합계는 int64,
  방향 int8, 체결강도 float64(DB 값과 비트 동일 — 피처 골든). 틱 하루 190만 행이
  ``read_sql`` 223MB → 약 65MB.
- **닫힌 날**(KST 오늘 이전)은 :class:`~.store.ColumnStore` 에 저장하고 다음부터 mmap 으로
  연다(DB 왕복 없음). 오늘은 수집 중이라 매번 DB 에서 읽는다.

:meth:`CodeDay.frame` 은 ``ts`` 를 복원하고 정수열을 int64 로 올린 DataFrame 을 낸다 —
``lob.build_second_grid``·기존 ``read_sql`` 소비 코드가 그대로 받는다
(int32 끼리 곱해 넘치는 일 방지).
배열을 직접 쓸 때(:meth:`CodeDay.get`)는 곱하기 전에 ``astype(np.int64)``/``float`` 할 것.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..market.session import KST
from .db import code_list, connect, fetch_frame, is_postgres, sql_literal
from .store import ColumnStore

__all__ = [
    "CodeDay",
    "DatasetSpec",
    "MINUTE_BARS",
    "QUOTES",
    "TICKS",
    "is_closed_day",
    "load_day",
    "load_minute_bars",
    "load_quotes",
    "load_ticks",
]


@dataclass(frozen=True)
class DatasetSpec:
    """장중 테이블 하나의 읽기 규칙. ``columns`` 는 (DB 열, 저장 자료형) — ``code``·``ts`` 제외."""

    name: str
    table: str
    columns: tuple[tuple[str, str], ...]
    order: tuple[str, ...] = ("code", "ts")
    version: int = 1

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c for c, _ in self.columns)


TICKS = DatasetSpec(
    "ticks",
    "ticks",
    (
        ("seq", "int32"), ("price", "int32"), ("volume", "int64"), ("side", "int8"),
        ("strength", "float64"), ("cum_volume", "int64"), ("best_bid", "int32"),
        ("best_ask", "int32"),
    ),
    order=("code", "ts", "seq"),
)  # fmt: skip

_LV = range(1, 11)
QUOTES = DatasetSpec(
    "quotes",
    "quotes",
    tuple(
        [(f"bid{i}", "int32") for i in _LV]
        + [(f"bidqty{i}", "int32") for i in _LV]
        + [(f"ask{i}", "int32") for i in _LV]
        + [(f"askqty{i}", "int32") for i in _LV]
        + [("total_bid_qty", "int64"), ("total_ask_qty", "int64")]
    ),
)

MINUTE_BARS = DatasetSpec(
    "minute_bars",
    "minute_bars",
    (
        ("open", "int32"), ("high", "int32"), ("low", "int32"), ("close", "int32"),
        ("volume", "int64"), ("trade_value", "float64"),
    ),
)  # fmt: skip


#: pandas 가 파이썬 datetime 을 받을 때 쓰는 단위(3.x ``us``, 2.x ``ns``) — ``read_sql`` 과 같게.
TS_UNIT = "us" if int(pd.__version__.split(".")[0]) >= 3 else "ns"


def _today_kst() -> date:
    return datetime.now(KST).date()


def is_closed_day(day: date, *, today: date | None = None) -> bool:
    """캐시해도 되는 날인가 — KST 오늘보다 이전."""
    return day < (today or _today_kst())


@dataclass(frozen=True)
class CodeDay:
    """하루치 장중 데이터, 종목별 CSR. ``codes`` 는 정렬된 6자리 코드, 종목 ``i`` 의 행은
    ``ptr[i]:ptr[i+1]``. ``cols`` 의 배열은 모두 행 길이(``sec`` 포함). mmap 이면 읽기 전용이다.
    """

    dataset: str
    day: date
    codes: NDArray[np.str_]
    ptr: NDArray[np.int64]
    cols: Mapping[str, NDArray[Any]]
    _index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.ptr) != len(self.codes) + 1:
            raise ValueError("ptr must have len(codes) + 1 entries")
        n = int(self.ptr[-1]) if len(self.ptr) else 0
        for k, v in self.cols.items():
            if len(v) != n:
                raise ValueError(f"column {k!r} has {len(v)} rows, ptr says {n}")
        self._index.update({str(c): i for i, c in enumerate(self.codes)})

    def __len__(self) -> int:
        return int(self.ptr[-1])

    def __contains__(self, code: object) -> bool:
        return code in self._index

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    @property
    def n_codes(self) -> int:
        return len(self.codes)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.cols)

    @property
    def nbytes(self) -> int:
        """열 배열 바이트 합(mmap 이면 디스크에 매핑된 크기)."""
        return int(sum(a.nbytes for a in self.cols.values()) + self.ptr.nbytes + self.codes.nbytes)

    def rows(self, code: str) -> slice:
        i = self._index[code]
        return slice(int(self.ptr[i]), int(self.ptr[i + 1]))

    def get(self, code: str) -> dict[str, NDArray[Any]]:
        """종목 하나의 열 배열(복사 없는 뷰). 없는 종목이면 ``KeyError``."""
        s = self.rows(code)
        return {k: v[s] for k, v in self.cols.items()}

    def items(self) -> Iterator[tuple[str, dict[str, NDArray[Any]]]]:
        for c in self._index:
            yield c, self.get(c)

    def ts(self, sec: NDArray[Any]) -> NDArray[np.datetime64]:
        """``sec`` → ``datetime64`` (그날 자정 + 초, tz 없음 — DB 와 같은 KST 벽시계).

        단위는 :data:`TS_UNIT`(설치된 pandas 가 ``read_sql`` 로 내는 단위).
        """
        base = np.datetime64(self.day.isoformat(), "s")
        ts = base + np.asarray(sec, np.int64).astype("timedelta64[s]")
        return ts.astype(f"datetime64[{TS_UNIT}]")

    def frame(self, code: str | None = None, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """DataFrame 브리지 — ``ts`` 복원, 정수열 int64(``read_sql`` 과 같은 의미).

        ``code`` 를 주면 그 종목만(``code`` 열 없음), ``None`` 이면 하루 전체(``code`` 범주형 열).
        """
        names = [c for c in (columns or self.columns) if c != "sec"]
        if code is not None:
            s = self.rows(code)
            data: dict[str, Any] = {}
        else:
            s = slice(0, len(self))
            counts = np.diff(self.ptr)
            cat = pd.Categorical.from_codes(
                np.repeat(np.arange(self.n_codes, dtype=np.int32), counts),
                categories=pd.Index(self.codes.astype(str)),
            )
            data = {"code": cat}
        data["ts"] = self.ts(self.cols["sec"][s])
        for c in names:
            a = self.cols[c][s]
            data[c] = a.astype(np.int64) if a.dtype.kind in "iu" else np.asarray(a)
        return pd.DataFrame(data)

    def subset(self, codes: Sequence[str]) -> CodeDay:
        """종목 부분집합(있는 것만, 정렬 유지). 행은 복사된다."""
        keep = sorted({c for c in codes if c in self._index})
        idx = [self._index[c] for c in keep]
        lens = np.array([self.ptr[i + 1] - self.ptr[i] for i in idx], np.int64)
        take = (
            np.concatenate([np.arange(self.ptr[i], self.ptr[i + 1]) for i in idx])
            if idx
            else np.zeros(0, np.int64)
        )
        ptr = np.zeros(len(keep) + 1, np.int64)
        np.cumsum(lens, out=ptr[1:])
        return CodeDay(
            self.dataset,
            self.day,
            np.array(keep, dtype="<U6"),
            ptr,
            {k: np.asarray(v)[take] for k, v in self.cols.items()},
        )


def _select(spec: DatasetSpec, day: date, codes: Sequence[str] | None) -> str:
    cols = ", ".join(("code", "ts", *spec.names))
    lo = sql_literal(day)
    hi = sql_literal(day + timedelta(days=1))
    where = f"ts >= {lo} AND ts < {hi}"
    if codes:
        where += f" AND code IN {code_list(sorted(set(codes)))}"
    return f"SELECT {cols} FROM {spec.table} WHERE {where} ORDER BY {', '.join(spec.order)}"


def _compact(values: pd.Series, dtype: str, name: str) -> NDArray[Any]:
    a = values.to_numpy()
    t = np.dtype(dtype)
    if t.kind in "iu":
        if a.dtype.kind == "f" and np.isnan(a).any():
            raise ValueError(f"column {name!r} has NULLs; cannot store as {dtype}")
        info = np.iinfo(t)
        if len(a) and (a.min() < info.min or a.max() > info.max):
            raise OverflowError(f"column {name!r} does not fit {dtype}")
        if a.dtype.kind == "f" and not np.array_equal(a, np.round(a)):
            raise ValueError(f"column {name!r} has non-integer values; cannot store as {dtype}")
    return np.ascontiguousarray(a.astype(t))


def _from_db(
    spec: DatasetSpec, day: date, codes: Sequence[str] | None, conn: Any
) -> dict[str, Any]:
    names = ("code", "ts", *spec.names)
    # Postgres(COPY CSV)는 C 파서가 목표 자료형으로 바로 읽는다 — 결측·범위 초과면 파서가 실패한다
    # (조용한 잘림 없음). 그 밖의 연결은 받은 뒤 :func:`_compact` 가 검사하며 줄인다.
    dtypes: dict[str, Any] = {"code": "category"}  # 행마다 파이썬 str 을 만들지 않는다
    if is_postgres(conn):
        dtypes.update(dict(spec.columns))
    df = fetch_frame(conn, _select(spec, day, codes), names, dtypes=dtypes, timestamps=("ts",))
    cat = df["code"].astype("category").cat
    code = np.asarray(cat.categories, dtype="<U6")[cat.codes.to_numpy()]
    del cat
    ts = df["ts"].to_numpy("datetime64[s]")
    base = np.datetime64(day.isoformat(), "s")
    sec = (ts - base).astype(np.int64)
    if len(sec) and (sec.min() < 0 or sec.max() >= 86400):
        raise ValueError(f"{spec.name} {day}: rows outside the day")
    if len(code) > 1 and (code[1:] < code[:-1]).any():
        raise ValueError(f"{spec.name} {day}: rows are not ordered by code")
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]]) if len(code) else np.zeros(0, int)
    out: dict[str, Any] = {
        "codes": code[starts].astype("<U6"),
        "ptr": np.r_[starts, len(code)].astype(np.int64),
        "sec": sec.astype(np.int32),
    }
    for c, t in spec.columns:
        out[c] = _compact(df[c], t, c)
    return out


def _wrap(
    spec: DatasetSpec, day: date, parts: Mapping[str, Any], columns: Sequence[str]
) -> CodeDay:
    cols = {"sec": parts["sec"], **{c: parts[c] for c in columns}}
    return CodeDay(spec.name, day, np.asarray(parts["codes"]), np.asarray(parts["ptr"]), cols)


def load_day(
    spec: DatasetSpec,
    day: date,
    *,
    codes: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    conn: Any = None,
    store: ColumnStore | bool | None = None,
    mmap: bool = True,
    today: date | None = None,
) -> CodeDay:
    """``spec`` 테이블의 하루. 닫힌 날이면 캐시(없으면 하루 전체를 한 번 받아 저장)에서 연다.

    Args:
        codes: 이 종목들만(캐시에서 읽으면 잘라내고, 오늘이면 그 종목만 쿼리).
        columns: 낼 열(``sec`` 는 항상 포함). 기본 ``spec`` 전체.
        conn: DB 연결. 없으면 캐시 미스일 때만 :func:`~.db.connect` 로 열고 닫는다.
        store: ``None`` → 기본 :class:`ColumnStore`, ``False`` → 캐시 안 씀, 또는 직접 준 store.
        mmap: 캐시를 메모리 맵(읽기 전용)으로. ``False`` 면 힙으로 읽는다.
        today: 테스트용 "오늘"(KST).
    """
    if isinstance(day, datetime):
        day = day.date()
    want = tuple(columns) if columns is not None else spec.names
    bad = [c for c in want if c not in spec.names and c != "sec"]
    if bad:
        raise KeyError(f"{spec.name} has no columns {bad}")
    want = tuple(c for c in want if c != "sec")
    cache: ColumnStore | None
    if store is False or not is_closed_day(day, today=today):
        cache = None
    elif store is None or store is True:
        cache = ColumnStore(spec.name, version=spec.version)
    else:
        cache = store
    key = day.isoformat()

    if cache is not None and cache.has(key, ("codes", "ptr", "sec", *spec.names)):
        pass
    else:
        own = conn is None
        c = connect() if own else conn
        try:
            parts = _from_db(spec, day, None if cache is not None else codes, c)
        finally:
            if own:
                c.close()
        if cache is None:
            return _wrap(spec, day, parts, want)
        _write_day(cache, key, parts)
    parts = _read_day(cache, key, want, mmap=mmap)
    out = _wrap(spec, day, parts, want)
    return out.subset(codes) if codes is not None else out


def _write_day(cache: ColumnStore, key: str, parts: Mapping[str, Any]) -> None:
    rows = {k: v for k, v in parts.items() if k not in ("codes", "ptr")}
    cache.write(key, rows, aux={"codes": parts["codes"], "ptr": parts["ptr"]})


def _read_day(cache: ColumnStore, key: str, want: Sequence[str], *, mmap: bool) -> dict[str, Any]:
    out = cache.read(key, ["codes", "ptr"], mmap=False)
    out.update(cache.read(key, ["sec", *want], mmap=mmap))
    return out


def load_ticks(day: date, **kwargs: Any) -> CodeDay:
    """하루치 체결(:data:`TICKS`). 인자는 :func:`load_day`."""
    return load_day(TICKS, day, **kwargs)


def load_quotes(day: date, *, levels: int = 10, **kwargs: Any) -> CodeDay:
    """하루치 호가 스냅샷(:data:`QUOTES`). ``levels`` 단계까지의 가격·잔량만 낸다.

    캐시는 10단계 전체를 저장한다(열 단위로 필요한 것만 읽힌다).
    """
    if not 1 <= levels <= 10:
        raise ValueError("levels must be 1..10")
    if "columns" not in kwargs:
        kwargs["columns"] = [
            f"{p}{i}" for p in ("bid", "bidqty", "ask", "askqty") for i in range(1, levels + 1)
        ]
    return load_day(QUOTES, day, **kwargs)


def load_minute_bars(day: date, **kwargs: Any) -> CodeDay:
    """하루치 분봉(:data:`MINUTE_BARS`). ``sec`` 는 분 시작 시각(초)."""
    return load_day(MINUTE_BARS, day, **kwargs)
