"""KRX 거래일 달력 — **관측된 거래일**로만 만든다.

휴장일 표를 하드코딩하지 않는 이유: 어느 레포에도 검증된 표가 없고, 임시공휴일·
선거일·연말 휴장처럼 해마다 바뀌는 날을 손으로 적으면 조용히 틀린다. 틀린 휴장일
하나가 "전 거래일" 을 하루 밀어 lookahead 나 라벨 누락을 만든다. 대신 DB 에 실제로
가격 행이 있는 날(예 ``select distinct date from daily_bars``)을 달력으로 쓴다.

그 대가로 **관측 범위 밖은 모른다.** 마지막 관측일 이후의 다음 거래일을 추측하지
않고 ``LookupError`` 를 던진다 — 주말만 건너뛰는 식의 추측이 가장 흔한 버그원이다.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator
from datetime import date, datetime

import pandas as pd

__all__ = ["PUBLISHED_COLUMN", "TradingCalendar", "align_to_trading_days"]


def _as_date(value: date | datetime | str) -> date:
    """``date``·``datetime``(``pd.Timestamp`` 포함)·ISO 문자열을 ``date`` 로."""
    if isinstance(value, datetime):  # pd.Timestamp 도 datetime 하위형
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


class TradingCalendar:
    """관측 거래일 집합 위의 조회 연산. 불변이다."""

    __slots__ = ("_days", "_set")

    def __init__(self, days: Iterable[date | datetime | str]) -> None:
        self._days: tuple[date, ...] = tuple(sorted({_as_date(d) for d in days}))
        self._set = frozenset(self._days)

    @classmethod
    def from_dates(cls, dates: Iterable[date | datetime | str]) -> TradingCalendar:
        """거래일 목록(중복·순서 무관, ``pd.Series`` 가능)으로 달력을 만든다."""
        return cls(dates)

    # --- 기본 ---------------------------------------------------------------
    @property
    def sessions(self) -> tuple[date, ...]:
        return self._days

    @property
    def first(self) -> date:
        return self._require_nonempty()[0]

    @property
    def last(self) -> date:
        return self._require_nonempty()[-1]

    def __len__(self) -> int:
        return len(self._days)

    def __iter__(self) -> Iterator[date]:
        return iter(self._days)

    def __contains__(self, d: object) -> bool:
        try:
            return _as_date(d) in self._set  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        if not self._days:
            return "TradingCalendar(empty)"
        return f"TradingCalendar({self._days[0]}..{self._days[-1]}, n={len(self._days)})"

    def _require_nonempty(self) -> tuple[date, ...]:
        if not self._days:
            raise LookupError("trading calendar is empty")
        return self._days

    # --- 조회 ---------------------------------------------------------------
    def is_session(self, d: date | datetime | str) -> bool:
        """``d`` 가 관측된 거래일인가. 범위 밖이면 False(모르는 날)."""
        return _as_date(d) in self._set

    def previous_session(self, d: date | datetime | str) -> date:
        """``d`` **이전**(당일 제외) 마지막 거래일. 없으면 ``LookupError``."""
        day = _as_date(d)
        i = bisect.bisect_left(self._days, day)
        if i == 0:
            raise LookupError(f"no session before {day} in calendar")
        return self._days[i - 1]

    def next_session(self, d: date | datetime | str) -> date:
        """``d`` **이후**(당일 제외) 첫 거래일. 관측 범위를 넘으면 ``LookupError``."""
        day = _as_date(d)
        i = bisect.bisect_right(self._days, day)
        if i >= len(self._days):
            raise LookupError(f"no session after {day} in calendar")
        return self._days[i]

    def sessions_between(
        self, start: date | datetime | str, end: date | datetime | str
    ) -> list[date]:
        """``start`` ~ ``end`` (양끝 포함) 사이의 거래일. ``start > end`` 면 빈 목록."""
        a, b = _as_date(start), _as_date(end)
        lo = bisect.bisect_left(self._days, a)
        hi = bisect.bisect_right(self._days, b)
        return list(self._days[lo:hi])

    def offset(self, d: date | datetime | str, n: int) -> date:
        """거래일 ``d`` 에서 ``n`` 거래일 뒤(음수면 앞).

        ``d`` 가 거래일이 아니면 ``ValueError`` — 휴장일 기준 "1거래일 뒤"는 이월
        규칙(당일 포함인지)에 따라 답이 갈리므로 호출부가 먼저
        :meth:`next_session`/:meth:`previous_session` 으로 정해야 한다.
        범위를 벗어나면 ``LookupError``.
        """
        day = _as_date(d)
        if day not in self._set:
            raise ValueError(f"{day} is not a session in calendar")
        i = bisect.bisect_left(self._days, day) + n
        if not 0 <= i < len(self._days):
            raise LookupError(f"{day} offset {n} is outside calendar range")
        return self._days[i]


# --- 휴장일 기사 이월 (scalp-it news/calendar.py 이식) -----------------------

PUBLISHED_COLUMN = "published_date"


def align_to_trading_days(articles: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """기사의 ``date`` 를 그날 이후 첫 거래일로 옮긴다(당일이 거래일이면 그대로).

    scalp-it ``news/calendar.py`` 의 정확한 이식이다. 이 단계가 없으면 **주말·공휴일
    기사가 통째로 사라진다** — 기사 날짜에 가격 행이 없어 ``adv`` 가 ``NaN`` 이 되고,
    유동성 하한 비교가 ``False`` 라 조용히 탈락한다.

    **이월은 앞으로만 한다.** 토요일 기사를 금요일에 알 수는 없으므로 뒤로 당기면
    lookahead 다. 원래 게재일은 ``published_date`` 로 보존한다.

    Args:
        articles: 최소한 ``date`` 컬럼을 가진 프레임. 나머지 컬럼은 그대로 보존된다.
        calendar: 거래일 목록. 보통 가격 데이터의 ``date`` 를 넘긴다.

    Returns:
        ``date`` 가 거래일로 바뀌고 ``published_date`` 가 추가된 프레임.
        달력 마지막 거래일을 넘어선 기사는 **버린다** — 이월할 곳이 없고,
        남겨두면 라벨이 생기지 않아 뒤에서 조용히 떨어진다.
    """
    out = articles.copy()
    if PUBLISHED_COLUMN not in out.columns:
        out[PUBLISHED_COLUMN] = out["date"]

    if out.empty:
        return out

    trading_days = pd.Index(pd.to_datetime(pd.Series(calendar)).unique()).sort_values()
    if trading_days.empty:
        return out.iloc[0:0]

    published = pd.to_datetime(out[PUBLISHED_COLUMN])
    # searchsorted "left" 는 거래일 당일이면 자기 자신을, 휴장일이면 다음 거래일을 준다.
    position = trading_days.searchsorted(published, side="left")
    within = position < len(trading_days)

    out = out.loc[within.tolist()].copy()
    out["date"] = trading_days[position[within]]
    return out.reset_index(drop=True)
