"""관측 거래일 달력 + 휴장일 기사 이월(scalp-it tests/test_news_calendar.py 이식)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from krx_quant_core.market.calendar import TradingCalendar, align_to_trading_days

# 2026-08: 14(금) · 17(월) · 18(화) · 20(목, 19 는 가상의 휴장일)
_DAYS = ["2026-08-18", "2026-08-14", "2026-08-17", "2026-08-20", "2026-08-17"]


@pytest.fixture()
def cal() -> TradingCalendar:
    return TradingCalendar.from_dates(_DAYS)


def test_from_dates_dedupes_sorts_and_accepts_mixed_types():
    c = TradingCalendar.from_dates(
        [pd.Timestamp("2026-08-17"), date(2026, 8, 14), datetime(2026, 8, 17, 15, 30), "2026-08-18"]
    )
    assert c.sessions == (date(2026, 8, 14), date(2026, 8, 17), date(2026, 8, 18))
    assert len(c) == 3
    assert c.first == date(2026, 8, 14)
    assert c.last == date(2026, 8, 18)


def test_from_series_like_db_distinct_dates():
    c = TradingCalendar.from_dates(pd.Series(pd.to_datetime(_DAYS)))
    assert len(c) == 4


def test_is_session(cal):
    assert cal.is_session(date(2026, 8, 14))
    assert cal.is_session(pd.Timestamp("2026-08-17"))
    assert not cal.is_session(date(2026, 8, 15))
    assert not cal.is_session(date(2026, 8, 19))
    assert date(2026, 8, 20) in cal
    assert "garbage" not in cal


def test_previous_session_is_strictly_before(cal):
    assert cal.previous_session(date(2026, 8, 17)) == date(2026, 8, 14)
    assert cal.previous_session(date(2026, 8, 16)) == date(2026, 8, 14)
    assert cal.previous_session(date(2026, 8, 20)) == date(2026, 8, 18)
    assert cal.previous_session(date(2026, 8, 19)) == date(2026, 8, 18)
    with pytest.raises(LookupError):
        cal.previous_session(date(2026, 8, 14))


def test_next_session_is_strictly_after_and_never_guesses_past_range(cal):
    assert cal.next_session(date(2026, 8, 14)) == date(2026, 8, 17)
    assert cal.next_session(date(2026, 8, 15)) == date(2026, 8, 17)
    assert cal.next_session(date(2026, 8, 18)) == date(2026, 8, 20)
    assert cal.next_session(date(2026, 1, 1)) == date(2026, 8, 14)
    with pytest.raises(LookupError):
        cal.next_session(date(2026, 8, 20))  # 21(금)을 추측하지 않는다


def test_sessions_between_inclusive(cal):
    assert cal.sessions_between(date(2026, 8, 14), date(2026, 8, 18)) == [
        date(2026, 8, 14), date(2026, 8, 17), date(2026, 8, 18)]
    assert cal.sessions_between(date(2026, 8, 15), date(2026, 8, 19)) == [
        date(2026, 8, 17), date(2026, 8, 18)]
    assert cal.sessions_between(date(2026, 8, 20), date(2026, 8, 14)) == []


def test_offset(cal):
    assert cal.offset(date(2026, 8, 14), 0) == date(2026, 8, 14)
    assert cal.offset(date(2026, 8, 14), 3) == date(2026, 8, 20)
    assert cal.offset(date(2026, 8, 20), -2) == date(2026, 8, 17)
    with pytest.raises(ValueError):
        cal.offset(date(2026, 8, 15), 1)
    with pytest.raises(LookupError):
        cal.offset(date(2026, 8, 18), 2)
    with pytest.raises(LookupError):
        cal.offset(date(2026, 8, 14), -1)


def test_empty_calendar():
    c = TradingCalendar.from_dates([])
    assert len(c) == 0
    assert not c.is_session(date(2026, 8, 14))
    with pytest.raises(LookupError):
        _ = c.first
    with pytest.raises(LookupError):
        c.next_session(date(2026, 8, 14))


# ---- align_to_trading_days (scalp-it 이식) ----------------------------------


@pytest.fixture()
def calendar() -> pd.Series:
    """금(8/14) · 월(8/17) · 화(8/18) 만 거래일. 주말은 휴장."""
    return pd.Series(pd.to_datetime(["2026-08-14", "2026-08-17", "2026-08-18"]))


def _articles(dates: list[str], news_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "code": ["005930"] * len(dates),
            "outlet": ["한국경제"] * len(dates),
            "news_id": news_ids,
            "title": ["제목"] * len(dates),
        }
    )


def test_trading_day_article_keeps_its_date(calendar):
    out = align_to_trading_days(_articles(["2026-08-14"], ["n1"]), calendar)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-14")
    assert out.iloc[0]["published_date"] == pd.Timestamp("2026-08-14")


def test_saturday_article_moves_to_monday(calendar):
    out = align_to_trading_days(_articles(["2026-08-15"], ["n1"]), calendar)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")
    assert out.iloc[0]["published_date"] == pd.Timestamp("2026-08-15")


def test_sunday_article_also_moves_to_monday(calendar):
    out = align_to_trading_days(_articles(["2026-08-16"], ["n1"]), calendar)
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_weekend_articles_collapse_onto_the_same_trading_day(calendar):
    out = align_to_trading_days(
        _articles(["2026-08-15", "2026-08-16"], ["n1", "n2"]), calendar
    )
    assert set(out["date"]) == {pd.Timestamp("2026-08-17")}
    assert len(out) == 2


def test_article_after_the_last_trading_day_is_dropped(calendar):
    out = align_to_trading_days(_articles(["2026-08-19"], ["n1"]), calendar)
    assert out.empty


def test_forward_only_never_moves_backwards(calendar):
    out = align_to_trading_days(_articles(["2026-08-15"], ["n1"]), calendar)
    assert (out["date"] >= out["published_date"]).all()


def test_empty_input_returns_empty_with_columns(calendar):
    out = align_to_trading_days(_articles([], []), calendar)
    assert out.empty
    assert "published_date" in out.columns


def test_other_columns_are_preserved(calendar):
    out = align_to_trading_days(_articles(["2026-08-15"], ["n1"]), calendar)
    assert out.iloc[0]["news_id"] == "n1"
    assert out.iloc[0]["code"] == "005930"
    assert out.iloc[0]["outlet"] == "한국경제"


def test_empty_calendar_drops_everything():
    empty_cal = pd.Series([], dtype="datetime64[ns]")
    out = align_to_trading_days(_articles(["2026-08-15"], ["n1"]), empty_cal)
    assert out.empty
