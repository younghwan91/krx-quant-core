"""daytrade-it tests/unit/infrastructure/test_dart_disclosure_db.py 이식.

Same-day DART disclosures for the entry risk gate, read from scalp-it's dart.db."""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import time
from pathlib import Path

import pytest

from krx_quant_core.market.session import KST
from krx_quant_core.risk import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_STALENESS,
    DartDisclosureDB,
    DisclosureDataUnavailable,
)

NOW = dt.datetime(2026, 9, 14, 10, 0, tzinfo=KST)


def _make_db(path: Path, rows: list[tuple[str, str, str]], *, age: dt.timedelta) -> Path:
    """rows: (rcept_dt YYYYMMDD, code, report_nm). mtime = NOW - age."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE disclosures (rcept_no TEXT PRIMARY KEY, rcept_dt TEXT NOT NULL, "
        "code TEXT NOT NULL, corp_name TEXT, corp_cls TEXT, report_nm TEXT, "
        "material_type TEXT, is_correction INTEGER)"
    )
    for i, (rcept_dt, code, report_nm) in enumerate(rows):
        con.execute(
            "INSERT INTO disclosures VALUES (?, ?, ?, '', 'K', ?, '', 0)",
            (str(i), rcept_dt, code, report_nm),
        )
    con.commit()
    con.close()
    mtime = (NOW - age).timestamp()
    os.utime(path, (mtime, mtime))
    return path


def _db(tmp_path: Path, rows: list[tuple[str, str, str]], age_min: int = 5) -> DartDisclosureDB:
    path = _make_db(tmp_path / "dart.db", rows, age=dt.timedelta(minutes=age_min))
    return DartDisclosureDB(path, clock=lambda: NOW)


def test_returns_hard_disclosure_within_lookback(tmp_path: Path) -> None:
    db = _db(tmp_path, [("20260909", "005930", "횡령ㆍ배임혐의발생")])
    assert db.hard_disclosure("005930") == "횡령ㆍ배임혐의발생"


def test_routine_disclosures_return_none(tmp_path: Path) -> None:
    db = _db(
        tmp_path,
        [
            ("20260914", "005930", "단일판매ㆍ공급계약체결"),
            ("20260914", "005930", "주권매매거래정지 (단일판매공급계약)"),
        ],
    )
    assert db.hard_disclosure("005930") is None


def test_other_tickers_do_not_leak(tmp_path: Path) -> None:
    db = _db(tmp_path, [("20260914", "000660", "상장폐지결정")])
    assert db.hard_disclosure("005930") is None


def test_disclosures_older_than_lookback_are_ignored(tmp_path: Path) -> None:
    # 7 calendar days back from 09-14 is 09-07 (inclusive).
    db = _db(
        tmp_path,
        [("20260906", "005930", "상장폐지결정"), ("20260907", "000660", "상장폐지결정")],
    )
    assert db.hard_disclosure("005930") is None
    assert db.hard_disclosure("000660") == "상장폐지결정"


def test_most_recent_hard_disclosure_wins(tmp_path: Path) -> None:
    db = _db(
        tmp_path,
        [("20260908", "005930", "파산신청"), ("20260912", "005930", "관리종목지정")],
    )
    assert db.hard_disclosure("005930") == "관리종목지정"


def test_missing_file_is_unavailable(tmp_path: Path) -> None:
    db = DartDisclosureDB(tmp_path / "nope.db", clock=lambda: NOW)
    with pytest.raises(DisclosureDataUnavailable, match="not found"):
        db.hard_disclosure("005930")


def test_stale_file_is_unavailable(tmp_path: Path) -> None:
    # A gate over data nobody refreshed would pass everything while looking wired.
    db = _db(tmp_path, [], age_min=31)
    with pytest.raises(DisclosureDataUnavailable, match="stale"):
        db.hard_disclosure("005930")


def test_unreadable_file_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "dart.db"
    path.write_text("not a sqlite file")
    now = time.time()
    os.utime(path, (now, now))
    db = DartDisclosureDB(path, clock=lambda: dt.datetime.fromtimestamp(now, KST))
    with pytest.raises(DisclosureDataUnavailable, match="unreadable"):
        db.hard_disclosure("005930")


def test_defaults_match_daytrade_it() -> None:
    assert DEFAULT_MAX_STALENESS == dt.timedelta(minutes=30)
    assert DEFAULT_LOOKBACK_DAYS == 7


def test_stale_message_format(tmp_path: Path) -> None:
    db = _db(tmp_path, [], age_min=45)
    with pytest.raises(DisclosureDataUnavailable) as exc:
        db.hard_disclosure("005930")
    assert str(exc.value) == (
        "dart.db stale: last refreshed 2026-09-14 09:15 KST (45 min ago, limit 30 min)"
    )


def test_exactly_at_staleness_limit_is_still_fresh(tmp_path: Path) -> None:
    db = _db(tmp_path, [("20260914", "005930", "상장폐지결정")], age_min=30)
    assert db.hard_disclosure("005930") == "상장폐지결정"


def test_mechanical_halt_then_hard_filing_same_day(tmp_path: Path) -> None:
    # 같은 날엔 rcept_no 내림차순 — 기계적 정지는 건너뛰고 HARD 를 찾는다.
    db = _db(
        tmp_path,
        [
            ("20260914", "005930", "주권매매거래정지 (무상증자)"),
            ("20260914", "005930", "감사보고서제출(감사의견거절)"),
        ],
    )
    assert db.hard_disclosure("005930") == "감사보고서제출(감사의견거절)"
