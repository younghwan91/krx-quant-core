"""리뷰(2026-10-08) 지적 회귀 — 연도 캐시 행 수 변화, 정수 wrap, 빈 날, 달력 지연, codes 인자 등."""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from krx_quant_core import data as D
from krx_quant_core.backtest.ticktime import forward_max_last, trailing_sums
from krx_quant_core.data import daily as daily_mod
from krx_quant_core.data import intraday as intraday_mod
from krx_quant_core.market.ticks import tick_size_array
from krx_quant_core.stats import cluster_bootstrap_diff_ci

TODAY = date(2026, 10, 8)


def _daily_db():
    con = sqlite3.connect(":memory:")
    for t in ("daily_bars", "daily_bars_adjusted"):
        con.execute(
            f"create table {t} (code text, date date, open real, high real, low real, close real, "
            "volume int, trade_value int, source text)"
        )
    return con


def _add_day(con, d, code="005930", c=100.0, table="daily_bars_adjusted"):
    con.execute(f"insert into {table} values (?,?,?,?,?,?,?,?,?)",
                (code, d, c, c, c, c, 1, 1, "k"))  # fmt: skip


def test_year_cache_survives_new_rows(tmp_path):  # H1
    con = _daily_db()
    _add_day(con, "2026-10-06")
    _add_day(con, "2026-10-07")
    st = D.ColumnStore("daily_bars_adjusted", root=tmp_path)
    assert len(D.load_daily_bars("2026-01-01", "2026-12-31", conn=con, store=st)) == 2
    _add_day(con, "2026-10-08")
    got = D.load_daily_bars("2026-01-01", "2026-12-31", conn=con, store=st)
    assert len(got) == 3 and st.meta("2026")["gen"] == 1
    for k in range(3):  # 세대가 쌓여도 직전 것 하나만 남는다
        _add_day(con, f"2026-11-0{k + 2}")
        D.load_daily_bars("2026-01-01", "2026-12-31", conn=con, store=st)
    gens = sorted(p.name for p in st.path("2026").glob("g*"))
    assert gens == ["g3", "g4"]


class _FakePg:
    __module__ = "psycopg.fake"

    def close(self):
        pass


def test_pg_path_reads_int64_then_range_checks(monkeypatch):  # H2
    seen = {}

    def fake_fetch(conn, sql, names, *, dtypes=None, timestamps=()):
        seen.update(dtypes)
        df = pd.DataFrame({n: [0] for n in names})
        df["code"] = "005930"
        df["ts"] = pd.Timestamp("2026-10-07 09:00:00")
        df["price"] = 3_000_000_000  # int32 범위 밖
        return df

    monkeypatch.setattr(intraday_mod, "fetch_frame", fake_fetch)
    with pytest.raises(OverflowError, match="price"):
        D.load_ticks(date(2026, 10, 7), conn=_FakePg(), store=False, today=TODAY)
    assert seen["price"] == "int64" and seen["side"] == "int64" and seen["strength"] == "float64"


def _tick_db(rows):
    con = sqlite3.connect(":memory:")
    con.execute(
        "create table ticks (code text, ts timestamp, seq int, price int, volume int, side int, "
        "strength real, cum_volume int, best_bid int, best_ask int, raw text)"
    )
    con.executemany("insert into ticks values (?,?,?,?,?,?,?,?,?,?,'{}')", rows)
    return con


def test_empty_day_not_cached_and_refresh(tmp_path):  # M1
    con = _tick_db([])
    st = D.ColumnStore("ticks", root=tmp_path)
    day = date(2026, 10, 7)
    assert len(D.load_ticks(day, conn=con, store=st, today=TODAY)) == 0
    assert not st.columns(day.isoformat())
    con.execute(
        "insert into ticks values ('005930','2026-10-07 09:00:01',0,100,1,1,100,1,99,100,'')"
    )
    assert len(D.load_ticks(day, conn=con, store=st, today=TODAY)) == 1  # 백필이 보인다
    con.execute(
        "insert into ticks values ('005930','2026-10-07 09:00:02',0,100,1,1,100,1,99,100,'')"
    )
    assert len(D.load_ticks(day, conn=con, store=st, today=TODAY)) == 1  # 닫힌 날 캐시
    assert len(D.load_ticks(day, conn=con, store=st, today=TODAY, refresh=True)) == 2


def test_subsecond_timestamps_refused():  # M6
    con = _tick_db([("005930", "2026-10-07 09:00:01.500", 0, 100, 1, 1, 100.0, 1, 99, 100)])
    with pytest.raises(ValueError, match="sub-second"):
        D.load_ticks(date(2026, 10, 7), conn=con, store=False, today=TODAY)


def test_codes_argument_is_strict(tmp_path):  # M3
    con = _tick_db([("005930", "2026-10-07 09:00:01", 0, 100, 1, 1, 100.0, 1, 99, 100)])
    with pytest.raises(TypeError):
        D.load_ticks(date(2026, 10, 7), conn=con, store=False, today=TODAY, codes="005930")
    for store in (False, D.ColumnStore("ticks", root=tmp_path)):
        assert len(D.load_ticks(date(2026, 10, 7), conn=con, store=store, today=TODAY,
                                codes=[])) == 0  # fmt: skip
    with pytest.raises(TypeError):
        D.load_daily_bars("2026-01-01", "2026-01-02", conn=_daily_db(), store=False, codes="005930")


def test_calendar_not_cached_when_previous_weekday_missing(tmp_path, monkeypatch):  # M2
    con = _daily_db()
    for d in ("2026-10-05", "2026-10-06"):
        _add_day(con, d, table="daily_bars")
    daily_mod._CAL_MEMO.clear()
    monkeypatch.setattr(daily_mod, "_STALE_TTL", 0.0)  # 지연 달력은 바로 만료
    st = D.ColumnStore("cal", root=tmp_path)
    cal = D.trading_calendar(conn=con, store=st, today=TODAY)  # 10-07 수집 실패 상태
    assert cal.previous_session(TODAY) == date(2026, 10, 6)
    assert not st.columns(f"2010-01-01@{TODAY}")
    _add_day(con, "2026-10-07", table="daily_bars")  # 재시도로 채워짐
    cal = D.trading_calendar(conn=con, store=st, today=TODAY)
    assert cal.previous_session(TODAY) == date(2026, 10, 7)
    assert st.columns(f"2010-01-01@{TODAY}")


def test_ticktime_rejects_bad_window_and_nan():  # M4
    with pytest.raises(ValueError):
        trailing_sums([1, 2, 3], [1.0, 1.0, 1.0], window=0)
    with pytest.raises(ValueError):
        trailing_sums([1, np.nan, 3], [1.0, 1.0, 1.0], window=5)
    with pytest.raises(ValueError):
        forward_max_last([1, 2], [1.0, 2.0], -1)


def test_diff_ci_mixed_date_types_pair_same_days():  # M5
    days = pd.date_range("2026-01-01", periods=20)
    rng = np.random.default_rng(0)
    da, db = rng.choice(days, 200), rng.choice(days, 200)
    a, b = rng.normal(1, 1, 200), rng.normal(0, 1, 200)
    ref = cluster_bootstrap_diff_ci(a, pd.Index(da).date, b, pd.Index(db).date, n_boot=300)
    mixed = cluster_bootstrap_diff_ci(a, da.astype("datetime64[ns]"), b, pd.Index(db).date,
                                      n_boot=300)  # fmt: skip
    assert ref[:3] == mixed[:3]


def test_panel_rejects_missing_code():  # L1
    bars = pd.DataFrame({
        "code": pd.Categorical(["A00000", None]),
        "date": pd.to_datetime(["2026-01-02"] * 2),
        "close": [1.0, 2.0],
    })  # fmt: skip
    with pytest.raises(ValueError, match="missing codes"):
        D.daily_panel(bars)


def test_sql_literal_numpy_and_nonfinite():  # L3
    assert D.sql_literal(np.float64(1.5)) == "1.5" and D.sql_literal(np.int64(3)) == "3"
    for bad in (float("nan"), np.float64("inf")):
        with pytest.raises(ValueError):
            D.sql_literal(bad)


def test_tick_size_array_scalar():  # L4
    assert tick_size_array(5000) == 10.0 and tick_size_array([[1, 2000]]).shape == (1, 2)


def _writer(root, k):
    st = D.ColumnStore("x", root=root)
    st.write("day", {"a": np.full(1000 + k, k, np.int32)}, aux={"n": np.array([k])}, replace=True)


def test_concurrent_replace_writers_leave_consistent_key(tmp_path):
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_writer, args=(tmp_path, k)) for k in range(8)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    st = D.ColumnStore("x", root=tmp_path)
    got = st.read("day", ["a", "n"])
    k = int(got["n"][0])
    assert len(got["a"]) == 1000 + k and (got["a"] == k).all()  # 한 작성자의 열들만
