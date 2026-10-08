"""data — 열 캐시·하루치 CSR·일봉 지문 캐시·달력. Postgres 대신 sqlite 로 같은 스키마를 만든다."""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from krx_quant_core import data as D
from krx_quant_core.backtest.lob import build_second_grid
from krx_quant_core.data import daily as daily_mod
from krx_quant_core.data import intraday as intraday_mod

DAY = date(2026, 10, 7)
TODAY = date(2026, 10, 8)


def _no_db(*_a, **_k):
    raise AssertionError("DB must not be touched on a cache hit")


@pytest.fixture
def db():
    rng = np.random.default_rng(0)
    con = sqlite3.connect(":memory:")
    con.execute(
        "create table ticks (code text, ts timestamp, seq int, price int, volume int, side int, "
        "strength real, cum_volume int, best_bid int, best_ask int, raw text)"
    )
    q_cols = [f"{p}{i}" for p in ("bid", "bidqty", "ask", "askqty") for i in range(1, 11)]
    con.execute(
        "create table quotes (code text, ts timestamp, "
        + ", ".join(f"{c} int" for c in q_cols)
        + ", total_bid_qty int, total_ask_qty int, vi_flag int, raw text)"
    )
    con.execute(
        "create table minute_bars (code text, ts timestamp, open real, high real, low real, "
        "close real, volume int, trade_value real)"
    )
    for t in ("daily_bars", "daily_bars_adjusted"):
        con.execute(
            f"create table {t} (code text, date date, open real, high real, low real, close real, "
            "volume int, trade_value int, source text)"
        )
    rows_t, rows_q, rows_m = [], [], []
    for day in (DAY, TODAY):
        for code in ("005930", "000660", "123456"):
            px = 10000
            for k in range(400):
                s = 9 * 3600 + int(k * 50 + rng.integers(0, 3))
                ts = f"{day} {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
                px += int(rng.integers(-1, 2)) * 10
                side = int(rng.choice([-1, 1]))
                rows_t.append((code, ts, k, px, int(rng.integers(1, 500)), side,
                               float(rng.uniform(50, 150)), k * 10, px - 10, px))  # fmt: skip
                if k % 2 == 0:
                    lv = [px - 10 * i for i in range(10)] + list(rng.integers(1, 9000, 10))
                    lv += [px + 10 * i for i in range(1, 11)] + list(rng.integers(1, 9000, 10))
                    rows_q.append((code, ts, *map(int, lv), 100000, 90000, None, "{}"))
            for m in range(30):
                ts = f"{day} 09:{m:02d}:00"
                rows_m.append((code, ts, 1000.0, 1010.0, 990.0, 1005.0, 100, 100500.0))
    con.executemany(f"insert into ticks values ({','.join('?' * 10)}, '{{}}')", rows_t)
    con.executemany(f"insert into quotes values ({','.join('?' * 46)})", rows_q)
    con.executemany(f"insert into minute_bars values ({','.join('?' * 8)})", rows_m)
    for t in ("daily_bars", "daily_bars_adjusted"):
        for d in pd.bdate_range("2026-09-01", "2026-10-08"):
            for code, base in (("005930", 70000), ("000660", 150000)):
                c = base + d.day * 100
                con.execute(
                    f"insert into {t} values (?,?,?,?,?,?,?,?,?)",
                    (code, d.date().isoformat(), c, c + 50, c - 50, c, 1000, c * 1000, "kiwoom"),
                )
    con.commit()
    return con


def test_store_roundtrip(tmp_path):
    st = D.ColumnStore("x", root=tmp_path)
    st.write("k", {"a": np.arange(5, dtype=np.int32), "b": np.ones(5)}, aux={"idx": np.arange(2)})
    assert st.has("k", ("a", "b", "idx"))
    got = st.read("k", ["a", "idx"])
    np.testing.assert_array_equal(got["a"], np.arange(5))
    with pytest.raises(ValueError):
        got["a"][0] = 9  # mmap 은 읽기 전용
    with pytest.raises(ValueError):
        st.write("k", {"c": np.ones(4)})  # 행 수 불일치
    with pytest.raises(TypeError):
        st.write("k", {"d": np.array(["a"] * 5, dtype=object)})
    with pytest.raises(ValueError):
        D.ColumnStore("../evil", root=tmp_path)


def test_ticks_cache_and_parity(db, tmp_path, monkeypatch):
    st = D.ColumnStore("ticks", root=tmp_path)
    cd = D.load_ticks(DAY, conn=db, store=st, today=TODAY)
    assert list(cd.codes) == ["000660", "005930", "123456"] and len(cd) == 1200
    assert cd.cols["price"].dtype == np.int32 and cd.cols["side"].dtype == np.int8
    ref = pd.read_sql(
        "select code, ts, seq, price, volume, side, strength, cum_volume, best_bid, best_ask "
        "from ticks where ts >= '2026-10-07' and ts < '2026-10-08' order by code, ts, seq",
        db,
        parse_dates=["ts"],
    )
    f = cd.frame("005930")
    r = ref[ref.code == "005930"].drop(columns="code").reset_index(drop=True)
    pd.testing.assert_frame_equal(f[r.columns], r)
    full = cd.frame()
    assert full["code"].dtype == "category" and len(full) == 1200

    # 두 번째는 캐시(mmap)만 — DB 를 건드리면 실패
    monkeypatch.setattr(intraday_mod, "connect", _no_db)
    cd2 = D.load_ticks(DAY, store=st, today=TODAY, columns=["price", "volume"], codes=["005930"])
    assert cd2.columns == ("sec", "price", "volume") and list(cd2.codes) == ["005930"]
    np.testing.assert_array_equal(cd2.get("005930")["price"], cd.get("005930")["price"])


def test_today_is_not_cached(db, tmp_path):
    st = D.ColumnStore("ticks", root=tmp_path)
    cd = D.load_ticks(TODAY, conn=db, store=st, today=TODAY, codes=["000660"])
    assert list(cd.codes) == ["000660"]
    assert not st.columns(TODAY.isoformat())


def test_quotes_levels_and_grid_parity(db, tmp_path):
    tk = D.load_ticks(DAY, conn=db, store=D.ColumnStore("ticks", root=tmp_path), today=TODAY)
    qs = D.load_quotes(DAY, conn=db, store=D.ColumnStore("quotes", root=tmp_path), today=TODAY,
                       levels=3)  # fmt: skip
    assert "bid4" not in qs.cols and qs.cols["bidqty3"].dtype == np.int32
    ref_t = pd.read_sql("select * from ticks where code='000660' and ts < '2026-10-08' "
                        "order by ts, seq", db, parse_dates=["ts"])  # fmt: skip
    ref_q = pd.read_sql("select * from quotes where code='000660' and ts < '2026-10-08' "
                        "order by ts", db, parse_dates=["ts"])  # fmt: skip
    a, pa = build_second_grid(tk.frame("000660"), qs.frame("000660"), warmup_sec=0, tail_sec=0)
    b, pb = build_second_grid(ref_t, ref_q, warmup_sec=0, tail_sec=0)
    pd.testing.assert_frame_equal(a, b, check_exact=True)  # 비트 동일
    for k in pa:
        np.testing.assert_array_equal(pa[k], pb[k])


def test_minute_bars(db, tmp_path):
    mb = D.load_minute_bars(DAY, conn=db, store=D.ColumnStore("mb", root=tmp_path), today=TODAY)
    assert mb.cols["close"].dtype == np.int32 and mb.n_codes == 3
    assert mb.get("005930")["sec"][1] == 9 * 3600 + 60


def test_null_and_fraction_refused(db, tmp_path):
    db.execute("update minute_bars set close = 1005.5 where code='005930'")
    with pytest.raises(ValueError, match="non-integer"):
        D.load_minute_bars(DAY, conn=db, store=False, today=TODAY)


def test_daily_fingerprint_cache(db, tmp_path, monkeypatch):
    st = D.ColumnStore("daily_bars_adjusted", root=tmp_path)
    df = D.load_daily_bars("2026-09-01", "2026-10-08", conn=db, store=st)
    assert df["code"].dtype == "category" and set(df["code"]) == {"005930", "000660"}
    calls = []
    real = daily_mod.fetch_frame
    monkeypatch.setattr(daily_mod, "fetch_frame", lambda *a, **k: calls.append(1) or real(*a, **k))
    df2 = D.load_daily_bars("2026-09-01", "2026-10-08", conn=db, store=st)
    assert not calls
    pd.testing.assert_frame_equal(df, df2)
    # 수정주가 재계산 → 지문이 바뀌면 다시 받는다
    db.execute("update daily_bars_adjusted set close = close / 2 where code='000660'")
    df3 = D.load_daily_bars("2026-09-01", "2026-10-08", conn=db, store=st, codes=["000660"])
    assert calls and set(df3["code"]) == {"000660"}
    assert df3["close"].iloc[0] == pytest.approx(df[df.code == "000660"]["close"].iloc[0] / 2)


def test_daily_panel_orientation():
    bars = pd.DataFrame({
        "code": ["A00000", "A00000", "B00000"],
        "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-05"]),
        "close": [-100.0, 101.0, 50.0],
    })  # fmt: skip
    p = D.daily_panel(bars, ["close"], dtype=np.float32)
    assert p.array("close").shape == (2, 2) and p.array("close").dtype == np.float32
    assert p.array("close")[0, 0] == 100.0 and np.isnan(p.array("close")[1, 0])
    f = p.frame("close", orient="date_x_code")
    assert list(f.columns) == ["A00000", "B00000"] and f.index[0] == pd.Timestamp("2026-01-02")
    assert p.frame("close", orient="code_x_date").shape == (2, 2)
    with pytest.raises(ValueError):
        p.frame("close", orient="wide")  # type: ignore[arg-type]


def test_trading_calendar_memo(db, tmp_path, monkeypatch):
    daily_mod._CAL_MEMO.clear()
    cal = D.trading_calendar(conn=db, store=D.ColumnStore("cal", root=tmp_path), today=TODAY)
    assert cal.previous_session(date(2026, 10, 5)) == date(2026, 10, 2)
    daily_mod._CAL_MEMO.clear()
    monkeypatch.setattr(daily_mod, "connect", _no_db)
    cal2 = D.trading_calendar(store=D.ColumnStore("cal", root=tmp_path), today=TODAY)
    assert cal2.sessions == cal.sessions


def test_resolve_dsn(tmp_path, monkeypatch):
    monkeypatch.delenv("KR_QUANT_DB", raising=False)
    monkeypatch.delenv("KQC_ENV_FILE", raising=False)
    (tmp_path / "quant-airflow").mkdir()
    (tmp_path / "quant-airflow" / ".env").write_text('X=1\nKR_QUANT_DB="postgresql://u:p@h/db"\n')
    sub = tmp_path / "repo" / "pkg"
    sub.mkdir(parents=True)
    assert D.resolve_dsn(search_from=sub) == "postgresql://u:p@h/db"
    monkeypatch.setenv("KR_QUANT_DB", "postgresql://env")
    assert D.resolve_dsn(search_from=sub) == "postgresql://env"
    assert D.resolve_dsn("explicit") == "explicit"


def test_sql_literal_rejects_injection():
    assert D.sql_literal(date(2026, 1, 2)) == "'2026-01-02'"
    assert D.sql_literal("005930") == "'005930'"
    for bad in ("0059'; drop table x; --", "abc", True, None):
        with pytest.raises(ValueError):
            D.sql_literal(bad)


def test_daily_years_partition_and_order(tmp_path, monkeypatch):
    con = sqlite3.connect(":memory:")
    con.execute(
        "create table daily_bars_adjusted (code text, date date, open real, high real, low real, "
        "close real, volume int, trade_value int, source text)"
    )
    rows = []
    for d in pd.bdate_range("2024-12-20", "2026-01-10"):
        for code in ("000660", "005930") + (("999999",) if d.year == 2025 else ()):
            c = 1000.0 + d.dayofyear
            rows.append((code, d.date().isoformat(), c, c, c, c + 0.25, 10, 1000, "k"))
    con.executemany("insert into daily_bars_adjusted values (?,?,?,?,?,?,?,?,?)", rows)
    st = D.ColumnStore("daily_bars_adjusted", root=tmp_path)
    got = D.load_daily_bars("2024-12-30", "2026-01-05", conn=con, store=st)
    ref = pd.read_sql(
        "select code, date, open, high, low, close, volume, trade_value from daily_bars_adjusted "
        "where date >= '2024-12-30' and date <= '2026-01-05' order by code, date",
        con,
        parse_dates=["date"],
    )
    pd.testing.assert_frame_equal(got.assign(code=got["code"].astype(str)), ref, check_dtype=False)
    assert list(got["code"].cat.categories) == ["000660", "005930", "999999"]

    fetched = []
    real = daily_mod._fetch
    monkeypatch.setattr(
        daily_mod, "_fetch", lambda c, t, lo, hi: fetched.append(lo.year) or real(c, t, lo, hi)
    )
    con.execute("update daily_bars_adjusted set close = close + 1 where date >= '2026-01-01'")
    again = D.load_daily_bars("2024-12-30", "2026-01-05", conn=con, store=st, codes=["005930"])
    assert fetched == [2026]  # 바뀐 해만 다시 받는다
    assert set(again["code"]) == {"005930"} and again["date"].is_monotonic_increasing


def test_daily_panel_unsorted_duplicates_keep_first():
    bars = pd.DataFrame({
        "code": ["B00000", "A00000", "A00000", "A00000"],
        "date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-02", "2026-01-05"]),
        "close": [50.0, 101.0, 100.0, 999.0],
    })  # fmt: skip
    p = D.daily_panel(bars, ["close"])
    np.testing.assert_array_equal(p.array("close"), [[100.0, 101.0], [np.nan, 50.0]])
