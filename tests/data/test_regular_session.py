"""정규장 방어 — 애프터마켓 분봉 걸러내기, 일봉 session="regular", 수급 항등식 기준가.

sqlite 로 같은 스키마를 만든다. 2026-09-14 부터 분봉에 16:00~19:59 애프터마켓 봉이 섞이고 일봉
고가·저가·종가·거래량이 그것을 포함한다는 실제 결함을 가짜 데이터로 재현한다.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from krx_quant_core import data as D
from krx_quant_core.data import daily as daily_mod

D1, D2, D3 = date(2026, 10, 7), date(2026, 10, 8), date(2026, 10, 12)  # 10-09 휴장
P1, P2 = date(2026, 9, 10), date(2026, 9, 11)  # 애프터마켓 전

# 005930 D2 분봉: (시각, open, high, low, close, volume) — 15:35 장후 시간외, 16:00~ 애프터마켓.
M_005930_D2 = [
    ("09:00:00", 269500, 270000, 269000, 269800, 100),
    ("12:00:00", 265000, 266000, 262000, 263000, 300),
    ("15:19:00", 264000, 264500, 264000, 264500, 50),
    ("15:32:00", 262000, 262000, 262000, 262000, 400),
    ("15:35:00", 262000, 262000, 262000, 262000, 7),
    ("16:00:00", 263000, 263500, 263000, 263500, 20),
    ("19:59:00", 263000, 263000, 261000, 263000, 30),
]
M_005930_D1 = [
    ("09:00:00", 269500, 279500, 269500, 270000, 100),
    ("15:30:00", 268500, 268500, 268500, 268500, 200),
    ("17:00:00", 269000, 269500, 268000, 269000, 10),
]
M_111111_D2 = [("09:00:00", 5000, 5100, 4990, 5050, 10), ("15:30:00", 5070, 5070, 5070, 5070, 5)]
M_222222_D2 = [("09:00:00", 7000, 7100, 6990, 7050, 10), ("15:19:00", 7080, 7080, 7080, 7080, 5)]
# 분봉이 애프터마켓에만 있는 종목 — 정규장 필터 뒤 빈 종목이 된다.
M_777777_D2 = [("16:30:00", 1000, 1000, 1000, 1000, 1)]


def _mins(code, day, rows):
    return [
        (code, f"{day} {t}", o, h, lo, c, v, float(c * v)) for t, o, h, lo, c, v in rows
    ]  # fmt: skip


@pytest.fixture
def db():
    con = sqlite3.connect(":memory:")
    con.execute(
        "create table minute_bars (code text, ts timestamp, open real, high real, low real, "
        "close real, volume int, trade_value real)"
    )
    for t in ("daily_bars", "daily_bars_adjusted", "daily_bars_raw"):
        con.execute(
            f"create table {t} (code text, date date, open real, high real, low real, close real, "
            "volume int, trade_value int, source text)"
        )
    con.execute(
        "create table supply_demand (code text, date date, close int, flu_rt real, source text)"
    )
    mins = (
        _mins("005930", D1, M_005930_D1)
        + _mins("005930", D2, M_005930_D2)
        + _mins("111111", D2, M_111111_D2)
        + _mins("222222", D2, M_222222_D2)
        + _mins("777777", D2, M_777777_D2)
    )
    con.executemany("insert into minute_bars values (?,?,?,?,?,?,?,?)", mins)
    # 일봉(통합 — 애프터마켓 포함). 수정주가 테이블은 가격 × 0.5, 거래량 × 2(분할 소급처럼).
    daily = [
        ("005930", P1, 70000, 71000, 69000, 70500, 1000, 70),
        ("005930", P2, 70500, 72000, 70000, 71000, 1000, 71),
        ("005930", D1, 269500, 279500, 268000, 269000, 310, 84),
        ("005930", D2, 269500, 270000, 261000, 263000, 907, 238),
        ("000660", D1, 150000, 156000, 149000, 155000, 50, 8),
        ("000660", D2, 155000, 160000, 154000, 158000, 60, 9),
        ("000660", D3, 158000, 161000, 157000, 160000, 70, 11),
        ("222222", D2, 7000, 7100, 6990, 7100, 15, 1),
        ("555555", P1, 3000, 3000, 3000, 3000, 1, 0),
    ]
    for code, d, o, h, lo, c, v, tv in daily:
        con.execute(
            "insert into daily_bars values (?,?,?,?,?,?,?,?,?)",
            (code, d.isoformat(), o, h, lo, c, v, tv, "kiwoom"),
        )
        con.execute(
            "insert into daily_bars_adjusted values (?,?,?,?,?,?,?,?,?)",
            (code, d.isoformat(), o * 0.5, h * 0.5, lo * 0.5, c * 0.5, v * 2, tv, "adj"),
        )
    con.execute(
        "insert into daily_bars_raw values (?,?,?,?,?,?,?,?,?)",
        ("444444", P1.isoformat(), 0, 0, 0, 4100, 1, 0, "raw"),
    )
    # 수급: close 는 통합 종가(애프터마켓 포함), flu_rt 는 KRX 기준가(전일 정규장 종가) 대비 ×100.
    # 000660 정규장 종가: D1 154,500 / D2 157,000(분봉 없음 — 항등식으로만 안다).
    sd = [
        ("005930", P1, 70500, 0.0),
        ("005930", P2, 71000, 71.0),  # 71000/70500-1 = 70.9bp → 71
        ("005930", D2, 263000, -205.0),  # 기준 268,500 → 263000/268500-1 = -204.8bp
        ("005930", D3, 251000, -420.0),  # 기준 262,000 → -420.0bp
        ("000660", D1, 155000, 0.0),
        ("000660", D2, 158000, 227.0),  # 158000/154500-1 = 226.5bp → 227(반올림 경계 쪽)
        ("000660", D3, 160000, 191.0),  # 160000/157000-1 = 191.1bp
        ("333333", P1, -2500, 0.0),  # 키움 부호 관례 — 절댓값
    ]
    con.executemany(
        "insert into supply_demand values (?,?,?,?,'kiwoom')",
        [(c, d.isoformat(), cl, fr) for c, d, cl, fr in sd],
    )
    con.commit()
    return con


# ── 분봉 ──────────────────────────────────────────────────────────────────────


def test_regular_session_drops_aftermarket(db):
    allb = D.load_minute_bars(D2, conn=db, store=False)
    reg = D.load_minute_bars(D2, conn=db, store=False, session="regular")
    assert len(allb) == len(M_005930_D2) + 2 + 2 + 1  # 기본은 테이블 그대로
    secs = reg.cols["sec"]
    assert secs.max() == 15 * 3600 + 32 * 60 and secs.min() == 9 * 3600
    assert list(reg.codes) == list(allb.codes)  # 빈 종목도 자리는 남는다
    assert len(reg.get("777777")["sec"]) == 0
    np.testing.assert_array_equal(reg.get("005930")["close"], [269800, 263000, 264500, 262000])
    np.testing.assert_array_equal(reg.get("111111")["close"], [5050, 5070])
    # regular_session 은 로드와 같은 결과(함수만 따로 써도 된다)
    again = D.regular_session(allb)
    np.testing.assert_array_equal(again.ptr, reg.ptr)
    with pytest.raises(ValueError):
        D.load_minute_bars(D2, conn=db, store=False, session="nxt")


# ── 항등식 ────────────────────────────────────────────────────────────────────


def test_base_from_change_identity_and_grid():
    got = D.base_from_change(
        [251000, 2072000, -263000, 0, 158000, np.nan],
        [-420.0, 58.0, -205.0, 10.0, 227.0, 1.0],
    )
    np.testing.assert_array_equal(got[:3], [262000, 2060000, 268500])
    assert np.isnan(got[3]) and np.isnan(got[5])
    assert got[4] == 154500
    # 개편 전 KOSDAQ 고가주(100원 호가): 500원 격자는 등락률을 재현하지 못해 100원 격자로 내려간다.
    base, close = 300100, 309000
    flu = round((close / base - 1) * 1e4)
    assert D.base_from_change(close, flu) == base


def test_base_from_change_matches_round_trip():
    rng = np.random.default_rng(1)
    ticks = [(1, 1500), (5, 3000), (10, 9000), (50, 30000), (100, 120000), (500, 300000)]
    for tick, level in ticks:
        base = (level + rng.integers(-200, 200, 500) * tick).astype(float)
        close = np.round(base * (1 + rng.uniform(-0.29, 0.29, 500)) / tick) * tick
        flu = np.round((close / base - 1) * 1e4)
        np.testing.assert_array_equal(D.base_from_change(close, flu), base)


# ── 분봉 정규장 집계 ───────────────────────────────────────────────────────────


def test_regular_session_bars_aggregate(db):
    s = D.regular_session_bars(D2, D2, conn=db).set_index("code")
    assert list(s.index) == ["005930", "111111", "222222"]  # 애프터마켓만 있는 종목은 없다
    r = s.loc["005930"]
    assert (r.open, r.high, r.low, r.close) == (269500, 270000, 262000, 262000)
    assert r.volume == 100 + 300 + 50 + 400
    tv = sum(c * v for t, _, _, _, c, v in M_005930_D2 if t <= "15:32:00")
    assert r.trade_value == pytest.approx(tv / D.TRADE_VALUE_UNIT)
    assert r.last_ts == pd.Timestamp("2026-10-08 15:32:00")
    assert s.loc["222222"].last_ts == pd.Timestamp("2026-10-08 15:19:00")
    only = D.regular_session_bars(D1, D2, codes=["111111"], conn=db)
    assert list(only["code"]) == ["111111"] and list(only["date"]) == [pd.Timestamp(D2)]
    assert D.regular_session_bars(D2, D2, codes=[], conn=db).empty
    with pytest.raises(TypeError):
        D.regular_session_bars(D2, D2, codes="005930", conn=db)


# ── 일봉 session ──────────────────────────────────────────────────────────────


def test_daily_default_unchanged(db):
    base = D.load_daily_bars(P1, D3, adjusted=False, conn=db, store=False)
    assert tuple(base.columns) == ("code", "date", *D.DAILY_COLUMNS)  # 표시 열 없음
    explicit = D.load_daily_bars(P1, D3, adjusted=False, conn=db, store=False, session="all")
    pd.testing.assert_frame_equal(base, explicit)
    row = base[(base.code == "005930") & (base.date == pd.Timestamp(D2))].iloc[0]
    assert row.close == 263000  # 기본값은 테이블(통합 종가) 그대로
    with pytest.raises(ValueError):
        D.load_daily_bars(P1, D3, conn=db, store=False, session="nxt")


@pytest.mark.parametrize("adjusted", [False, True])
def test_daily_regular(db, adjusted):
    k = 0.5 if adjusted else 1.0
    allb = D.load_daily_bars(P1, D3, adjusted=adjusted, conn=db, store=False)
    reg = D.load_daily_bars(P1, D3, adjusted=adjusted, conn=db, store=False, session="regular")
    assert tuple(reg.columns) == ("code", "date", *D.DAILY_COLUMNS, *D.REGULAR_FLAGS)
    pd.testing.assert_frame_equal(reg[list(allb.columns)].iloc[:0], allb.iloc[:0])
    key = reg.set_index(["code", "date"])

    def at(code, d):
        return key.loc[(code, pd.Timestamp(d))]

    # 애프터마켓 전 행은 그대로, 표시 True
    pre = reg["date"] < pd.Timestamp(D.AFTERMARKET_START)
    pd.testing.assert_frame_equal(
        reg.loc[pre, list(allb.columns)], allb.loc[pre.to_numpy(), list(allb.columns)]
    )
    assert reg.loc[pre, list(D.REGULAR_FLAGS)].all().all()
    # 분봉 있는 행: 정규장 값 × 조정계수(행 시가 / 분봉 시가), 거래량은 ÷ 조정계수
    r = at("005930", D2)
    assert (r.open, r.high, r.low, r.close) == (269500 * k, 270000 * k, 262000 * k, 262000 * k)
    assert r.volume == pytest.approx(850 / k)
    assert r.trade_value == pytest.approx(
        sum(c * v for t, _, _, _, c, v in M_005930_D2 if t <= "15:32:00") / 1e6
    )
    assert r.regular_ok and r.close_ok
    assert at("005930", D1).close == 268500 * k
    # 분봉 없는 행: 다음 거래일 수급 항등식으로 종가만 복원(10-09 휴장을 건너 D3 를 쓴다)
    g1, g2, g3 = at("000660", D1), at("000660", D2), at("000660", D3)
    assert g1.close == 154500 * k and g2.close == 157000 * k
    assert (not g1.regular_ok) and g1.close_ok and g1.high == 156000 * k  # 고가는 테이블 값
    assert (not g3.regular_ok) and (not g3.close_ok) and g3.close == 160000 * k  # 다음날 없음
    # 15:19 가 마지막 봉인 종목도 분봉이 있으면 정규장 집계(일봉 규칙은 15:32 이하 마지막 봉)
    assert at("222222", D2).close == 7080 * k


def test_daily_regular_codes_subset(db):
    reg = D.load_daily_bars(D1, D2, adjusted=False, codes=["000660"], conn=db, store=False,
                            session="regular")  # fmt: skip
    assert list(reg["close"]) == [154500, 157000]


# ── 가격제한 기준가 ────────────────────────────────────────────────────────────


def test_base_price_after_aftermarket(db):
    bp = D.base_price(D3, ["005930", "111111", "222222", "000660", "999999"], conn=db)
    assert list(bp["code"]) == ["005930", "111111", "222222", "000660", "999999"]
    b = bp.set_index("code")
    assert (b.loc["005930"].base, b.loc["005930"].src) == (262000, "supply_demand_identity")
    assert (b.loc["000660"].base, b.loc["000660"].src) == (157000, "supply_demand_identity")
    assert (b.loc["111111"].base, b.loc["111111"].src) == (5070, "minute_bars")
    # 종가 단일가 봉이 없고 전일이 09-14 이후 → 일봉(통합 종가)으로 돌아가지 않는다
    assert np.isnan(b.loc["222222"].base) and b.loc["222222"].src is None
    assert np.isnan(b.loc["999999"].base)
    assert (bp["prev_date"] == pd.Timestamp(D2)).all()
    from krx_quant_core.market.limits import limit_up_price

    assert limit_up_price(b.loc["005930"].base) == 340500  # 일봉 종가 263,000 이면 341,500
    allb = D.base_price(D3, conn=db)
    assert set(allb["code"]) == {"005930", "000660", "111111"}
    with pytest.raises(TypeError):
        D.base_price(D3, "005930", conn=db)


def test_base_price_before_aftermarket_fallbacks(db):
    bp = D.base_price(P2, ["005930", "333333", "444444", "555555"], conn=db).set_index("code")
    assert (bp.loc["005930"].base, bp.loc["005930"].src) == (70500, "supply_demand_identity")
    assert (bp.loc["333333"].base, bp.loc["333333"].src) == (2500, "supply_demand")
    assert (bp.loc["444444"].base, bp.loc["444444"].src) == (4100, "daily_bars_raw")
    assert (bp.loc["555555"].base, bp.loc["555555"].src) == (3000, "daily_bars")
    assert tuple(daily_mod.BASE_CLOSE_FROM) == (
        "supply_demand_identity", "minute_bars", "supply_demand", "daily_bars_raw", "daily_bars",
    )  # fmt: skip
