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
D4 = date(2026, 10, 13)
P1, P2 = date(2026, 9, 10), date(2026, 9, 11)  # 애프터마켓 전
Q1, Q2 = date(2022, 6, 2), date(2022, 6, 3)  # 2023-01-25 호가 개편 전

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
# 888888: D2 분봉이 D4 권리락(−20%)이 소급된 수정 가격으로 재수집됐다(009730·220100 실례).
# 원 가격: 시가 2,000 · 고가 2,100 · 저가 1,990 · 정규장 종가 2,075 (통합 종가 2,090).
M_888888_D2 = [
    ("09:00:00", 1600, 1650, 1592, 1640, 10), ("15:30:00", 1660, 1680, 1660, 1660, 5),
    ("17:00:00", 1672, 1672, 1672, 1672, 1),
]  # fmt: skip
# 141414: 첫 봉(09:00)이 빠져 분봉 시가 13,850 ≠ 일봉 시가 14,100(009300 2026-10-07 실례) — 소급
# 사건 없음. 정규장 종가 13,990 은 맞다.
M_141414_D2 = [
    ("09:05:00", 13850, 14030, 13800, 13900, 5), ("15:30:00", 13990, 13990, 13990, 13990, 3),
]  # fmt: skip
# 121212: 분봉 정상(원 가격), D3 이 권리락 −0.77% — D3 기준가 = 5,070 × 0.9923 → 5,030.
M_121212_D2 = [("09:00:00", 5000, 5100, 4990, 5050, 10), ("15:30:00", 5070, 5070, 5070, 5070, 5)]
# 171717: 분봉·원주가 정상인데 daily_bars 가 D4 권리락(×0.9923)으로 소급됐다(207940 실례).
M_171717_D2 = [
    ("09:00:00", 1415000, 1420000, 1390000, 1400000, 10),
    ("15:30:00", 1396000, 1396000, 1396000, 1396000, 5),
]  # fmt: skip


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
    con.execute("create table stocks (code text, name text, market text, sector text, kind text)")
    con.execute("create table delisted_stocks (code text, name text)")
    con.execute("create table kind_delistings (code text, delist_date date, name text)")
    con.execute(
        "create table corporate_actions (code text, date date, upd_stkpc_tp text, upd_rt real, "
        "kind text, source text)"
    )
    # 주식 마스터·폐지 목록 — 069500(ETF)은 어디에도 없다.
    equities = ("005930", "000660", "111111", "222222", "333333", "777777", "888888", "141414",
                "121212", "171717", "151515", "161616", "131313")  # fmt: skip
    con.executemany("insert into stocks values (?,'','','','A')", [(c,) for c in equities])
    con.execute("insert into delisted_stocks values ('444444', '')")
    con.execute("insert into kind_delistings values ('555555', '2020-01-01', '')")
    con.executemany(
        "insert into corporate_actions values (?,?,'256',?,'rights_ex','kiwoom_ka10081')",
        [("888888", D4.isoformat(), -20.0), ("121212", D3.isoformat(), -0.77),
         ("171717", D4.isoformat(), -0.77), ("151515", D2.isoformat(), -5.0),
         ("131313", Q2.isoformat(), -10.0)],
    )  # fmt: skip
    mins = (
        _mins("005930", D1, M_005930_D1)
        + _mins("005930", D2, M_005930_D2)
        + _mins("111111", D2, M_111111_D2)
        + _mins("222222", D2, M_222222_D2)
        + _mins("777777", D2, M_777777_D2)
        + _mins("888888", D2, M_888888_D2)
        + _mins("141414", D2, M_141414_D2)
        + _mins("121212", D2, M_121212_D2)
        + _mins("171717", D2, M_171717_D2)
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
        ("111111", D2, 5000, 5100, 4990, 5080, 16, 0),
        ("555555", P1, 3000, 3000, 3000, 3000, 1, 0),
        ("888888", D2, 2000, 2110, 1990, 2090, 20, 0),
        ("141414", D2, 14100, 14100, 13800, 14000, 9, 0),
        ("121212", D2, 5000, 5100, 4990, 5080, 16, 0),
        ("171717", D2, 1415000, 1420000, 1390000, 1401000, 16, 22),
        ("151515", D1, 3000, 3100, 2900, 3050, 5, 0),
        ("161616", D1, 10000, 10000, 10000, 10000, 5, 0),
        ("069500", D1, 108700, 109000, 108000, 108550, 50, 5),
        ("069500", D2, 107000, 107500, 105500, 105650, 50, 5),
        ("131313", Q1, 10000, 10000, 10000, 10000, 1, 0),
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
        if d >= D.AFTERMARKET_START:  # 원주가(애프터마켓 뒤만 — 그 전 원천 순서 시험을 지키려고)
            con.execute(
                "insert into daily_bars_raw values (?,?,?,?,?,?,?,?,?)",
                (code, d.isoformat(), o, h, lo, c, v, tv, "kiwoom_raw"),
            )
    # 171717: daily_bars 는 D4 권리락이 소급된 값(×0.9923), 원주가는 그대로.
    con.execute(
        "update daily_bars set open = 1404105, high = 1409066, low = 1379297, close = 1390202 "
        "where code = '171717'"
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
        ("888888", D2, 2090, 0.0),
        ("888888", D3, 2100, 120.0),  # 기준 2,075 → 2100/2075-1 = 120.5bp → 120
        ("141414", D2, 14000, 0.0),
        ("141414", D3, 14000, 7.0),  # 기준 13,990 → 7.1bp
        ("151515", D1, 3050, 0.0),
        ("151515", D2, 2900, 0.0),  # D2 권리락 — 항등식이 조정 기준가 2,900(전일 종가 아님)
        ("161616", D1, 10000, 0.0),
        ("161616", D2, 50000, 0.0),  # 사건 표에 빠진 병합 — 기준가 50,000 은 전일 종가일 수 없다
        ("131313", Q1, 10000, 0.0),
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
    d2 = (M_005930_D2, M_111111_D2, M_222222_D2, M_777777_D2, M_888888_D2, M_141414_D2,
          M_121212_D2, M_171717_D2)  # fmt: skip
    assert len(allb) == sum(map(len, d2))  # 기본은 테이블 그대로
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
    # 애프터마켓만 있는 종목(777777)은 없다
    assert list(s.index) == ["005930", "111111", "121212", "141414", "171717", "222222", "888888"]
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
    assert set(allb["code"]) == {
        "005930", "000660", "111111", "069500", "121212", "141414", "171717", "888888",
    }  # fmt: skip
    assert set(allb.columns) == {"code", "base", "src", "prev_date", "adj"}
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


# ── 리뷰 결함 재현(2026-10-13) ────────────────────────────────────────────────


def test_base_price_unscales_retro_minute_bars(db):
    """MAJOR 1: 분봉이 이후 권리락 소급 값(×0.8)이면 원 가격으로 되돌린다(전엔 1,660 그대로)."""
    pre = list(D.BASE_CLOSE_FROM[1:])  # 장 전 — 그날 수급 행이 아직 없다
    b = D.base_price(D3, ["888888"], conn=db, sources=pre).iloc[0]
    assert (b.base, b.src, b.adj) == (2075, "minute_bars", 1.0)
    # 원주가 시가와 안 맞고 어느 사건 계수로도 설명 안 되는 어긋남(첫 봉 빠짐)은 버린다.
    b = D.base_price(D3, ["141414"], conn=db, sources=pre).iloc[0]
    assert np.isnan(b.base) and b.src is None
    # 그날 수급 행이 있으면 항등식이 이긴다(같은 값).
    assert D.base_price(D3, ["888888", "141414"], conn=db)["base"].tolist() == [2075, 13990]
    with pytest.raises(ValueError):
        D.base_price(D3, ["888888"], conn=db, sources=["minute"])
    # 호가가 아닌 분봉 종가(2,000~5,000원은 5원 호가)는 버린다 — 스팩 실례 2,073.
    db.execute("update minute_bars set close = 5073 where code = '121212' and close = 5070")
    assert np.isnan(D.base_price(D3, ["121212"], conn=db, sources=pre)["base"].iloc[0])


def test_base_price_rebase_day_adjusts(db):
    """MAJOR 1: 재산정일(권리락 −0.77%)엔 전일 종가 × 계수를 호가 반올림(207940 10-02 실례)."""
    b = D.base_price(D3, ["121212"], conn=db).iloc[0]
    assert (b.base, b.src) == (5030, "minute_bars") and b.adj == pytest.approx(0.9923)
    # 호가 개편 전 날짜는 현행 호가표로 반올림할 수 없어 버린다(daily_bars 도 — 소급됐을지 모른다).
    q = D.base_price(Q2, ["131313"], conn=db).iloc[0]
    assert np.isnan(q.base) and q.src is None


def test_base_price_prev_session_not_from_daily_only(db):
    """MAJOR 2: 일봉 적재가 하루 빠져도 전 거래일을 이틀 전으로 잡지 않는다(경고)."""
    db.execute("delete from daily_bars where date = ?", (D2.isoformat(),))
    with pytest.warns(RuntimeWarning, match="daily_bars"):
        bp = D.base_price(D3, ["111111"], conn=db)
    b = bp.iloc[0]
    assert b.prev_date == pd.Timestamp(D2)
    assert (b.base, b.src) == (5070, "minute_bars")  # 전엔 D1 을 전일로 봐서 NaN


def test_etf_not_gated(db):
    """MAJOR 3: ETF·ETN 은 애프터마켓 대상이 아니라 09-14 뒤에도 일봉 종가가 정규장 값이다."""
    b = D.base_price(D3, ["069500"], conn=db).iloc[0]
    assert (b.base, b.src) == (105650, "daily_bars_raw")
    reg = D.load_daily_bars(D1, D2, adjusted=False, codes=["069500"], conn=db, store=False,
                            session="regular")  # fmt: skip
    assert reg["regular_ok"].all() and reg["close_ok"].all()
    assert reg["close"].tolist() == [108550, 105650]


def test_daily_regular_open_mismatch_and_retro(db):
    """MINOR(_to_regular): 시가 배율을 그대로 곱하지 않는다.

    - 141414: 첫 봉 빠짐(시가 13,850 vs 14,100) — 전엔 13,990 × 1.018 = 14,242.5 를 냈다. 이제
      분봉을 버리고 다음날 항등식으로 종가 13,990 복원.
    - 888888: 분봉이 소급 값(×0.8) — 정규장 값으로 쓰지 않고 항등식으로 종가 2,075 복원.
    - 171717: daily_bars 가 소급된 행(배율 0.9923) — 결과는 테이블처럼 원 단위 정수.
    """
    reg = D.load_daily_bars(D2, D2, adjusted=False, codes=["141414", "888888", "171717"],
                            conn=db, store=False, session="regular").set_index("code")  # fmt: skip
    r = reg.loc["141414"]
    assert (not r.regular_ok) and r.close_ok and r.close == 13990
    r = reg.loc["888888"]
    assert (not r.regular_ok) and r.close_ok and r.close == 2075 and r.high == 2110
    r = reg.loc["171717"]
    assert r.regular_ok and r.close == round(1396000 * 1404105 / 1415000)
    assert r.close == int(r.close) and r.high == int(r.high) and r.volume == int(r.volume)
    # 호가가 아닌 분봉 종가 → 분봉을 버리고 항등식 복원(888888 처럼)
    db.execute("update minute_bars set close = 13993 where code = '141414' and close = 13990")
    db.execute("update daily_bars_raw set open = 13850 where code = '141414'")
    db.execute("update daily_bars set open = 13850 where code = '141414'")
    r = D.load_daily_bars(D2, D2, adjusted=False, codes=["141414"], conn=db, store=False,
                          session="regular").iloc[0]  # fmt: skip
    assert (not r.regular_ok) and r.close_ok and r.close == 13990
    # 수정주가 테이블: 배율(0.5)만 곱한다
    adj = D.load_daily_bars(D2, D2, codes=["171717"], conn=db, store=False, session="regular")
    assert adj["close"].iloc[0] == 1396000 * 0.5


def test_restore_close_rejects_rebase_and_impossible(db):
    """MINOR(_restore_close): 다음날이 재산정일이거나 복원값이 가격제한 폭 밖이면 close_ok=False."""
    reg = D.load_daily_bars(D1, D1, adjusted=False, codes=["151515", "161616", "000660"],
                            conn=db, store=False, session="regular").set_index("code")  # fmt: skip
    assert not reg.loc["151515"].close_ok and reg.loc["151515"].close == 3050  # 테이블 값
    assert not reg.loc["161616"].close_ok and reg.loc["161616"].close == 10000
    assert reg.loc["000660"].close_ok and reg.loc["000660"].close == 154500


def test_regular_session_bars_skips_holidays(db, monkeypatch):
    """MINOR: 공휴일(10-09 금)은 묻지 않는다. 일봉·수급표 마지막 날 뒤 평일은 묻는다(적재 지연)."""
    asked = []
    real = daily_mod._session_day

    def spy(conn, day, codes):
        asked.append(day)
        return real(conn, day, codes)

    monkeypatch.setattr(daily_mod, "_session_day", spy)
    D.regular_session_bars(D1, date(2026, 10, 14), conn=db)
    assert asked == [D1, D2, D3, D4, date(2026, 10, 14)]


def test_unscale_and_snap_helpers():
    # 009730 2026-09-23: 1604 / 0.8173 → 1962·1963 둘 다 1604 로 반올림 — 유일하지 않아 NaN.
    got = daily_mod._unscale([1604, 1696, 5070, 1660], [0.8173, 0.8173, 1.0, np.nan], False)
    assert np.isnan(got[0]) and got[1] == 2075 and got[2] == 5070 and np.isnan(got[3])
    ev = {"888888": (np.array(["2026-10-13", "2026-10-20"], dtype="datetime64[D]"),
                     np.array([0.8, 5.0]))}  # fmt: skip
    days = np.array(["2026-10-08"] * 5, dtype="datetime64[D]")
    obs = [0.8, 4.0, 1.001, 0.9, 1.01]
    got = daily_mod._snap_retro(obs, ["888888", "888888", "888888", "888888", "000660"], days, ev)
    np.testing.assert_allclose(got[:3], [0.8, 4.0, 1.0])
    assert np.isnan(got[3]) and np.isnan(got[4])  # 설명 안 되는 어긋남
    # 사건 이후 날짜에는 그 계수가 후보가 아니다
    late = daily_mod._snap_retro([0.8], ["888888"], np.array(["2026-10-13"], "datetime64[D]"), ev)
    assert np.isnan(late[0])
