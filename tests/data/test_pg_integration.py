"""실 Postgres COPY 경로 — ``KQC_PG_TESTS=1`` 이고 ``KR_QUANT_DB`` 가 있을 때만(simnode).

sqlite 테스트는 커서 경로만 탄다. 여기서 psycopg·psycopg2 COPY CSV 경로가 ``read_sql`` 과 같은 값을
내는지, 캐시 없이(``store=False``) 한 종목·하루로 본다. 읽기 전용.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from krx_quant_core import data as D

pytestmark = pytest.mark.skipif(
    os.environ.get("KQC_PG_TESTS") != "1", reason="set KQC_PG_TESTS=1 (needs KR_QUANT_DB)"
)


def _drivers():
    return [m for m in ("psycopg", "psycopg2") if importlib.util.find_spec(m)]


@pytest.fixture(scope="module")
def day():
    cal = D.trading_calendar(store=False)
    return cal.previous_session(date.today())


@pytest.mark.parametrize("driver", _drivers())
def test_copy_path_matches_read_sql(driver, day):
    mod = __import__(driver)
    conn = mod.connect(D.resolve_dsn())
    try:
        code = "005930"
        tk = D.load_ticks(day, conn=conn, store=False, codes=[code], today=day + timedelta(1))
        qs = D.load_quotes(day, conn=conn, store=False, codes=[code], levels=10)
        mb = D.load_minute_bars(day, conn=conn, store=False, codes=[code])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lo, hi = day.isoformat(), (day + timedelta(1)).isoformat()
            rt = pd.read_sql(
                "select ts, seq, price, volume, side, strength, cum_volume, best_bid, best_ask "
                f"from ticks where code='{code}' and ts >= '{lo}' and ts < '{hi}' order by ts, seq",
                conn,
            )
            rq = pd.read_sql(
                f"select * from quotes where code='{code}' and ts >= '{lo}' and ts < '{hi}' "
                "order by ts",
                conn,
            )
            rm = pd.read_sql(
                f"select * from minute_bars where code='{code}' and ts >= '{lo}' and ts < '{hi}' "
                "order by ts",
                conn,
            )
        assert len(tk) == len(rt) > 0
        f = tk.frame(code)
        for c in rt.columns:
            np.testing.assert_array_equal(f[c].to_numpy(), rt[c].to_numpy(), err_msg=c)
        fq = qs.frame(code)
        for c in [c for c in fq.columns if c != "ts"]:
            np.testing.assert_array_equal(fq[c].to_numpy(), rq[c].to_numpy(), err_msg=c)
        np.testing.assert_array_equal(mb.frame(code)["close"].to_numpy(), rm["close"].to_numpy())
    finally:
        conn.close()


def test_regular_session_005930_2026_10_08():
    """애프터마켓 오염 실례: 005930 2026-10-08 통합 종가 263,000, 정규장(15:32 봉) 262,000."""
    conn = D.connect()
    try:
        d = date(2026, 10, 8)
        kw = {"adjusted": False, "codes": ["005930"], "conn": conn, "store": False}
        allb = D.load_daily_bars(d, d, **kw)
        reg = D.load_daily_bars(d, d, session="regular", **kw)
        assert allb["close"].iloc[0] == 263000
        assert reg["close"].iloc[0] == 262000 and bool(reg["regular_ok"].iloc[0])
        mb = D.load_minute_bars(d, conn=conn, store=False, codes=["005930"], session="regular")
        assert mb.get("005930")["close"][-1] == 262000
        assert mb.get("005930")["sec"].max() <= 15 * 3600 + 32 * 60
        bp = D.base_price(date(2026, 10, 12), ["005930"], conn=conn)
        assert bp["base"].iloc[0] == 262000 and bp["src"].iloc[0] == "supply_demand_identity"
        # 분할 전 원 종가(daily_bars.close 는 소급 수정 41,200)
        old = D.base_price(date(2017, 4, 3), ["005930"], conn=conn)
        assert old["base"].iloc[0] == 2060000
    finally:
        conn.close()


def test_review_cases_2026_10():
    """리뷰 결함 실례(2026-10-13): 소급 분봉·재산정일·ETF·소급 분봉 행의 정규장 보정."""
    conn = D.connect()
    try:
        pre = list(D.BASE_CLOSE_FROM[1:])  # 장 전 — 그날 수급 항등식 없이
        cases = [
            # (날짜, 종목, 기준가, adj) — 009730 분봉은 10-12 권리락(−18.27%), 220100 은 09-17
            # 권리락(−16.67%)이 소급된 값이다(전엔 1,629·6,891 을 기준가로 냈다).
            (date(2026, 10, 8), "009730", 1993, 1.0),
            (date(2026, 10, 12), "009730", 1696, 0.8173),  # 재산정일: 2,075 × 0.8173
            (date(2026, 9, 16), "220100", 8270, 1.0),
            (date(2026, 9, 17), "220100", 6750, 0.8333),
            (date(2026, 10, 2), "207940", 1418000, 0.9923),  # 전엔 1,429,000(조정 전)
        ]
        for d, code, base, adj in cases:
            b = D.base_price(d, [code], conn=conn, sources=pre).iloc[0]
            assert (b.base, b.src) == (base, "minute_bars"), (d, code)
            assert b.adj == pytest.approx(adj)
            full = D.base_price(d, [code], conn=conn).iloc[0]  # 수급 항등식과 같은 값
            assert (full.base, full.src) == (base, "supply_demand_identity")
        # ETF 는 애프터마켓 대상이 아니다 — 09-14 뒤에도 전일 일봉 종가(전엔 영구 NaN).
        etf = D.base_price(date(2026, 10, 13), ["069500"], conn=conn,
                           sources=["daily_bars_raw", "daily_bars"]).iloc[0]  # fmt: skip
        assert etf.base == 101605 and etf.src == "daily_bars_raw"
        # 소급 분봉 행은 정규장 값으로 안 쓰고 다음날 항등식으로 종가만 복원(전엔 1,962.5).
        reg = D.load_daily_bars(date(2026, 9, 22), date(2026, 9, 23), adjusted=False,
                                codes=["009730", "069500"], conn=conn, store=False,
                                session="regular").set_index(["code", "date"])  # fmt: skip
        r = reg.loc[("009730", pd.Timestamp("2026-09-23"))]
        assert (not r.regular_ok) and r.close_ok and r.close == 1962
        e = reg.loc["069500"]
        assert e["regular_ok"].all() and e["close"].tolist() == [111880, 113145]
    finally:
        conn.close()
