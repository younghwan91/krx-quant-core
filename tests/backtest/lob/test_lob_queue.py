"""backtest.lob.queue — 지정가 대기열 모델: 손으로 만든 장면 + 모델 간 순서 불변식."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.backtest.lob import (
    QUEUE_MODELS,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_NOT_PLACED,
    STATUS_PARTIAL,
    BookGrid,
    LevelTrades,
    build_book_grid,
    build_level_trades,
    simulate_limit_orders,
)


def _book(n, bid=(1000.0, 999.0), ask=(1001.0, 1002.0), bq=(100.0, 100.0), aq=(100.0, 100.0)):
    L = len(bid)
    return BookGrid(
        bid_px=np.tile(np.array(bid), (n, 1)),
        bid_qty=np.tile(np.array(bq), (n, 1)),
        ask_px=np.tile(np.array(ask), (n, 1)).reshape(n, L),
        ask_qty=np.tile(np.array(aq), (n, 1)),
    )


def _trades(n, rows):
    """rows: [(sec, price, buy_vol, sell_vol)]"""
    rows = sorted(rows)
    counts = np.bincount([r[0] for r in rows], minlength=n) if rows else np.zeros(n, int)
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=ptr[1:])
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for s, p, _, _ in rows:
        lo[s] = p if np.isnan(lo[s]) else min(lo[s], p)
        hi[s] = p if np.isnan(hi[s]) else max(hi[s], p)
    return LevelTrades(
        ptr=ptr,
        price=np.array([r[1] for r in rows], float),
        buy_vol=np.array([r[2] for r in rows], float),
        sell_vol=np.array([r[3] for r in rows], float),
        lo=lo,
        hi=hi,
    )


def _one(book, trades, model, **kw):
    args = dict(side=1, price=1000.0, qty=10.0, place_sec=0)
    args.update({k: kw.pop(k) for k in list(kw) if k in args})
    kw.setdefault("max_wait", 8)
    return simulate_limit_orders(**args, book=book, trades=trades, queue_model=model, **kw)


def test_risk_averse_needs_queue_ahead_to_trade_out():
    # 매수 1,000 에 100주 앞. 매도 주도 체결 60 → 60 → 앞 −20 → 10주 체결.
    n = 10
    book = _book(n)
    tr = _trades(n, [(2, 1000.0, 0, 60), (3, 1000.0, 0, 60)])
    r = _one(book, tr, "risk_averse")
    assert r.status[0] == STATUS_FILLED and r.first_fill_sec[0] == 3 and r.done_sec[0] == 3
    assert r.avg_price[0] == 1000.0 and r.taker_qty[0] == 0
    # 체결 60 한 번이면 못 받는다
    r = _one(book, _trades(n, [(2, 1000.0, 0, 60)]), "risk_averse")
    assert r.status[0] == STATUS_CANCELED and r.done_sec[0] == 9 and np.isnan(r.avg_price[0])


def test_touch_fills_on_first_trade_at_price_through_does_not():
    n = 10
    book = _book(n)
    tr = _trades(n, [(2, 1000.0, 0, 4)])
    r = _one(book, tr, "touch")
    assert r.status[0] == STATUS_PARTIAL and r.filled_qty[0] == 4
    r = _one(book, tr, "through")
    assert r.status[0] == STATUS_CANCELED
    # 체결가가 내 가격 아래로 → 관통, 모든 모델 전량 체결(내 가격에)
    tr = _trades(n, [(4, 999.0, 0, 1)])
    for model in QUEUE_MODELS:
        r = _one(book, tr, model)
        assert r.status[0] == STATUS_FILLED and r.done_sec[0] == 4 and r.avg_price[0] == 1000.0


def test_buy_side_ignores_buyer_initiated_volume():
    n = 10
    r = _one(_book(n), _trades(n, [(2, 1000.0, 500, 0)]), "touch")
    assert r.status[0] == STATUS_CANCELED


def test_cancellations_ahead_only_help_prob_models():
    # 앞 100주. 3초에 잔량 100 → 20 (체결 없음 = 취소 80). 4초 체결 30.
    n = 10
    book = _book(n)
    book.bid_qty[3:, 0] = 20.0
    tr = _trades(n, [(4, 1000.0, 0, 30)])
    ra = _one(book, tr, "risk_averse")
    # RA: 앞 = min(100, 20) = 20 → 30 체결에 10주 전량
    assert ra.status[0] == STATUS_FILLED
    # 체결 15 면 RA 는 앞 20 에 못 미친다. 확률 모델은 취소 일부를 앞에서 뺀다 — 그래도
    # 새 잔량 20 이 상한이라 여기선 같다. 잔량 100 → 60 으로 줄여 차이를 만든다.
    book2 = _book(n)
    book2.bid_qty[3:, 0] = 60.0
    tr2 = _trades(n, [(4, 1000.0, 0, 50)])
    ra = _one(book2, tr2, "risk_averse")
    pp = _one(book2, tr2, "prob_power", power=1.0)
    assert ra.status[0] == STATUS_CANCELED
    # power=1, 앞 100·뒤 0 → 뒤 확률 0 → 취소 40 전부 앞 → 앞 60, 체결 50 으로도 부족
    assert pp.status[0] == STATUS_CANCELED
    # 내가 늦게 선 경우: 앞 40·뒤 60 을 만들려고 도착 때 잔량을 40 으로 두고 뒤에 60 쌓기
    book3 = _book(n)
    book3.bid_qty[:, 0] = 40.0
    book3.bid_qty[2, 0] = 100.0  # 뒤에 60 추가
    book3.bid_qty[3:, 0] = 70.0  # 취소 30
    tr3 = _trades(n, [(4, 1000.0, 0, 38)])
    ra = _one(book3, tr3, "risk_averse")
    pp = _one(book3, tr3, "prob_power", power=2.0)
    assert ra.filled_qty[0] == 0  # 앞 40 그대로, 체결 38
    assert pp.filled_qty[0] > 0  # 취소 일부가 앞에서 났다고 추정


def test_marketable_arrival_sweeps_then_rests():
    # 매수 1,002 × 10: 도착(1초) 스냅샷 매도 1,001×3·1,002×4 를 쳐서 7주, 3주는 1,002 에 선다.
    n = 10
    book = _book(n, ask=(1001.0, 1002.0), aq=(3.0, 4.0))
    r = _one(book, _trades(n, []), "risk_averse", price=1002.0, qty=10.0)
    assert r.taker_qty[0] == 7.0 and r.first_fill_sec[0] == 1
    # 2초 스냅샷도 매도1호가 1,001 ≤ 1,002 → 남은 3주 관통 체결(내 가격)
    assert r.status[0] == STATUS_FILLED and r.done_sec[0] == 2
    assert r.avg_price[0] == pytest.approx((3 * 1001 + 4 * 1002 + 3 * 1002) / 10)


def test_latency_and_not_placed():
    n = 5
    book = _book(n)
    tr = _trades(n, [(1, 999.0, 0, 1), (3, 999.0, 0, 1)])
    r = _one(book, tr, "through", latency=2)  # 도착 2초, 판정 3초부터 → 3초 관통
    assert r.done_sec[0] == 3
    r = simulate_limit_orders(
        1, 1000.0, 10.0, [0, 4, -9], book, tr, max_wait=3, queue_model="touch"
    )
    assert list(r.status) == [STATUS_FILLED, STATUS_NOT_PLACED, STATUS_NOT_PLACED]


def test_sell_side_symmetric():
    n = 10
    book = _book(n)
    tr = _trades(n, [(2, 1001.0, 150, 0)])
    r = simulate_limit_orders(-1, 1001.0, 10.0, 0, book, tr, max_wait=5, queue_model="risk_averse")
    assert r.status[0] == STATUS_FILLED and r.avg_price[0] == 1001.0


def test_validation():
    n = 5
    book = _book(n)
    tr = _trades(n, [])
    with pytest.raises(ValueError, match="queue_model"):
        simulate_limit_orders(1, 1000.0, 1, 0, book, tr, max_wait=1, queue_model="fifo")
    with pytest.raises(ValueError, match="side"):
        simulate_limit_orders(0, 1000.0, 1, 0, book, tr, max_wait=1)
    with pytest.raises(ValueError, match="grid"):
        simulate_limit_orders(1, 1000.0, 1, 0, book, _trades(n + 1, []), max_wait=1)


# --- 무작위 장면: 모델 간 순서 --------------------------------------------------


def _random_market(seed, n=400, L=5):
    rng = np.random.default_rng(seed)
    mid = 1000 + np.cumsum(rng.choice([-1, 0, 1], n, p=[0.2, 0.6, 0.2]))
    bid1 = mid.astype(float)
    bid_px = bid1[:, None] - np.arange(L)[None, :]
    ask_px = bid1[:, None] + 1 + np.arange(L)[None, :]
    bid_qty = rng.integers(1, 300, (n, L)).astype(float)
    ask_qty = rng.integers(1, 300, (n, L)).astype(float)
    book = BookGrid(bid_px, bid_qty, ask_px, ask_qty)
    rows = []
    for s in range(n):
        for _ in range(rng.integers(0, 3)):
            if rng.random() < 0.5:
                rows.append((s, bid1[s], 0.0, float(rng.integers(1, 200))))
            else:
                rows.append((s, bid1[s] + 1, float(rng.integers(1, 200)), 0.0))
    df = pd.DataFrame(rows, columns=["s", "p", "b", "v"]).groupby(["s", "p"]).sum().reset_index()
    return book, _trades(n, list(df.itertuples(index=False, name=None))), bid1, rng


@pytest.mark.parametrize("seed", range(5))
def test_model_ordering_invariant(seed):
    book, tr, bid1, rng = _random_market(seed)
    m = 300
    place = rng.integers(0, 380, m)
    side = rng.choice([-1, 1], m)
    offs = rng.integers(0, 3, m)
    price = np.where(side > 0, bid1[place] - offs, bid1[place] + 1 + offs)
    qty = rng.integers(1, 400, m).astype(float)
    res = {
        md: simulate_limit_orders(side, price, qty, place, book, tr, max_wait=30, queue_model=md)
        for md in QUEUE_MODELS
    }
    f = {k: v.filled_qty for k, v in res.items()}
    assert (f["touch"] >= f["prob_power"] - 1e-9).all()
    assert (f["touch"] >= f["prob_log"] - 1e-9).all()
    assert (f["prob_power"] >= f["risk_averse"] - 1e-9).all()
    assert (f["prob_log"] >= f["risk_averse"] - 1e-9).all()
    assert (f["risk_averse"] >= f["through"] - 1e-9).all()
    assert f["touch"].sum() > f["risk_averse"].sum() > f["through"].sum() > 0
    for v in res.values():
        assert (v.filled_qty <= qty + 1e-9).all()
        filled = v.filled_qty > 0
        assert np.all(v.avg_price[filled] > 0)


def test_build_book_and_trades_from_frames():
    d = pd.Timestamp("2026-09-10")
    t = lambda s: d + pd.Timedelta(seconds=9 * 3600 + s)  # noqa: E731
    q = {"ts": [t(1), t(1), t(3)]}
    for i in (1, 2):
        q[f"bid{i}"] = [1000 - i + 1, 900 - i + 1, 1001 - i + 1]
        q[f"ask{i}"] = [1001 + i - 1, 901 + i - 1, 1002 + i - 1]
        q[f"bidqty{i}"] = [10 * i, 20 * i, 30 * i]
        q[f"askqty{i}"] = [11 * i, 21 * i, 31 * i]
    q["bid2"][2] = 0  # 없는 단계
    book = build_book_grid(pd.DataFrame(q), levels=2)
    assert np.isnan(book.bid_px[0]).all()
    assert list(book.bid_px[1]) == [900.0, 899.0]  # 같은 초 마지막 스냅샷
    assert list(book.bid_px[2]) == [900.0, 899.0]  # 전방채움
    assert (
        book.bid_px[3, 0] == 1001.0 and np.isnan(book.bid_px[3, 1]) and np.isnan(book.bid_qty[3, 1])
    )
    ticks = pd.DataFrame(
        {
            "ts": [t(1), t(1), t(1), t(2)],
            "price": [1000, 1000, 1001, 999],
            "volume": [5, 7, 3, 2],
            "side": [-1, -1, 1, 0],
        }
    )
    tr = build_level_trades(ticks)
    assert tr.ptr[1] == 0 and tr.ptr[2] == 2 and tr.ptr[3] == 3
    assert list(tr.price[:3]) == [1000.0, 1001.0, 999.0]
    assert list(tr.sell_vol[:3]) == [12.0, 0.0, 2.0] and list(tr.buy_vol[:3]) == [0.0, 3.0, 2.0]
    assert tr.lo[1] == 1000 and tr.hi[1] == 1001
