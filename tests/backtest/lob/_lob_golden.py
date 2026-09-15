"""골든 기준 — scalp-it 원본을 **고치지 않고** 옮겨 둔 사본(scalp-it main 5683adf).

- ``build_code`` 등: ``scripts/orderflow_80/of_build.py``
- ``strength_exit_labels``: ``scripts/rl_82/dataset.py::_strength_exit_labels`` (numba 데코레이터만 뺌)
- ``random_control``: ``scripts/rl_82/common.py::random_control`` (``Episode``·``COST`` 를 인자로)

이 파일은 비교 기준이라 lint 대상에서 뺀다(원본 스타일 보존).
"""
# ruff: noqa
import numpy as np
import pandas as pd

COST = 0.0023
HS = (5, 15, 30, 60, 120, 300)
T0, T1 = 9 * 3600, 15 * 3600 + 20 * 60  # 09:00~15:20
N = T1 - T0


def _sec(ts):
    return (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy() - T0


def ffill(a):
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.argmax(~np.isnan(a)) if (~np.isnan(a)).any() else len(a)
    out[:first] = np.nan
    return out


def _tick(p):
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if p < lim:
            return t
    return 1000


def roll_sum(a, w):
    c = np.concatenate([[0.0], np.cumsum(a)])
    out = np.full(len(a), np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def fwd_max(a, h):
    # max over (t, t+h]
    s = pd.Series(a[::-1]).rolling(h, min_periods=1).max().to_numpy()[::-1]
    out = np.full(len(a), np.nan)
    out[:-1] = s[1:]
    # s[i+1] = max a[i+1 .. i+h]
    return out


def build_code(tk, qt):
    n = N
    sec_t = _sec(tk.ts)
    m = (sec_t >= 0) & (sec_t < n)
    tk = tk[m]
    sec_t = sec_t[m]
    val = (tk.price * tk.volume).to_numpy(float)
    side = tk.side.to_numpy()
    buy = np.bincount(sec_t, weights=np.where(side > 0, val, 0), minlength=n)
    sell = np.bincount(sec_t, weights=np.where(side < 0, val, 0), minlength=n)
    cnt = np.bincount(sec_t, minlength=n)
    last = np.full(n, np.nan)
    last[sec_t] = tk.price.to_numpy(float)  # 마지막 값이 남는다(정렬됨)
    stg = np.full(n, np.nan)
    stg[sec_t] = tk.strength.to_numpy(float)
    bb = np.full(n, np.nan); ba = np.full(n, np.nan)
    bb[sec_t] = tk.best_bid.to_numpy(float); ba[sec_t] = tk.best_ask.to_numpy(float)

    bq1 = np.full(n, np.nan); aq1 = np.full(n, np.nan)
    bd3 = np.full(n, np.nan); ad3 = np.full(n, np.nan)
    if len(qt):
        sec_q = _sec(qt.ts)
        mq = (sec_q >= 0) & (sec_q < n)
        qt = qt[mq]; sec_q = sec_q[mq]
        # 같은 초면 호가 스냅샷을 틱 best 보다 우선하지 않고, 둘 다 있으면 호가를 쓴다
        b1 = qt.bid1.to_numpy(float); a1 = qt.ask1.to_numpy(float)
        ok = (b1 > 0) & (a1 > 0)
        bb[sec_q[ok]] = b1[ok]; ba[sec_q[ok]] = a1[ok]
        bq1[sec_q[ok]] = qt.bidqty1.to_numpy(float)[ok]
        aq1[sec_q[ok]] = qt.askqty1.to_numpy(float)[ok]
        bd3[sec_q[ok]] = (qt[["bidqty1", "bidqty2", "bidqty3"]].sum(axis=1).to_numpy(float) * b1)[ok]
        ad3[sec_q[ok]] = (qt[["askqty1", "askqty2", "askqty3"]].sum(axis=1).to_numpy(float) * a1)[ok]
    bb = ffill(bb); ba = ffill(ba); last = ffill(last); stg = ffill(stg)
    bq1 = ffill(bq1); aq1 = ffill(aq1); bd3 = ffill(bd3); ad3 = ffill(ad3)
    bb[bb <= 0] = np.nan; ba[ba <= 0] = np.nan
    mid = (bb + ba) / 2

    # OFI(Cont) level1
    pb, pa = np.roll(bb, 1), np.roll(ba, 1)
    pbq, paq = np.roll(bq1, 1), np.roll(aq1, 1)
    e = ((bb >= pb) * bq1 - (bb <= pb) * pbq - (ba <= pa) * aq1 + (ba >= pa) * paq)
    e[0] = 0
    e = np.nan_to_num(e)
    depth1 = np.nan_to_num((bq1 + aq1) / 2)

    tot = buy + sell
    v30 = roll_sum(tot, 30)
    v600 = roll_sum(tot, 600)
    base = np.full(n, np.nan)
    base[30:] = v600[:-30] / 20.0  # 30초 창 20개 평균(직전 30초 제외)
    f = {
        "sec": np.arange(n),
        "cnt": cnt,
        "acc": v30 / (base + 1e6),
        "v30": v30,
        "v600": v600,
        "buyshare30": roll_sum(buy, 30) / (v30 + 1),
        "buyshare10": roll_sum(buy, 10) / (roll_sum(tot, 10) + 1),
        "ret30": mid / np.roll(mid, 30) - 1,
        "ret300": mid / np.roll(mid, 300) - 1,
        "ofi10": roll_sum(e, 10) / (pd.Series(depth1).rolling(60, min_periods=1).mean().to_numpy() + 1),
        "imb3": (bd3 - ad3) / (bd3 + ad3),
        "spread_bp": (ba - bb) / mid * 1e4,
        "strength": stg,
        "dstr10": stg - np.roll(stg, 10),
        "dstr30": stg - np.roll(stg, 30),
        "dstr60": stg - np.roll(stg, 60),
        "spread_ticks": (ba - bb) / np.array([_tick(x) for x in np.nan_to_num(ba, nan=1.0)]),
        "bid": bb, "ask": ba,
    }
    ask_e = np.roll(ba, -1)  # 1초 지연 진입
    for h in HS:
        exit_b = np.full(n, np.nan)
        exit_b[: n - 1 - h] = bb[1 + h:]
        f[f"net{h}"] = exit_b / ask_e - 1 - COST
        mx = fwd_max(bb, h)
        mfe = np.full(n, np.nan)
        mfe[: n - 1] = mx[1:]
        f[f"mfe{h}"] = mfe / ask_e - 1 - COST
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    pr = tk.price.to_numpy(float)
    lo_s = pd.Series(pr).groupby(sec_t).min(); hi_s = pd.Series(pr).groupby(sec_t).max()
    lo[lo_s.index.to_numpy()] = lo_s.to_numpy(); hi[hi_s.index.to_numpy()] = hi_s.to_numpy()
    path = {"bid": bb.astype(np.float32), "ask": ba.astype(np.float32),
            "lo": lo.astype(np.float32), "hi": hi.astype(np.float32),
            "bidval1": np.nan_to_num(bq1 * bb).astype(np.float32),
            "bidval3": np.nan_to_num(bd3).astype(np.float32),
            "ofi": np.cumsum(e).astype(np.float32), "strength": stg.astype(np.float32)}
    df = pd.DataFrame(f)
    keep = (cnt > 0) & (df.sec >= 300) & (df.sec <= N - 600) & df.ask.notna() & df.bid.notna() \
        & (df.ask > df.bid) & df.acc.notna()
    df = df[keep]
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df, path


def strength_exit_labels(secs, bid, ask, stg, sdrop, tmax, cost):
    n = len(secs)
    out = np.full(n, np.nan, np.float32)
    hold = np.zeros(n, np.int32)
    L = len(bid)
    for i in range(n):
        e = secs[i] + 1
        if e >= L - 1:
            continue
        a = ask[e]
        if not (a > 0):
            continue
        peak = stg[e]
        end = min(e + tmax, L - 1)
        px = bid[end]
        xs = end
        for s in range(e + 1, end + 1):
            v = stg[s]
            if v == v:
                if not (peak == peak) or v > peak:
                    peak = v
                if v <= peak - sdrop:
                    px = bid[s]
                    xs = s
                    break
        if px > 0:
            out[i] = px / a - 1.0 - cost
            hold[i] = xs - e
    return out, hold


def trade_ret(ep, i_in, i_out, cost):
    return ep.sell[i_out] / ep.buy[i_in] - 1.0 - cost


def random_control(trades, eps, cost, seed=82):
    """같은 종목·날, 같은 보유 시간, 진입 초만 무작위 — 사전등록 §6-6."""
    rng = np.random.default_rng(seed)
    rets = []
    for r in trades.itertuples():
        ep = eps[(r.day, r.code)]
        hold = int(r.hold)
        cand = np.nonzero(ep.secs + hold <= ep.secs[-1])[0]
        if len(cand) == 0:
            rets.append(np.nan)
            continue
        i = int(rng.choice(cand))
        j = int(np.searchsorted(ep.secs, ep.secs[i] + hold))
        j = min(j, len(ep.secs) - 1)
        rets.append(trade_ret(ep, i, j, cost))
    return trades.assign(ret_ctrl=rets)


def synthetic_day(seed, *, n_ticks=6000, n_quotes=6000, base=1990.0, day="2026-09-08"):
    """경계(2,000원 밴드)를 오가는 가짜 틱·호가. 결측·0 호가·같은 초 여러 건을 일부러 섞는다."""
    rng = np.random.default_rng(seed)
    d0 = pd.Timestamp(day)
    span = N + 120  # 장 밖 몇 초도 섞어서 잘리는지 본다
    def ts(k):
        s = np.sort(rng.uniform(-60, span - 60, k))
        return d0 + pd.to_timedelta(T0 + s, unit="s")
    walk = base + np.cumsum(rng.choice([-5.0, -1.0, 0.0, 1.0, 5.0], n_ticks, p=[.1, .2, .4, .2, .1]))
    price = np.maximum(np.round(walk), 1.0)
    bb = price - rng.choice([0.0, 1.0], n_ticks)
    ba = bb + rng.choice([1.0, 2.0, 5.0], n_ticks)
    bb[rng.random(n_ticks) < 0.03] = np.nan
    ba[rng.random(n_ticks) < 0.03] = 0.0
    stg = 100 + np.cumsum(rng.normal(0, 1.5, n_ticks))
    stg[rng.random(n_ticks) < 0.05] = np.nan
    ticks = pd.DataFrame({
        "ts": ts(n_ticks), "seq": np.arange(n_ticks), "price": price,
        "volume": rng.integers(1, 500, n_ticks), "side": rng.choice([-1, 1], n_ticks),
        "strength": stg, "best_bid": bb, "best_ask": ba,
    })
    qwalk = base + np.cumsum(rng.choice([-5.0, -1.0, 0.0, 1.0, 5.0], n_quotes, p=[.1, .2, .4, .2, .1]))
    b1 = np.maximum(np.round(qwalk), 1.0)
    a1 = b1 + rng.choice([1.0, 2.0, 5.0, 0.0], n_quotes, p=[.6, .2, .15, .05])
    b1[rng.random(n_quotes) < 0.02] = 0.0
    quotes = pd.DataFrame({"ts": ts(n_quotes), "bid1": b1, "ask1": a1})
    for c in ("bidqty1", "bidqty2", "bidqty3", "askqty1", "askqty2", "askqty3"):
        quotes[c] = rng.integers(0, 5000, n_quotes)
    return ticks, quotes
