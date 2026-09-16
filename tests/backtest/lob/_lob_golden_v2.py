"""골든 기준 v2 — scalp-it 원본을 **고치지 않고** 옮겨 둔 사본(scalp-it main 1b443c9).

- ``_tick_np``·``ffill``·``rsum``·``lag``·``_per_second``·``_window_trade_stats``·``_roll_extreme``·
  ``build_code``: ``scripts/flow_83/features.py`` (``_labels`` 는 라벨이라 뺐다)
- ``_trade_runs``·``_extra``: ``scripts/flow_84/library.py``
- ``features_by_code``: ``scripts/flow_84/library.py::build_day`` 의 피처 부분만 — DB 조회·재료
  (공시·뉴스·테마)·결정 행 필터·라벨을 빼고, 격자 전체 열을 돌려준다. 순위 두 열
  (``cum_rank``·``dayret_rank``)의 식은 원본 줄 그대로다.

``numba.njit(cache=True)`` 는 numba 가 있으면 그대로(운용 원본과 같은 경로), 없으면 파이썬으로
돈다. 이 파일은 비교 기준이라 lint 대상에서 뺀다(원본 스타일 보존).
"""
# ruff: noqa
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import numba
except ImportError:  # numba 없는 환경: 같은 함수를 파이썬으로
    class numba:  # type: ignore[no-redef]
        @staticmethod
        def njit(*a, **k):
            if a and callable(a[0]):
                return a[0]
            return lambda f: f

#: scalp-it ``scripts/rl_93/features.json`` (main 1b443c9) — 93·94번 정책 입력 37열, 순서 고정.
FEATURES_JSON = json.loads('["sec", "log_cumval", "spread_bp", "cum_rank", "n_tr10", "dist_lo300", "ret300", "same_size_val10", "sweep_ticks3", "dist_hi300", "micro_dev", "imb3", "ntr60", "n_tr5", "buyval3", "n_tr1", "burst_secs10", "buyval1", "sv20", "bv30", "buyval10", "sv60", "sellval5", "sellval3", "maxbuy5", "dayret_rank", "sweep10", "sellval1", "sweep_ticks1", "dstr3", "dstr5", "dS2", "ofiw20", "ofiw30", "ofiw60", "ofiw1", "ofiw2"]')

# ---- scripts/flow_83/features.py ------------------------------------------------
T0 = 9 * 3600
N = 22800                      # 09:00~15:20 초

W = (1, 3, 5, 10)
FEATURES: tuple[str, ...] = (
    *[f"n_tr{w}" for w in W], *[f"buyval{w}" for w in W], *[f"sellval{w}" for w in W],
    *[f"maxbuy{w}" for w in W], "bigbuy_share10", "bigbuy_share3",
    "same_size_cnt10", "same_size_val10", "burst_secs10", "sweep_ticks1", "sweep_ticks3", "persist10",
    *[f"dstr{k}" for k in (1, 3, 5, 10, 30)], *[f"ofi{k}" for k in (1, 3, 5, 10)],
    "bidq1_chg1", "bidq1_chg3", "askq1_depl1", "askq1_depl3", "val_accel5",
    *[f"ret{k}" for k in (1, 3, 5, 10, 30, 300)],
    "dist_hi60", "dist_lo60", "dist_hi300", "dist_lo300", "since_hi60", "since_lo60", "rebound_lo30",
    "spread_bp", "spread_ticks", "imb3", "cum_rank", "log_cumval", "sec",
)

def _tick_np(prices: np.ndarray) -> np.ndarray:
    from kiwoom_client.tick_size import tick_size

    u, inv = np.unique(np.nan_to_num(prices, nan=1000.0), return_inverse=True)
    t = np.array([float(tick_size(Decimal(int(max(p, 1))))) for p in u])
    return t[inv]


def ffill(a: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(a)
    if not ok.any():
        return a
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[: np.argmax(ok)] = np.nan
    return out


def rsum(a: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(a))])
    out = np.zeros(len(a))
    out[w - 1:] = c[w:] - c[:-w]
    out[: w - 1] = c[1:w]
    return out


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    out[k:] = a[:-k]
    return out


@numba.njit(cache=True)
def _per_second(sec, price, vol, side, n):
    n_tr = np.zeros(n)
    buyval = np.zeros(n)
    sellval = np.zeros(n)
    maxbuy = np.zeros(n)
    n_buy = np.zeros(n)
    bhi = np.full(n, np.nan)
    blo = np.full(n, np.nan)
    for i in range(len(sec)):
        s = sec[i]
        v = price[i] * vol[i]
        n_tr[s] += 1
        if side[i] > 0:
            buyval[s] += v
            n_buy[s] += 1
            if v > maxbuy[s]:
                maxbuy[s] = v
            if not (bhi[s] == bhi[s]) or price[i] > bhi[s]:
                bhi[s] = price[i]
            if not (blo[s] == blo[s]) or price[i] < blo[s]:
                blo[s] = price[i]
        elif side[i] < 0:
            sellval[s] += v
    return n_tr, buyval, sellval, maxbuy, n_buy, bhi, blo


@numba.njit(cache=True)
def _window_trade_stats(sec, price, vol, side, n, big_thr_by_min):
    """10초 창 같은 수량 반복·큰 체결 매수 비중(10·3초). big_thr 는 분 단위(직전 10분 중앙값×20)."""
    same_cnt = np.zeros(n)
    same_val = np.zeros(n)
    big10 = np.zeros(n)
    big3 = np.zeros(n)
    m = len(sec)
    lo = 0
    hi = 0
    for s in range(n):
        while hi < m and sec[hi] <= s:
            hi += 1
        while lo < hi and sec[lo] < s - 9:
            lo += 1
        if hi == lo:
            continue
        thr = big_thr_by_min[s // 60]
        # 큰 체결
        tot10 = 0.0
        b10 = 0.0
        tot3 = 0.0
        b3 = 0.0
        nb = 0
        vols = np.empty(hi - lo)
        pxs = np.empty(hi - lo)
        for i in range(lo, hi):
            if side[i] > 0:
                v = price[i] * vol[i]
                tot10 += v
                if v >= thr:
                    b10 += v
                if sec[i] >= s - 2:
                    tot3 += v
                    if v >= thr:
                        b3 += v
                vols[nb] = vol[i]
                pxs[nb] = price[i]
                nb += 1
        if tot10 > 0:
            big10[s] = b10 / tot10
        if tot3 > 0:
            big3[s] = b3 / tot3
        if nb >= 2:
            vs = np.sort(vols[:nb])
            best = 1
            run = 1
            bestv = vs[0]
            for j in range(1, nb):
                if vs[j] == vs[j - 1]:
                    run += 1
                else:
                    run = 1
                if run > best:
                    best = run
                    bestv = vs[j]
            same_cnt[s] = best
            same_val[s] = best * bestv * pxs[nb - 1]
    return same_cnt, same_val, big10, big3


@numba.njit(cache=True)
def _roll_extreme(x, w, want_max):
    """최근 w초(포함) 극값과 그 뒤 경과 초 — 단조 덱."""
    n = len(x)
    val = np.full(n, np.nan)
    since = np.full(n, np.nan)
    dq = np.empty(n, np.int64)
    h = 0
    t = 0
    for i in range(n):
        xi = x[i]
        if xi == xi:
            while t > h:
                last = x[dq[t - 1]]
                if (want_max and last <= xi) or ((not want_max) and last >= xi):
                    t -= 1
                else:
                    break
            dq[t] = i
            t += 1
        while t > h and dq[h] <= i - w:
            h += 1
        if t > h:
            val[i] = x[dq[h]]
            since[i] = i - dq[h]
    return val, since


def build_code(tk: pd.DataFrame, qt: pd.DataFrame) -> tuple[dict, dict]:
    sec = (tk.ts.dt.hour * 3600 + tk.ts.dt.minute * 60 + tk.ts.dt.second).to_numpy() - T0
    m = (sec >= 0) & (sec < N)
    tk, sec = tk[m], sec[m].astype(np.int64)
    price = tk.price.to_numpy(np.float64)
    vol = tk.volume.to_numpy(np.float64)
    side = tk.side.to_numpy(np.int64)
    n_tr, buyval, sellval, maxbuy, n_buy, bhi, blo = _per_second(sec, price, vol, side, N)
    tv = price * vol
    # 분 단위 큰 체결 문턱: 직전 10분 체결 1건 금액 중앙값 × 20
    thr = np.full(N // 60 + 1, np.inf)
    for mi in range(N // 60 + 1):
        s0 = mi * 60
        a, b = np.searchsorted(sec, s0 - 600), np.searchsorted(sec, s0)
        if b - a >= 20:
            thr[mi] = float(np.median(tv[a:b])) * 20.0
    same_cnt, same_val, big10, big3 = _window_trade_stats(sec, price, vol, side, N, thr)

    stg = np.full(N, np.nan)
    stg[sec] = tk.strength.to_numpy(np.float64)
    last_px = np.full(N, np.nan)
    last_px[sec] = price
    hi = np.full(N, np.nan)
    s_hi = pd.Series(price).groupby(sec).max()
    hi[s_hi.index.to_numpy()] = s_hi.to_numpy()
    bb = np.full(N, np.nan)
    ba = np.full(N, np.nan)
    bb[sec] = tk.best_bid.to_numpy(np.float64)
    ba[sec] = tk.best_ask.to_numpy(np.float64)
    bq1 = np.full(N, np.nan)
    aq1 = np.full(N, np.nan)
    bd3 = np.full(N, np.nan)
    ad3 = np.full(N, np.nan)
    if len(qt):
        qs = (qt.ts.dt.hour * 3600 + qt.ts.dt.minute * 60 + qt.ts.dt.second).to_numpy() - T0
        mq = (qs >= 0) & (qs < N)
        qt, qs = qt[mq], qs[mq]
        b1 = qt.bid1.to_numpy(np.float64)
        a1 = qt.ask1.to_numpy(np.float64)
        ok = (b1 > 0) & (a1 > 0)
        bb[qs[ok]] = b1[ok]
        ba[qs[ok]] = a1[ok]
        bq1[qs[ok]] = qt.bidqty1.to_numpy(np.float64)[ok]
        aq1[qs[ok]] = qt.askqty1.to_numpy(np.float64)[ok]
        bd3[qs[ok]] = qt[["bidqty1", "bidqty2", "bidqty3"]].sum(axis=1).to_numpy(np.float64)[ok] * b1[ok]
        ad3[qs[ok]] = qt[["askqty1", "askqty2", "askqty3"]].sum(axis=1).to_numpy(np.float64)[ok] * a1[ok]
    bb, ba, stg = ffill(bb), ffill(ba), ffill(stg)
    bq1, aq1, bd3, ad3 = ffill(bq1), ffill(aq1), ffill(bd3), ffill(ad3)
    bb[bb <= 0] = np.nan
    ba[ba <= 0] = np.nan
    mid = (bb + ba) / 2
    tick_at = _tick_np(ba)

    pb, pa, pbq, paq = lag(bb, 1), lag(ba, 1), lag(bq1, 1), lag(aq1, 1)
    e = np.nan_to_num((bb >= pb) * bq1 - (bb <= pb) * pbq - (ba <= pa) * aq1 + (ba >= pa) * paq)
    depth = pd.Series(np.nan_to_num((bq1 + aq1) / 2)).rolling(60, min_periods=1).mean().to_numpy() + 1

    val = buyval + sellval
    f: dict[str, np.ndarray] = {}
    for w in W:
        f[f"n_tr{w}"] = rsum(n_tr, w)
        f[f"buyval{w}"] = np.log1p(rsum(buyval, w))
        f[f"sellval{w}"] = np.log1p(rsum(sellval, w))
        f[f"maxbuy{w}"] = np.log1p(pd.Series(maxbuy).rolling(w, min_periods=1).max().to_numpy())
    f["bigbuy_share10"], f["bigbuy_share3"] = big10, big3
    f["same_size_cnt10"], f["same_size_val10"] = same_cnt, np.log1p(same_val)
    f["burst_secs10"] = rsum((n_buy >= 5).astype(float), 10)
    sw = np.nan_to_num((bhi - blo) / tick_at)
    f["sweep_ticks1"] = sw
    f["sweep_ticks3"] = pd.Series(sw).rolling(3, min_periods=1).max().to_numpy()
    f["persist10"] = rsum((n_buy > 0).astype(float), 10)
    for k in (1, 3, 5, 10, 30):
        f[f"dstr{k}"] = stg - lag(stg, k)
    for k in (1, 3, 5, 10):
        f[f"ofi{k}"] = rsum(e, k) / depth
    f["bidq1_chg1"] = (bq1 - lag(bq1, 1)) / (lag(bq1, 1) + 1)
    f["bidq1_chg3"] = (bq1 - lag(bq1, 3)) / (lag(bq1, 3) + 1)
    f["askq1_depl1"] = (lag(aq1, 1) - aq1) / (lag(aq1, 1) + 1)
    f["askq1_depl3"] = (lag(aq1, 3) - aq1) / (lag(aq1, 3) + 1)
    f["val_accel5"] = rsum(val, 5) / (rsum(val, 300) / 60 + 1e5)
    for k in (1, 3, 5, 10, 30, 300):
        f[f"ret{k}"] = (mid / lag(mid, k) - 1) * 1e4
    for w in (60, 300):
        hv, hs = _roll_extreme(mid, w, True)
        lv, ls = _roll_extreme(mid, w, False)
        f[f"dist_hi{w}"] = (mid / hv - 1) * 1e4
        f[f"dist_lo{w}"] = (mid / lv - 1) * 1e4
        if w == 60:
            f["since_hi60"], f["since_lo60"] = hs, ls
    lv30, _ = _roll_extreme(mid, 30, False)
    f["rebound_lo30"] = (mid / lv30 - 1) * 1e4
    f["spread_bp"] = (ba - bb) / mid * 1e4
    f["spread_ticks"] = (ba - bb) / tick_at
    f["imb3"] = (bd3 - ad3) / (bd3 + ad3)
    f["cumval"] = np.cumsum(val)
    f["n_tr_sec"] = n_tr
    path = {"bid": bb, "ask": ba, "hi": hi, "tick": tick_at, "last": ffill(last_px),
            "buyval": buyval, "sellval": sellval, "strength": stg, "e_ofi": e}
    return f, path


# ---- scripts/flow_84/library.py --------------------------------------------------
class F83:  # 원본의 `import features as F83` 자리
    N = N
    T0 = T0
    rsum = staticmethod(rsum)
    lag = staticmethod(lag)
    ffill = staticmethod(ffill)
    _roll_extreme = staticmethod(_roll_extreme)
    build_code = staticmethod(build_code)


W8 = (1, 2, 3, 5, 10, 20, 30, 60)



@numba.njit
def _trade_runs(sec, side, price, vol, n, big_thr_by_min):
    run = np.zeros(n)            # 초 끝 시점 연속 매수 체결 길이
    since_big = np.full(n, np.nan)
    cur = 0
    last_big = -1
    j = 0
    m = len(sec)
    for s in range(n):
        while j < m and sec[j] == s:
            if side[j] > 0:
                cur += 1
                if price[j] * vol[j] >= big_thr_by_min[s // 60]:
                    last_big = s
            elif side[j] < 0:
                cur = 0
            j += 1
        run[s] = cur
        if last_big >= 0:
            since_big[s] = s - last_big
    return run, since_big


def _extra(tk: pd.DataFrame, qt: pd.DataFrame, f: dict, p: dict) -> dict:
    """83번 격자 위에 얹는 추가 열(전부 t 이하)."""
    N = F83.N
    rs, lag = F83.rsum, F83.lag
    buy, sell = p["buyval"].astype(np.float64), p["sellval"].astype(np.float64)
    bid, ask = p["bid"].astype(np.float64), p["ask"].astype(np.float64)
    mid = (bid + ask) / 2
    stg = p["strength"].astype(np.float64)
    e = p["e_ofi"].astype(np.float64)
    sec = (tk.ts.dt.hour * 3600 + tk.ts.dt.minute * 60 + tk.ts.dt.second).to_numpy() - F83.T0
    ok = (sec >= 0) & (sec < N)
    tk, sec = tk[ok], sec[ok].astype(np.int64)
    price, vol, side = tk.price.to_numpy(float), tk.volume.to_numpy(float), tk.side.to_numpy(np.int64)
    volsec = np.bincount(sec, weights=vol, minlength=N).astype(float)
    tv = price * vol
    thr = np.full(N // 60 + 1, np.inf)
    for mi in range(N // 60 + 1):
        a, b = np.searchsorted(sec, mi * 60 - 600), np.searchsorted(sec, mi * 60)
        if b - a >= 20:
            thr[mi] = float(np.median(tv[a:b])) * 20.0
    run, since_big = _trade_runs(sec, side, price, vol, N, thr)

    bq1 = np.full(N, np.nan)
    aq1 = np.full(N, np.nan)
    if len(qt):
        qs = (qt.ts.dt.hour * 3600 + qt.ts.dt.minute * 60 + qt.ts.dt.second).to_numpy() - F83.T0
        mq = (qs >= 0) & (qs < N)
        bq1[qs[mq]] = qt.bidqty1.to_numpy(float)[mq]
        aq1[qs[mq]] = qt.askqty1.to_numpy(float)[mq]
    bq1, aq1 = F83.ffill(bq1), F83.ffill(aq1)
    sellvol = np.bincount(sec[side < 0], weights=vol[side < 0], minlength=N).astype(float)

    x: dict[str, np.ndarray] = {}
    tot = buy + sell
    sgn = np.sign(buy - sell)
    for w in W8:
        bw, sw_ = rs(buy, w), rs(sell, w)
        x[f"bv{w}"] = np.log1p(bw)
        x[f"sv{w}"] = np.log1p(sw_)
        x[f"bshare{w}"] = bw / (bw + sw_ + 1)
        x[f"ntr{w}"] = rs(f["n_tr_sec"], w)
        x[f"dS{w}"] = stg - lag(stg, w)
        x[f"ofiw{w}"] = rs(e, w) / (pd.Series(np.nan_to_num((bq1 + aq1) / 2)).rolling(60, min_periods=1).mean().to_numpy() + 1)
        x[f"r{w}"] = (mid / lag(mid, w) - 1) * 1e4
        vw = rs(tot, w) / (rs(volsec, w) + 1e-9)
        x[f"vwapdev{w}"] = np.where(rs(volsec, w) > 0, (mid / vw - 1) * 1e4, 0.0)
        x[f"bq1chg{w}"] = (bq1 - lag(bq1, w)) / (lag(bq1, w) + 1)
        x[f"aq1depl{w}"] = (lag(aq1, w) - aq1) / (lag(aq1, w) + 1)
    for w in (10, 30):
        hv, hs = F83._roll_extreme(mid, w, True)
        lv, ls = F83._roll_extreme(mid, w, False)
        x[f"dhi{w}"], x[f"dlo{w}"] = (mid / hv - 1) * 1e4, (mid / lv - 1) * 1e4
        x[f"sincehi{w}"], x[f"sincelo{w}"] = hs, ls
        x[f"burst{w}"] = rs((f["n_tr_sec"] >= 5).astype(float), w)
        x[f"sgnac{w}"] = rs(sgn * lag(np.nan_to_num(sgn), 1), w) / w
    x["sweep10"] = pd.Series(np.nan_to_num(f["sweep_ticks1"])).rolling(10, min_periods=1).max().to_numpy()
    micro = (bid * aq1 + ask * bq1) / (bq1 + aq1)
    x["micro_dev"] = (micro / mid - 1) * 1e4
    x["micro_dev5"] = pd.Series(x["micro_dev"]).rolling(5, min_periods=1).mean().to_numpy()
    x["l1_imb"] = (bq1 - aq1) / (bq1 + aq1)
    same_bid = bid == lag(bid, 1)
    cancel = np.where(same_bid, np.clip(lag(bq1, 1) - bq1 - sellvol, 0, None), 0.0) / (lag(bq1, 1) + 1)
    x["cancel3"], x["cancel10"] = rs(np.nan_to_num(cancel), 3), rs(np.nan_to_num(cancel), 10)
    x["buy_run"], x["since_bigbuy"] = run, since_big
    v5 = rs(tot, 5)
    x["val_dd5"] = np.log1p(v5) - 2 * np.log1p(lag(v5, 5)) + np.log1p(lag(v5, 10))
    return x




def features_by_code(tk, qt, prevclose):
    """``build_day`` 피처 부분(원본 줄 그대로, DB·재료·행 필터·라벨 제외) → {code: {열: 격자 전체}}."""
    qg = dict(tuple(qt.groupby("code")))
    tg = dict(tuple(tk.groupby("code")))
    per = {c: F83.build_code(g, qg.get(c, qt.iloc[:0])) for c, g in tg.items() if len(g) >= 500}
    codes = sorted(per)
    cum = np.stack([per[c][0]["cumval"] for c in codes])
    rank = (-cum).argsort(axis=0).argsort(axis=0) + 1
    dayret = np.stack([(per[c][1]["last"] / prevclose.get(c, np.nan) - 1) * 100 for c in codes])
    dayret_rank = (-np.nan_to_num(dayret, nan=-1e9)).argsort(axis=0).argsort(axis=0) + 1
    out = {}
    for ci, c in enumerate(codes):
        f, p = per[c]
        f["cum_rank"], f["log_cumval"], f["sec"] = rank[ci].astype(float), np.log1p(f["cumval"]), np.arange(N, dtype=float)
        x = _extra(tg[c], qg.get(c, qt.iloc[:0]), f, p)
        cols = {k: f[k] for k in FEATURES}
        cols.update(x)
        cols["dayret_rank"] = dayret_rank[ci].astype(float)
        out[c] = {k: cols[k] for k in FEATURES_JSON}
    return out


# ---- 합성 데이터(원본 아님) ---------------------------------------------------------
def synthetic_market(seed, *, codes=("000010", "000020", "000030", "000040"), n_ticks=5000, n_quotes=5000,
                     day="2026-09-08"):
    """여러 종목의 가짜 틱·호가 + 전일 종가.

    일부러 섞는 것: 장 초반 호가 없는 구간(빈 호가창 초)·중간 체결 없는 긴 공백·같은 초 폭주
    매수(체결 5건 이상·같은 수량 반복·여러 호가 쓸기)·side 0 체결·nan/0 best 호가·무효 스냅샷
    (bid1=0 인데 잔량은 있음)·nan 잔량·2,000원 호가단위 경계·늦게 첫 체결하는 종목(누적대금 0 동점)·
    전일 종가 없는 종목(nan 동점).
    """
    rng = np.random.default_rng(seed)
    d0 = pd.Timestamp(day)
    tks, qts = [], []
    for ci, code in enumerate(codes):
        base = 1990.0 + 3 * ci
        # 체결 시각: 공백 [8000, 8900) 초, 3번째부터는 1500초부터(누적대금 0·last nan 동점)
        s = rng.uniform(-60 if ci < 2 else 1500, N + 60, n_ticks)
        s = s[(s < 8000) | (s >= 8900)]
        bursts = rng.choice(np.arange(200 if ci < 2 else 1500, N - 10), 150)
        bs = np.repeat(bursts, 8) + rng.uniform(0, 0.99, 150 * 8)
        s = np.sort(np.concatenate([s, bs]))
        k = len(s)
        walk = base + np.cumsum(rng.choice([-5.0, -1.0, 0.0, 1.0, 5.0], k, p=[.1, .2, .4, .2, .1]))
        price = np.maximum(np.round(walk), 1.0)
        vol = np.where(rng.random(k) < 0.4, rng.choice([10, 50, 100], k), rng.integers(1, 500, k))
        side = rng.choice([-1, 0, 1], k, p=[.4, .05, .55])
        bb = price - rng.choice([0.0, 1.0], k)
        ba = bb + rng.choice([1.0, 2.0, 5.0], k)
        bb[rng.random(k) < 0.03] = np.nan
        ba[rng.random(k) < 0.03] = 0.0
        early = s < 120  # 장 초반은 틱에도 호가가 없다 → 빈 호가창
        bb[early] = np.nan
        ba[early] = np.nan
        stg = 100 + np.cumsum(rng.normal(0, 1.5, k))
        stg[rng.random(k) < 0.05] = np.nan
        tks.append(pd.DataFrame({
            "code": code, "ts": d0 + pd.to_timedelta(T0 + s, unit="s"), "seq": np.arange(k),
            "price": price, "volume": vol, "side": side, "strength": stg,
            "best_bid": bb, "best_ask": ba,
        }))
        q = np.sort(rng.uniform(150, N + 60, n_quotes))
        q = q[(q < 3000) | (q >= 3400)]  # 호가 스냅샷 공백
        m = len(q)
        qwalk = base + np.cumsum(rng.choice([-5.0, -1.0, 0.0, 1.0, 5.0], m, p=[.1, .2, .4, .2, .1]))
        b1 = np.maximum(np.round(qwalk), 1.0)
        a1 = b1 + rng.choice([1.0, 2.0, 5.0, 0.0], m, p=[.6, .2, .15, .05])
        b1[rng.random(m) < 0.03] = 0.0
        qd = pd.DataFrame({"code": code, "ts": d0 + pd.to_timedelta(T0 + q, unit="s"),
                           "bid1": b1, "ask1": a1})
        for c in ("bidqty1", "bidqty2", "bidqty3", "askqty1", "askqty2", "askqty3"):
            v = rng.integers(0, 5000, m).astype(float)
            v[rng.random(m) < 0.02] = np.nan
            v[rng.random(m) < 0.02] = 0.0
            qd[c] = v
        qts.append(qd)
    ticks = pd.concat(tks, ignore_index=True)
    quotes = pd.concat(qts, ignore_index=True)
    prevclose = {c: 1985.0 + 7 * i for i, c in enumerate(codes[:-1])}  # 마지막 종목은 전일 종가 없음
    return ticks, quotes, prevclose
