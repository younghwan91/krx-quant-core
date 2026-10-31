"""벡터화한 기존 함수가 원래 구현과 비트 단위로 같은지 — 원래 코드를 여기 두고 무작위 대조한다.

수치 동일성이 계약이다(CLAUDE.md). 빠른 길을 고치면 이 파일의 원본과 다시 맞아야 한다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from krx_quant_core.backtest import crosssectional as cs
from krx_quant_core.backtest import drift, panels, replay
from krx_quant_core.execution.engine import Bar, Quote, Trade
from krx_quant_core.market.session import KST
from krx_quant_core.stats import metrics, trials


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


# --- 원래 구현 (c4fb378) -------------------------------------------------------------------


def _old_trailing_adv(V, nD, adv_window, like):
    adv = np.full_like(like, np.nan)
    for j in range(adv_window, nD):
        adv[:, j] = np.nanmean(V[:, j - adv_window:j], axis=1)
    return adv


def _old_rank_buckets(frame, value, n, date_col="date", ascending=False):
    def _bucket(s):
        r = s.rank(method="first", ascending=ascending)
        k = s.notna().sum()
        if k == 0:
            return pd.Series(np.nan, index=s.index)
        b = np.floor((r - 1) * n / k)
        return b.clip(lower=0, upper=n - 1)

    b = frame.groupby(date_col, sort=False, group_keys=False)[value].apply(_bucket)
    return b.reindex(frame.index)


def _old_adv_panel(prices, window=20):
    tv = prices[["code", "date", "trade_value"]].copy()
    tv["trade_value"] = tv["trade_value"].abs()
    tv = tv.sort_values(["code", "date"])
    tv["adv"] = tv.groupby("code")["trade_value"].transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    return tv.dropna(subset=["adv"])[["code", "date", "adv"]].reset_index(drop=True)


def _old_paired_bootstrap_draws(a, b, block, n_boot, seed, ppy):
    n = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    d_sharpe = np.empty(n_boot)
    d_cagr = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        d_sharpe[i] = metrics.ann_sharpe(a[idx], ppy) - metrics.ann_sharpe(b[idx], ppy)
        d_cagr[i] = metrics.cagr(a[idx], ppy) - metrics.cagr(b[idx], ppy)
    return d_sharpe, d_cagr


# --- 대조 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_trailing_adv_and_ffill_match_original(seed):
    rng = np.random.default_rng(seed)
    nC, nD = int(rng.integers(1, 30)), int(rng.integers(1, 90))
    V = rng.lognormal(10, 2, (nC, nD))
    V[rng.random((nC, nD)) < rng.uniform(0, 0.6)] = np.nan
    for w in (0, 1, 2, 5, 20, nD - 1, nD, nD + 3):
        with np.errstate(all="ignore"), _quiet():
            assert _same(cs._trailing_adv(V, nD, w, V), _old_trailing_adv(V, nD, w, V))
    C = V.copy()
    C[:, :1][rng.random((nC, 1)) < 0.3] = -0.0
    assert _same(cs._ffill_axis1_nb(C), pd.DataFrame(C).ffill(axis=1).to_numpy(float))


@pytest.mark.parametrize("seed", range(40))
def test_rank_buckets_match_original(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 400))
    f = pd.DataFrame({
        "date": rng.integers(0, max(1, m // 20), m),
        "v": np.round(rng.normal(size=m), int(rng.integers(0, 3))),  # 반올림으로 동점을 만든다
    })
    f.loc[rng.random(m) < 0.2, "v"] = np.nan
    f.index = rng.permutation(m) * 3
    for n in (1, 3, 5, 10):
        for asc in (False, True):
            new = drift.rank_buckets(f, "v", n=n, ascending=asc)
            old = _old_rank_buckets(f, "v", n=n, ascending=asc)
            pd.testing.assert_series_equal(new, old, check_exact=True, check_names=False)


def test_rank_buckets_nullable_all_na_stays_float64():
    f = pd.DataFrame({"date": [1, 1, 2], "v": pd.array([None, None, None], dtype="Float64")})
    b = drift.rank_buckets(f, "v", n=3)
    assert b.dtype == np.float64 and b.isna().all()


@pytest.mark.parametrize("seed", range(30))
def test_adv_panel_matches_original(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 600))
    prices = pd.DataFrame({
        "code": rng.choice(["005930", "000660", "035720", "A1"], m).astype(object),
        "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.permutation(m), "D"),
        "trade_value": rng.normal(1e9, 5e8, m),
    })
    prices.loc[rng.random(m) < 0.05, "trade_value"] = np.nan
    if seed % 3 == 0:
        prices.index = rng.integers(0, max(1, m // 2), m)  # 겹치는 인덱스
    if seed % 5 == 0:
        prices.loc[prices.index[: max(1, m // 10)], "code"] = np.nan
    for w in (1, 3, 20):
        pd.testing.assert_frame_equal(
            panels.adv_panel(prices, window=w), _old_adv_panel(prices, window=w), check_exact=True
        )


def _event_frames(rng, m):
    base = pd.Timestamp("2026-10-29 09:00")
    ts = base + pd.to_timedelta(np.sort(rng.integers(0, 20000, m)), "s")
    code = rng.choice(["005930", "000660", "069500"], m).astype(object)
    q = pd.DataFrame({"ts": ts, "code": code, "bid": rng.integers(1000, 9000, m),
                      "ask": rng.integers(1000, 9000, m).astype(float)})
    t = pd.DataFrame({"ts": ts.tz_localize(KST), "code": code,
                      "price": rng.uniform(1000, 9000, m), "qty": rng.integers(1, 99, m)})
    b = pd.DataFrame({"ts": ts.astype(str), "code": code, "open": rng.uniform(1, 9, m),
                      "high": rng.uniform(1, 9, m), "low": rng.uniform(1, 9, m),
                      "close": rng.uniform(1, 9, m), "volume": rng.integers(0, 9, m)})
    return [(q, Quote), (t, Trade), (b, Bar), (q.drop(columns="ask"), Quote)]


@pytest.mark.parametrize("seed", range(20))
def test_frame_events_fast_path_matches_rows(seed):
    rng = np.random.default_rng(seed)
    for df, cls in _event_frames(rng, int(rng.integers(0, 200))):
        names = [n for n in (f.name for f in replay.fields(cls)) if n in df.columns]
        fast = _outcome(replay._frame_events, df, cls)
        rows = _outcome(replay._frame_events_rows, df, cls, names) if len(df) else ("ok", [])
        assert fast == rows
        if fast[0] == "ok":
            assert [e.ts.tzinfo for e in fast[1]] == [e.ts.tzinfo for e in rows[1]]


def _outcome(f, *args):
    try:
        return ("ok", f(*args))
    except Exception as e:  # 원래 경로가 예외면 빠른 길도 같은 종류의 예외여야 한다
        return ("err", type(e))


def test_frame_events_object_none_still_raises():
    # 원래처럼 float(None) 이 막는다 — 조용히 NaN 가격 이벤트가 되면 안 된다.
    df = pd.DataFrame({"ts": [pd.Timestamp("2026-10-29 09:00")], "code": ["005930"],
                       "price": pd.Series([None], dtype=object), "qty": [1.0]})
    with pytest.raises(TypeError):
        replay._frame_events(df, Trade)


@pytest.mark.parametrize("seed", range(25))
def test_paired_bootstrap_matches_original(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 400))
    idx = pd.RangeIndex(n)
    a = pd.Series(rng.normal(0.001, 0.02, n), index=idx)
    b = pd.Series(rng.normal(0.0005, 0.02, n), index=idx)
    if seed % 4 == 0:
        b[:] = 0.001  # 분산 0
    block, n_boot = int(rng.integers(1, 8)), int(rng.integers(1, 300))
    got = metrics.paired_bootstrap(a, b, block=block, n_boot=n_boot, seed=seed)
    ds, dc = _old_paired_bootstrap_draws(a.to_numpy(), b.to_numpy(), block, n_boot, seed,
                                         metrics.PPY)
    ds, dc = ds[np.isfinite(ds)], dc[np.isfinite(dc)]
    want_s = (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))) if len(ds) else None
    if want_s is not None:
        assert got["d_sharpe_ci"] == want_s
        assert got["prob_a_better_sharpe"] == float((ds > 0).mean())
    if len(dc):
        assert got["d_cagr_ci"] == (float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5)))


def test_record_trials_ledger_bytes_match_one_by_one(tmp_path):
    rng = np.random.default_rng(0)
    cfgs = [{"a": int(rng.integers(0, 5)), "b": [int(rng.integers(0, 3))]} for _ in range(60)]
    one, batch = tmp_path / "one", tmp_path / "batch"
    fps_one = [trials.record_trial("x", c, logs_dir=one, note="n") for c in cfgs]
    fps_batch = trials.record_trials("x", cfgs[:30], logs_dir=batch, note="n")
    fps_batch += trials.record_trials("x", cfgs[30:], logs_dir=batch, note="n")
    assert fps_one == fps_batch
    assert (trials.ledger_path("x", logs_dir=one).read_bytes()
            == trials.ledger_path("x", logs_dir=batch).read_bytes())
    assert trials.count_trials("x", logs_dir=batch) == len({json.dumps(c, sort_keys=True)
                                                            for c in cfgs})


def test_record_trial_retries_after_failed_append(tmp_path, monkeypatch):
    # 쓰기가 실패한 시도는 다음 호출에서 다시 적혀야 한다(캐시가 '이미 적음'으로 기억하면 안 된다).
    cfg = {"k": 1}
    real_open = type(tmp_path).open

    def broken(self, *a, **k):
        if a and a[0] == "a":
            raise OSError("disk full")
        return real_open(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "open", broken)
    with pytest.raises(OSError):
        trials.record_trial("y", cfg, logs_dir=tmp_path)
    monkeypatch.setattr(type(tmp_path), "open", real_open)
    assert trials.count_trials("y", logs_dir=tmp_path) == 0
    trials.record_trial("y", cfg, logs_dir=tmp_path)
    assert trials.count_trials("y", logs_dir=tmp_path) == 1


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)
