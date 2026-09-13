"""백테스트 성과 지표 — Sharpe·CAGR·낙폭·HAC t·순위상관·페어드 부트스트랩.

swing-it ``engine/metrics.py`` 를 **수치 동일**로 옮겼다(시그니처·기본값 그대로).
swing-it 에서도 이미 "엔진의 단일 정본" 이었고, scalp-it·daytrade-it 이 같은 지표를
각자 다시 짜면 같은 수익열에서 레포마다 다른 Sharpe 가 나온다. 그래서 여기로 올린다.

``PPY=12`` (월간 수익열) 기본값은 swing-it 의 연구 관례다. 일간 수익열이면 호출부가
``ppy=252`` 를 넘겨야 한다 — 기본값을 바꾸면 swing-it 의 발표 수치가 조용히 바뀐다.

페어드 부트스트랩이 **두 종류**다(의미가 다르므로 이름을 나눴다):

- :func:`paired_bootstrap` (swing-it): 두 수익열을 **블록 단위로 같이** 재표본해
  ΔSharpe·ΔCAGR 의 CI 를 낸다. 자기상관을 보존한다. 입력은 인덱스로 정렬된 Series.
- :func:`paired_date_bootstrap` (scalp-it ``validate/compare.py`` 의 ``paired_bootstrap``):
  ``date`` 로 머지한 ``mean_excess`` **차이**를 i.i.d. 재표본해 평균 차이의 CI 를 낸다.
  자기상관을 보존하지 않고 Sharpe 가 아니라 평균을 본다. 공통 날짜가 없으면 예외.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "PPY",
    "ann_sharpe",
    "cagr",
    "max_drawdown",
    "newey_west_t",
    "paired_bootstrap",
    "paired_date_bootstrap",
    "quantile_summary",
    "regime_buckets",
    "spearman",
    "summarize_periods",
]

PPY = 12  # periods per year (monthly return series)


def ann_sharpe(r: np.ndarray, ppy: int = PPY) -> float:
    """연환산 Sharpe = mean/std(ddof=0)·√ppy. 무위험수익률 차감 없음. 퇴화면 NaN."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std() < 1e-12:  # degenerate/no-dispersion → Sharpe undefined
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ppy))


def cagr(r: np.ndarray, ppy: int = PPY) -> float:
    """기간수익열의 연복리 성장률. 유한값이 없으면 NaN."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float((1.0 + r).prod() ** (ppy / r.size) - 1.0)


def max_drawdown(r: np.ndarray) -> float:
    """기간수익열의 최대낙폭(음수, 예 -0.2). 유한값이 없으면 NaN."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    # 초기자본 1.0 을 첫 peak 로 세운다. 이게 없으면 **첫 구간의 손실이 안 잡힌다** —
    # 자기 자신이 peak 가 되어 drawdown 0 이 되기 때문이다.
    # 실측 반례: max_drawdown([-0.20, +0.05, +0.05]) 가 0.0 을 돌려줬다(정답 -0.20).
    equity = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def newey_west_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    """평균과 Newey-West(HAC) t — 직렬상관에 강건.

    보유기간이 겹치는 수익은 자기상관이 있어 평범한 t 가 유의성을 부풀린다.
    ``lag`` 짜리 Bartlett 커널 HAC 분산으로 그걸 교정한다.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 2:
        return float("nan"), float("nan")
    mu = x.mean()
    d = x - mu
    var = (d @ d) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * ((d[k:] @ d[:-k]) / n)
    se = np.sqrt(var / n)
    return float(mu), float(mu / se) if se > 0 else float("nan")


def summarize_periods(periods: pd.DataFrame, horizon: int) -> dict:
    """리밸런스 기간표(``net``·``turnover`` 열) → 연환산 순 Sharpe·t·누적수익·손익비.

    연환산은 ``252/horizon`` 기간/년(거래일 기준)이다.
    """
    if periods.empty:
        return {"n": 0, "sharpe": float("nan"), "t_stat": float("nan"),
                "mean_net": float("nan"), "hit_rate": float("nan"),
                "cum_net": float("nan"), "avg_turnover": float("nan")}
    net = periods["net"].to_numpy()
    per_year = 252 / horizon
    std = net.std()
    ann = (1 + net.mean()) ** per_year - 1
    wins = net[net > 0]
    losses = net[net < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0
    return {
        "n": len(net),
        "sharpe": float(ann / (std * np.sqrt(per_year))) if std > 0 else float("nan"),
        "t_stat": float(net.mean() / (std / np.sqrt(len(net)))) if std > 0 else float("nan"),
        "mean_net": float(net.mean()),
        "hit_rate": float((net > 0).mean()),
        "cum_net": float((1 + net).prod() - 1),
        "avg_turnover": float(periods["turnover"].mean()),
        # 손익 프로파일: 승률이 낮아도 payoff_ratio > 1 이면 비대칭·볼록한 전략이다.
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(avg_win / avg_loss) if avg_loss > 0 else float("nan"),
        "best": float(net.max()),
        "worst": float(net.min()),
    }


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman 순위상관(순위에 Pearson — scipy 의존 없음)."""
    if len(a) < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def quantile_summary(merged: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """점수 분위별 평균 선행수익·적중률 (Q1 = 최고 점수). ``score``·``fwd_ret`` 열 필요."""
    cols = ["quantile", "n", "mean_fwd", "hit_rate"]
    if len(merged) < quantiles:
        return pd.DataFrame(columns=cols)
    # 내림차순 순위라 Q1 이 최고 점수 버킷; 라벨 1..quantiles.
    q = pd.qcut(merged["score"].rank(method="first", ascending=False), quantiles,
                labels=False) + 1
    out = (
        merged.assign(_q=q)
        .groupby("_q")
        .agg(n=("fwd_ret", "size"), mean_fwd=("fwd_ret", "mean"),
             hit_rate=("fwd_ret", lambda s: (s > 0).mean()))
        .reset_index()
        .rename(columns={"_q": "quantile"})
    )
    return out[cols]


def regime_buckets(returns: pd.Series, *, n: int = 4) -> list[dict]:
    """수익열을 시간순 ``n`` 등분해 구간별 평균·부호를 보고 — 국면 지속성 점검(3+/4 양수 기대)."""
    r = returns.to_numpy(float)
    out: list[dict] = []
    if len(r) < n:
        return out
    b = len(r) // n
    for k in range(n):
        seg = r[k * b:(k + 1) * b if k < n - 1 else len(r)]
        m = float(np.nanmean(seg))
        out.append({"start": returns.index[k * b], "mean": m, "positive": m > 0})
    return out


def paired_bootstrap(
    ret_a: pd.Series,
    ret_b: pd.Series,
    *,
    block: int = 6,
    n_boot: int = 2000,
    seed: int = 0,
    ppy: int = PPY,
) -> dict:
    """A−B 의 ΔSharpe·ΔCAGR **블록** 부트스트랩 (swing-it 판).

    두 수익열에서 같은 인덱스 블록을 함께 뽑아(짝·자기상관 보존) ΔSharpe·ΔCAGR 을
    다시 계산하고 95% CI 와 P(A>B) 를 낸다. CI 가 0 을 **배제**할 때만 사전등록 기준을
    넘은 것이다.

    Returns:
        ``{"d_sharpe_ci", "d_cagr_ci", "prob_a_better_sharpe", "n"}`` — CI 는 2.5/97.5 분위.
    """
    a = ret_a.reindex(ret_b.index).to_numpy(float)
    b = ret_b.to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < block + 1:
        return {"d_sharpe_ci": (float("nan"),) * 2, "d_cagr_ci": (float("nan"),) * 2,
                "prob_a_better_sharpe": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    d_sharpe = np.empty(n_boot)
    d_cagr = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        d_sharpe[i] = ann_sharpe(a[idx], ppy) - ann_sharpe(b[idx], ppy)
        d_cagr[i] = cagr(a[idx], ppy) - cagr(b[idx], ppy)
    ds = d_sharpe[np.isfinite(d_sharpe)]
    dc = d_cagr[np.isfinite(d_cagr)]
    nan2 = (float("nan"), float("nan"))
    return {
        "d_sharpe_ci": ((float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))
                        if ds.size else nan2),
        "d_cagr_ci": ((float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5)))
                      if dc.size else nan2),
        "prob_a_better_sharpe": float((ds > 0).mean()) if ds.size else float("nan"),
        "n": n,
    }


def paired_date_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """날짜로 짝지은 ``a - b`` 평균 차이의 i.i.d. 부트스트랩 (scalp-it 판).

    같은 날짜의 두 포트폴리오를 짝지어 차이를 보므로 시장 공통 변동이 상쇄된다.
    :func:`paired_bootstrap` 과 달리 블록이 아니고 Sharpe 가 아니라 **평균**을 본다.

    Args:
        a, b: ``date``·``mean_excess`` 열을 가진 프레임(scalp-it ``portfolio_returns`` 결과).

    Returns:
        ``mean_diff`` (관측 평균 차이), ``ci_low``/``ci_high`` (95% 신뢰구간),
        ``p_greater`` (부트스트랩 표본 중 차이가 0 초과인 비율), ``n_dates``.

    Raises:
        ValueError: 공통 날짜가 없을 때 — 폴드가 어긋났다는 뜻이라 조용히 NaN 을 내지 않는다.
    """
    merged = a.merge(b, on="date", suffixes=("_a", "_b"))
    diff = (merged["mean_excess_a"] - merged["mean_excess_b"]).to_numpy()
    n = len(diff)
    if n == 0:
        raise ValueError("두 층의 공통 날짜가 없다 — 폴드가 어긋났다")

    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_greater": float((means > 0).mean()),
        "n_dates": int(n),
    }
