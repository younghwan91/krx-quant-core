"""위험모델 입력 — kr_quant 의 수익률·시총·ADV·업종·스타일 원천을 **그 시점에 알던 값**으로.

:mod:`krx_quant_core.risk.factor_model` 이 받는 ``code × date`` 패널을 한 번에 만든다.
날짜 t 열의 값은 **t 장 마감까지 알 수 있던 정보만** 쓴다(노출은 전일 마감 기준이라는
위험모델 규약은 소비 쪽이 ``X[:, t-1]`` 로 지킨다).

원천과 시점 규칙:

- 수익률: ``daily_bars_adjusted`` 종가 대 종가(분할 반영). 상장폐지 종목 포함(네이버 백필분).
- 시총 = **원 종가** × 그때 알던 상장주식수(swing-it 정정 G21-1). 원 종가는 ``daily_bars_raw`` →
  ``supply_demand.close`` → 수정 종가 순(:data:`CAP_SOURCES`, 실측 2021: 원주가에 없는
  248종목이 전부 폐지 종목, 그중 205가 수급표 종가를 가진다). 주식수는
  ``shares_outstanding_history`` 의 ``knowledge_date`` **다음 거래일**부터 — 같은 시점에 여러
  판이면 기준일(``date``)이 가장 늦은 것.
- 업종: ``wics_membership.sector_cd``(WICS 10 섹터 — 산업군 ``industry_cd`` 는 2019 미디어 등이 비어
  있다). 주간 스냅샷을 다음 스냅샷 전까지 앞으로 채운다. WICS 에 없는 종목(스팩·리츠·ETF)은 ``None``
  — 추정 유니버스에서 빠진다. 2019 이전은 전부 ``None``.
- 가치(EP): TTM 순이익 / 시총. 실적 행은 max(``avail_date``, 정기보고서 접수일) **다음 거래일**부터
  (swing-it ``eps_asof`` 규약 — ``avail_date`` 는 법정 마감 근사라 18% 가 그보다 늦게 나왔다).
  Q4 행은 사업보고서 연간 값이다. TTM = 직전 연간 + Σ_{올해 공시 분기}(당기 − 전년 동기).

2026-09-14~ 종가·거래량에 애프터마켓이 섞였다(:data:`~.daily.AFTERMARKET_START`) — 이 로더는 그
구간을 아직 정규장으로 바꾸지 않으므로 ``end`` 가 그날 이후면 거부한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .daily import AFTERMARKET_START, TRADE_VALUE_UNIT, daily_panel, load_daily_bars
from .db import connect, fetch_frame, sql_literal

__all__ = [
    "CAP_SOURCES",
    "SHARE_EVENT_KINDS",
    "STYLE_NAMES",
    "RiskInputs",
    "load_risk_inputs",
    "style_panels",
]

#: 스타일 원천 이름(:func:`style_panels` 의 키, 위험모델 노출 순서).
STYLE_NAMES: tuple[str, ...] = ("size", "beta", "momentum", "resid_vol", "liquidity", "value")
#: 시총 원 종가의 출처 코드(:attr:`RiskInputs.cap_src`). -1 = 시총 없음.
CAP_SOURCES: tuple[str, ...] = ("daily_bars_raw", "supply_demand", "daily_bars_adjusted")
#: 주식수를 굴리는 ``corporate_actions.kind`` — 분할·병합·권리락(무상·유상)·감자.
#: 합병·인적분할은 제외(:func:`_roll_events`).
SHARE_EVENT_KINDS: tuple[str, ...] = ("split", "reverse_split", "rights_ex", "capital_reduction")
_NO_DAY = np.iinfo(np.int64).min // 2
#: 스타일 계산 창(거래일).
BETA_WINDOW, BETA_HALF_LIFE, BETA_MIN_OBS = 252, 63, 126
MOM_LONG, MOM_SKIP = 252, 21
RESID_WINDOW = 63
ADV_WINDOW = 20
#: 유동성 = 회전율(거래대금/시총) 1·3·12개월 로그 평균의 가중합(Barra CNE5 LIQUIDTY: STOM·STOQ·STOA
#: 0.35/0.35/0.30). 20일 ADV 하나로 쓴 v1 은 노출이 빨리 평균회귀해 고정 포트의 21일 보유 동안
#: 자기 노출이 0.90 으로 줄었고 바이어스 0.75 로 완료 조건 4 를 못 넘었다(스펙 §6).
LIQ_WINDOWS: tuple[tuple[int, float], ...] = ((21, 0.35), (63, 0.35), (252, 0.30))
#: 시작일 앞에서 미리 읽는 달력일(스타일 창 252 거래일 + 여유).
WARMUP_DAYS = 400
#: 주가지수(KOSPI) — ``sector_index`` 값은 ×100 정수.
MARKET_INDEX = "001"


@dataclass(frozen=True)
class RiskInputs:
    """``code × date`` 패널 묶음. 금액은 전부 **원**."""

    codes: NDArray[np.str_]
    dates: pd.DatetimeIndex
    ret: NDArray[np.float64]
    """종가 대 종가 수익률(t 열 = t 종가 / 전 거래일 종가 − 1). 첫 날·결측은 nan."""
    close: NDArray[np.float64]
    """수정 종가."""
    cap: NDArray[np.float64]
    cap_src: NDArray[np.int8]
    adv_value: NDArray[np.float64]
    """t 포함 직전 :data:`ADV_WINDOW` 거래일 평균 거래대금(원)."""
    trade_value: NDArray[np.float64]
    """일 거래대금(원)."""
    sector: NDArray[np.object_]
    market_ret: NDArray[np.float64]
    """KOSPI 일수익률(``dates`` 순서)."""
    ttm_netinc: NDArray[np.float64]

    def sector_at(self, t: int) -> NDArray[np.object_]:
        return self.sector[:, t]


def _day(d: date | datetime | str) -> date:
    return pd.Timestamp(d).date()


def _equity_codes(conn: Any) -> set[str]:
    rows = fetch_frame(
        conn,
        "SELECT code FROM stocks UNION SELECT code FROM delisted_stocks "
        "UNION SELECT code FROM kind_delistings",
        ["code"], dtypes={"code": str},
    )  # fmt: skip
    return set(rows["code"].astype(str))


def _pivot(
    frame: pd.DataFrame, value: str, codes: NDArray[np.str_], dates: pd.DatetimeIndex
) -> NDArray[np.float64]:
    out = np.full((len(codes), len(dates)), np.nan)
    if frame.empty:
        return out
    ci = pd.Index(codes).get_indexer(frame["code"].astype(str))
    di = dates.get_indexer(pd.to_datetime(frame["date"]))
    ok = (ci >= 0) & (di >= 0)
    out[ci[ok], di[ok]] = frame[value].to_numpy(np.float64)[ok]
    return out


def _avail_index(dates: pd.DatetimeIndex, known: pd.Series) -> NDArray[np.int64]:
    """알게 된 날 → 그 값을 처음 쓸 수 있는 ``dates`` 행(다음 거래일)."""
    return dates.searchsorted(pd.to_datetime(known).to_numpy(), side="right").astype(np.int64)


def _asof_shares(
    sh: pd.DataFrame, codes: NDArray[np.str_], dates: pd.DatetimeIndex
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """상장주식수 as-of 패널과 그 값의 기준일(``date``, 1970-01-01 부터의 일 수 — 없으면 최솟값).

    쓸 수 있게 된 판 중 기준일이 가장 늦은 것(같으면 늦게 안 판). 기준일은 :func:`_roll_events` 가
    "그 뒤 사건"을 고르는 데 쓴다.
    """
    out = np.full((len(codes), len(dates)), np.nan)
    ref = np.full((len(codes), len(dates)), _NO_DAY, dtype=np.int64)
    if sh.empty:
        return out, ref
    sh = sh.assign(a=_avail_index(dates, sh["knowledge_date"]), d=pd.to_datetime(sh["date"]))
    sh = sh[sh["a"] < len(dates)].sort_values(["code", "a", "d", "knowledge_date"], kind="stable")
    cidx = pd.Index(codes)
    for code, g in sh.groupby("code", sort=False):
        i = cidx.get_indexer([code])[0]
        if i < 0:
            continue
        best_d = None
        for a, d, v in zip(
            g["a"].to_numpy(),
            g["d"].to_numpy(),
            g["shares_outstanding"].to_numpy(np.float64),
            strict=True,
        ):
            # 옛 기준일의 늦은 정정은 더 최신 기준일 값을 덮지 않는다(같은 기준일 정정은 덮는다).
            if best_d is None or d >= best_d:
                best_d = d
                out[i, a:] = v
                ref[i, a:] = _days(d)
    return out, ref


def _days(d: Any) -> int:
    return int(np.datetime64(d, "D").astype(np.int64))


def _roll_events(
    shares: NDArray[np.float64],
    ref: NDArray[np.int64],
    events: pd.DataFrame,
    codes: NDArray[np.str_],
    dates: pd.DatetimeIndex,
) -> NDArray[np.float64]:
    """주식수 기준일 뒤, t 까지 일어난 분할·병합·권리락·감자를 주식수에 반영한다.

    DART 주식수는 연 1행(12-31 기준, 80~90일 뒤 공시)이라 그 사이 분할이 있으면 원 종가만 떨어지고
    주식수는 그대로다(035720 2021-04 5:1 분할 뒤 시총이 실제의 0.19배 — 리뷰 M5). ``events`` 는
    ``corporate_actions`` 의 (code, date, upd_rt): ``date`` 는 새 기준 첫 거래일이고 그 전 가격 ×
    (1 + upd_rt/100) 이 조정 가격이므로 주식수 배수는 그 역수다. 합병·인적분할(kind=merger)은
    가격 계수가 주식수 변화와 다르므로(017670 2021-11 −82.75%, 실제 주식수 ×3.04) 넣지 않는다 —
    :data:`SHARE_EVENT_KINDS`. 유상 권리락은 이론가 기준이라 실제 신주보다 적게 잡힌다(근사).
    상장 종목만 사건이 있어 폐지 종목은 굴리지 않는다.
    """
    out = shares.copy()
    if events.empty:
        return out
    day = dates.to_numpy("datetime64[D]").astype(np.int64)
    cidx = pd.Index(codes)
    for code, g in events.groupby("code", sort=False):
        i = cidx.get_indexer([code])[0]
        if i < 0:
            continue
        g = g.sort_values("date")
        de = g["date"].to_numpy("datetime64[D]").astype(np.int64)
        cum = np.r_[0.0, np.cumsum(-np.log1p(g["upd_rt"].to_numpy(np.float64) / 100.0))]
        k_t = np.searchsorted(de, day, side="right")
        k_r = np.searchsorted(de, ref[i], side="right")
        out[i] = shares[i] * np.exp(cum[k_t] - cum[k_r])
    return out


def _ttm_panel(
    earn: pd.DataFrame, codes: NDArray[np.str_], dates: pd.DatetimeIndex
) -> NDArray[np.float64]:
    """TTM 순이익 as-of 패널. ``earn`` 열: code, period('YYYYQk'), known, netinc, netinc_prior."""
    out = np.full((len(codes), len(dates)), np.nan)
    if earn.empty:
        return out
    earn = earn.assign(a=_avail_index(dates, earn["known"]))
    earn = earn[earn["a"] < len(dates)].sort_values(["code", "a", "known"], kind="stable")
    cidx = pd.Index(codes)
    for code, g in earn.groupby("code", sort=False):
        i = cidx.get_indexer([code])[0]
        if i < 0:
            continue
        state: dict[tuple[int, int], tuple[float, float]] = {}
        for a, per, ni, nip in zip(
            g["a"].to_numpy(), g["period"].astype(str), g["netinc"].to_numpy(np.float64),
            g["netinc_prior"].to_numpy(np.float64), strict=True,
        ):  # fmt: skip
            try:
                y, q = int(per[:4]), int(per[-1])
            except ValueError:
                continue
            state[(y, q)] = (ni, nip)
            out[i, a:] = _ttm(state)
    return out


def _ttm(state: dict[tuple[int, int], tuple[float, float]]) -> float:
    y, q = max(state)
    if q == 4:
        return state[(y, 4)][0]
    annual = state.get((y - 1, 4))
    if annual is None or not np.isfinite(annual[0]):
        return np.nan
    adj = 0.0
    for j in range(1, q + 1):
        cur = state.get((y, j))
        if cur is None or not (np.isfinite(cur[0]) and np.isfinite(cur[1])):
            return annual[0]  # 올해 분기가 빠졌으면 직전 연간으로
        adj += cur[0] - cur[1]
    return annual[0] + adj


def load_risk_inputs(
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    conn: Any = None,
    warmup_days: int = WARMUP_DAYS,
) -> RiskInputs:
    """[start − warmup, end] 를 읽어 패널로. ``dates`` 는 그 전체(스타일 창 계산용 앞부분 포함).

    호출부는 ``dates`` 에서 start 이후 열만 asof 로 쓴다. 상장폐지 종목이 DB 에는 있는데
    결과에 없으면 ``AssertionError``(생존편향 가드 — swing-it ``read_prices`` 와 같은 원칙).
    """
    a, b = _day(start), _day(end)
    if b >= AFTERMARKET_START:
        raise ValueError(
            f"end {b} 가 애프터마켓 혼입 구간({AFTERMARKET_START}~)이다 — "
            "정규장 보정 경로가 아직 없다"
        )
    lo = a - timedelta(days=warmup_days)
    own = conn is None
    c = connect() if own else conn
    try:
        return _load(c, lo, a, b)
    finally:
        if own:
            c.close()


def _load(conn: Any, lo: date, a: date, b: date) -> RiskInputs:
    equities = _equity_codes(conn)
    bars = load_daily_bars(lo, b, adjusted=True, conn=conn)
    bars = bars[bars["code"].astype(str).isin(equities)]
    bars = bars.assign(code=bars["code"].astype(str))
    panel = daily_panel(bars, ("close", "trade_value"))
    codes, dates = panel.codes, panel.dates
    close = panel.array("close")
    tv = panel.array("trade_value") * TRADE_VALUE_UNIT

    ret = np.full_like(close, np.nan)
    ret[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    adv = pd.DataFrame(tv.T).rolling(ADV_WINDOW, min_periods=ADV_WINDOW // 2).mean().to_numpy().T

    span = f"date BETWEEN {sql_literal(lo)} AND {sql_literal(b)}"
    raw = fetch_frame(
        conn,
        f"SELECT code, date, close FROM daily_bars_raw WHERE {span}",
        ["code", "date"] + ["close"],
        dtypes={"code": str, "close": float},
        timestamps=["date"],
    )
    sd = fetch_frame(
        conn,
        f"SELECT code, date, abs(close) FROM supply_demand WHERE {span} AND close IS NOT NULL",
        ["code", "date", "close"],
        dtypes={"code": str, "close": float},
        timestamps=["date"],
    )
    px = np.full_like(close, np.nan)
    cap_src = np.full(close.shape, -1, dtype=np.int8)
    for k, p in enumerate(
        (_pivot(raw, "close", codes, dates), _pivot(sd, "close", codes, dates), close)
    ):
        fill = np.isnan(px) & np.isfinite(p) & (p > 0)
        px[fill] = p[fill]
        cap_src[fill] = k

    sh = fetch_frame(
        conn,
        f"SELECT code, date, shares_outstanding, knowledge_date FROM shares_outstanding_history "
        f"WHERE knowledge_date <= {sql_literal(b)}",
        ["code", "date", "shares_outstanding", "knowledge_date"],
        dtypes={"code": str, "shares_outstanding": float}, timestamps=["date", "knowledge_date"],
    )  # fmt: skip
    shares, ref = _asof_shares(sh, codes, dates)
    kinds = ", ".join(f"'{k}'" for k in SHARE_EVENT_KINDS)
    ev = fetch_frame(
        conn,
        f"SELECT code, date, upd_rt FROM corporate_actions WHERE kind IN ({kinds}) "
        f"AND upd_rt <> 0 AND upd_rt > -100 AND date <= {sql_literal(b)}",
        ["code", "date", "upd_rt"],
        dtypes={"code": str, "upd_rt": float},
        timestamps=["date"],
    )
    shares = _roll_events(shares, ref, ev, codes, dates)
    cap = px * shares
    cap_src[~np.isfinite(cap)] = -1

    wics = fetch_frame(
        conn,
        f"SELECT code, date, sector_cd FROM wics_membership WHERE {span} AND sector_cd IS NOT NULL",
        ["code", "date", "sector_cd"], dtypes={"code": str, "sector_cd": str}, timestamps=["date"],
    )  # fmt: skip
    sector = _sector_panel(wics, codes, dates)

    idx = fetch_frame(
        conn, f"SELECT date, close FROM sector_index WHERE code = '{MARKET_INDEX}' AND {span}",
        ["date", "close"], dtypes={"close": float}, timestamps=["date"],
    )  # fmt: skip
    lvl = pd.Series(idx["close"].to_numpy(), index=pd.to_datetime(idx["date"])).reindex(dates)
    market_ret = lvl.pct_change(fill_method=None).to_numpy()

    earn = fetch_frame(
        conn,
        "SELECT e.code, e.period, GREATEST(e.avail_date, p.per_dt), NULLIF(e.netinc, 'NaN'), "
        "NULLIF(e.netinc_prior, 'NaN') "
        "FROM earnings e LEFT JOIN (SELECT code, period, min(rcept_dt) AS per_dt "
        "FROM earnings_disclosures "
        "WHERE kind = 'periodic' GROUP BY code, period) p USING (code, period) "
        f"WHERE GREATEST(e.avail_date, p.per_dt) <= {sql_literal(b)}",
        ["code", "period", "known", "netinc", "netinc_prior"],
        dtypes={"code": str, "period": str, "netinc": float, "netinc_prior": float},
        timestamps=["known"],
    )  # fmt: skip
    ttm = _ttm_panel(earn, codes, dates)

    _assert_delisted(conn, codes, a, b)
    return RiskInputs(
        codes=codes,
        dates=dates,
        ret=ret,
        close=close,
        cap=cap,
        cap_src=cap_src,
        adv_value=adv,
        trade_value=tv,
        sector=sector,
        market_ret=market_ret,
        ttm_netinc=ttm,
    )


def _sector_panel(
    wics: pd.DataFrame, codes: NDArray[np.str_], dates: pd.DatetimeIndex
) -> NDArray[np.object_]:
    out = np.full((len(codes), len(dates)), None, dtype=object)
    if wics.empty:
        return out
    wide = wics.pivot_table(index="date", columns="code", values="sector_cd", aggfunc="first")
    snap = pd.to_datetime(wide.index)
    # 스냅샷은 그날 구성 — 그날부터 다음 스냅샷 전까지. 스냅샷에 없는 종목은 그 주 None(WICS 밖).
    pos = snap.searchsorted(dates, side="right") - 1
    wide = wide.reindex(columns=pd.Index(codes))
    vals = wide.to_numpy(dtype=object)
    ok = pos >= 0
    out[:, ok] = vals[pos[ok]].T
    out[pd.isna(out)] = None
    return out


def _assert_delisted(conn: Any, codes: NDArray[np.str_], a: date, b: date) -> None:
    """창 안에 폐지된 종목의 95% 이상이 패널에 있어야 한다(1개만 있어도 통과하던 v1)."""
    dl = fetch_frame(
        conn,
        "SELECT code FROM delisted_stocks WHERE last_trade_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' "
        f"AND last_trade_date::date BETWEEN {sql_literal(a)} AND {sql_literal(b)}",
        ["code"],
        dtypes={"code": str},
    )
    if len(dl):
        have = np.isin(dl["code"].astype(str).to_numpy(), codes).mean()
        if have < 0.95:
            raise AssertionError(
                f"창 안 상장폐지 {len(dl)}종목 중 {have:.0%} 만 패널에 있다 — "
                "생존편향(데이터 경로를 확인)"
            )


def style_panels(x: RiskInputs) -> dict[str, NDArray[np.float64]]:
    """:data:`STYLE_NAMES` 원천 패널(표준화 전). t 열은 t 마감까지의 정보만."""
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.log(x.cap)
        liquidity = _liquidity(x.trade_value, x.cap)
        value = x.ttm_netinc / x.cap
        logp = np.log(x.close)
    mom = np.full_like(x.close, np.nan)
    mom[:, MOM_LONG:] = logp[:, MOM_LONG - MOM_SKIP : -MOM_SKIP] - logp[:, :-MOM_LONG]
    beta, rvol = _beta_resid_vol(x.ret, x.market_ret)
    return {"size": size, "beta": beta, "momentum": mom, "resid_vol": rvol,
            "liquidity": liquidity, "value": value}  # fmt: skip


def _beta_resid_vol(
    ret: NDArray[np.float64], mkt: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n, t_len = ret.shape
    beta = np.full((n, t_len), np.nan)
    rvol = np.full((n, t_len), np.nan)
    lam = 0.5 ** (1.0 / BETA_HALF_LIFE)
    w_full = lam ** np.arange(BETA_WINDOW - 1, -1, -1, dtype=np.float64)
    for t in range(BETA_WINDOW, t_len):
        r = ret[:, t - BETA_WINDOW + 1 : t + 1]
        m = mkt[t - BETA_WINDOW + 1 : t + 1]
        ok = np.isfinite(r) & np.isfinite(m)[None, :]
        w = np.where(ok, w_full[None, :], 0.0)
        sw = w.sum(1)
        nobs = ok.sum(1)
        mm = np.where(ok, m[None, :], 0.0)
        rr = np.where(ok, r, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mbar = (w * mm).sum(1) / sw
            rbar = (w * rr).sum(1) / sw
            cov = (w * (mm - mbar[:, None]) * (rr - rbar[:, None])).sum(1) / sw
            var = (w * (mm - mbar[:, None]) ** 2).sum(1) / sw
            b = cov / var
        b[nobs < BETA_MIN_OBS] = np.nan
        beta[:, t] = b
        tail = slice(BETA_WINDOW - RESID_WINDOW, BETA_WINDOW)
        res = rr[:, tail] - b[:, None] * mm[:, tail]
        okt = ok[:, tail]
        cnt = okt.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(okt, res, 0.0).sum(1) / cnt
            sd = np.sqrt(np.where(okt, (res - mu[:, None]) ** 2, 0.0).sum(1) / (cnt - 1))
        sd[cnt < RESID_WINDOW // 2] = np.nan
        rvol[:, t] = sd
    return beta, rvol


def _liquidity(tv: NDArray[np.float64], cap: NDArray[np.float64]) -> NDArray[np.float64]:
    """:data:`LIQ_WINDOWS` 가중합. 창마다 ln(21 × 평균 일 회전율) — 창의 절반 미만 관측이면 NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        turn = pd.DataFrame((tv / cap).T)
        out = np.zeros(tv.shape)
        for win, wt in LIQ_WINDOWS:
            m = turn.rolling(win, min_periods=win // 2).mean().to_numpy().T
            out = out + wt * np.log(21.0 * m)
    return out
