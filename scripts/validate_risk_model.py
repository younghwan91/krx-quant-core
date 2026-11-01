"""위험모델 실데이터 검증 — 스펙 2026-11-01 §4 완료 조건 4(바이어스 통계).

kr_quant 2019-07~2023-12 월말마다 그날까지의 데이터로 :func:`build_risk_model` 을 만들고,
다음 21거래일 실현 수익 / 예측 표준편차(z)의 표준편차를 포트별로 잰다. 1 이면 예측이 맞다. 대상:

(a) 무작위 롱온리 50종목 동일가중 100개(매월 새로 뽑는다) (b) 시총가중 추정 유니버스
(c) 스타일 6개 각각의 팩터 모방 포트(그날 회귀의 순수 팩터 포트 — 롱숏).

노출 규칙: 스타일 원천을 날마다 :func:`standardize` 하고, 추정 유니버스(WICS 업종·시총·수익률 있음)
안의 결측 노출은 0(유니버스 평균)으로 둔다 — 모르는 노출을 평균으로 보는 관례. 2024~ 는 swing-it
봉인 소모 구간이라 쓰지 않는다.

    set -a; . ~/git/quant-airflow/.env; set +a
    uv run --extra db python scripts/validate_risk_model.py [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from krx_quant_core.data.risk_inputs import STYLE_NAMES, load_risk_inputs, style_panels
from krx_quant_core.risk.factor_model import bias_statistic, build_risk_model, standardize

START, END = "2019-07-01", "2023-12-28"
HORIZON = 21
N_RANDOM, RANDOM_NAMES = 100, 50
SEED = 20261101


def exposures(x, raw: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for k, v in raw.items():
        e = np.full_like(v, np.nan)
        for t in range(v.shape[1]):
            m = mask[:, t]
            if m.sum() < 30:
                continue
            col = standardize(v[:, t], x.cap[:, t], m)
            col[m & ~np.isfinite(col)] = 0.0
            e[:, t] = col
        out[k] = e
    return out


def month_ends(dates, lo: str, last_index: int) -> list[int]:
    idx = []
    for i in range(len(dates) - 1):
        if dates[i] >= np.datetime64(lo) and dates[i].month != dates[i + 1].month:
            if i + HORIZON <= last_index:
                idx.append(i)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    t0 = time.time()
    x = load_risk_inputs("2019-01-02", END)
    sector = x.sector
    has_sec = np.vectorize(lambda s: s is not None, otypes=[bool])(sector)
    mask = has_sec & np.isfinite(x.cap) & (x.cap > 0) & np.isfinite(x.close)
    raw = style_panels(x)
    X = exposures(x, raw, mask)
    print(f"입력 {time.time() - t0:.0f}s — 종목 {len(x.codes)}, 날짜 {len(x.dates)}", flush=True)

    first = int(np.argmax(has_sec.any(axis=0)))  # WICS 첫 날 — 그 전 열은 버린다
    sl = slice(first, None)
    R, cap, msk, lab = x.ret[:, sl], x.cap[:, sl], mask[:, sl], sector[:, sl]
    Xs = {k: v[:, sl] for k, v in X.items()}
    dates = x.dates[sl]
    asofs = month_ends(dates, START, R.shape[1] - 1)
    rng = np.random.default_rng(SEED)
    rng_diag = np.random.default_rng(SEED + 1)  # 진단 추첨이 바 대상의 난수열을 바꾸지 않게
    rows = {"random": [], "capw": [], **{f"style:{k}": [] for k in STYLE_NAMES}}
    # 사후 진단(리뷰 M1·M2 — 바가 못 보는 특이위험·업종·꼬리). 합격 바가 아니다.
    diag = {"random5": [], "hivol_decile": [], "small_decile": []}
    sectors = sorted({s for s in lab[:, -1] if s is not None})
    diag.update({f"sector_ew:{g}": [] for g in sectors})
    diag.update({f"sector_pure:{g}": [] for g in sectors})
    for n, a in enumerate(asofs):
        m = build_risk_model(R, Xs, lab, cap, msk, a, codes=np.arange(R.shape[0]))
        idx = m.codes.astype(int)
        fwd = np.nan_to_num(R[idx, a + 1 : a + 1 + HORIZON], nan=0.0).sum(axis=1)

        def z(w, fwd=fwd, m=m):
            return float(w @ fwd), float(np.sqrt(HORIZON * m.var(w)))

        rnd = []
        for _ in range(N_RANDOM):
            w = np.zeros(len(idx))
            w[rng.choice(len(idx), RANDOM_NAMES, replace=False)] = 1.0 / RANDOM_NAMES
            rnd.append(z(w))
        rows["random"].append(rnd)
        cw = cap[idx, a] / cap[idx, a].sum()
        rows["capw"].append(z(cw))
        # 순수 팩터 포트: (XᵀWX)⁻¹XᵀW 의 스타일 행(W = √시총, 회귀 가중과 같다).
        W = np.sqrt(cap[idx, a])
        P = np.linalg.pinv((m.X * W[:, None]).T @ m.X) @ (m.X * W[:, None]).T
        for k in STYLE_NAMES:
            rows[f"style:{k}"].append(z(P[m.names.index(k)]))
        r5 = []
        for _ in range(N_RANDOM):
            w = np.zeros(len(idx))
            w[rng_diag.choice(len(idx), 5, replace=False)] = 0.2
            r5.append(z(w))
        diag["random5"].append(r5)
        for key, raw_k, top in (
            ("hivol_decile", "resid_vol", True),
            ("small_decile", "size", False),
        ):
            v = raw[raw_k][idx, a + first]
            ok = np.isfinite(v)
            cut = np.nanquantile(v, 0.9 if top else 0.1)
            sel = ok & ((v >= cut) if top else (v <= cut))
            diag[key].append(z(sel / sel.sum()))
        labs_a = lab[idx, a]
        for g in sectors:
            sel = labs_a == g
            diag[f"sector_ew:{g}"].append(z(sel / sel.sum()) if sel.any() else (np.nan, np.nan))
            diag[f"sector_pure:{g}"].append(
                z(P[m.names.index(g)]) if g in m.names else (np.nan, np.nan)
            )
        if n % 6 == 0:
            print(f"  {dates[a].date()} 종목 {len(idx)} ({time.time() - t0:.0f}s)", flush=True)

    result = {
        "window": [str(dates[asofs[0]].date()), str(dates[asofs[-1]].date())],
        "n_asof": len(asofs),
        "horizon": HORIZON,
    }
    rr = np.array(rows["random"])  # (A, P, 2)
    result["random"] = bias_statistic((rr[:, :, 1]).T, (rr[:, :, 0]).T)
    for key in ["capw", *[f"style:{k}" for k in STYLE_NAMES]]:
        arr = np.array(rows[key])
        result[key] = bias_statistic(arr[:, 1], arr[:, 0])
    r5 = np.array(diag.pop("random5"))
    post = {"random5": bias_statistic(r5[:, :, 1].T, r5[:, :, 0].T)}
    for key, vals in diag.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr).all(axis=1)]
        post[key] = bias_statistic(arr[:, 1], arr[:, 0])
    result["post_hoc"] = post
    print(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, ensure_ascii=False, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
