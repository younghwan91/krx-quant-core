"""성과 귀속 — 포트 수익을 팩터 기여와 특이 기여로 쪼갠다.

위험모델 r_t = X_{t-1} f_t + u_t (노출은 전일 마감 기준) 아래에서 날짜 t 의 포트 수익은

    Σ_i w_{i,t} r_{i,t} = Σ_k (w_tᵀ X_{t-1})_k f_{t,k} + w_tᵀ u_t

로 정확히 갈린다. :func:`factor_attribution` 이 이 우변을 열별로 낸다. ``total`` 은 그 열들의
합이고, 같은 X·f·u 로 만든 실제 포트 수익과 부동소수 반올림(≲1e-12) 안에서 같다.

입력 규약(``risk.factor_model.FactorReturns`` 와 같은 모양 — 그 모듈을 import 하지는 않는다):

- ``w`` : ``ndarray[N, T]`` — 열 t = 날짜 t 동안 **쥔** 비중(t-1 마감에 정한 것). NaN 은 0(안 쥠).
- ``X`` : ``ndarray[T, N, K]`` 또는 길이 T 의 ``ndarray[N, K]`` 열(쌓아서 쓴다). 슬롯 t 는
  날짜 t 의 수익 f_t 를 추정할 때 쓴 노출, 즉 **전일 마감 X_{t-1}** 이다 — 이 함수는 밀지
  않는다(호출부가 맞춘 대로).
- ``f`` : ``ndarray[T, K]`` — 날짜 t 의 팩터 수익. 열 순서 = ``names``.
- ``u`` : ``ndarray[N, T]`` — 특이(잔차) 수익.

결측: 비중이 0 인(또는 NaN 인) 종목의 X·u 결측은 무시한다. **쥔 종목**의 X·u 가 NaN 이면 그날
해당 열과 ``total`` 이 NaN 이 된다 — 0 으로 채워 틀린 귀속을 내지 않는다.

숫자만 낸다(판정 키 없음).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["factor_attribution"]

_RESERVED = ("date", "specific", "total")


def factor_attribution(
    w: np.ndarray,
    X: np.ndarray | Sequence[np.ndarray],
    f: np.ndarray,
    u: np.ndarray,
    *,
    names: Sequence[str],
    dates: Sequence,
) -> pd.DataFrame:
    """날짜별 팩터 기여 ``(wᵀX)_k f_k``·특이 기여 ``wᵀu``·합계.

    Args:
        w: ``[N, T]`` 보유 비중.
        X: ``[T, N, K]`` 노출(슬롯 t = X_{t-1}) 또는 그런 ``[N, K]`` 의 길이-T 열.
        f: ``[T, K]`` 팩터 수익.
        u: ``[N, T]`` 특이 수익.
        names: 길이 K 팩터 이름(``f`` 열 순서). ``date``·``specific``·``total`` 은 쓸 수 없다.
        dates: 길이 T 날짜 라벨.

    Returns:
        ``DataFrame[date, <names…>, specific, total]`` — T 행. ``total = Σ names + specific``.
    """
    W = np.asarray(w, dtype=float)
    Xa = np.asarray(X, dtype=float)               # 길이-T 열이면 [T, N, K] 로 쌓인다
    F = np.asarray(f, dtype=float)
    U = np.asarray(u, dtype=float)
    names = list(names)
    dates = list(dates)

    if W.ndim != 2:
        raise ValueError(f"w 는 [N, T] 여야 한다: {W.shape}")
    N, T = W.shape
    if Xa.ndim != 3 or Xa.shape[:2] != (T, N):
        raise ValueError(f"X 는 [T={T}, N={N}, K] 여야 한다: {Xa.shape}")
    K = Xa.shape[2]
    if F.shape != (T, K):
        raise ValueError(f"f 는 [T={T}, K={K}] 여야 한다: {F.shape}")
    if U.shape != (N, T):
        raise ValueError(f"u 는 [N={N}, T={T}] 여야 한다: {U.shape}")
    if len(names) != K:
        raise ValueError(f"names 길이 {len(names)} != K={K}")
    if len(set(names)) != K:
        raise ValueError("names 에 중복이 있다")
    bad = [n for n in names if n in _RESERVED]
    if bad:
        raise ValueError(f"names 에 예약 열 이름이 있다: {bad}")
    if len(dates) != T:
        raise ValueError(f"dates 길이 {len(dates)} != T={T}")

    Wz = np.where(np.isfinite(W), W, 0.0)          # [N, T]
    held = Wz != 0.0
    Xz = np.where(held.T[:, :, None], Xa, 0.0)     # [T, N, K] — 안 쥔 종목의 결측은 무시
    Uz = np.where(held, U, 0.0)                    # [N, T]

    expo = np.einsum("tnk,nt->tk", Xz, Wz)         # 포트 노출 wᵀX_{t-1}
    contrib = expo * F                             # [T, K]
    specific = np.einsum("nt,nt->t", Wz, Uz)       # wᵀu
    total = contrib.sum(axis=1) + specific

    out = pd.DataFrame(contrib, columns=names)
    out.insert(0, "date", dates)
    out["specific"] = specific
    out["total"] = total
    return out
