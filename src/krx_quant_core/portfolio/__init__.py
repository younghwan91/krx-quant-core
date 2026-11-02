"""포트폴리오 구성 — 팩터 위험모델 위의 제약 평균-분산 최적화.

cvxpy 는 선택 의존성(``pip install krx-quant-core[portfolio]``)이다. 이 패키지 import 는 cvxpy 없이
되고, :func:`optimize` 를 부를 때 없으면 안내 ``ImportError`` 가 난다.
"""

from krx_quant_core.portfolio.optimize import Constraints, CostParams, OptResult, optimize

__all__ = ["Constraints", "CostParams", "OptResult", "optimize"]
