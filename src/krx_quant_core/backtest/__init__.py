"""백테스트 부품 — 호가 충격·지정가 체결 판정·트레이드 원장/성과지표·횡단면 시뮬레이션.

엔진(이벤트 루프)은 여기 없다. 루프는 소비자마다 다르지만(틱 스캘핑·일봉 데이트레이딩·
월간 리밸런스), 그 루프가 부르는 "체결됐나/얼마에/순손익이 얼마고 성과가 어떤가" 는
세 레포가 같은 답을 내야 한다.
"""

from .crosssectional import rank_ic, rank_tilt_backtest, staggered_tranche_backtest
from .fills import EPS, FILL_BASES, FillBasis, limit_buy_filled, limit_sell_filled
from .ledger import PerformanceMetrics, Trade, TradeResult, apply_costs, performance_metrics
from .orderbook import Level, liquidity_size_multiplier, roundtrip_bp, sweep_vwap
from .panels import adv_panel, forward_returns, lookup_panel, panel_pivot

__all__ = [
    "EPS",
    "FILL_BASES",
    "FillBasis",
    "Level",
    "PerformanceMetrics",
    "Trade",
    "TradeResult",
    "adv_panel",
    "apply_costs",
    "forward_returns",
    "limit_buy_filled",
    "limit_sell_filled",
    "liquidity_size_multiplier",
    "lookup_panel",
    "panel_pivot",
    "performance_metrics",
    "rank_ic",
    "rank_tilt_backtest",
    "roundtrip_bp",
    "staggered_tranche_backtest",
    "sweep_vwap",
]
