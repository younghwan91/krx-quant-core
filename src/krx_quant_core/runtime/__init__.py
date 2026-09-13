"""실행 환경 가드 — 어느 호스트에서 무엇을 돌려도 되는가."""

from krx_quant_core.runtime.host import BACKTEST_HOST, require_backtest_host, require_host

__all__ = ["BACKTEST_HOST", "require_backtest_host", "require_host"]
