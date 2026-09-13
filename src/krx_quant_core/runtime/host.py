"""무거운 작업의 호스트 배치 규칙.

백테스트는 simnode 에서만 돈다(사용자 지시, 2026-09-13). trader 호스트는 실시간
수집·주문 데몬이 도는 곳이라 CPU 를 뺏기면 실매매가 늦어지고, simnode 에 CPU 와
primary DB 가 있다. 엔진 내부가 아니라 **진입점**(CLI·MCP 도구)에서 막아야 엔진
자체는 어디서든 테스트할 수 있다.

daytrade-it ``shared/runtime_host.py`` 와 scalp-it ``cli_closebet._require_simnode_host``
두 벌을 여기로 모았다.

테스트에서 호스트명을 바꿀 때는 ``socket.gethostname`` 을 monkeypatch 한다 — 이 모듈은
``socket`` 모듈 속성으로 조회하므로 ``krx_quant_core.runtime.host.socket.gethostname``
어느 경로로 패치해도 같다.
"""

from __future__ import annotations

import socket

__all__ = ["BACKTEST_HOST", "require_backtest_host", "require_host"]

BACKTEST_HOST = "simnode"


def require_host(expected: str, *, what: str = "this command", hint: str = "") -> None:
    """현재 호스트명이 ``expected`` 가 아니면 ``RuntimeError``.

    메시지에 기대 호스트와 현재 호스트를 둘 다 적는다 — 잘못된 곳에서 돌렸을 때
    어디로 가야 하는지 바로 보이게.
    """
    host = socket.gethostname()
    if host != expected:
        message = f"{what} runs only on {expected!r} (current host: {host!r})."
        if hint:
            message = f"{message} {hint}"
        raise RuntimeError(message)


def require_backtest_host() -> None:
    """simnode 가 아니면 ``RuntimeError`` (daytrade-it 과 같은 메시지)."""
    host = socket.gethostname()
    if host != BACKTEST_HOST:
        raise RuntimeError(
            f"Backtests run only on {BACKTEST_HOST!r} (current host: {host!r}). "
            "ssh simnode-local and run it there."
        )
