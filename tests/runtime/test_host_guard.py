"""호스트 가드 — daytrade-it tests/unit/shared/test_runtime_host.py 이식."""

from __future__ import annotations

import pytest

from krx_quant_core.runtime import BACKTEST_HOST, require_backtest_host, require_host


def _host(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr("krx_quant_core.runtime.host.socket.gethostname", lambda: name)


def test_simnode_is_allowed(monkeypatch):
    _host(monkeypatch, "simnode")
    require_backtest_host()


def test_trader_is_refused(monkeypatch):
    _host(monkeypatch, "trader")
    with pytest.raises(RuntimeError, match="simnode"):
        require_backtest_host()


def test_backtest_message_is_daytrade_compatible(monkeypatch):
    _host(monkeypatch, "trader")
    with pytest.raises(RuntimeError) as exc:
        require_backtest_host()
    assert str(exc.value) == (
        "Backtests run only on 'simnode' (current host: 'trader'). "
        "ssh simnode-local and run it there."
    )


def test_patching_socket_globally_also_works(monkeypatch):
    """소비 레포가 ``socket.gethostname`` 을 어느 경로로 패치해도 가드가 본다."""
    monkeypatch.setattr("socket.gethostname", lambda: "trader")
    with pytest.raises(RuntimeError):
        require_backtest_host()


def test_backtest_host_constant():
    assert BACKTEST_HOST == "simnode"


def test_require_host_generic(monkeypatch):
    _host(monkeypatch, "trader")
    require_host("trader")
    with pytest.raises(RuntimeError) as exc:
        require_host("simnode", what="scalp-closebet", hint="simnode-local 로 ssh 해서 실행하라.")
    msg = str(exc.value)
    assert "scalp-closebet" in msg and "'simnode'" in msg and "'trader'" in msg
    assert msg.endswith("simnode-local 로 ssh 해서 실행하라.")


def test_require_host_default_what(monkeypatch):
    _host(monkeypatch, "trader")
    with pytest.raises(RuntimeError, match="this command runs only on 'simnode'"):
        require_host("simnode")
