"""``PositionBook`` — 평단·실현손익(비용 차감)을 메모리 + journal 로 추적한다.

``OrderManager``(Task 6)가 체결마다 :meth:`PositionBook.apply` 를 호출해 장부를
갱신한다. 크래시 뒤 재시작해도 상태가 사라지지 않도록, ``journal`` 을 주면 모든
fill 을 JSONL 로 append(``fsync``)하고 :meth:`PositionBook.load` 로 재생해 복원한다.

비용 모델은 기본으로 **슬리피지 0** 인 :class:`KoreanCostModel` 을 쓴다 — 여기 오는
``price`` 는 이미 실제 체결가라 슬리피지를 또 반영하면 이중 계산이 된다(비용
모델 원래 용도는 백테스트에서 아직 안 일어난 체결가를 추정하는 것).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from krx_quant_core.costs.model import CostModelConfig, KoreanCostModel

from .events import Fill, Holding

__all__ = ["PositionBook"]

_TRADE_COLUMNS = ["code", "entry_ts", "exit_ts", "qty", "entry_price", "exit_price", "pnl"]


@dataclass
class _Position:
    """보유 1종목의 내부 상태.

    평단은 ``Decimal`` 로 들고 있다가 :class:`Holding` 으로 낼 때만 float 로 바꾼다.
    """

    qty: int
    avg_price: Decimal
    entry_ts: datetime


def _zero_slippage_cost_model() -> KoreanCostModel:
    """기본 비용 모델 — 슬리피지 0(체결가가 이미 실제 가격), 세율·수수료는 기본값."""
    return KoreanCostModel(
        CostModelConfig(kospi_slippage_pct=Decimal("0"), kosdaq_slippage_pct=Decimal("0"))
    )


class PositionBook:
    """체결로 갱신되는 포지션 장부. 매수는 평단 가중평균, 매도는 비용 차감 실현손익."""

    def __init__(
        self,
        *,
        journal: Path | str | None = None,
        market_of: Callable[[str], str] = lambda c: "KOSPI",
        cost_model: KoreanCostModel | None = None,
    ) -> None:
        self._journal_path = Path(journal) if journal is not None else None
        self._market_of = market_of
        self._cost_model = cost_model or _zero_slippage_cost_model()
        self._positions: dict[str, _Position] = {}
        self._trades: list[dict[str, Any]] = []
        self.realized_krw: float = 0.0
        self.n_round_trips: int = 0

    def apply(self, fill: Fill) -> float:
        """체결 1건을 반영한다. 이 체결로 실현된 순손익(원)을 돌려준다(매수는 0.0)."""
        realized = self._apply_no_journal(fill)
        if self._journal_path is not None:
            self._write_journal(fill)
        return realized

    def _apply_no_journal(self, fill: Fill) -> float:
        if fill.side == "buy":
            return self._apply_buy(fill)
        if fill.side == "sell":
            return self._apply_sell(fill)
        raise ValueError(f"알 수 없는 매매구분: {fill.side!r}")

    def _apply_buy(self, fill: Fill) -> float:
        pos = self._positions.get(fill.code)
        price = Decimal(fill.price)
        if pos is None or pos.qty == 0:
            self._positions[fill.code] = _Position(
                qty=fill.qty, avg_price=price, entry_ts=fill.ts
            )
        else:
            new_qty = pos.qty + fill.qty
            new_avg = (pos.avg_price * pos.qty + price * fill.qty) / new_qty
            self._positions[fill.code] = _Position(
                qty=new_qty, avg_price=new_avg, entry_ts=pos.entry_ts
            )
        return 0.0

    def _apply_sell(self, fill: Fill) -> float:
        pos = self._positions.get(fill.code)
        held = pos.qty if pos is not None else 0
        if fill.qty > held:
            raise ValueError(
                f"보유 초과 매도: {fill.code} 매도수량={fill.qty} 보유수량={held}"
            )
        assert pos is not None  # held > 0 이면 pos 가 있어야 한다(held 는 pos.qty 에서 옴)

        market = self._market_of(fill.code)
        trade_date = fill.ts.date()
        avg = pos.avg_price
        qty = Decimal(fill.qty)
        sell_price = Decimal(fill.price)

        buy_cost = self._cost_model.cost_of_trade(avg, qty, "BUY", market, trade_date)
        sell_cost = self._cost_model.cost_of_trade(sell_price, qty, "SELL", market, trade_date)
        gross = (sell_price - avg) * qty
        pnl = gross - buy_cost.commission - sell_cost.commission - sell_cost.tax
        pnl_f = float(pnl)

        self._trades.append(
            {
                "code": fill.code,
                "entry_ts": pos.entry_ts,
                "exit_ts": fill.ts,
                "qty": fill.qty,
                "entry_price": float(avg),
                "exit_price": fill.price,
                "pnl": pnl_f,
            }
        )
        self.realized_krw += pnl_f
        self.n_round_trips += 1

        remaining = pos.qty - fill.qty
        if remaining == 0:
            del self._positions[fill.code]
        else:
            self._positions[fill.code] = replace(pos, qty=remaining)

        return pnl_f

    def position(self, code: str) -> Holding | None:
        """``code`` 의 현재 보유. 없으면 ``None``."""
        pos = self._positions.get(code)
        if pos is None or pos.qty == 0:
            return None
        return Holding(code=code, qty=pos.qty, avg_price=float(pos.avg_price))

    def positions(self) -> dict[str, Holding]:
        """수량이 0 초과인 보유 전체."""
        return {
            code: Holding(code=code, qty=pos.qty, avg_price=float(pos.avg_price))
            for code, pos in self._positions.items()
            if pos.qty > 0
        }

    def to_frame(self) -> pd.DataFrame:
        """청산 원장: 매도 fill 하나당 한 행."""
        return pd.DataFrame(self._trades, columns=_TRADE_COLUMNS)

    def _write_journal(self, fill: Fill) -> None:
        row = {
            "ord_no": fill.ord_no,
            "code": fill.code,
            "side": fill.side,
            "qty": fill.qty,
            "price": fill.price,
            "ts": fill.ts.isoformat(),
        }
        assert self._journal_path is not None
        with open(self._journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @classmethod
    def load(cls, journal: Path | str, **kw: Any) -> PositionBook:
        """journal 을 재생해 상태를 복원한다(재기록 안 함). 이후 ``apply`` 는 이어서 append."""
        path = Path(journal)
        book = cls(journal=None, **kw)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    fill = Fill(
                        ord_no=row["ord_no"],
                        code=row["code"],
                        side=row["side"],
                        qty=int(row["qty"]),
                        price=int(row["price"]),
                        ts=datetime.fromisoformat(row["ts"]),
                    )
                    book._apply_no_journal(fill)
        book._journal_path = path
        return book
