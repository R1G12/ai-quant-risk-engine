"""Tests for historical FX conversion on tracker P&L."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.portfolio.tracker.currency import (
    apply_display_currency_to_closed,
    open_lot_sgd,
    realized_pnl_sgd,
)
from src.portfolio.tracker.fx import FxRateTable
from src.portfolio.tracker.performance import actual_equity_curve
from src.portfolio.tracker.schema import validate_and_normalize


def _fx() -> FxRateTable:
    return FxRateTable.from_dict(
        "USDSGD=X",
        {
            date(2025, 1, 1): 1.30,
            date(2025, 1, 2): 1.32,
            date(2025, 2, 1): 1.35,
        },
    )


def _trades(rows: list[dict]) -> pl.DataFrame:
    return validate_and_normalize(pl.DataFrame(rows))


def test_realized_pnl_uses_per_trade_fx() -> None:
    lot = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "AAPL",
                "trade_date": "2025-01-01",
                "side": "long",
                "action": "buy",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
            {
                "trade_id": "2",
                "ticker": "AAPL",
                "trade_date": "2025-02-01",
                "side": "long",
                "action": "sell",
                "quantity": 10,
                "price": 110.0,
                "fees": 1.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    fx = _fx()
    # buy: -1000 * 1.30 = -1300; sell: 1099 * 1.35 = 1483.65
    assert realized_pnl_sgd(lot, fx) == pytest.approx(183.65)


def test_open_unrealized_uses_trade_and_mark_fx() -> None:
    lot = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "AAPL",
                "trade_date": "2025-01-01",
                "side": "long",
                "action": "buy",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    fx = _fx()
    mv, ur = open_lot_sgd(
        lot,
        quantity=10.0,
        side="long",
        mark=110.0,
        mark_date=date(2025, 1, 2),
        fx=fx,
    )
    assert mv == pytest.approx(10 * 110 * 1.32)
    assert ur == pytest.approx(mv - 10 * 100 * 1.30)


def test_apply_display_currency_to_closed() -> None:
    trades = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "AAPL",
                "trade_date": "2025-01-01",
                "side": "long",
                "action": "buy",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
            {
                "trade_id": "2",
                "ticker": "AAPL",
                "trade_date": "2025-02-01",
                "side": "long",
                "action": "sell",
                "quantity": 10,
                "price": 110.0,
                "fees": 1.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    closed = pl.DataFrame(
        {
            "lot_id": ["L1"],
            "ticker": ["AAPL"],
            "side": ["long"],
            "open_date": [date(2025, 1, 1)],
            "close_date": [date(2025, 2, 1)],
            "holding_days": [31],
            "realized_pnl": [99.0],
            "total_fees": [1.0],
            "rolled_from_lot_id": [None],
        }
    )
    out = apply_display_currency_to_closed(closed, trades, _fx())
    assert float(out["realized_pnl"][0]) == pytest.approx(183.65)


def test_actual_equity_curve_with_fx() -> None:
    trades = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "AAPL",
                "trade_date": "2025-01-01",
                "side": "long",
                "action": "buy",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    panel = pl.DataFrame(
        {
            "date": [date(2025, 1, 1), date(2025, 1, 2)],
            "ticker": ["AAPL", "AAPL"],
            "close": [100.0, 110.0],
        }
    )
    fx = _fx()
    curve, _ = actual_equity_curve(
        trades,
        panel,
        initial_nav=1000.0,
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        fx=fx,
    )
    # Day 1: cash 1000 - 1300 = -300, mtm 10*100*1.30 = 1300 → 1000
    assert float(curve["equity"][0]) == pytest.approx(1000.0)
    # Day 2: cash -300, mtm 10*110*1.32 = 1452 → 1152
    assert float(curve["equity"][1]) == pytest.approx(1152.0)
