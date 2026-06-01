"""Tests for open/closed position derivation."""

from __future__ import annotations

import polars as pl
import pytest

from src.portfolio.tracker.positions import build_closed_positions, build_open_positions
from src.portfolio.tracker.schema import validate_and_normalize


def _trades(rows: list[dict]) -> pl.DataFrame:
    return validate_and_normalize(pl.DataFrame(rows))


def test_open_and_closed_long() -> None:
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
            {
                "trade_id": "3",
                "ticker": "MSFT",
                "trade_date": "2025-03-01",
                "side": "long",
                "action": "buy",
                "quantity": 5,
                "price": 200.0,
                "fees": 0.0,
                "lot_id": "L2",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    open_pos = build_open_positions(trades)
    closed = build_closed_positions(trades)
    assert open_pos.height == 1
    assert open_pos["ticker"][0] == "MSFT"
    assert closed.height == 1
    assert float(closed["realized_pnl"][0]) == pytest.approx(100.0 - 1.0)


def test_partial_open_lot() -> None:
    trades = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "NVDA",
                "trade_date": "2025-01-01",
                "side": "long",
                "action": "buy",
                "quantity": 10,
                "price": 50.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
            {
                "trade_id": "2",
                "ticker": "NVDA",
                "trade_date": "2025-02-01",
                "side": "long",
                "action": "sell",
                "quantity": 4,
                "price": 60.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    open_pos = build_open_positions(trades)
    assert open_pos.height == 1
    assert float(open_pos["quantity"][0]) == pytest.approx(6.0)
    assert float(open_pos["avg_cost"][0]) == pytest.approx(50.0)


def test_short_round_trip() -> None:
    trades = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "MSFT",
                "trade_date": "2025-01-01",
                "side": "short",
                "action": "sell",
                "quantity": 10,
                "price": 400.0,
                "fees": 0.0,
                "lot_id": "S1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
            {
                "trade_id": "2",
                "ticker": "MSFT",
                "trade_date": "2025-02-01",
                "side": "short",
                "action": "buy",
                "quantity": 10,
                "price": 380.0,
                "fees": 0.0,
                "lot_id": "S1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    closed = build_closed_positions(trades)
    assert closed.height == 1
    assert float(closed["realized_pnl"][0]) == pytest.approx(200.0)
