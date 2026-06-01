"""Derive open and closed lots from trade events."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.portfolio.tracker.schema import cash_flow, signed_quantity_delta


def _lot_summary(trades: pl.DataFrame, lot_id: str) -> dict:
    lot = trades.filter(pl.col("lot_id") == lot_id).sort("trade_date", "trade_id")
    if lot.is_empty():
        raise ValueError(f"unknown lot_id: {lot_id}")

    first = lot.row(0, named=True)
    last = lot.row(-1, named=True)
    ticker = first["ticker"]
    side = first["side"]

    qty = 0.0
    total_fees = 0.0
    cash = 0.0
    avg_cost = 0.0
    for row in lot.iter_rows(named=True):
        delta = signed_quantity_delta(row["side"], row["action"], row["quantity"])
        trade_qty = row["quantity"]
        trade_price = row["price"]
        qty += delta
        cash += cash_flow(row["side"], row["action"], trade_qty, trade_price, row["fees"])
        total_fees += row["fees"]
        if abs(qty) < 1e-12:
            avg_cost = 0.0
        elif delta > 0:
            prev_qty = qty - delta
            if prev_qty > 1e-12:
                avg_cost = (avg_cost * prev_qty + trade_price * trade_qty) / qty
            else:
                avg_cost = trade_price
        # sells: avg_cost unchanged (average-cost method)

    rolled = lot.filter(pl.col("rolled_from_lot_id").is_not_null())
    rolled_from = rolled["rolled_from_lot_id"][0] if rolled.height else None
    open_date: date = first["trade_date"]
    close_date: date | None = last["trade_date"] if abs(qty) < 1e-9 else None

    return {
        "lot_id": lot_id,
        "ticker": ticker,
        "side": side,
        "quantity": qty,
        "avg_cost": avg_cost,
        "open_date": open_date,
        "close_date": close_date,
        "last_trade_date": last["trade_date"],
        "realized_pnl": cash if abs(qty) < 1e-9 else None,
        "total_fees": total_fees,
        "rolled_from_lot_id": rolled_from,
        "trade_count": lot.height,
    }


def build_open_positions(trades: pl.DataFrame) -> pl.DataFrame:
    """Lots with non-zero remaining quantity."""
    lot_ids = trades["lot_id"].unique().to_list()
    rows = []
    for lot_id in lot_ids:
        summary = _lot_summary(trades, lot_id)
        if abs(summary["quantity"]) < 1e-9:
            continue
        rows.append(
            {
                "lot_id": summary["lot_id"],
                "ticker": summary["ticker"],
                "side": summary["side"],
                "quantity": abs(summary["quantity"]),
                "avg_cost": summary["avg_cost"],
                "open_date": summary["open_date"],
                "last_trade_date": summary["last_trade_date"],
                "rolled_from_lot_id": summary["rolled_from_lot_id"],
                "market_value": None,
                "unrealized_pnl": None,
            }
        )
    if not rows:
        return pl.DataFrame(
            {
                "lot_id": [],
                "ticker": [],
                "side": [],
                "quantity": [],
                "avg_cost": [],
                "open_date": [],
                "last_trade_date": [],
                "rolled_from_lot_id": [],
                "market_value": [],
                "unrealized_pnl": [],
            }
        )
    return pl.DataFrame(rows).sort("ticker", "lot_id")


def build_closed_positions(trades: pl.DataFrame) -> pl.DataFrame:
    """Fully closed lots with realized PnL."""
    lot_ids = trades["lot_id"].unique().to_list()
    rows = []
    for lot_id in lot_ids:
        summary = _lot_summary(trades, lot_id)
        if abs(summary["quantity"]) >= 1e-9:
            continue
        open_d: date = summary["open_date"]
        close_d: date = summary["close_date"] or summary["last_trade_date"]
        holding_days = (close_d - open_d).days
        rows.append(
            {
                "lot_id": summary["lot_id"],
                "ticker": summary["ticker"],
                "side": summary["side"],
                "open_date": open_d,
                "close_date": close_d,
                "holding_days": holding_days,
                "realized_pnl": summary["realized_pnl"],
                "total_fees": summary["total_fees"],
                "rolled_from_lot_id": summary["rolled_from_lot_id"],
            }
        )
    if not rows:
        return pl.DataFrame(
            {
                "lot_id": [],
                "ticker": [],
                "side": [],
                "open_date": [],
                "close_date": [],
                "holding_days": [],
                "realized_pnl": [],
                "total_fees": [],
                "rolled_from_lot_id": [],
            }
        )
    return pl.DataFrame(rows).sort("close_date", "ticker")
