"""Trade ledger column schema and validation."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

import polars as pl

TRADE_COLUMNS = [
    "trade_id",
    "ticker",
    "trade_date",
    "side",
    "action",
    "quantity",
    "price",
    "fees",
    "lot_id",
    "rolled_from_lot_id",
    "notes",
]

VALID_SIDES = frozenset({"long", "short"})
VALID_ACTIONS = frozenset({"buy", "sell"})


def _normalize_ticker(value: Any) -> str:
    return str(value).strip().upper()


def _parse_trade_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    return date.fromisoformat(text)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return None
    return text


def signed_quantity_delta(side: str, action: str, quantity: float) -> float:
    """Signed share delta for a lot (positive = more long exposure / more short shares)."""
    side_l = side.lower()
    action_l = action.lower()
    if side_l == "long":
        return quantity if action_l == "buy" else -quantity
    if side_l == "short":
        return quantity if action_l == "sell" else -quantity
    raise ValueError(f"invalid side: {side}")


def economic_position_delta(side: str, action: str, quantity: float) -> float:
    """Signed position delta for NAV MTM (long +, short -)."""
    side_l = side.lower()
    action_l = action.lower()
    if side_l == "long":
        return quantity if action_l == "buy" else -quantity
    if side_l == "short":
        return -quantity if action_l == "sell" else quantity
    raise ValueError(f"invalid side: {side}")


def cash_flow(side: str, action: str, quantity: float, price: float, fees: float) -> float:
    """Cash received (+) or paid (-) for one trade."""
    side_l = side.lower()
    action_l = action.lower()
    notional = quantity * price
    if side_l == "long":
        if action_l == "buy":
            return -notional - fees
        return notional - fees
    if side_l == "short":
        if action_l == "sell":
            return notional - fees
        return -notional - fees
    raise ValueError(f"invalid side: {side}")


def validate_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Cast, normalize, and validate trade rows."""
    missing = [c for c in TRADE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in trades sheet: {missing}")

    rows: list[dict[str, Any]] = []
    for row in df.select(TRADE_COLUMNS).iter_rows(named=True):
        ticker = _normalize_ticker(row["ticker"])
        if not ticker:
            raise ValueError("ticker is required")

        side = str(row["side"]).strip().lower()
        action = str(row["action"]).strip().lower()
        if side not in VALID_SIDES:
            raise ValueError(f"invalid side {side!r} for {ticker}")
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid action {action!r} for {ticker}")

        quantity = float(row["quantity"])
        price = float(row["price"])
        fees = float(row["fees"]) if row["fees"] is not None else 0.0
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0 for trade {row.get('trade_id')}")
        if price <= 0:
            raise ValueError(f"price must be > 0 for trade {row.get('trade_id')}")

        lot_id = _optional_str(row["lot_id"])
        if not lot_id:
            raise ValueError(f"lot_id is required for {ticker}")

        trade_id = _optional_str(row["trade_id"]) or str(uuid.uuid4())
        rolled = _optional_str(row["rolled_from_lot_id"])
        notes = _optional_str(row["notes"])

        rows.append(
            {
                "trade_id": trade_id,
                "ticker": ticker,
                "trade_date": _parse_trade_date(row["trade_date"]),
                "side": side,
                "action": action,
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "lot_id": lot_id,
                "rolled_from_lot_id": rolled,
                "notes": notes,
            }
        )

    out = pl.DataFrame(rows).with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("quantity").cast(pl.Float64),
        pl.col("price").cast(pl.Float64),
        pl.col("fees").cast(pl.Float64),
    )
    return out.sort("trade_date", "trade_id")
