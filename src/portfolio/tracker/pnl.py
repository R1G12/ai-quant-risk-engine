"""Mark-to-market and PnL helpers for open positions."""

from __future__ import annotations

import polars as pl


def attach_unrealized_pnl(
    open_positions: pl.DataFrame,
    mark_prices: dict[str, float],
) -> pl.DataFrame:
    """Add market_value and unrealized_pnl from latest marks per ticker."""
    if open_positions.is_empty():
        return open_positions

    rows = []
    for row in open_positions.iter_rows(named=True):
        ticker = row["ticker"]
        mark = mark_prices.get(ticker)
        qty = float(row["quantity"])
        avg = float(row["avg_cost"])
        side = row["side"]
        if mark is None:
            rows.append({**row, "market_value": None, "unrealized_pnl": None})
            continue
        market_value = qty * mark
        if side == "long":
            unrealized = qty * (mark - avg)
        else:
            unrealized = qty * (avg - mark)
        rows.append({**row, "market_value": market_value, "unrealized_pnl": unrealized})

    return pl.DataFrame(rows)


def exposure_by_ticker(
    open_positions: pl.DataFrame,
    mark_prices: dict[str, float],
) -> pl.DataFrame:
    """Signed notional exposure per ticker for vs-model comparison."""
    if open_positions.is_empty():
        return pl.DataFrame({"ticker": [], "notional": [], "weight": []})

    rows = []
    for row in open_positions.iter_rows(named=True):
        ticker = row["ticker"]
        mark = mark_prices.get(ticker)
        if mark is None:
            continue
        qty = float(row["quantity"])
        sign = 1.0 if row["side"] == "long" else -1.0
        rows.append({"ticker": ticker, "notional": sign * qty * mark})

    if not rows:
        return pl.DataFrame({"ticker": [], "notional": [], "weight": []})

    df = pl.DataFrame(rows).group_by("ticker").agg(pl.col("notional").sum().alias("notional"))
    gross = float(df["notional"].abs().sum())
    if gross < 1e-12:
        return df.with_columns(pl.lit(0.0).alias("weight"))
    return df.with_columns((pl.col("notional").abs() / gross).alias("weight"))
