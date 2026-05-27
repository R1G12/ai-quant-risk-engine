"""Shared long-to-wide returns loading for pipeline stages."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.paths import RISK_DATASET_PATH


def returns_long_to_wide(long: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    """Pivot ticker returns to wide format with timestamp deduplication."""
    if long.is_empty():
        return pl.DataFrame({"timestamp": []})

    long = long.group_by(["timestamp", "ticker"]).agg(pl.col("returns").mean())
    wide = long.pivot(on="ticker", index="timestamp", values="returns").sort("timestamp")
    ticker_cols = [t for t in tickers if t in wide.columns]
    if not ticker_cols:
        return wide

    wide = (
        wide.group_by("timestamp")
        .agg([pl.col(t).mean().alias(t) for t in ticker_cols])
        .sort("timestamp")
    )
    return wide.filter(pl.any_horizontal([pl.col(t).is_not_null() for t in ticker_cols]))


def load_returns_long(tickers: list[str], path: Path = RISK_DATASET_PATH) -> pl.DataFrame:
    """Scan risk dataset and collect long returns for the given tickers."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("ticker").is_in(tickers))
        .filter(pl.col("returns").is_not_null())
        .select("timestamp", "ticker", "returns")
        .collect()
    )


def load_returns_wide(
    tickers: list[str],
    path: Path = RISK_DATASET_PATH,
) -> pl.DataFrame:
    """Load returns from risk dataset as a wide, deduplicated DataFrame."""
    long = load_returns_long(tickers, path)
    return returns_long_to_wide(long, tickers)
