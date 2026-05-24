"""Liquidity proxy features."""

from __future__ import annotations

import polars as pl


def add_liquidity_features(lf: pl.LazyFrame, window: int = 21) -> pl.LazyFrame:
    """Volume z-score as a simple liquidity proxy."""
    return lf.with_columns(
        (
            (pl.col("volume") - pl.col("volume").rolling_mean(window).over("ticker"))
            / pl.col("volume").rolling_std(window).over("ticker")
        ).alias("volume_zscore"),
    )
