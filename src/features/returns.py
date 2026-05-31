"""Return feature transformations."""

from __future__ import annotations

import polars as pl


def add_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add simple and log returns per ticker."""
    return lf.with_columns(
        pl.col("close").pct_change().over("ticker").alias("returns"),
        (pl.col("close").log() - pl.col("close").shift(1).log())
        .over("ticker")
        .alias("log_returns"),
    )
