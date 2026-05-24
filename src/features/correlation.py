"""Rolling correlation features (stretch)."""

from __future__ import annotations

import polars as pl

from src.utils.config import FeatureConfig


def add_rolling_correlation_to_index(
    lf: pl.LazyFrame,
    cfg: FeatureConfig,
    benchmark_ticker: str = "AAPL",
) -> pl.LazyFrame:
    """Rolling correlation of returns to a benchmark ticker."""
    window = cfg.correlation_window
    bench = (
        lf.filter(pl.col("ticker") == benchmark_ticker)
        .select("timestamp", pl.col("returns").alias("bench_returns"))
    )
    return (
        lf.join(bench, on="timestamp", how="left")
        .with_columns(
            pl.rolling_corr(
                pl.col("returns"),
                pl.col("bench_returns"),
                window_size=window,
            )
            .over("ticker")
            .alias(f"rolling_corr_{benchmark_ticker}"),
        )
        .drop("bench_returns")
    )
