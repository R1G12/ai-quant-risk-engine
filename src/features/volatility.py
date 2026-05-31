"""Volatility feature transformations."""

from __future__ import annotations

import polars as pl

from src.utils.config import FeatureConfig


def add_volatility_features(lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    """Add rolling volatility and annualized volatility per ticker."""
    window = cfg.volatility_window
    ann = cfg.annualization_factor ** 0.5
    return lf.with_columns(
        pl.col("returns")
        .rolling_std(window_size=window)
        .over("ticker")
        .alias("volatility"),
        (
            pl.col("returns").rolling_std(window_size=window).over("ticker") * ann
        ).alias("annualized_volatility"),
    )
