"""Technical indicator features."""

from __future__ import annotations

import polars as pl

from src.utils.config import FeatureConfig


def add_technical_features(lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    """SMA, momentum, and drawdown features."""
    exprs: list[pl.Expr] = []
    for w in cfg.sma_windows:
        exprs.append(
            pl.col("close").rolling_mean(window_size=w).over("ticker").alias(f"sma_{w}")
        )
    exprs.extend(
        [
            (
                pl.col("close") / pl.col("close").shift(cfg.momentum_window).over("ticker")
                - 1.0
            ).alias("momentum"),
            (
                pl.col("close")
                / pl.col("close").cum_max().over("ticker")
                - 1.0
            ).alias("drawdown"),
        ]
    )
    return lf.with_columns(exprs)
