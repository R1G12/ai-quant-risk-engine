"""Simple volatility regime flags."""

from __future__ import annotations

import polars as pl

from src.utils.config import VolatilityRiskConfig


def add_vol_regime_flags(
    lf: pl.LazyFrame,
    cfg: VolatilityRiskConfig,
    *,
    vol_col: str = "ewma_vol",
) -> pl.LazyFrame:
    """High-vol regime when EWMA vol z-score vs rolling median exceeds threshold."""
    return (
        lf.with_columns(
            pl.col(vol_col).rolling_median(window_size=cfg.long_vol_window).alias("vol_median"),
            pl.col(vol_col).rolling_std(window_size=cfg.long_vol_window).alias("vol_disp"),
        )
        .with_columns(
            ((pl.col(vol_col) - pl.col("vol_median")) / pl.col("vol_disp").clip(lower_bound=1e-8))
            .alias("vol_zscore")
        )
        .with_columns(
            (pl.col("vol_zscore") > cfg.regime_zscore_threshold)
            .cast(pl.Int8)
            .alias("vol_regime_flag")
        )
    )
