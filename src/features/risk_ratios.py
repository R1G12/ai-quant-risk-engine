"""Risk ratio features (Sharpe, etc.)."""

from __future__ import annotations

import polars as pl

from src.utils.config import FeatureConfig


def add_rolling_sharpe(lf: pl.LazyFrame, cfg: FeatureConfig) -> pl.LazyFrame:
    """Rolling Sharpe ratio using config risk-free rate (annualized)."""
    window = cfg.sharpe_window
    rf_daily = cfg.risk_free_rate / cfg.annualization_factor
    excess = pl.col("returns") - rf_daily
    return lf.with_columns(
        (
            excess.rolling_mean(window_size=window).over("ticker")
            / pl.col("returns").rolling_std(window_size=window).over("ticker")
        ).alias("rolling_sharpe"),
    )
