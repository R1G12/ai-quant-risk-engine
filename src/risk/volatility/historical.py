"""Rolling historical volatility."""

from __future__ import annotations

import polars as pl

from src.utils.config import VolatilityRiskConfig


def add_rolling_volatility(
    lf: pl.LazyFrame,
    cfg: VolatilityRiskConfig,
    *,
    return_col: str = "portfolio_return",
) -> pl.LazyFrame:
    """Rolling std of returns, annualized: sigma * sqrt(252)."""
    ann = cfg.annualization_factor**0.5
    return lf.with_columns(
        pl.col(return_col)
        .rolling_std(window_size=cfg.rolling_vol_window)
        .alias("rolling_vol")
    ).with_columns((pl.col("rolling_vol") * ann).alias("rolling_vol_ann"))
