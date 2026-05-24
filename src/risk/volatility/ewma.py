"""EWMA volatility."""

from __future__ import annotations

import polars as pl

from src.utils.config import VolatilityRiskConfig


def add_ewma_volatility(
    lf: pl.LazyFrame,
    cfg: VolatilityRiskConfig,
    *,
    return_col: str = "portfolio_return",
) -> pl.LazyFrame:
    """EWMA variance: var_t = lambda*var_{t-1} + (1-lambda)*r_t^2; lambda = 1 - 2/(span+1)."""
    lam = 1.0 - 2.0 / (cfg.ewma_span + 1.0)
    ann = cfg.annualization_factor**0.5
    return (
        lf.with_columns(pl.col(return_col).pow(2).ewm_mean(alpha=1.0 - lam).alias("ewma_var"))
        .with_columns(pl.col("ewma_var").sqrt().alias("ewma_vol"))
        .with_columns((pl.col("ewma_vol") * ann).alias("ewma_vol_ann"))
    )
