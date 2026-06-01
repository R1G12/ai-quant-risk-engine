"""Portfolio exposure metrics."""

from __future__ import annotations

import polars as pl

from src.risk.portfolio.holdings import load_portfolio_weights, portfolio_weighting_mode


def exposure_table(weights: dict[str, float], *, weighting: str | None = None) -> pl.DataFrame:
    """Static exposure snapshot from weights."""
    df = pl.DataFrame(
        {
            "asset": list(weights.keys()),
            "weight": list(weights.values()),
            "exposure": list(weights.values()),
        }
    )
    if weighting is not None:
        df = df.with_columns(pl.lit(weighting).alias("weighting_mode"))
    return df


def load_exposure_table(app) -> pl.DataFrame:
    """Load exposures using run-profile weighting (equal, manual, partial, optimised)."""
    return exposure_table(
        load_portfolio_weights(app),
        weighting=portfolio_weighting_mode(app),
    )
