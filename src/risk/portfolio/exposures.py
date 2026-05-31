"""Portfolio exposure metrics."""

from __future__ import annotations

import polars as pl

from src.risk.portfolio.holdings import load_weights


def exposure_table(weights: dict[str, float]) -> pl.DataFrame:
    """Static exposure snapshot from weights."""
    return pl.DataFrame(
        {
            "asset": list(weights.keys()),
            "weight": list(weights.values()),
            "exposure": list(weights.values()),
        }
    )


def load_exposure_table(app) -> pl.DataFrame:
    """Load exposures from config holdings."""
    return exposure_table(load_weights(app))
