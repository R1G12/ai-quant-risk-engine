"""Historical VaR."""

from __future__ import annotations

import polars as pl


def historical_var(returns: pl.Series, confidence: float) -> float:
    """VaR as empirical quantile at level (1 - confidence) for loss = negative return."""
    r = returns.drop_nulls()
    if r.len() == 0:
        return float("nan")
    alpha = 1.0 - confidence
    return float(r.quantile(alpha))
