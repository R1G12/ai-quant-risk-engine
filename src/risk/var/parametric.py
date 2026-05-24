"""Parametric (Gaussian) VaR."""

from __future__ import annotations

import polars as pl
from scipy import stats


def parametric_var(returns: pl.Series, confidence: float) -> float:
    """VaR = mu + z_alpha * sigma (loss convention: negative tail)."""
    r = returns.drop_nulls()
    if r.len() == 0:
        return float("nan")
    mu = float(r.mean())
    sigma = float(r.std())
    z = stats.norm.ppf(1.0 - confidence)
    return mu + z * sigma
