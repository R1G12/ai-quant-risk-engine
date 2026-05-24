"""CVaR / Expected Shortfall."""

from __future__ import annotations

import polars as pl


def historical_cvar(returns: pl.Series, confidence: float) -> float:
    """ES = E[r | r <= VaR_alpha]."""
    r = returns.drop_nulls().sort()
    if r.len() == 0:
        return float("nan")
    alpha = 1.0 - confidence
    var = float(r.quantile(alpha))
    tail = r.filter(r <= var)
    return float(tail.mean()) if tail.len() else var


def parametric_cvar(returns: pl.Series, confidence: float) -> float:
    """Gaussian ES approximation using scipy."""
    from scipy import stats

    r = returns.drop_nulls()
    if r.len() == 0:
        return float("nan")
    mu = float(r.mean())
    sigma = float(r.std())
    z = stats.norm.ppf(1.0 - confidence)
    pdf = stats.norm.pdf(z)
    es = mu - sigma * pdf / (1.0 - confidence)
    return es
