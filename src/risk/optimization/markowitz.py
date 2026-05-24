"""Markowitz mean-variance optimization (scipy boundary)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.risk.optimization.constraints import PortfolioConstraints


def portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov: np.ndarray,
) -> tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe) for weights."""
    ret = float(weights @ mean_returns)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = ret / vol if vol > 1e-12 else 0.0
    return ret, vol, sharpe


def min_variance_weights(
    cov: np.ndarray,
    constraints: PortfolioConstraints,
    n_assets: int,
) -> np.ndarray:
    """Minimum variance portfolio."""
    x0 = np.ones(n_assets) / n_assets

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    cons = [constraints.budget_constraint()]
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=constraints.bounds(n_assets),
        constraints=cons,
    )
    return res.x if res.success else x0


def max_sharpe_weights(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    constraints: PortfolioConstraints,
    risk_free: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Maximum Sharpe ratio portfolio."""
    n = len(mean_returns)
    x0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        ret, vol, _ = portfolio_stats(w, mean_returns, cov)
        return -(ret - risk_free) / vol if vol > 1e-12 else 0.0

    cons = [constraints.budget_constraint()]
    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=constraints.bounds(n),
        constraints=cons,
    )
    return (res.x if res.success else x0, bool(res.success))
