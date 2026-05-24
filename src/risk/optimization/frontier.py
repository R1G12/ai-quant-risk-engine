"""Efficient frontier generation."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize

from src.risk.optimization.constraints import PortfolioConstraints
from src.risk.optimization.markowitz import portfolio_stats


def efficient_frontier(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    constraints: PortfolioConstraints,
    n_points: int = 25,
) -> pl.DataFrame:
    """Trace efficient frontier via target-return optimization."""
    n = len(mean_returns)
    rets = []
    vols = []
    weights_list = []

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)

    for target in target_returns:
        x0 = np.ones(n) / n

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        cons = [
            constraints.budget_constraint(),
            {"type": "eq", "fun": lambda w, t=target: w @ mean_returns - t},
        ]
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=constraints.bounds(n),
            constraints=cons,
        )
        w = res.x if res.success else x0
        r, v, s = portfolio_stats(w, mean_returns, cov)
        rets.append(r)
        vols.append(v)
        weights_list.append(w.tolist())

    return pl.DataFrame(
        {
            "expected_return": rets,
            "volatility": vols,
            "weights": weights_list,
        }
    )
