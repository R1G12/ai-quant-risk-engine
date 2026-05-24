"""Optimization tests."""

import numpy as np

from src.risk.optimization.constraints import PortfolioConstraints
from src.risk.optimization.markowitz import max_sharpe_weights, min_variance_weights, portfolio_stats


def test_weights_sum_to_one() -> None:
    mean = np.array([0.10, 0.12])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    cons = PortfolioConstraints(long_only=True, max_weight=0.9)
    w = min_variance_weights(cov, cons, 2)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()


def test_max_sharpe_beats_equal_weight() -> None:
    mean = np.array([0.08, 0.15])
    cov = np.array([[0.02, 0.005], [0.005, 0.05]])
    cons = PortfolioConstraints(long_only=True, max_weight=1.0)
    w_opt, ok = max_sharpe_weights(mean, cov, cons)
    w_eq = np.array([0.5, 0.5])
    assert ok
    _, _, s_opt = portfolio_stats(w_opt, mean, cov)
    _, _, s_eq = portfolio_stats(w_eq, mean, cov)
    assert s_opt >= s_eq - 1e-6
