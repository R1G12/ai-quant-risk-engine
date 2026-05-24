"""Parameter sensitivity checks."""

from __future__ import annotations

import numpy as np

from src.simulation.stochastic_processes.gbm import simulate_gbm_paths


def terminal_volatility(mu: float, sigma: float, *, seed: int, n_paths: int = 2000) -> float:
    """Std of terminal simple returns for sensitivity comparison."""
    paths = simulate_gbm_paths(mu, sigma, n_paths=n_paths, n_steps=63, dt=1.0 / 252.0, seed=seed)
    rets = paths[:, -1] / paths[:, 0] - 1.0
    return float(np.std(rets))


def assert_sigma_sensitivity(mu: float, sigma: float, seed: int, rtol: float = 0.15) -> None:
    """Higher sigma should produce higher terminal return dispersion."""
    base = terminal_volatility(mu, sigma, seed=seed)
    bumped = terminal_volatility(mu, sigma * 1.1, seed=seed)
    if bumped <= base * (1.0 - rtol):
        raise ValueError(f"Sigma sensitivity failed: {bumped} vs {base}")
