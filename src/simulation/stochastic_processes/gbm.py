"""Geometric Brownian Motion (NumPy boundary)."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    mu: float,
    sigma: float,
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: int,
    s0: float = 1.0,
) -> np.ndarray:
    """Simulate GBM wealth paths shape (n_paths, n_steps+1).

    dS/S = mu*dt + sigma*dW
    """
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths
