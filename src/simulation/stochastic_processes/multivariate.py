"""Correlated multivariate GBM via Cholesky."""

from __future__ import annotations

import numpy as np


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    """Extract correlation matrix from covariance."""
    std = np.sqrt(np.diag(cov))
    std[std < 1e-12] = 1e-12
    return cov / np.outer(std, std)


def simulate_multivariate_gbm(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """Asset price paths shape (n_paths, n_assets, n_steps+1)."""
    n_assets = len(mu)
    corr = correlation_from_cov(cov)
    chol = np.linalg.cholesky(corr)
    std = np.sqrt(np.diag(cov))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps, n_assets))
    corr_shocks = z @ chol.T
    paths = np.ones((n_paths, n_assets, n_steps + 1))
    for t in range(n_steps):
        log_r = (mu - 0.5 * std**2) * dt + std * np.sqrt(dt) * corr_shocks[:, t, :]
        paths[:, :, t + 1] = paths[:, :, t] * np.exp(log_r)
    return paths
