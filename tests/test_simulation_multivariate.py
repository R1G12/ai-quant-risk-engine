"""Multivariate GBM tests."""

import numpy as np

from src.simulation.stochastic_processes.multivariate import correlation_from_cov, simulate_multivariate_gbm


def test_correlation_from_cov_diagonal() -> None:
    cov = np.diag([0.04, 0.09])
    corr = correlation_from_cov(cov)
    assert np.allclose(np.diag(corr), 1.0)


def test_multivariate_paths_shape() -> None:
    mu = np.array([0.05, 0.06])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    paths = simulate_multivariate_gbm(mu, cov, n_paths=80, n_steps=15, dt=1 / 252, seed=3)
    assert paths.shape == (80, 2, 16)


def test_correlated_assets_differ() -> None:
    mu = np.zeros(2)
    cov = np.array([[0.04, 0.035], [0.035, 0.04]])
    paths = simulate_multivariate_gbm(mu, cov, n_paths=500, n_steps=30, dt=1 / 252, seed=11)
    r0 = paths[:, 0, -1] / paths[:, 0, 0] - 1
    r1 = paths[:, 1, -1] / paths[:, 1, 0] - 1
    assert np.corrcoef(r0, r1)[0, 1] > 0.5
