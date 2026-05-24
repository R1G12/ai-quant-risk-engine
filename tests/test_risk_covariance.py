"""Covariance tests."""

import numpy as np
import polars as pl

from src.risk.correlations.covariance import ledoit_wolf_shrinkage, sample_covariance_matrix


def test_covariance_symmetric() -> None:
    wide = pl.DataFrame(
        {
            "timestamp": range(50),
            "A": np.random.default_rng(0).normal(0, 0.02, 50),
            "B": np.random.default_rng(1).normal(0, 0.03, 50),
        }
    )
    cov = sample_covariance_matrix(wide, ["A", "B"])
    assert np.allclose(cov, cov.T)


def test_shrinkage_reduces_off_diagonal_spread() -> None:
    cov = np.array([[0.04, 0.035], [0.035, 0.09]])
    shrunk = ledoit_wolf_shrinkage(cov)
    assert abs(shrunk[0, 1]) <= abs(cov[0, 1]) + 1e-6
