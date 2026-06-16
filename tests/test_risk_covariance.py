"""Covariance tests."""

import numpy as np
import polars as pl
import pytest

from src.risk.correlations.covariance import (
    align_optimization_tickers,
    ledoit_wolf_shrinkage,
    sample_covariance_matrix,
)


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


def test_single_ticker_covariance_is_2d() -> None:
    wide = pl.DataFrame(
        {
            "timestamp": range(30),
            "A": np.random.default_rng(0).normal(0, 0.02, 30),
        }
    )
    with pytest.raises(ValueError, match="at least 2 tickers"):
        sample_covariance_matrix(wide, ["A"])


def test_align_optimization_tickers_excludes_missing() -> None:
    wide = pl.DataFrame(
        {
            "timestamp": range(10),
            "A": np.random.default_rng(0).normal(0, 0.02, 10),
            "B": np.random.default_rng(1).normal(0, 0.03, 10),
        }
    )
    tickers = align_optimization_tickers(wide, ["A", "B", "SPCX"])
    assert tickers == ["A", "B"]


def test_numpy_cov_scalar_guard() -> None:
    """np.cov on one column is 0-d; atleast_2d keeps shrinkage safe if mis-called."""
    arr = np.random.default_rng(0).normal(0, 0.02, (30, 1))
    cov = np.atleast_2d(np.asarray(np.cov(arr, rowvar=False), dtype=float))
    assert cov.shape == (1, 1)
    shrunk = ledoit_wolf_shrinkage(cov)
    assert shrunk.shape == (1, 1)
