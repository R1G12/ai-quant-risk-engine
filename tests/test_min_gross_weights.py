"""Tests for minimum gross weight floor 1/(divisor*n)."""

from __future__ import annotations

import numpy as np

from src.portfolio.weights import (
    min_gross_per_ticker,
    optimize_max_sharpe_gross,
    resolve_weights,
)


def test_min_gross_per_ticker_formula() -> None:
    assert min_gross_per_ticker(7, divisor=5) == 1.0 / 35.0


def test_optimised_max_sharpe_respects_floor() -> None:
    n = 7
    tickers = [f"T{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    mu = rng.normal(0.0005, 0.0002, n)
    mu[0] = 0.01
    mu[1] = 0.009
    cov = np.diag(rng.uniform(0.0001, 0.0005, n))
    min_w = min_gross_per_ticker(n)
    w, ok = optimize_max_sharpe_gross(
        mu,
        cov,
        tickers,
        allow_shorts=True,
        max_gross_per_ticker=0.5,
        min_gross_divisor=5.0,
    )
    assert ok
    assert abs(np.sum(np.abs(w)) - 1.0) < 1e-5
    assert np.all(np.abs(w) >= min_w - 1e-6)
    assert np.sum(np.abs(w) > min_w * 2) >= 3


def test_equal_weights_respect_floor() -> None:
    tickers = ["A", "B", "C", "D", "E", "F", "G"]
    min_w = min_gross_per_ticker(len(tickers))
    w = resolve_weights(
        "equal",
        tickers,
        allow_shorts=True,
        max_gross_per_ticker=0.5,
        position_sides=None,
    )
    assert np.all(np.abs(w) >= min_w - 1e-9)
