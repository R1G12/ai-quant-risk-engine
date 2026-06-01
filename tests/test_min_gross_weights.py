"""Tests for minimum gross per-ticker floor."""

from __future__ import annotations

import numpy as np
import pytest

from src.portfolio.weights import (
    enforce_min_gross_per_ticker,
    min_gross_per_ticker,
    optimize_max_sharpe_gross,
    _check_min_gross_feasible,
)


def test_min_gross_formula_seven_names() -> None:
    assert min_gross_per_ticker(7) == pytest.approx(1 / 35)


def test_infeasible_min_gross_raises() -> None:
    with pytest.raises(ValueError, match="exceeds max_gross"):
        _check_min_gross_feasible(7, 1 / 35, max_gross_per_ticker=0.02)


def test_enforce_min_gross_raises_magnitude() -> None:
    w = np.array([0.001, 0.5, -0.001, 0.5])
    out = enforce_min_gross_per_ticker(w, 0.1)
    assert np.all(np.abs(out) >= 0.1 - 1e-12)


def test_optimize_spreads_weight_across_names() -> None:
    n = 7
    mean_r = np.full(n, 0.0002)
    cov = np.diag(np.full(n, 0.0004))
    w, ok = optimize_max_sharpe_gross(
        mean_r,
        cov,
        allow_shorts=True,
        max_gross_per_ticker=0.5,
        min_gross_divisor=5.0,
    )
    assert ok
    assert len(w) == n
    assert np.sum(np.abs(w)) == pytest.approx(1.0, rel=1e-4)
    assert np.all(np.abs(w) >= min_gross_per_ticker(n) - 1e-4)
