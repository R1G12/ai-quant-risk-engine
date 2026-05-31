"""Tests for notebooks/trading_notebook_utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.portfolio.weights import (
    aggregate_portfolio_paths,
    exposure_summary,
    gbm_paths,
    normalize_gross_weights,
    optimize_partial_weights,
    resolve_weights,
    simulate_trailing_stops,
)


def test_normalize_gross_weights() -> None:
    w = normalize_gross_weights(np.array([0.5, -0.3, 0.2]))
    assert abs(np.sum(np.abs(w)) - 1.0) < 1e-9


def test_exposure_summary_short() -> None:
    s = exposure_summary(np.array([0.4, -0.2, 0.4]))
    assert abs(s["gross"] - 1.0) < 1e-9
    assert s["short"] < 0


def test_partial_anchor_no_budget_for_free_tickers() -> None:
    tickers = ["A", "B", "C"]
    mu = np.array([0.1, 0.08, 0.05])
    cov = np.diag([0.04, 0.03, 0.02])
    with pytest.raises(ValueError, match="no room left for free tickers"):
        optimize_partial_weights(mu, cov, tickers, {"A": 0.5, "B": 0.5})


def test_resolve_partial_inline_without_w_sharpe() -> None:
    tickers = ["A", "B", "C"]
    mu = np.array([0.1, 0.08, 0.05])
    cov = np.diag([0.04, 0.03, 0.02])
    w = resolve_weights(
        "partial",
        tickers,
        allow_shorts=True,
        max_gross_per_ticker=1.0,
        position_sides=None,
        anchor_weights={"A": 0.4},
        mean_returns=mu,
        cov=cov,
    )
    assert abs(np.sum(np.abs(w)) - 1.0) < 1e-6


def test_resolve_equal_with_short_side() -> None:
    w = resolve_weights(
        "equal",
        ["A", "B"],
        allow_shorts=True,
        max_gross_per_ticker=1.0,
        position_sides={"B": "short"},
    )
    assert w[1] < 0
    assert abs(np.sum(np.abs(w)) - 1.0) < 1e-9


def test_long_trailing_stop_fires() -> None:
    # peak 100 -> 90 is -10% drawdown
    paths = np.array([[100.0, 100.0], [100.0, 90.0]])
    stops = [{"level": -0.05, "exit_fraction": 1.0, "label": "S1"}]
    out = simulate_trailing_stops(paths, stops, side="long")
    assert out[1, 1] < 100.0


def test_short_gbm_position_loses_on_rally() -> None:
    paths = gbm_paths(
        horizon=1,
        n_paths=1,
        start_value=100.0,
        drift_daily=0.01,
        vol_daily=0.0,
        seed=0,
        side="short",
    )
    assert paths[1, 0] < 100.0


def test_short_trailing_stop_on_position_path() -> None:
    paths = np.array([[100.0, 100.0], [100.0, 88.0]])
    stops = [{"level": -0.05, "exit_fraction": 1.0, "label": "S1"}]
    out = simulate_trailing_stops(paths, stops, side="long")
    assert out[1, 1] <= 88.0


def test_aggregate_portfolio_paths() -> None:
    p1 = np.array([[1.0], [2.0]])
    p2 = np.array([[1.0], [3.0]])
    w = np.array([0.6, -0.4])
    agg = aggregate_portfolio_paths({"A": p1, "B": p2}, w, ["A", "B"])
    assert agg[1, 0] == pytest.approx(0.6 * 2.0 + 0.4 * 3.0)
