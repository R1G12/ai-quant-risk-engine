"""Tests for shared yfinance scalar/series helpers used by Tracker."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.market.adapters.yfinance import yfinance_close_series, yfinance_scalar_float
from src.portfolio.tracker.fx import FxRateTable
from src.portfolio.tracker.prices import _yfinance_last_close
from src.simulation.monte_carlo.engine import path_summary_stats, tail_metrics_from_paths
import numpy as np


def test_yfinance_scalar_float_from_series() -> None:
    assert yfinance_scalar_float(pd.Series([123.45])) == pytest.approx(123.45)


def test_yfinance_close_series_multiindex() -> None:
    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])
    hist = pd.DataFrame(
        {
            ("Close", "MSFT"): [100.0, 101.0],
            ("Open", "MSFT"): [99.0, 100.0],
        },
        index=idx,
    )
    hist.columns = pd.MultiIndex.from_tuples(hist.columns)
    series = yfinance_close_series(hist)
    assert series is not None
    assert float(series.iloc[-1]) == pytest.approx(101.0)


def test_fx_rate_table_multiindex_history() -> None:
    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])
    hist = pd.DataFrame({("Close", "USDSGD=X"): [1.30, 1.31]}, index=idx)
    hist.columns = pd.MultiIndex.from_tuples(hist.columns)

    from unittest.mock import patch

    with patch("src.portfolio.tracker.fx.download_symbol_history", return_value=hist):
        table = FxRateTable.load("USDSGD=X", date(2025, 1, 1), date(2025, 1, 2))
    assert table.rate_on(date(2025, 1, 2)) == pytest.approx(1.31)


def test_yfinance_last_close_multiindex() -> None:
    yesterday = date.today() - timedelta(days=1)
    idx = pd.to_datetime([yesterday - timedelta(days=1), yesterday])
    hist = pd.DataFrame({("Close", "AAPL"): [50.0, 51.0]}, index=idx)
    hist.columns = pd.MultiIndex.from_tuples(hist.columns)

    from unittest.mock import patch

    with patch("src.portfolio.tracker.prices.download_symbol_history", return_value=hist):
        price, as_of = _yfinance_last_close("AAPL", lookback_days=7)
    assert price == pytest.approx(51.0)
    assert as_of == yesterday


def test_path_summary_handles_zero_start() -> None:
    paths = np.array([[0.0, 0.0, 1.0], [1.0, 1.1, 1.2]])
    summary = path_summary_stats(paths)
    assert summary.height == 2
    bad = summary.row(0, named=True)["terminal_return"]
    assert isinstance(bad, float) and np.isnan(bad)


def test_tail_metrics_ignores_non_finite() -> None:
    paths = np.array([[0.0, 0.0], [1.0, 1.1], [1.0, 0.9]])
    tail = tail_metrics_from_paths(paths)
    assert tail.height == 1
    assert np.isfinite(tail["mean_return"][0])
