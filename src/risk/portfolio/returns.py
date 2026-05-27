"""Portfolio return series from risk dataset and holdings."""

from __future__ import annotations

import polars as pl

from src.features.wide_returns import load_returns_wide
from src.risk.portfolio.holdings import load_weights
from src.utils.config import AppConfig


def build_portfolio_returns(app: AppConfig) -> pl.DataFrame:
    """Weighted portfolio daily returns: r_p = sum_i w_i * r_i (eager pivot for correctness)."""
    weights = load_weights(app)
    tickers = list(weights.keys())
    wide = load_returns_wide(tickers)

    port = pl.lit(0.0)
    for ticker, w in weights.items():
        if ticker in wide.columns:
            port = port + pl.col(ticker).fill_null(0.0) * w

    return wide.with_columns(port.alias("portfolio_return")).select("timestamp", "portfolio_return")


__all__ = ["build_portfolio_returns"]
