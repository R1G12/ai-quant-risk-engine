"""Backtesting tests."""

from datetime import datetime

import polars as pl

from src.backtesting.costs.fees import trading_cost
from src.backtesting.engine.walk_forward import run_historical_backtest
from src.backtesting.validation.leakage import assert_no_future_timestamps
from src.utils.config import load_app_config


def test_trading_cost_positive() -> None:
    assert trading_cost(1.0, 10.0, 5.0) > 0


def test_weights_sum_used_in_backtest() -> None:
    app = load_app_config()
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    ts = pl.date_range(
        pl.date(2024, 1, 1), pl.date(2024, 2, 1), interval="1d", eager=True
    ).cast(pl.Datetime)
    wide = pl.DataFrame(
        {
            "timestamp": ts,
            "AAPL": [0.001] * len(ts),
            "MSFT": [0.001] * len(ts),
        }
    )
    equity, rolling, trades = run_historical_backtest(wide, weights, app)
    assert equity.height == len(ts)
    assert "equity" in equity.columns
    assert "window_id" in equity.columns
    assert trades.height >= 1
    assert equity.filter(~pl.col("is_rebalance"))["cost"].max() == 0.0


def test_no_lookahead_guard() -> None:
    feats = pl.DataFrame({"timestamp": ["2024-01-01"]}).with_columns(
        pl.col("timestamp").str.to_datetime()
    )
    assert_no_future_timestamps(feats, datetime(2024, 1, 2))
