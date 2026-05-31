"""Walk-forward and rolling backtest engine."""

from __future__ import annotations

import polars as pl

from src.backtesting.costs.fees import trading_cost
from src.backtesting.rebalancing.calendar import rebalance_mask, walk_forward_window_ids
from src.utils.config import AppConfig


def run_historical_backtest(
    returns_wide: pl.DataFrame,
    weights: dict[str, float],
    app: AppConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Walk-forward historical backtest with rebalance-only costs.

    Weights are static (Phase 3 optimal_weights). Train window labels warmup rows;
    test windows roll forward for reporting. Equity compounds across the full timeline.
    """
    tickers = [t for t in weights if t in returns_wide.columns]
    if not tickers:
        raise ValueError("No weight tickers found in returns wide frame")

    bt = app.research.backtest
    turnover = sum(abs(weights[t]) for t in tickers)
    rebal_cost = trading_cost(turnover, bt.tc_bps, bt.slippage_bps)

    ret_cols = [pl.col(t).fill_null(0.0) * weights[t] for t in tickers]
    port = returns_wide.select(
        pl.col("timestamp"),
        pl.sum_horizontal(ret_cols).alias("gross_return"),
    ).sort("timestamp")

    is_rebal = rebalance_mask(port["timestamp"], bt.rebalance_freq)
    cost = pl.when(is_rebal).then(pl.lit(rebal_cost)).otherwise(0.0)

    port = port.with_columns(
        cost.alias("cost"),
        is_rebal.alias("is_rebalance"),
    ).with_columns((pl.col("gross_return") - pl.col("cost")).alias("portfolio_return"))

    window_ids = walk_forward_window_ids(
        port.height, bt.walk_forward_train_days, bt.walk_forward_test_days
    )
    port = port.with_columns(pl.Series("window_id", window_ids)).with_columns(
        (1 + pl.col("portfolio_return")).cum_prod().alias("equity"),
    ).with_columns((pl.col("equity") / pl.col("equity").cum_max() - 1.0).alias("drawdown"))

    rolling = port.with_columns(
        (pl.col("portfolio_return").rolling_mean(21) * app.features.annualization_factor).alias(
            "rolling_mean_ann"
        ),
        (
            pl.col("portfolio_return").rolling_std(21)
            * (app.features.annualization_factor**0.5)
        ).alias("rolling_vol_ann"),
    ).with_columns(
        (
            (pl.col("rolling_mean_ann") - app.features.risk_free_rate)
            / pl.col("rolling_vol_ann").clip(lower_bound=1e-8)
        ).alias("rolling_sharpe")
    )

    trades = (
        port.filter(pl.col("is_rebalance"))
        .select("timestamp", "cost")
        .with_columns(pl.lit(turnover).alias("turnover"))
        .select("timestamp", "turnover", "cost")
    )

    return port, rolling, trades
