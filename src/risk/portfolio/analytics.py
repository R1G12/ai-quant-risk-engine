"""Portfolio-level risk analytics."""

from __future__ import annotations

import polars as pl

from src.utils.config import AppConfig


def add_portfolio_analytics(lf: pl.LazyFrame, app: AppConfig) -> pl.LazyFrame:
    """Sharpe, Sortino, drawdown, rolling metrics on portfolio returns."""
    rf = app.features.risk_free_rate
    ann = app.features.annualization_factor
    window = app.risk.portfolio.rolling_metrics_window

    return (
        lf.with_columns((1 + pl.col("portfolio_return")).cum_prod().alias("wealth"))
        .with_columns(
            (pl.col("wealth") / pl.col("wealth").cum_max() - 1.0).alias("drawdown")
        )
        .with_columns(
            (pl.col("portfolio_return").rolling_mean(window) * ann).alias("rolling_mean_ann"),
            (pl.col("portfolio_return").rolling_std(window) * ann**0.5).alias("rolling_vol_ann"),
            pl.when(pl.col("portfolio_return") < 0)
            .then(pl.col("portfolio_return"))
            .otherwise(0.0)
            .alias("_downside"),
        )
        .with_columns(
            ((pl.col("rolling_mean_ann") - rf) / pl.col("rolling_vol_ann").clip(lower_bound=1e-8)).alias(
                "rolling_sharpe"
            ),
            (
                (pl.col("rolling_mean_ann") - rf)
                / pl.col("_downside").rolling_std(window).clip(lower_bound=1e-8)
            ).alias("rolling_sortino"),
        )
        .drop("_downside")
    )


def max_drawdown(returns: pl.Series) -> float:
    """Maximum drawdown on cumulative wealth."""
    wealth = (1 + returns.fill_null(0)).cum_prod()
    dd = wealth / wealth.cum_max() - 1.0
    return float(dd.min())


def rolling_beta(
    asset_returns: pl.Series,
    market_returns: pl.Series,
    window: int,
) -> pl.Series:
    """Rolling beta = Cov(r_a, r_m) / Var(r_m)."""
    df = pl.DataFrame({"asset": asset_returns, "market": market_returns})
    cov = pl.col("asset").rolling_cov(pl.col("market"), window_size=window)
    var = pl.col("market").rolling_var(window_size=window)
    return df.select((cov / var.clip(lower_bound=1e-8)).alias("beta"))["beta"]
