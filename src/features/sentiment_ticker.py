"""Map news/sentiment rows to portfolio tickers."""

from __future__ import annotations

import polars as pl

from src.utils.config import AppConfig


def ticker_from_source_map(app: AppConfig) -> pl.Expr:
    """Publisher name → ticker via sentiment_map (sample / legacy rows)."""
    mapping = app.sentiment_map.get("source_to_ticker", {}) or {}
    default = app.sentiment_map.get(
        "default_ticker",
        app.market.default_sentiment_ticker,
    )
    ticker_expr = pl.lit(str(default))
    for source, ticker in mapping.items():
        ticker_expr = (
            pl.when(pl.col("source") == source)
            .then(pl.lit(str(ticker)))
            .otherwise(ticker_expr)
        )
    return ticker_expr


def resolve_ticker_expr(app: AppConfig, *, has_ticker_column: bool) -> pl.Expr:
    """Prefer explicit ``ticker`` when present; else map by news ``source``."""
    mapped = ticker_from_source_map(app)
    if not has_ticker_column:
        return mapped
    return (
        pl.when(pl.col("ticker").is_not_null() & (pl.col("ticker").str.strip_chars() != ""))
        .then(pl.col("ticker").str.to_uppercase())
        .otherwise(mapped)
    )


def resolve_ticker_series(df: pl.DataFrame, app: AppConfig) -> pl.Series:
    """Eager ticker resolution for dashboard loaders."""
    has_ticker = "ticker" in df.columns
    return df.select(resolve_ticker_expr(app, has_ticker_column=has_ticker).alias("ticker"))[
        "ticker"
    ]
