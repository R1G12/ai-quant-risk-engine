"""Sentiment aggregation features from Phase 1 output."""

from __future__ import annotations

import polars as pl

from src.features.sentiment_ticker import resolve_ticker_expr
from src.utils.config import AppConfig


def map_sentiment_to_tickers(lf: pl.LazyFrame, app: AppConfig) -> pl.LazyFrame:
    """Map news rows to tickers (explicit ``ticker`` column or publisher map)."""
    has_ticker = "ticker" in lf.collect_schema().names()

    lf = lf.with_columns(
        pl.col("date").str.to_datetime(time_zone="UTC").alias("timestamp"),
        pl.col("source").alias("source"),
        pl.col("text").alias("headline"),
        pl.col("sentiment_label").alias("sentiment"),
        pl.col("sentiment_score").alias("confidence"),
    )

    return lf.with_columns(
        resolve_ticker_expr(app, has_ticker_column=has_ticker).alias("ticker")
    )


def aggregate_sentiment_daily(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate sentiment to daily frequency per ticker."""
    return (
        lf.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by(["date", "ticker"])
        .agg(
            pl.col("confidence").mean().alias("confidence"),
            pl.col("sentiment").mode().first().alias("sentiment"),
            (pl.col("sentiment") == "positive").mean().alias("bullish_ratio"),
            pl.len().alias("article_count"),
        )
        .with_columns(pl.col("date").cast(pl.Datetime(time_zone="UTC")).alias("timestamp"))
        .drop("date")
    )
