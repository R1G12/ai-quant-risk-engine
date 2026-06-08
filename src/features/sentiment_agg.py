"""Sentiment aggregation features from Phase 1 output."""

from __future__ import annotations

import polars as pl

from src.ingestion.news_schema import apply_ticker_mapping
from src.utils.config import AppConfig


def _timestamp_from_date_expr(lf: pl.LazyFrame, col: str = "date") -> pl.Expr:
    """Build UTC timestamp from ``date`` whether Utf8 (sample) or Date (yfinance)."""
    dtype = lf.collect_schema()[col]
    c = pl.col(col)
    if dtype == pl.Date:
        return c.cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    if isinstance(dtype, pl.Datetime) or str(dtype).startswith("Datetime"):
        if getattr(dtype, "time_zone", None) is None:
            return c.dt.replace_time_zone("UTC")
        return c
    return c.str.to_datetime(time_zone="UTC", strict=False)


def map_sentiment_to_tickers(lf: pl.LazyFrame, app: AppConfig) -> pl.LazyFrame:
    """Map news rows to tickers (ticker column first, else sentiment_map)."""
    lf = lf.with_columns(
        _timestamp_from_date_expr(lf).alias("timestamp"),
        pl.col("source").alias("source"),
        pl.col("text").alias("headline"),
        pl.col("sentiment_label").alias("sentiment"),
        pl.col("sentiment_score").alias("confidence"),
    )
    return apply_ticker_mapping(lf, app)


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
