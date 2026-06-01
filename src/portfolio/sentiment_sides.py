"""FinBERT-derived position sides and shared 30d sentiment scores."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import polars as pl

from src.ingestion.news_schema import apply_ticker_mapping
from src.portfolio.stops import SENTIMENT_BEAR_THRESHOLD, SENTIMENT_BULL_THRESHOLD
from src.utils.config import AppConfig
from src.utils.paths import PROCESSED_SENTIMENT_PATH

Side = Literal["long", "short"]


def _timestamp_expr(df: pl.DataFrame) -> pl.Expr:
    date_col = df.schema.get("date")
    if date_col == pl.Date:
        return pl.col("date").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    if date_col == pl.Datetime(time_unit="us", time_zone="UTC") or str(date_col).startswith("Datetime"):
        return pl.col("date")
    return pl.col("date").str.to_datetime(time_zone="UTC", strict=False)


def map_sentiment_dataframe(df: pl.DataFrame, app: AppConfig) -> pl.DataFrame:
    """Map sentiment rows to tickers and normalize timestamp."""
    mapped = apply_ticker_mapping(df.lazy(), app).collect()
    return mapped.with_columns(_timestamp_expr(mapped).alias("timestamp"))


def _label_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    label = pl.col("sentiment_label").str.to_lowercase()
    return lf.with_columns(
        (label == "positive").cast(pl.Float64).alias("is_positive"),
        (label == "negative").cast(pl.Float64).alias("is_negative"),
    )


def load_finbert_window(
    app: AppConfig,
    *,
    window_days: int = 30,
    tickers: list[str] | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Per-ticker summary and daily scores: mean(positive) - mean(negative)."""
    if not PROCESSED_SENTIMENT_PATH.is_file():
        return None, None

    raw = pl.read_parquet(PROCESSED_SENTIMENT_PATH)
    if raw.is_empty() or "sentiment_label" not in raw.columns:
        return None, None

    mapped = map_sentiment_dataframe(raw, app)
    max_ts = mapped["timestamp"].max()
    if max_ts is None:
        return None, None
    end = max_ts.date() if hasattr(max_ts, "date") else date.today()
    start = end - timedelta(days=window_days)

    if tickers is None:
        from src.risk.portfolio.holdings import load_weights

        universe = list(load_weights(app).keys())
    else:
        universe = tickers

    lf = (
        mapped.lazy()
        .filter(pl.col("ticker").is_in(universe))
        .filter(pl.col("timestamp").dt.date() >= start)
        .pipe(_label_flags)
        .rename({"sentiment_score": "finbert_confidence"})
    )

    daily = (
        lf.with_columns(pl.col("timestamp").dt.date().alias("day"))
        .group_by(["day", "ticker"])
        .agg(
            (pl.col("is_positive").mean() - pl.col("is_negative").mean()).alias("sentiment_score"),
            pl.col("is_positive").mean().alias("bullish_ratio"),
            pl.col("is_negative").mean().alias("negative_ratio"),
            pl.len().alias("article_count"),
            pl.col("finbert_confidence").mean().alias("avg_confidence"),
        )
        .with_columns(pl.col("day").cast(pl.Datetime(time_unit="us", time_zone="UTC")).alias("timestamp"))
        .sort(["ticker", "timestamp"])
        .collect()
    )

    if daily.is_empty():
        return None, None

    summary = (
        daily.group_by("ticker")
        .agg(
            pl.col("sentiment_score").mean().alias("sentiment_score"),
            pl.col("bullish_ratio").mean().alias("bullish_ratio"),
            pl.col("negative_ratio").mean().alias("negative_ratio"),
            pl.col("article_count").sum().alias("article_count"),
            pl.col("avg_confidence").mean().alias("avg_confidence"),
        )
        .sort("sentiment_score", descending=True)
    )
    return summary, daily


def finbert_scores_by_ticker(
    app: AppConfig,
    tickers: list[str],
    *,
    window_days: int = 30,
) -> dict[str, float]:
    """Map ticker -> 30d sentiment score; missing tickers omitted."""
    summary, _ = load_finbert_window(app, window_days=window_days, tickers=tickers)
    if summary is None or summary.is_empty():
        return {}
    return {str(r["ticker"]): float(r["sentiment_score"]) for r in summary.iter_rows(named=True)}


def infer_position_side(score: float) -> Side | None:
    """Return long/short when score crosses bull/bear bands; else None (default long)."""
    if score <= SENTIMENT_BEAR_THRESHOLD:
        return "short"
    if score >= SENTIMENT_BULL_THRESHOLD:
        return "long"
    return None


def merge_position_sides_from_sentiment(
    tickers: list[str],
    explicit_sides: dict[str, str],
    scores: dict[str, float],
) -> dict[str, str]:
    """YAML position_sides override FinBERT inference per ticker."""
    merged: dict[str, str] = {}
    for t in tickers:
        if t in explicit_sides:
            merged[t] = explicit_sides[t].strip().lower()
            continue
        side = infer_position_side(scores.get(t, 0.0))
        merged[t] = side if side is not None else "long"
    return merged


def effective_position_sides(
    app: AppConfig,
    tickers: list[str],
) -> dict[str, str] | None:
    """Merged long/short map for prepare/optimization, or None to use explicit YAML only."""
    if app.run is None:
        return None
    port = app.run.portfolio
    explicit = {k: v for k, v in (port.position_sides or {}).items() if k in tickers}
    if not port.sentiment_position_sides or not port.allow_shorts:
        return explicit or None
    scores = finbert_scores_by_ticker(app, tickers, window_days=port.sentiment_sides_window_days)
    merged = merge_position_sides_from_sentiment(tickers, explicit, scores)
    return merged or None
