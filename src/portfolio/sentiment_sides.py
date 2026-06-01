"""FinBERT-derived position sides for portfolio weighting (Signals-aligned)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.features.sentiment_ticker import resolve_ticker_series
from src.portfolio.stops import SENTIMENT_BEAR_THRESHOLD, SENTIMENT_BULL_THRESHOLD
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.paths import PROCESSED_SENTIMENT_PATH

LOGGER = get_logger(__name__)


def _map_sentiment_to_tickers(df: pl.DataFrame, app: AppConfig) -> pl.DataFrame:
    """Map news rows to tickers (same rules as feature stage)."""
    date_col = df.schema.get("date")
    if date_col == pl.Date:
        ts = pl.col("date").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    elif date_col == pl.Datetime(time_unit="us", time_zone="UTC") or str(date_col).startswith("Datetime"):
        ts = pl.col("date")
    else:
        ts = pl.col("date").str.to_datetime(time_zone="UTC", strict=False)

    tickers = resolve_ticker_series(df, app)
    return df.with_columns(ts.alias("timestamp")).with_columns(tickers.alias("ticker"))


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
    """Load per-ticker summary and daily FinBERT scores over the last ``window_days``.

    Returns (summary_df, daily_df). Either may be None if sentiment file missing/empty.
    """
    if not PROCESSED_SENTIMENT_PATH.is_file():
        return None, None

    raw = pl.read_parquet(PROCESSED_SENTIMENT_PATH)
    if raw.is_empty() or "sentiment_label" not in raw.columns:
        return None, None

    mapped = _map_sentiment_to_tickers(raw, app)
    max_ts = mapped["timestamp"].max()
    if max_ts is None:
        return None, None
    end = max_ts.date() if hasattr(max_ts, "date") else date.today()
    start = end - timedelta(days=window_days)

    if tickers is None:
        tickers = list(app.market.tickers)
    universe = [str(t).strip().upper() for t in tickers if str(t).strip()]

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


def ticker_sentiment_scores(
    app: AppConfig,
    tickers: list[str],
    *,
    window_days: int = 30,
) -> dict[str, float]:
    """Per-ticker mean FinBERT score over ``window_days`` (same metric as Signals dashboard)."""
    summary, _ = load_finbert_window(app, window_days=window_days, tickers=tickers)
    if summary is None or summary.is_empty():
        return {}
    return {
        str(row["ticker"]): float(row["sentiment_score"])
        for row in summary.iter_rows(named=True)
    }


def merge_position_sides_from_sentiment(
    tickers: list[str],
    explicit: dict[str, str],
    scores: dict[str, float],
    *,
    bear_threshold: float = SENTIMENT_BEAR_THRESHOLD,
    bull_threshold: float = SENTIMENT_BULL_THRESHOLD,
) -> dict[str, str]:
    """Fill ``long``/``short`` for tickers not in ``explicit`` using FinBERT score bands."""
    out = {str(k): str(v).strip().lower() for k, v in explicit.items()}
    for t in tickers:
        if t in out:
            continue
        score = scores.get(t)
        if score is None:
            continue
        if score <= bear_threshold:
            out[t] = "short"
        elif score >= bull_threshold:
            out[t] = "long"
    return out


def effective_position_sides(
    app: AppConfig,
    tickers: list[str],
    explicit: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Merge user ``position_sides`` with FinBERT-derived sides when enabled."""
    port = app.run.portfolio if app.run is not None else None
    base = {k: v for k, v in (explicit or (port.position_sides if port else {}) or {}).items() if k in tickers}

    if port is None or not port.sentiment_position_sides:
        return base or None
    if not port.allow_shorts:
        return base or None

    scores = ticker_sentiment_scores(
        app,
        tickers,
        window_days=port.sentiment_sides_window_days,
    )
    if not scores:
        LOGGER.debug("No FinBERT scores for sentiment_position_sides; using explicit sides only")
        return base or None

    merged = merge_position_sides_from_sentiment(tickers, base, scores)
    inferred = {t: merged[t] for t in tickers if t in merged and t not in base}
    if inferred:
        LOGGER.info("FinBERT position sides (explicit config unchanged): %s", inferred)
    return merged or None
