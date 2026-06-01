"""Load FinBERT scores, regimes, and stop tables for the Signals dashboard page."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.portfolio.stops import TrailingStopSet, trailing_stop_set
from src.risk.portfolio.holdings import load_weights
from src.utils.config import AppConfig
from src.utils.paths import PROCESSED_SENTIMENT_PATH, RISK_REGIMES_PATH


def _map_sentiment_to_tickers(df: pl.DataFrame, app: AppConfig) -> pl.DataFrame:
    """Map news rows to tickers using sentiment_map (same rules as feature stage)."""
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

    date_col = df.schema.get("date")
    if date_col == pl.Date:
        ts = pl.col("date").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    elif date_col == pl.Datetime(time_unit="us", time_zone="UTC") or str(date_col).startswith("Datetime"):
        ts = pl.col("date")
    else:
        ts = pl.col("date").str.to_datetime(time_zone="UTC", strict=False)

    return df.with_columns(ts.alias("timestamp"), ticker_expr.alias("ticker"))


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
    """Load per-ticker summary and daily scores over the last ``window_days``.

    Returns (summary_df, daily_df). Either may be None if sentiment file missing/empty.
    summary columns: ticker, sentiment_score, bullish_ratio, negative_ratio, article_count,
    avg_confidence
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

    universe = tickers if tickers is not None else list(load_weights(app).keys())
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


def load_trailing_stops_table(summary: pl.DataFrame) -> pl.DataFrame:
    """Expand sentiment summary into per-ticker stop levels (3 tranches)."""
    rows: list[dict[str, object]] = []
    for row in summary.iter_rows(named=True):
        ticker = str(row["ticker"])
        score = float(row["sentiment_score"])
        tset: TrailingStopSet = trailing_stop_set(score)
        rows.append(
            {
                "ticker": ticker,
                "sentiment_score": score,
                "regime_tag": tset.regime_tag,
                "tranche_1_level": tset.tranche_1_level,
                "stop_1_level": tset.stop_1["level"],
                "stop_1_exit": tset.stop_1["exit_fraction"],
                "stop_2_level": tset.stop_2["level"],
                "stop_2_exit": tset.stop_2["exit_fraction"],
                "stop_3_level": tset.stop_3["level"],
                "stop_3_exit": tset.stop_3["exit_fraction"],
            }
        )
    return pl.DataFrame(rows)


def load_latest_regime() -> tuple[str | None, pl.DataFrame | None]:
    """Return (latest_regime_label, full regimes frame) or (None, None)."""
    if not RISK_REGIMES_PATH.is_file():
        return None, None
    regimes = pl.read_parquet(RISK_REGIMES_PATH).sort("timestamp")
    if regimes.is_empty() or "regime_label" not in regimes.columns:
        return None, regimes
    latest = regimes["regime_label"][-1]
    return str(latest) if latest is not None else None, regimes
