"""Shared news row schema and ticker mapping for ingest / features / Signals."""

from __future__ import annotations

from typing import Any

import polars as pl
import yaml

from src.utils.config import AppConfig
from src.utils.paths import CONFIGS_DIR

RAW_NEWS_COLUMNS = ["date", "ticker", "source", "title", "content"]

HEADLINE_CAP_MIN = 20
HEADLINE_CAP_MAX = 60


def clamp_headline_cap(n: int) -> int:
    """Clamp per-ticker headline cap to [20, 60]."""
    return max(HEADLINE_CAP_MIN, min(HEADLINE_CAP_MAX, int(n)))


def load_ingestion_news_config() -> dict[str, Any]:
    """Load configs/ingestion.yaml with defaults."""
    path = CONFIGS_DIR / "ingestion.yaml"
    defaults = {"max_headlines_per_ticker": 40, "lookback_days": 30}
    if not path.is_file():
        return defaults
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "max_headlines_per_ticker": int(raw.get("max_headlines_per_ticker", defaults["max_headlines_per_ticker"])),
        "lookback_days": int(raw.get("lookback_days", defaults["lookback_days"])),
    }


def source_to_ticker_expr(app: AppConfig, source_col: str = "source") -> pl.Expr:
    """Map news source names to tickers via sentiment_map."""
    mapping = app.sentiment_map.get("source_to_ticker", {}) or {}
    default = app.sentiment_map.get(
        "default_ticker",
        app.market.default_sentiment_ticker,
    )
    expr = pl.lit(str(default))
    for source, ticker in mapping.items():
        expr = (
            pl.when(pl.col(source_col) == source)
            .then(pl.lit(str(ticker)))
            .otherwise(expr)
        )
    return expr


def ticker_expr_from_map(app: AppConfig, source_col: str = "source") -> pl.Expr:
    """Prefer explicit ticker column when set; else sentiment_map on source."""
    mapped = source_to_ticker_expr(app, source_col=source_col)
    return (
        pl.when(pl.col("ticker").is_not_null() & (pl.col("ticker").cast(pl.Utf8).str.strip_chars() != ""))
        .then(pl.col("ticker").cast(pl.Utf8).str.to_uppercase())
        .otherwise(mapped)
    )


def apply_ticker_mapping(lf: pl.LazyFrame, app: AppConfig) -> pl.LazyFrame:
    """Add normalized ticker column (ticker-first, then sentiment_map)."""
    cols = lf.collect_schema().names()
    if "ticker" in cols:
        return lf.with_columns(ticker_expr_from_map(app).alias("ticker"))
    return lf.with_columns(source_to_ticker_expr(app).alias("ticker"))
