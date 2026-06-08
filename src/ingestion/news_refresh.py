"""Detect when on-disk news/sentiment is stale for the active run profile."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.paths import PROCESSED_SENTIMENT_PATH, RAW_NEWS_PATH


def _tickers_in_parquet(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    df = pl.read_parquet(path)
    if df.is_empty() or "ticker" not in df.columns:
        return set()
    return {str(t).upper() for t in df["ticker"].unique().to_list()}


def holdings_missing_from_news(holdings_tickers: list[str], news_path: Path | None = None) -> bool:
    """True when no holding ticker appears in raw news (wrong source or empty ingest)."""
    path = news_path or RAW_NEWS_PATH
    hold = {str(t).strip().upper() for t in holdings_tickers if str(t).strip()}
    if not hold:
        return False
    news_tickers = _tickers_in_parquet(path)
    if not news_tickers:
        return True
    return hold.isdisjoint(news_tickers)


def sentiment_missing_holdings(holdings_tickers: list[str], sentiment_path: Path | None = None) -> bool:
    """True when processed sentiment has no overlap with holdings."""
    path = sentiment_path or PROCESSED_SENTIMENT_PATH
    return holdings_missing_from_news(holdings_tickers, path)
