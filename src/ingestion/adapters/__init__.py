"""News ingestion adapters.

Each adapter returns a Polars DataFrame with at least: ``date``, ``source``,
``title``, ``content``. Live adapters may add ``ticker`` for direct attribution.
Register new sources in ``_ADAPTERS`` (e.g. a future NewsAPI module).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import polars as pl

from src.ingestion.adapters.sample import load_sample_news
from src.ingestion.adapters.yfinance import load_yfinance_news

AdapterFn = Callable[..., pl.DataFrame]

_ADAPTERS: dict[str, AdapterFn] = {
    "sample": load_sample_news,
    "yfinance": load_yfinance_news,
}


def get_news_source() -> str:
    """Resolve news source from env or params.yaml."""
    if env := os.getenv("NEWS_SOURCE"):
        return env.strip().lower()
    import yaml

    from src.utils.paths import PROJECT_ROOT

    params_path = PROJECT_ROOT / "params.yaml"
    params: dict[str, Any] = {}
    if params_path.is_file():
        with params_path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
    return str(params.get("ingestion", {}).get("source", "sample")).lower()


def load_news(
    source: str | None = None,
    *,
    tickers: list[str] | None = None,
    max_headlines_per_ticker: int | None = None,
) -> pl.DataFrame:
    """Load news via the configured adapter."""
    key = (source or get_news_source()).lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown news source {key!r}; available: {sorted(_ADAPTERS)}")

    if key == "yfinance":
        if not tickers:
            raise ValueError("yfinance news adapter requires tickers=...")
        cap = 10 if max_headlines_per_ticker is None else max_headlines_per_ticker
        return load_yfinance_news(tickers, max_headlines_per_ticker=cap)

    return _ADAPTERS[key]()
