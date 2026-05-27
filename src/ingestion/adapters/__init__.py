"""News ingestion adapters."""

from __future__ import annotations

import os
from typing import Callable

import polars as pl

from src.ingestion.adapters.sample import load_sample_news

Adapters = dict[str, Callable[[], pl.DataFrame]]

_ADAPTERS: Adapters = {
    "sample": load_sample_news,
}


def get_news_source() -> str:
    """Resolve news source from env or params.yaml."""
    if env := os.getenv("NEWS_SOURCE"):
        return env.strip().lower()
    import yaml

    from src.utils.paths import PROJECT_ROOT

    params_path = PROJECT_ROOT / "params.yaml"
    params: dict = {}
    if params_path.is_file():
        with params_path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
    return str(params.get("ingestion", {}).get("source", "sample")).lower()


def load_news(source: str | None = None) -> pl.DataFrame:
    """Load news via the configured adapter."""
    key = (source or get_news_source()).lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown news source {key!r}; available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[key]()
