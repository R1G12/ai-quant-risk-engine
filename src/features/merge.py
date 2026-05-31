"""Merge feature sets into risk dataset."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.utils.paths import RISK_DATASET_METADATA_PATH

JOIN_KEYS = ("timestamp", "ticker")

# Columns owned by the returns branch (market OHLCV + return features).
_BASE_COLUMNS = frozenset(
    {
        *JOIN_KEYS,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "year",
        "month",
        "ingested_at",
        "returns",
        "log_returns",
    }
)


def _engineered_columns(lf: pl.LazyFrame) -> list[str]:
    """Feature-only columns from a branch parquet (excludes OHLCV and return duplicates)."""
    names = lf.collect_schema().names()
    return [c for c in names if c not in _BASE_COLUMNS]


def merge_feature_frames(
    returns_lf: pl.LazyFrame,
    volatility_lf: pl.LazyFrame,
    technical_lf: pl.LazyFrame,
    sentiment_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Left-join feature sets on timestamp and ticker.

    Each branch contributes only its engineered columns (not duplicate OHLCV).
    """
    merged = returns_lf
    for branch in (volatility_lf, technical_lf):
        cols = list(JOIN_KEYS) + _engineered_columns(branch)
        merged = merged.join(branch.select(cols), on=list(JOIN_KEYS), how="left")

    if sentiment_lf is not None and len(sentiment_lf.collect_schema().names()) > 0:
        cols = list(JOIN_KEYS) + _engineered_columns(sentiment_lf)
        merged = merged.join(sentiment_lf.select(cols), on=list(JOIN_KEYS), how="left")

    return merged


def write_metadata(path: Path, columns: list[str], feature_version: str = "phase2-v1") -> None:
    """Write merged dataset metadata sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"feature_version": feature_version, "columns": columns}
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
