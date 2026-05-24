"""Merge feature sets into risk dataset."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.utils.paths import RISK_DATASET_METADATA_PATH


def merge_feature_frames(
    returns_lf: pl.LazyFrame,
    volatility_lf: pl.LazyFrame,
    technical_lf: pl.LazyFrame,
    sentiment_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Left-join feature sets on timestamp and ticker."""
    vol_cols = ["timestamp", "ticker", "volatility", "annualized_volatility", "rolling_sharpe"]
    tech_cols = ["timestamp", "ticker", "sma_10", "sma_20", "momentum", "drawdown", "volume_zscore"]
    sent_cols = ["timestamp", "ticker", "confidence", "bullish_ratio", "article_count"]

    vol = volatility_lf.select([c for c in vol_cols if c in volatility_lf.collect_schema().names()])
    tech = technical_lf.select([c for c in tech_cols if c in technical_lf.collect_schema().names()])
    sent = sentiment_lf.select([c for c in sent_cols if c in sentiment_lf.collect_schema().names()])

    merged = (
        returns_lf.join(vol, on=["timestamp", "ticker"], how="left")
        .join(tech, on=["timestamp", "ticker"], how="left")
        .join(sent, on=["timestamp", "ticker"], how="left")
    )
    return merged


def write_metadata(path: Path, columns: list[str], feature_version: str = "phase2-v1") -> None:
    """Write merged dataset metadata sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"feature_version": feature_version, "columns": columns}
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
