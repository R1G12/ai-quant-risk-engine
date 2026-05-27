"""Simple feature drift vs trailing baseline (z-score)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.utils.paths import RISK_DATASET_PATH


@dataclass(frozen=True)
class DriftReport:
    feature: str
    zscore: float
    flagged: bool


def compute_feature_drift(
    feature: str = "returns",
    *,
    z_threshold: float = 2.5,
) -> list[DriftReport]:
    if not RISK_DATASET_PATH.is_file():
        return []
    df = pl.read_parquet(RISK_DATASET_PATH).sort("timestamp")
    if feature not in df.columns:
        return []
    recent = df[feature].tail(21).drop_nulls()
    baseline = df[feature].drop_nulls()
    if recent.len() < 5 or baseline.len() < 30:
        return []
    mu = float(baseline.mean())
    sigma = float(baseline.std())
    if sigma == 0:
        return []
    z = abs((float(recent.mean()) - mu) / sigma)
    return [
        DriftReport(feature=feature, zscore=z, flagged=z > z_threshold),
    ]
