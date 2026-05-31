"""DVCLive metrics helpers for pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from dvclive import Live


def log_stage_metrics(
    metrics_dir: Path,
    metrics: dict[str, Any],
    *,
    lazy_frame: pl.LazyFrame | None = None,
) -> None:
    """Log metrics with DVCLive (dvc experiment saving disabled)."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if lazy_frame is not None:
        stats = lazy_frame.select(
            pl.len().alias("row_count"),
            pl.col("timestamp").min().alias("min_timestamp"),
            pl.col("timestamp").max().alias("max_timestamp"),
        ).collect()
        if stats.height:
            metrics["row_count"] = stats["row_count"][0]
            metrics["min_timestamp"] = str(stats["min_timestamp"][0])
            metrics["max_timestamp"] = str(stats["max_timestamp"][0])

    with Live(dir=metrics_dir, save_dvc_exp=False, dvcyaml=False) as live:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                live.log_metric(key, value)
            else:
                live.log_metric(key, str(value))

    meta_path = metrics_dir / "metrics.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
