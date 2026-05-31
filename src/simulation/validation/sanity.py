"""Simulation output sanity checks."""

from __future__ import annotations

import polars as pl


def validate_path_summaries(df: pl.DataFrame) -> None:
    """Raise if required columns missing or all-null."""
    required = ["path_id", "terminal_return", "max_drawdown"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing path summary columns: {missing}")
    if df.height == 0:
        raise ValueError("Empty path summary frame")
