"""Lookahead bias guards."""

from __future__ import annotations

from datetime import datetime

import polars as pl


def assert_no_future_timestamps(
    features: pl.DataFrame,
    decision_time: datetime,
) -> None:
    """Ensure all feature timestamps are <= decision_time."""
    if features.height == 0:
        return
    max_ts = features["timestamp"].max()
    if max_ts is not None and max_ts > decision_time:
        raise ValueError(f"Lookahead detected: max timestamp {max_ts} > {decision_time}")
