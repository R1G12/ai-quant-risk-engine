"""Rebalance calendar helpers."""

from __future__ import annotations

import polars as pl


def rebalance_mask(timestamps: pl.Series, freq: str) -> pl.Series:
    """Boolean mask: True on rebalance days (including first observation)."""
    if timestamps.len() == 0:
        return timestamps

    if freq == "monthly":
        month = timestamps.dt.month()
        changed = month != month.shift(1)
        return changed.fill_null(True)

    if freq == "weekly":
        week = timestamps.dt.week()
        changed = week != week.shift(1)
        return changed.fill_null(True)

    # daily: rebalance every day
    return pl.lit(True)


def walk_forward_window_ids(n_rows: int, train_days: int, test_days: int) -> list[int]:
    """Assign window id per row; -1 during train warmup, then 0,1,2,... per test block."""
    ids: list[int] = []
    for i in range(n_rows):
        if i < train_days:
            ids.append(-1)
        else:
            ids.append((i - train_days) // test_days)
    return ids
