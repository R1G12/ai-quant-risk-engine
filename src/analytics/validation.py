"""Chart data validation helpers."""

from __future__ import annotations

import polars as pl


def assert_unique_timestamps(df: pl.DataFrame, col: str = "timestamp") -> None:
    dupes = df.group_by(col).len().filter(pl.col("len") > 1)
    if dupes.height > 0:
        raise ValueError(f"Duplicate timestamps in {col}: {dupes.height} groups")


def assert_equity_sane(
    equity: pl.DataFrame,
    *,
    max_daily_move: float = 0.25,
    equity_col: str = "equity",
    return_col: str = "portfolio_return",
) -> None:
    """Raise if equity series has duplicate dates or implausible jumps."""
    assert_unique_timestamps(equity)
    if equity.height == 0:
        raise ValueError("Empty equity frame")
    max_ret = float(equity[return_col].abs().max())
    if max_ret > max_daily_move:
        raise ValueError(f"Daily return exceeds sanity bound: {max_ret:.4f} > {max_daily_move}")


def assert_histogram_counts(df: pl.DataFrame, expected: int, col: str = "path_id") -> None:
    if df.height != expected:
        raise ValueError(f"Expected {expected} paths, got {df.height}")


def assert_stress_ordering(stress: pl.DataFrame, baseline_var: float) -> None:
    if "scenario" not in stress.columns or "var" not in stress.columns:
        return
    vol = stress.filter(pl.col("scenario") == "vol_spike")
    if vol.height and float(vol["var"][0]) > baseline_var + 1e-9:
        raise ValueError("vol_spike VaR should be <= baseline VaR")
