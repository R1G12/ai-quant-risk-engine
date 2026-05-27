"""Shared Plotly helpers."""

from __future__ import annotations

import polars as pl


def series_xy(df: pl.DataFrame, x_col: str, y_col: str) -> tuple[list, list]:
    """Convert polars columns to JSON-safe lists for Plotly."""
    sub = df.filter(pl.col(y_col).is_not_null()).sort(x_col)
    dtype = sub[x_col].dtype
    if isinstance(dtype, pl.Datetime) or dtype == pl.Date:
        x = sub[x_col].dt.strftime("%Y-%m-%d").to_list()
    else:
        x = sub[x_col].to_list()
    y = sub[y_col].to_list()
    return x, y
