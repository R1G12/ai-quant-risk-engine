"""Schema validation using Polars lazy/eager frames."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
import yaml

from src.utils.paths import CONFIGS_DIR


def _load_schema(schema_name: str) -> dict:
    path = CONFIGS_DIR / "schemas" / f"{schema_name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_columns(
    df: pl.DataFrame | pl.LazyFrame,
    required_columns: Iterable[str],
    *,
    optional_columns: Iterable[str] | None = None,
) -> None:
    """Raise ValueError if required columns are missing."""
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    names = set(schema.names())
    missing = [c for c in required_columns if c not in names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if optional_columns:
        allowed = set(required_columns) | set(optional_columns)
        extra = names - allowed
        if extra:
            # Allow partition/metadata columns
            pass


def validate_schema_file(
    df: pl.DataFrame | pl.LazyFrame,
    schema_name: str,
) -> None:
    """Validate a frame against a YAML schema in configs/schemas/."""
    spec = _load_schema(schema_name)
    validate_columns(
        df,
        spec.get("required_columns", []),
        optional_columns=spec.get("optional_columns"),
    )
