"""Tests for project-relative path display helpers."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import PROJECT_ROOT, display_path, sanitize_display_text


def test_display_path_relative_to_project() -> None:
    p = PROJECT_ROOT / "data" / "risk" / "optimization" / "optimal_weights.parquet"
    assert display_path(p) == "data/risk/optimization/optimal_weights.parquet"


def test_sanitize_display_text_strips_absolute_root() -> None:
    abs_path = PROJECT_ROOT / "data" / "foo.parquet"
    msg = f"No such file: {abs_path}"
    cleaned = sanitize_display_text(msg)
    assert str(PROJECT_ROOT) not in cleaned
    assert "data/foo.parquet" in cleaned.replace("\\", "/")
