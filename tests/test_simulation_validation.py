"""Simulation validation tests."""

import polars as pl
import pytest

from src.simulation.validation.sanity import validate_path_summaries


def test_validate_path_summaries_ok() -> None:
    df = pl.DataFrame(
        {"path_id": [0, 1], "terminal_return": [0.1, -0.05], "max_drawdown": [-0.2, -0.3]}
    )
    validate_path_summaries(df)


def test_validate_path_summaries_missing_column() -> None:
    df = pl.DataFrame({"path_id": [0]})
    with pytest.raises(ValueError, match="Missing"):
        validate_path_summaries(df)


def test_validate_path_summaries_empty() -> None:
    df = pl.DataFrame({"path_id": [], "terminal_return": [], "max_drawdown": []})
    with pytest.raises(ValueError, match="Empty"):
        validate_path_summaries(df)
