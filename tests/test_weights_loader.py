"""Tests for optimization weights loader."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.risk.portfolio.weights_loader import load_optimization_weights
from src.utils.config import load_app_config


def test_load_optimization_weights_from_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opt_dir = tmp_path / "data" / "risk" / "optimization"
    opt_dir.mkdir(parents=True)
    opt_path = opt_dir / "optimal_weights.parquet"
    pl.DataFrame(
        {
            "portfolio": ["max_sharpe", "max_sharpe", "min_var"],
            "asset": ["AAPL", "MSFT", "AAPL"],
            "weight": [0.6, 0.4, 1.0],
        }
    ).write_parquet(opt_path)

    monkeypatch.setattr("src.risk.portfolio.weights_loader.RISK_OPT_WEIGHTS_PATH", opt_path)
    app = load_app_config()
    weights, label = load_optimization_weights(app, portfolio="max_sharpe")
    assert label == "max_sharpe"
    assert weights["AAPL"] == pytest.approx(0.6)
    assert weights["MSFT"] == pytest.approx(0.4)


def test_load_optimization_weights_fallback_holdings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.risk.portfolio.weights_loader.RISK_OPT_WEIGHTS_PATH",
        tmp_path / "missing.parquet",
    )
    holdings = tmp_path / "holdings.parquet"
    pl.DataFrame({"asset": ["X", "Y"], "weight": [0.5, 0.5]}).write_parquet(holdings)
    monkeypatch.setattr("src.risk.portfolio.holdings.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.risk.portfolio.weights_loader.load_weights",
        lambda _app: {"X": 0.5, "Y": 0.5},
    )

    app = load_app_config()
    weights, label = load_optimization_weights(app)
    assert "fallback" in label
    assert weights == {"X": 0.5, "Y": 0.5}
