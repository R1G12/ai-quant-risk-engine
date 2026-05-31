"""Tests for aqre prepare / materialize_run."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.portfolio.prepare import materialize_run
from src.utils.config import load_app_config


def test_materialize_run_equal_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.portfolio.prepare.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("AQRE_SKIP_TICKER_VALIDATION", "1")
    monkeypatch.delenv("MARKET_SOURCE", raising=False)
    app = load_app_config()
    if app.run is None:
        pytest.skip("configs/run.yaml not present")
    holdings = tmp_path / "holdings.parquet"
    app.risk.portfolio.holdings_path = str(holdings)
    materialize_run(app)
    assert holdings.is_file()
    df = pl.read_parquet(holdings)
    assert len(df) == len(app.market.tickers)
    assert abs(float(df["weight"].abs().sum()) - 1.0) < 1e-6
    manifest = tmp_path / "data" / "run_manifest.json"
    assert manifest.is_file()
    assert manifest.read_text(encoding="utf-8").find("weighting") >= 0
