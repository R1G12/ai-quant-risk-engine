"""Tests for portfolio holdings bootstrap and refresh."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.risk.portfolio.holdings import ensure_holdings
from src.utils.config import load_app_config


@pytest.fixture
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("src.risk.portfolio.holdings.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.risk.portfolio.holdings.MANIFEST_PATH", tmp_path / "data" / "run_manifest.json")
    return tmp_path


def test_ensure_holdings_refreshes_stale_tickers(isolated_root: Path) -> None:
    holdings_dir = isolated_root / "data" / "raw" / "portfolio"
    holdings_dir.mkdir(parents=True)
    holdings_path = holdings_dir / "holdings.parquet"
    pl.DataFrame({"asset": ["CVX", "XLE", "QQQ"], "weight": [1 / 3, 1 / 3, 1 / 3]}).write_parquet(
        holdings_path
    )
    manifest_path = isolated_root / "data" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '{"tickers": ["CVX", "XLE", "QQQ"], "weighting": "partial"}',
        encoding="utf-8",
    )

    app = load_app_config(Path("configs/run.ci.yaml"))
    path = ensure_holdings(app)
    df = pl.read_parquet(path)
    assert set(df["asset"].to_list()) == set(app.market.tickers)
    # Second call should reuse refreshed holdings despite stale manifest.
    assert ensure_holdings(app) == path


def test_ensure_holdings_keeps_matching_manifest(isolated_root: Path) -> None:
    app = load_app_config(Path("configs/run.ci.yaml"))
    holdings_path = isolated_root / app.risk.portfolio.holdings_path
    holdings_path.parent.mkdir(parents=True, exist_ok=True)
    tickers = app.market.tickers
    weights = [0.3, 0.25, 0.2, 0.15, 0.1][: len(tickers)]
    pl.DataFrame({"asset": tickers, "weight": weights}).write_parquet(holdings_path)
    manifest_path = isolated_root / "data" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        f'{{"tickers": {tickers!r}}}',
        encoding="utf-8",
    )

    app = load_app_config(Path("configs/run.ci.yaml"))
    path = ensure_holdings(app)
    df = pl.read_parquet(path)
    assert df["weight"].to_list() == pytest.approx(weights)
