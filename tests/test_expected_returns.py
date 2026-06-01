"""Tests for blended expected returns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.portfolio.expected_returns import build_expected_returns
from src.utils.config import (
    AppConfig,
    Config,
    FeatureConfig,
    PortfolioRunConfig,
    RunConfig,
    RunMarketOverrides,
    load_app_config,
)


@pytest.fixture
def risk_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "risk_dataset.parquet"
    rows = []
    for t in ["A", "B"]:
        for i in range(10):
            rows.append({"ticker": t, "returns": 0.001 * (i + 1), "bullish_ratio": 0.6})
    pl.DataFrame(rows).write_parquet(p)
    monkeypatch.setattr("src.portfolio.expected_returns.RISK_DATASET_PATH", p)
    return p


def test_build_expected_returns_blend(risk_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = tmp_path / "sentiment.parquet"
    pl.DataFrame(
        {
            "date": ["2026-01-01"] * 4,
            "ticker": ["A", "A", "B", "B"],
            "source": ["x"] * 4,
            "title": ["h"] * 4,
            "content": ["c"] * 4,
            "sentiment_label": ["positive", "negative", "positive", "positive"],
            "sentiment_score": [0.9, 0.9, 0.8, 0.8],
        }
    ).write_parquet(sent)
    monkeypatch.setattr("src.portfolio.sentiment_sides.PROCESSED_SENTIMENT_PATH", sent)

    base = load_app_config()
    run = RunConfig(
        mode="demo",
        market=RunMarketOverrides(tickers=["A", "B"]),
        portfolio=PortfolioRunConfig(sentiment_mu_blend=0.5, sentiment_mu_scale=0.01),
    )
    app = AppConfig(
        finbert=Config(),
        market=base.market,
        features=FeatureConfig(),
        sentiment_map={"source_to_ticker": {}, "default_ticker": "MARKET"},
        risk=base.risk,
        research=base.research,
        tracker=base.tracker,
        run=run,
    )
    mu, meta = build_expected_returns(app, ["A", "B"])
    assert len(mu) == 2
    assert meta["alpha_eff"] > 0
    assert meta["mu_mode"] == "vol_scaled"
