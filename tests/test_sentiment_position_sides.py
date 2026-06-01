"""Tests for FinBERT-derived position sides."""

from __future__ import annotations

import numpy as np
import pytest

from src.portfolio.sentiment_sides import (
    effective_position_sides,
    infer_position_side,
    merge_position_sides_from_sentiment,
)
from src.portfolio.weights import optimize_max_sharpe_gross
from src.utils.config import PortfolioRunConfig, RunConfig, RunMarketOverrides, load_app_config


def test_infer_position_side_bands() -> None:
    assert infer_position_side(-0.9) == "short"
    assert infer_position_side(0.5) == "long"
    assert infer_position_side(0.0) is None


def test_merge_respects_explicit_over_sentiment() -> None:
    merged = merge_position_sides_from_sentiment(
        ["PDD", "USO"],
        {"PDD": "long"},
        {"PDD": -0.9, "USO": -0.1},
    )
    assert merged["PDD"] == "long"
    assert merged["USO"] == "long"


def test_sentiment_sides_disabled_when_allow_shorts_false() -> None:
    run = RunConfig(
        mode="demo",
        market=RunMarketOverrides(tickers=["PDD"]),
        portfolio=PortfolioRunConfig(
            allow_shorts=False,
            sentiment_position_sides=True,
            position_sides={},
        ),
    )
    from src.utils.config import AppConfig, Config, FeatureConfig

    base = load_app_config()
    app = AppConfig(
        finbert=Config(),
        market=base.market,
        features=FeatureConfig(),
        sentiment_map={},
        risk=base.risk,
        research=base.research,
        tracker=base.tracker,
        run=run,
    )
    sides = effective_position_sides(app, ["PDD"])
    assert sides is None


def test_optimize_with_short_side_sign() -> None:
    mean_r = np.array([0.001, -0.001, 0.0005])
    cov = np.diag([0.0004, 0.0004, 0.0004])
    position_sides = {"A": "long", "B": "short", "C": "long"}
    w, ok = optimize_max_sharpe_gross(
        mean_r,
        cov,
        allow_shorts=True,
        max_gross_per_ticker=0.5,
    )
    assert ok
    assert w[1] < 0
