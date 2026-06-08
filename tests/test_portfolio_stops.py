"""Tests for sentiment-derived trailing stop helpers."""

from __future__ import annotations

import polars as pl
import pytest

from src.portfolio.stops import (
    BEAR_STOP_LEVELS,
    BULL_STOP_LEVELS,
    NEUTRAL_STOP_LEVELS,
    TrailingStopPolicy,
    TrailingStopTranchePolicy,
    build_stops,
    trailing_stop_policy_from_dict,
    trailing_stop_set,
)
from src.dashboards.core.signals_loaders import load_trailing_stops_table
from src.utils.config import AppConfig, Config, FeatureConfig, PortfolioRunConfig, RunConfig, RunMarketOverrides, load_app_config


def test_build_stops_bull() -> None:
    stops, tag = build_stops(0.5)
    assert tag == "Bullish-derived"
    assert [s["level"] for s in stops] == list(BULL_STOP_LEVELS)


def test_build_stops_neutral() -> None:
    stops, tag = build_stops(0.0)
    assert tag == "Neutral-derived"
    assert [s["level"] for s in stops] == list(NEUTRAL_STOP_LEVELS)


def test_build_stops_bear() -> None:
    stops, tag = build_stops(-0.5)
    assert tag == "Bearish-derived"
    assert [s["level"] for s in stops] == list(BEAR_STOP_LEVELS)


def test_tranche_1_is_tightest_level() -> None:
    tset = trailing_stop_set(0.0)
    assert tset.tranche_1_level == pytest.approx(-0.05)
    assert tset.tranche_1_level == max(s["level"] for s in tset.stops)


def test_load_trailing_stops_table() -> None:
    summary = pl.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "sentiment_score": [0.5, -0.5],
            "bullish_ratio": [0.6, 0.2],
            "negative_ratio": [0.1, 0.5],
            "article_count": [10, 8],
            "avg_confidence": [0.9, 0.85],
        }
    )
    out = load_trailing_stops_table(summary)
    assert out.height == 2
    assert "tranche_1_level" in out.columns
    aapl = out.filter(pl.col("ticker") == "AAPL").row(0, named=True)
    assert aapl["regime_tag"] == "Bullish-derived"


def test_trailing_stop_policy_from_dict() -> None:
    policy = trailing_stop_policy_from_dict(
        {
            "bull_threshold": 0.4,
            "bear_threshold": -0.4,
            "bull": {"levels": [-0.01, -0.02, -0.03], "fractions": [0.5, 0.25, 0.25]},
        }
    )
    assert policy.bull_threshold == pytest.approx(0.4)
    assert policy.bull.levels == (-0.01, -0.02, -0.03)
    stops, tag = build_stops(0.5, policy)
    assert tag == "Bullish-derived"
    assert [s["level"] for s in stops] == [-0.01, -0.02, -0.03]


def test_load_trailing_stops_table_uses_run_profile() -> None:
    base = load_app_config()
    run = RunConfig(
        mode="demo",
        market=RunMarketOverrides(tickers=["AAPL"]),
        portfolio=PortfolioRunConfig(
            trailing_stops=TrailingStopPolicy(
                bull=TrailingStopTranchePolicy((-0.01, -0.02, -0.03), (1 / 3, 1 / 3, 1 / 3)),
            ),
        ),
    )
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
    summary = pl.DataFrame(
        {
            "ticker": ["AAPL"],
            "sentiment_score": [0.5],
            "bullish_ratio": [0.6],
            "negative_ratio": [0.1],
            "article_count": [10],
            "avg_confidence": [0.9],
        }
    )
    out = load_trailing_stops_table(summary, app)
    row = out.row(0, named=True)
    assert row["tranche_1_level"] == pytest.approx(-0.01)
