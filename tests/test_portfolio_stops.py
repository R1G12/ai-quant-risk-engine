"""Tests for sentiment-derived trailing stop helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
    vol_scaled_stop_levels,
)
from src.dashboards.core.signals_loaders import load_trailing_stops_table
from src.utils.config import AppConfig, Config, FeatureConfig, PortfolioRunConfig, RunConfig, RunMarketOverrides, load_app_config


def test_build_stops_bull() -> None:
    stops, tag, _, _ = build_stops(0.5)
    assert tag == "Bullish-derived"
    assert [s["level"] for s in stops] == list(BULL_STOP_LEVELS)


def test_build_stops_neutral() -> None:
    stops, tag, _, _ = build_stops(0.0)
    assert tag == "Neutral-derived"
    assert [s["level"] for s in stops] == list(NEUTRAL_STOP_LEVELS)


def test_build_stops_bear() -> None:
    stops, tag, _, _ = build_stops(-0.5)
    assert tag == "Bearish-derived"
    assert [s["level"] for s in stops] == list(BEAR_STOP_LEVELS)


def test_tranche_1_is_tightest_level() -> None:
    tset = trailing_stop_set(0.0)
    assert tset.tranche_1_level == pytest.approx(-0.05)
    assert tset.tranche_1_level == max(s["level"] for s in tset.stops)


def test_vol_scaled_stop_levels_scale_with_vol() -> None:
    policy = trailing_stop_policy_from_dict({"mode": "vol_scaled"})
    high_vol_levels, _ = vol_scaled_stop_levels(0.04, policy, sentiment_mult=1.0)
    low_vol_levels, _ = vol_scaled_stop_levels(0.01, policy, sentiment_mult=1.0)
    assert high_vol_levels[0] < low_vol_levels[0]
    assert high_vol_levels[1] < low_vol_levels[1]


def test_vol_scaled_bull_wider_than_bear() -> None:
    policy = trailing_stop_policy_from_dict({"mode": "vol_scaled"})
    daily_vol = 0.025
    bull_stops, tag_bull, vol_used, _ = build_stops(0.5, policy, daily_vol=daily_vol)
    bear_stops, tag_bear, _, _ = build_stops(-0.5, policy, daily_vol=daily_vol)
    assert tag_bull == "Bullish-derived (vol-scaled)"
    assert tag_bear == "Bearish-derived (vol-scaled)"
    assert vol_used == pytest.approx(daily_vol)
    assert bull_stops[0]["level"] < bear_stops[0]["level"]


def test_vol_scaled_uses_fallback_when_vol_missing() -> None:
    policy = trailing_stop_policy_from_dict(
        {"mode": "vol_scaled", "fallback_daily_vol": 0.03}
    )
    _, _, vol_used, vol_scale = build_stops(0.0, policy, daily_vol=None)
    assert vol_used == pytest.approx(0.03)
    assert vol_scale is not None


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
    stops, tag, _, _ = build_stops(0.5, policy)
    assert tag == "Bullish-derived"
    assert [s["level"] for s in stops] == [-0.01, -0.02, -0.03]


def test_trailing_stop_policy_from_dict_vol_scaled() -> None:
    policy = trailing_stop_policy_from_dict(
        {
            "mode": "vol_scaled",
            "tranche_sigmas": [2.0, 3.0, 4.0],
            "horizon_days": 10,
            "sentiment_vol_mult": {"bull": 1.5, "neutral": 1.0, "bear": 0.5},
        }
    )
    assert policy.mode == "vol_scaled"
    assert policy.tranche_sigmas == (2.0, 3.0, 4.0)
    assert policy.sentiment_vol_mult["bull"] == pytest.approx(1.5)


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


def test_load_trailing_stops_table_vol_scaled_with_feature_vol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.dashboards.core import signals_loaders as loaders_mod
    from src.utils import paths as paths_mod

    vol_dir = tmp_path / "volatility"
    vol_dir.mkdir()
    pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            "ticker": ["AAPL", "AAPL"],
            "volatility": [0.02, 0.03],
            "annualized_volatility": [0.32, 0.48],
        }
    ).write_parquet(vol_dir / "volatility.parquet")
    monkeypatch.setattr(paths_mod, "FEATURES_VOLATILITY_DIR", vol_dir)
    monkeypatch.setattr(loaders_mod, "project_paths", paths_mod)

    policy = trailing_stop_policy_from_dict({"mode": "vol_scaled"})
    base = load_app_config()
    run = RunConfig(
        mode="demo",
        portfolio=PortfolioRunConfig(trailing_stops=policy),
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
            "sentiment_score": [0.0],
            "bullish_ratio": [0.5],
            "negative_ratio": [0.2],
            "article_count": [5],
            "avg_confidence": [0.9],
        }
    )
    out = load_trailing_stops_table(summary, app)
    row = out.row(0, named=True)
    assert row["daily_vol"] == pytest.approx(0.03)
    assert row["regime_tag"] == "Neutral-derived (vol-scaled)"
