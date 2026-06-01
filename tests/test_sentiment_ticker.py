"""Tests for sentiment ticker resolution."""

from __future__ import annotations

import polars as pl

from src.features.sentiment_agg import map_sentiment_to_tickers
from src.features.sentiment_ticker import resolve_ticker_series
from src.utils.config import AppConfig, IngestionConfig, MarketConfig, ResearchConfig, RiskConfig
from src.utils.config import (
    BacktestConfig,
    Config,
    FeatureConfig,
    OptimizationRiskConfig,
    PortfolioRiskConfig,
    ResearchMetaConfig,
    ScenarioConfig,
    SimulationConfig,
    VarRiskConfig,
    VolatilityRiskConfig,
)


def _minimal_app() -> AppConfig:
    risk = RiskConfig(
        volatility=VolatilityRiskConfig(),
        var=VarRiskConfig(),
        portfolio=PortfolioRiskConfig(),
        optimization=OptimizationRiskConfig(),
    )
    research = ResearchConfig(
        simulation=SimulationConfig(),
        backtest=BacktestConfig(),
        scenarios=ScenarioConfig(scenarios={}),
        meta=ResearchMetaConfig(),
    )
    return AppConfig(
        finbert=Config(),
        ingestion=IngestionConfig(),
        market=MarketConfig(default_sentiment_ticker="MARKET"),
        features=FeatureConfig(),
        sentiment_map={
            "source_to_ticker": {"Bloomberg": "AAPL"},
            "default_ticker": "MARKET",
        },
        risk=risk,
        research=research,
    )


def test_resolve_ticker_prefers_explicit_column() -> None:
    app = _minimal_app()
    df = pl.DataFrame(
        {
            "source": ["Bloomberg", "Reuters"],
            "ticker": ["QQQ", None],
        }
    )
    out = resolve_ticker_series(df, app)
    assert out.to_list() == ["QQQ", "MARKET"]


def test_map_sentiment_to_tickers_live_column() -> None:
    app = _minimal_app()
    lf = pl.LazyFrame(
        {
            "date": ["2026-01-01"],
            "source": ["yfinance"],
            "text": ["headline"],
            "sentiment_label": ["positive"],
            "sentiment_score": [0.9],
            "ticker": ["CVX"],
        }
    )
    out = map_sentiment_to_tickers(lf, app).collect()
    assert out["ticker"][0] == "CVX"


def test_map_sentiment_to_tickers_publisher_map() -> None:
    app = _minimal_app()
    lf = pl.LazyFrame(
        {
            "date": ["2026-01-01"],
            "source": ["Bloomberg"],
            "text": ["headline"],
            "sentiment_label": ["positive"],
            "sentiment_score": [0.9],
        }
    )
    out = map_sentiment_to_tickers(lf, app).collect()
    assert out["ticker"][0] == "AAPL"
