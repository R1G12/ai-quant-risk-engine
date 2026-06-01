"""Tests for effective portfolio weight loading."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.risk.portfolio.holdings import (
    load_portfolio_weights,
    portfolio_weighting_mode,
    read_holdings_weights,
)
from src.utils.config import AppConfig, IngestionConfig, MarketConfig, ResearchConfig, RiskConfig
from src.utils.config import (
    BacktestConfig,
    Config,
    FeatureConfig,
    OptimizationRiskConfig,
    PortfolioRiskConfig,
    ResearchMetaConfig,
    RunConfig,
    RunMarketOverrides,
    PortfolioRunConfig,
    ScenarioConfig,
    SimulationConfig,
    VarRiskConfig,
    VolatilityRiskConfig,
)
def _app(
    tmp_path: Path,
    *,
    weighting: str = "equal",
    holdings: dict[str, float] | None = None,
) -> AppConfig:
    holdings_path = tmp_path / "holdings.parquet"
    holdings = holdings or {"AAPL": 0.5, "MSFT": 0.5}
    pl.DataFrame({"asset": list(holdings), "weight": list(holdings.values())}).write_parquet(
        holdings_path
    )
    risk = RiskConfig(
        volatility=VolatilityRiskConfig(),
        var=VarRiskConfig(),
        portfolio=PortfolioRiskConfig(holdings_path=str(holdings_path)),
        optimization=OptimizationRiskConfig(weighting=weighting),
    )
    research = ResearchConfig(
        simulation=SimulationConfig(),
        backtest=BacktestConfig(weight_source="max_sharpe"),
        scenarios=ScenarioConfig(scenarios={}),
        meta=ResearchMetaConfig(),
    )
    return AppConfig(
        finbert=Config(),
        ingestion=IngestionConfig(),
        market=MarketConfig(tickers=list(holdings)),
        features=FeatureConfig(),
        sentiment_map={},
        risk=risk,
        research=research,
        run=RunConfig(
            portfolio=PortfolioRunConfig(weighting=weighting),
            market=RunMarketOverrides(tickers=list(holdings)),
        ),
    )


def test_portfolio_weighting_mode_from_run(tmp_path: Path) -> None:
    app = _app(tmp_path, weighting="optimised")
    assert portfolio_weighting_mode(app) == "optimised"


def test_load_portfolio_weights_equal_uses_holdings(tmp_path: Path) -> None:
    app = _app(tmp_path, weighting="equal", holdings={"AAPL": 0.6, "MSFT": 0.4})
    assert load_portfolio_weights(app) == {"AAPL": 0.6, "MSFT": 0.4}


def test_load_portfolio_weights_optimised_uses_opt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, weighting="optimised", holdings={"AAPL": 0.5, "MSFT": 0.5})
    opt_path = tmp_path / "optimal_weights.parquet"
    pl.DataFrame(
        {
            "portfolio": ["max_sharpe", "max_sharpe"],
            "asset": ["AAPL", "MSFT"],
            "weight": [0.7, 0.3],
        }
    ).write_parquet(opt_path)
    monkeypatch.setattr("src.risk.portfolio.holdings.RISK_OPT_WEIGHTS_PATH", opt_path)
    assert load_portfolio_weights(app) == {"AAPL": 0.7, "MSFT": 0.3}
    assert read_holdings_weights(app) == {"AAPL": 0.5, "MSFT": 0.5}
