"""Unit tests for individual copilot explainers."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.analytics.dashboard_kpis import WindowKpis
from src.copilot.attribution.portfolio import attribution_summary
from src.copilot.context.builder import PortfolioContext
from src.copilot.explanations.drawdown import explain_drawdown
from src.copilot.explanations.regime import explain_regime
from src.copilot.explanations.risk import explain_risk_increase, explain_var_contributors
from src.copilot.summarization.report import generate_portfolio_summary


def _ctx(**overrides) -> PortfolioContext:
    base = dict(
        experiment_id="t",
        tickers=["AAPL", "MSFT"],
        weights={"AAPL": 0.6, "MSFT": 0.4},
        exposures=pl.DataFrame(
            {"asset": ["AAPL", "MSFT"], "weight": [0.6, 0.4], "exposure": [0.6, 0.4]}
        ),
        kpis=WindowKpis(0.5, -0.1, -0.02),
        var_table=None,
        correlations=None,
        regimes=None,
        portfolio_metrics=None,
        sentiment_summary=None,
        as_of=date.today(),
        metadata={},
    )
    base.update(overrides)
    return PortfolioContext(**base)


def test_explain_var_with_correlations() -> None:
    corr = pl.DataFrame(
        {
            "asset_i": ["AAPL", "AAPL", "MSFT"],
            "asset_j": ["AAPL", "MSFT", "MSFT"],
            "metric": ["corr", "corr", "corr"],
            "value": [1.0, 0.5, 1.0],
        }
    )
    exp = explain_var_contributors(_ctx(correlations=corr))
    assert "AAPL" in exp.summary
    assert exp.evidence.get("top_correlation_contributors")


def test_explain_var_missing_data() -> None:
    exp = explain_var_contributors(_ctx())
    assert "not available" in exp.summary.lower() or "Correlation" in exp.summary


def test_explain_risk_increase_with_vol() -> None:
    pm = pl.DataFrame(
        {"timestamp": ["2024-01-01", "2024-01-02"], "ewma_vol": [0.10, 0.14]}
    ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
    exp = explain_risk_increase(_ctx(portfolio_metrics=pm))
    assert "EWMA" in exp.summary or "volatility" in exp.summary.lower()


def test_explain_regime() -> None:
    reg = pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "regime_label": ["low", "low", "high"],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
    exp = explain_regime(_ctx(regimes=reg))
    assert "high" in exp.summary or "low" in exp.summary


def test_generate_portfolio_summary() -> None:
    text = generate_portfolio_summary(_ctx())
    assert "experiment" in text.lower()
    assert "AAPL" in text


def test_attribution_summary(tmp_path, monkeypatch) -> None:
    from tests.helpers.platform_fixtures import patch_paths_to_root, write_minimal_platform_artifacts

    root = tmp_path / "r"
    write_minimal_platform_artifacts(root)
    patch_paths_to_root(monkeypatch, root)
    exp = attribution_summary(
        _ctx(weights={"AAPL": 0.6, "MSFT": 0.4}, tickers=["AAPL", "MSFT"])
    )
    assert "AAPL" in exp.summary or "attrib" in exp.title.lower()


def test_explain_drawdown_with_equity(tmp_path, monkeypatch) -> None:
    from tests.helpers.platform_fixtures import patch_paths_to_root, write_minimal_platform_artifacts

    root = tmp_path / "r"
    write_minimal_platform_artifacts(root)
    patch_paths_to_root(monkeypatch, root)
    from src.utils.config import load_app_config

    app = load_app_config()
    exp = explain_drawdown(
        _ctx(
            experiment_id=app.research.meta.experiment_id,
            weights={"AAPL": 0.5, "MSFT": 0.5},
        ),
        app,
    )
    assert "drawdown" in exp.summary.lower()
