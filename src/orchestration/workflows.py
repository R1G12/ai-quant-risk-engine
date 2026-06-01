"""Named workflows mapping to DVC stage bundles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    description: str
    dvc_stages: list[str]
    requires_network: bool = False


WORKFLOWS: dict[str, WorkflowSpec] = {
    "refresh_market": WorkflowSpec(
        name="refresh_market",
        description="Refresh sample market data and merged features",
        dvc_stages=[
            "ingest_market_data",
            "clean_market_data",
            "generate_returns",
            "generate_volatility_features",
            "generate_technical_features",
            "generate_sentiment_features",
            "merge_features",
        ],
    ),
    "risk_refresh": WorkflowSpec(
        name="risk_refresh",
        description="Recompute Phase 3 risk artifacts",
        dvc_stages=[
            "generate_correlations",
            "optimize_portfolios",
            "generate_portfolio_metrics",
            "generate_volatility_metrics",
            "generate_var_metrics",
            "generate_cvar_metrics",
            "generate_efficient_frontier",
        ],
    ),
    "research_refresh": WorkflowSpec(
        name="research_refresh",
        description="Simulation, backtest, and reports",
        dvc_stages=[
            "generate_simulations",
            "run_backtests",
            "run_stress_tests",
            "run_scenario_analysis",
            "evaluate_performance",
            "compare_experiments",
            "generate_research_reports",
        ],
    ),
    "platform_reports": WorkflowSpec(
        name="platform_reports",
        description="Generate Phase 5 institutional reports",
        dvc_stages=["generate_platform_reports"],
    ),
    "full_refresh": WorkflowSpec(
        name="full_refresh",
        description="Full pipeline repro",
        dvc_stages=[],  # empty => dvc repro
    ),
}
