"""Tests for copilot reasoning engine."""

from __future__ import annotations

import polars as pl

from src.copilot.context.builder import PortfolioContext, build_portfolio_context
from src.copilot.reasoning.engine import CopilotEngine
from src.analytics.dashboard_kpis import WindowKpis


def test_copilot_answers_var_question() -> None:
    engine = CopilotEngine()
    ctx = PortfolioContext(
        experiment_id="test",
        tickers=["AAPL"],
        weights={"AAPL": 1.0},
        exposures=pl.DataFrame({"asset": ["AAPL"], "weight": [1.0], "exposure": [1.0]}),
        kpis=WindowKpis(None, None, None),
        var_table=None,
        correlations=None,
        regimes=None,
        portfolio_metrics=None,
        sentiment_summary=None,
        as_of=__import__("datetime").date.today(),
    )
    resp = engine.ask("Which assets contribute most to VaR?", ctx)
    assert resp.title
    assert "VaR" in resp.title or "var" in resp.answer.lower()
    assert resp.requires_human_review


def test_copilot_help_fallback() -> None:
    resp = CopilotEngine().ask("hello there")
    assert "Copilot help" in resp.title or "help" in resp.answer.lower()


def test_build_portfolio_context_runs() -> None:
    ctx = build_portfolio_context()
    assert ctx.experiment_id
    assert isinstance(ctx.tickers, list)


def test_copilot_response_has_evidence_dict() -> None:
    resp = CopilotEngine().ask("Give me a portfolio summary")
    assert isinstance(resp.evidence, dict)
    assert resp.question == "Give me a portfolio summary"
