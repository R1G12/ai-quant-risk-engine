"""Portfolio endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.copilot.context.builder import build_portfolio_context
from src.copilot.summarization.report import generate_portfolio_summary
from src.utils.config import load_app_config

router = APIRouter()


@router.get("/summary")
def portfolio_summary() -> dict:
    ctx = build_portfolio_context()
    return {
        "experiment_id": ctx.experiment_id,
        "tickers": ctx.tickers,
        "weights": ctx.weights,
        "summary": generate_portfolio_summary(ctx),
    }


@router.get("/exposures")
def exposures() -> dict:
    ctx = build_portfolio_context()
    return {"rows": ctx.exposures.to_dicts()}


@router.get("/kpis")
def kpis() -> dict:
    ctx = build_portfolio_context()
    k = ctx.kpis
    return {
        "sharpe": k.sharpe,
        "max_drawdown": k.max_drawdown,
        "var_95": k.var_95,
        "experiment_id": load_app_config().research.meta.experiment_id,
    }
