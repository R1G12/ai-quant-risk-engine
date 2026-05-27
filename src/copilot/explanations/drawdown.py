"""Drawdown attribution explanations."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.copilot.context.builder import PortfolioContext
from src.copilot.explanations.risk import Explanation
from src.utils.config import AppConfig
from src.utils.paths import RESEARCH_BACKTESTS_DIR


def explain_drawdown(ctx: PortfolioContext, app: AppConfig | None = None) -> Explanation:
    """Explain latest drawdown using backtest equity and holding weights."""
    from src.utils.config import load_app_config

    app = app or load_app_config()
    exp = ctx.experiment_id
    path = RESEARCH_BACKTESTS_DIR / f"experiment_id={exp}" / "equity_curve.parquet"
    evidence: dict[str, object] = {"experiment_id": exp}

    if not path.is_file():
        return Explanation(
            title="Drawdown drivers",
            summary="Backtest equity curve not found. Run Phase 4 `run_backtests` first.",
            evidence=evidence,
            confidence="low",
        )

    eq = pl.read_parquet(path).sort("timestamp")
    if "drawdown" not in eq.columns or eq.height == 0:
        return Explanation(
            title="Drawdown drivers",
            summary="Equity curve missing drawdown column.",
            evidence=evidence,
            confidence="low",
        )

    worst_idx = eq["drawdown"].arg_min()
    worst_row = eq.row(worst_idx, named=True)
    evidence["worst_drawdown"] = worst_row

    top_weight = max(ctx.weights.items(), key=lambda x: x[1])
    summary = (
        f"Maximum drawdown in the backtest window: **{worst_row['drawdown']:.2%}** "
        f"around **{worst_row['timestamp']}**.\n\n"
        f"Largest static weight: **{top_weight[0]}** at {top_weight[1]:.1%}. "
        "Backtest uses fixed Phase 3 weights; intraday/asset-level P&L attribution "
        "requires position-level data not yet in this MVP."
    )

    return Explanation(
        title="What drove drawdown",
        summary=summary,
        evidence=evidence,
        confidence="medium",
    )
