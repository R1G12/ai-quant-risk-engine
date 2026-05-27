"""Market regime commentary."""

from __future__ import annotations

import polars as pl

from src.copilot.context.builder import PortfolioContext
from src.copilot.explanations.risk import Explanation


def explain_regime(ctx: PortfolioContext) -> Explanation:
    """Summarize current HMM regime and recent transitions."""
    evidence: dict[str, object] = {}

    if ctx.regimes is None or ctx.regimes.height < 2:
        return Explanation(
            title="Volatility regime",
            summary="Regime labels unavailable. Run Phase 3 portfolio metrics stage.",
            evidence=evidence,
            confidence="low",
        )

    reg = ctx.regimes.sort("timestamp")
    latest = reg.tail(1).row(0, named=True)
    label_col = "regime_label" if "regime_label" in reg.columns else "regime"
    current = latest.get(label_col, "unknown")
    evidence["current_regime"] = current

    labels = reg[label_col].to_list()
    transitions = sum(1 for a, b in zip(labels[:-1], labels[1:], strict=False) if a != b)
    evidence["transition_count"] = transitions

    summary = (
        f"Current volatility regime: **{current}** (as of {latest.get('timestamp')}).\n\n"
        f"Observed **{transitions}** regime switch(es) in the labeled history. "
        "High transition counts suggest unstable macro/vol clustering; "
        "consider regime-conditioned simulation outputs in the research dashboard."
    )

    return Explanation(
        title="Volatility regime commentary",
        summary=summary,
        evidence=evidence,
        confidence="high" if transitions == 0 else "medium",
    )
