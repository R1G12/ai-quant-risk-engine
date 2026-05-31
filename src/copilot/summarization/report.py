"""Generate NL portfolio summaries from structured context."""

from __future__ import annotations

from src.copilot.context.builder import PortfolioContext


def generate_portfolio_summary(ctx: PortfolioContext) -> str:
    """Executive summary paragraph for reports and copilot."""
    k = ctx.kpis
    parts = [
        f"Portfolio intelligence snapshot for experiment **{ctx.experiment_id}** "
        f"(as of {ctx.as_of}).",
        f"Holdings: {', '.join(ctx.tickers)}.",
    ]
    if k.sharpe is not None:
        parts.append(f"Window Sharpe: **{k.sharpe:.2f}**.")
    if k.max_drawdown is not None:
        parts.append(f"Max drawdown: **{k.max_drawdown:.1%}**.")
    if k.var_95 is not None:
        parts.append(f"VaR 95%: **{k.var_95:.2%}**.")
    parts.append(
        "All metrics are derived from reproducible DVC artifacts; "
        "human validation is required before investment decisions."
    )
    return " ".join(parts)
