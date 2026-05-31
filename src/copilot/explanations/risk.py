"""VaR and portfolio risk explanations."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.copilot.context.builder import PortfolioContext


@dataclass(frozen=True)
class Explanation:
    title: str
    summary: str
    evidence: dict[str, object]
    confidence: str = "medium"
    requires_human_review: bool = True


def explain_var_contributors(ctx: PortfolioContext) -> Explanation:
    """Identify assets with highest marginal correlation to portfolio tail risk."""
    weights = ctx.weights
    evidence: dict[str, object] = {"weights": weights}

    if ctx.correlations is None or not ctx.correlations.height:
        return Explanation(
            title="VaR contributors",
            summary="Correlation matrix not available. Run Phase 3 `generate_correlations` first.",
            evidence=evidence,
            confidence="low",
        )

    corr = ctx.correlations.filter(pl.col("metric") == "corr")
    if not corr.height:
        return Explanation(
            title="VaR contributors",
            summary="No pairwise correlations found in artifacts.",
            evidence=evidence,
            confidence="low",
        )

    # Weight-adjusted average correlation per asset (proxy for diversification benefit)
    scores: list[tuple[str, float]] = []
    for asset in weights:
        sub = corr.filter((pl.col("asset_i") == asset) | (pl.col("asset_j") == asset))
        if sub.height:
            avg_corr = float(sub["value"].mean())
            scores.append((asset, abs(avg_corr) * weights[asset]))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:3]
    evidence["top_correlation_contributors"] = top

    lines = [
        "Assets with **higher weight × |correlation|** tend to dominate portfolio tail risk "
        "(simplified marginal proxy; not full Euler VaR decomposition).",
        "",
    ]
    for ticker, score in top:
        lines.append(f"- **{ticker}**: score {score:.4f} (weight {weights[ticker]:.1%})")

    if ctx.var_table is not None and ctx.var_table.height:
        row = ctx.var_table.filter(pl.col("confidence") == 0.95).sort("var").head(1)
        if row.height:
            evidence["var_95_snapshot"] = row.row(0, named=True)
            lines.append(
                f"\nPortfolio VaR 95% ({row['method'][0]}): **{float(row['var'][0]):.2%}**"
            )

    return Explanation(
        title="VaR contributors",
        summary="\n".join(lines),
        evidence=evidence,
        confidence="medium",
    )


def explain_risk_increase(ctx: PortfolioContext) -> Explanation:
    """Explain factors that could increase portfolio risk."""
    bullets: list[str] = []
    evidence: dict[str, object] = {}

    if ctx.portfolio_metrics is not None and "ewma_vol" in ctx.portfolio_metrics.columns:
        pm = ctx.portfolio_metrics.sort("timestamp")
        if pm.height >= 2:
            last = float(pm["ewma_vol"][-1])
            prev = float(pm["ewma_vol"][-2])
            delta = last - prev
            evidence["ewma_vol_change"] = {"last": last, "prev": prev, "delta": delta}
            if delta > 0:
                bullets.append(
                    f"EWMA volatility **rose** ({prev:.2%} → {last:.2%}) on the latest observation."
                )
            else:
                bullets.append(
                    f"EWMA volatility **fell** ({prev:.2%} → {last:.2%}); risk may still be elevated vs history."
                )

    if ctx.regimes is not None and ctx.regimes.height:
        latest = ctx.regimes.sort("timestamp").tail(1)
        label = latest["regime_label"][0] if "regime_label" in latest.columns else "unknown"
        evidence["current_regime"] = str(label)
        bullets.append(f"HMM regime label: **{label}** (regime-conditioned vol may differ).")

    if ctx.sentiment_summary is not None and ctx.sentiment_summary.height:
        volatile = ctx.sentiment_summary.sort("return_vol", descending=True).head(1)
        if volatile.height:
            t = volatile["ticker"][0]
            bullets.append(f"**{t}** shows the highest return volatility among holdings in the merged dataset.")

    if not bullets:
        bullets.append(
            "Insufficient risk artifacts on disk. Re-run Phase 3+ pipelines (`dvc repro` through risk stages)."
        )

    return Explanation(
        title="Why portfolio risk may have increased",
        summary="\n".join(f"- {b}" for b in bullets),
        evidence=evidence,
        confidence="medium",
    )
