"""Static portfolio attribution from weights and features."""

from __future__ import annotations

import polars as pl

from src.copilot.context.builder import PortfolioContext
from src.copilot.explanations.risk import Explanation
from src.utils.paths import RISK_DATASET_PATH


def attribution_summary(ctx: PortfolioContext) -> Explanation:
    """Weight-based exposure and return contribution proxy."""
    weights = ctx.weights
    evidence: dict[str, object] = {"weights": weights}

    if not RISK_DATASET_PATH.is_file():
        return Explanation(
            title="Portfolio attribution",
            summary="Risk dataset missing; run `merge_features`.",
            evidence=evidence,
            confidence="low",
        )

    stats = (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(list(weights.keys())))
        .group_by("ticker")
        .agg(pl.col("returns").mean().alias("mean_return"))
        .collect()
    )

    rows: list[dict[str, object]] = []
    for row in stats.iter_rows(named=True):
        t = row["ticker"]
        w = weights.get(t, 0.0)
        mu = float(row["mean_return"] or 0.0)
        rows.append({"ticker": t, "weight": w, "mean_return": mu, "contrib": w * mu})

    contrib = pl.DataFrame(rows).sort("contrib", descending=True)
    evidence["attribution_table"] = contrib.to_dicts()

    lines = [
        "**Static attribution proxy** (weight × mean return; not a full Brinson model):",
        "",
    ]
    for r in contrib.head(5).iter_rows(named=True):
        lines.append(
            f"- **{r['ticker']}**: weight {r['weight']:.1%}, "
            f"mean return {r['mean_return']:.4f}, contrib {r['contrib']:.4f}"
        )

    dominant = contrib.row(0, named=True) if contrib.height else None
    if dominant:
        lines.append(
            f"\nLargest contributor: **{dominant['ticker']}** "
            f"({dominant['contrib']:.4f} return units)."
        )

    return Explanation(
        title="Portfolio attribution",
        summary="\n".join(lines),
        evidence=evidence,
        confidence="medium",
    )
