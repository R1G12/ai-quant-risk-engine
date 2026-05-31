"""Route natural-language questions to explainers (rule-based MVP)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

from src.copilot.attribution.portfolio import attribution_summary
from src.copilot.context.builder import PortfolioContext, build_portfolio_context
from src.copilot.explanations.drawdown import explain_drawdown
from src.copilot.explanations.regime import explain_regime
from src.copilot.explanations.risk import (
    Explanation,
    explain_risk_increase,
    explain_var_contributors,
)
from src.copilot.summarization.report import generate_portfolio_summary
from src.utils.config import AppConfig, load_app_config


@dataclass(frozen=True)
class CopilotResponse:
    question: str
    answer: str
    title: str
    evidence: dict[str, object]
    confidence: str
    requires_human_review: bool = True


def _match(question: str, patterns: list[str]) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in patterns)


class CopilotEngine:
    """Portfolio Q&A assistant (deterministic, auditable)."""

    def __init__(self, app: AppConfig | None = None) -> None:
        self._app = app or load_app_config()

    def ask(self, question: str, ctx: PortfolioContext | None = None) -> CopilotResponse:
        ctx = ctx or build_portfolio_context(self._app)
        explanation = self._route(question, ctx)
        return CopilotResponse(
            question=question,
            answer=explanation.summary,
            title=explanation.title,
            evidence=explanation.evidence,
            confidence=explanation.confidence,
            requires_human_review=explanation.requires_human_review,
        )

    def _route(self, question: str, ctx: PortfolioContext) -> Explanation:
        if _match(
            question,
            [r"var", r"value.at.risk", r"tail risk", r"contribute.*var", r"which assets"],
        ):
            if _match(question, [r"contribut", r"which asset", r"dominat"]):
                return explain_var_contributors(ctx)
            return explain_var_contributors(ctx)

        if _match(question, [r"drawdown", r"lost", r"decline today", r"drove.*down"]):
            return explain_drawdown(ctx, self._app)

        if _match(question, [r"regime", r"volatility regime", r"hmm", r"market state"]):
            return explain_regime(ctx)

        if _match(question, [r"attribution", r"contribution", r"what drove return"]):
            return attribution_summary(ctx)

        if _match(question, [r"sentiment", r"bullish", r"news"]):
            return self._sentiment_explanation(ctx)

        if _match(question, [r"correlation", r"corr structure", r"diversification"]):
            return self._correlation_explanation(ctx)

        if _match(question, [r"exposure", r"sector", r"weight", r"concentration"]):
            return self._exposure_explanation(ctx)

        if _match(question, [r"risk increase", r"why.*risk", r"higher risk"]):
            return explain_risk_increase(ctx)

        if _match(question, [r"summar", r"overview", r"executive"]):
            text = generate_portfolio_summary(ctx)
            return Explanation(
                title="Portfolio summary",
                summary=text,
                evidence={"tickers": ctx.tickers, "kpis": ctx.kpis.__dict__},
                confidence="high",
            )

        return Explanation(
            title="Copilot help",
            summary=(
                "I can explain: **VaR contributors**, **drawdown**, **regime**, "
                "**attribution**, **sentiment sensitivity**, **correlations**, "
                "**exposures**, and **risk increases**. "
                "Try: 'Which assets contribute most to VaR?' or 'What is the current volatility regime?'"
            ),
            evidence={},
            confidence="high",
            requires_human_review=False,
        )

    def _sentiment_explanation(self, ctx: PortfolioContext) -> Explanation:
        if ctx.sentiment_summary is None or not ctx.sentiment_summary.height:
            return Explanation(
                title="Sentiment sensitivity",
                summary="Sentiment aggregates unavailable in risk dataset.",
                evidence={},
                confidence="low",
            )
        top = ctx.sentiment_summary.sort("avg_bullish_ratio", descending=True).head(3)
        lines = ["Holdings ranked by average **bullish_ratio** in merged features:"]
        for row in top.iter_rows(named=True):
            bull = row["avg_bullish_ratio"]
            bull_s = f"{float(bull):.2f}" if bull is not None else "n/a"
            rv = row.get("return_vol")
            rv_s = f"{float(rv):.4f}" if rv is not None else "n/a"
            lines.append(f"- **{row['ticker']}**: bullish_ratio {bull_s}, return vol {rv_s}")
        return Explanation(
            title="Sentiment-sensitive positions",
            summary="\n".join(lines),
            evidence={"top_sentiment": top.to_dicts()},
            confidence="medium",
        )

    def _correlation_explanation(self, ctx: PortfolioContext) -> Explanation:
        if ctx.correlations is None:
            return Explanation(
                title="Correlation structure",
                summary="Run `generate_correlations` to materialize correlation artifacts.",
                evidence={},
                confidence="low",
            )
        corr = ctx.correlations
        if "metric" in corr.columns:
            corr = corr.filter(pl.col("metric") == "corr")
        pairs = corr.sort("value", descending=True).head(5) if corr.height else corr
        lines = ["Largest pairwise correlations (from latest artifact):"]
        for row in pairs.iter_rows(named=True):
            lines.append(
                f"- {row.get('asset_i')} / {row.get('asset_j')}: {row.get('value', 0):.3f}"
            )
        return Explanation(
            title="Correlation structure",
            summary="\n".join(lines) if lines else "No correlation rows.",
            evidence={"top_pairs": pairs.to_dicts() if pairs.height else []},
            confidence="medium",
        )

    def _exposure_explanation(self, ctx: PortfolioContext) -> Explanation:
        exp = ctx.exposures.sort("weight", descending=True)
        lines = ["**Portfolio weights (static holdings):**", ""]
        for row in exp.iter_rows(named=True):
            lines.append(f"- **{row['asset']}**: {row['weight']:.1%}")
        hhi = sum(w * w for w in ctx.weights.values())
        lines.append(f"\nConcentration (HHI): **{hhi:.3f}** (1.0 = single name).")
        return Explanation(
            title="Exposure breakdown",
            summary="\n".join(lines),
            evidence={"hhi": hhi, "weights": ctx.weights},
            confidence="high",
        )
