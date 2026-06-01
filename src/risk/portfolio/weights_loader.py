"""Load portfolio weights: optimized (Phase 3) with holdings fallback."""

from __future__ import annotations

import polars as pl

from src.risk.portfolio.holdings import load_weights
from src.utils.config import AppConfig
from src.utils.paths import RISK_OPT_WEIGHTS_PATH


def load_optimization_weights(
    app: AppConfig,
    *,
    portfolio: str | None = None,
) -> tuple[dict[str, float], str]:
    """Load weights from optimal_weights.parquet or fallback to holdings.

    Returns (weights dict, source label for UI).
    """
    source_name = portfolio or app.research.backtest.weight_source or "max_sharpe"

    if RISK_OPT_WEIGHTS_PATH.is_file():
        df = pl.read_parquet(RISK_OPT_WEIGHTS_PATH)
        df = df.filter(pl.col("asset") != "_stats_")
        sub = df.filter(pl.col("portfolio") == source_name)
        if sub.height:
            weights = dict(zip(sub["asset"].to_list(), sub["weight"].to_list(), strict=False))
            return weights, source_name

    return load_weights(app), "holdings (fallback)"
