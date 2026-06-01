"""HMM regime policy for scaling sentiment influence in optimization."""

from __future__ import annotations

import polars as pl

from src.utils.config import AppConfig
from src.utils.paths import RISK_REGIMES_PATH

DEFAULT_REGIME_MIX: dict[str, float] = {"low": 0.4, "mid": 0.75, "high": 1.0}


def latest_hmm_regime_label(_app: AppConfig) -> str | None:
    """Return the latest regime_label from portfolio regimes parquet."""
    if not RISK_REGIMES_PATH.is_file():
        return None
    regimes = pl.read_parquet(RISK_REGIMES_PATH).sort("timestamp")
    if regimes.is_empty() or "regime_label" not in regimes.columns:
        return None
    latest = regimes["regime_label"][-1]
    return str(latest) if latest is not None else None


def regime_sentiment_multiplier(
    label: str | None,
    mix: dict[str, float] | None = None,
) -> float:
    """Scale sentiment blend by HMM regime (default 1.0 when unknown)."""
    if label is None:
        return 1.0
    table = mix or DEFAULT_REGIME_MIX
    return float(table.get(label, 1.0))
