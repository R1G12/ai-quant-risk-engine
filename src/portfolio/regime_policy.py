"""HMM regime policy for scaling sentiment influence in optimization."""

from __future__ import annotations

import polars as pl

from src.utils.config import AppConfig
from src.utils.paths import RISK_REGIMES_PATH

DEFAULT_REGIME_MIX: dict[str, float] = {"low": 0.4, "mid": 0.75, "high": 1.0}


def latest_hmm_regime_label(_app: AppConfig) -> str | None:
    """Return the latest ``regime_label`` from portfolio HMM output, or None if unavailable."""
    if not RISK_REGIMES_PATH.is_file():
        return None
    df = pl.read_parquet(RISK_REGIMES_PATH).sort("timestamp")
    if df.is_empty() or "regime_label" not in df.columns:
        return None
    latest = df["regime_label"][-1]
    return str(latest) if latest is not None else None


def regime_sentiment_multiplier(
    regime: str | None,
    mix: dict[str, float] | None = None,
) -> float:
    """Scale sentiment blend/tilt by HMM regime (``low`` / ``mid`` / ``high``)."""
    policy = mix or DEFAULT_REGIME_MIX
    key = (regime or "mid").strip().lower()
    if key in policy:
        return float(policy[key])
    return float(policy.get("mid", 0.75))
