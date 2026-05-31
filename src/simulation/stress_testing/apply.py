"""Apply stress shocks to simulation parameters."""

from __future__ import annotations


def apply_shock(
    mu: float,
    sigma: float,
    shock: dict[str, float],
) -> tuple[float, float]:
    """Apply vol_multiplier, drift_shift to base parameters."""
    vol_mult = float(shock.get("vol_multiplier", 1.0))
    drift = float(shock.get("drift_shift", 0.0))
    return mu + drift, sigma * vol_mult
