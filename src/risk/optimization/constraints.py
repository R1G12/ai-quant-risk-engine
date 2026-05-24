"""Optimization constraints for Markowitz problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PortfolioConstraints:
    """Weight bounds and budget constraint."""

    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    leverage_cap: float = 1.0

    def bounds(self, n_assets: int) -> list[tuple[float, float]]:
        lo = self.min_weight if self.long_only else -self.max_weight
        hi = min(self.max_weight, self.leverage_cap)
        return [(lo, hi)] * n_assets

    def budget_constraint(self) -> dict:
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
