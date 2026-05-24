"""Monte Carlo VaR engine (seeded, future-ready)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class MonteCarloVaREngine:
    """Fit Gaussian params and simulate for VaR/CVaR."""

    mu: float
    sigma: float
    seed: int

    @classmethod
    def fit(cls, returns: pl.Series, seed: int = 42) -> MonteCarloVaREngine:
        r = returns.drop_nulls()
        return cls(mu=float(r.mean()), sigma=float(r.std()), seed=seed)

    def simulate(self, n_paths: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(self.mu, self.sigma, size=n_paths)

    def var(self, confidence: float, n_simulations: int = 10_000) -> float:
        sims = self.simulate(n_simulations)
        return float(np.quantile(sims, 1.0 - confidence))

    def cvar(self, confidence: float, n_simulations: int = 10_000) -> float:
        sims = self.simulate(n_simulations)
        var = np.quantile(sims, 1.0 - confidence)
        tail = sims[sims <= var]
        return float(tail.mean()) if len(tail) else float("nan")
