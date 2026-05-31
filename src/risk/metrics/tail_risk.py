"""Tail risk metric aggregation."""

from __future__ import annotations

import polars as pl

from src.risk.cvar.expected_shortfall import historical_cvar, parametric_cvar
from src.risk.var.historical import historical_var
from src.risk.var.monte_carlo import MonteCarloVaREngine
from src.risk.var.parametric import parametric_var
from src.utils.config import VarRiskConfig


def compute_var_table(returns: pl.Series, cfg: VarRiskConfig) -> pl.DataFrame:
    """VaR for all configured methods and confidence levels."""
    rows = []
    for conf in cfg.confidence_levels:
        if cfg.methods.get("historical", True):
            rows.append(
                {
                    "method": "historical",
                    "confidence": conf,
                    "var": historical_var(returns, conf),
                }
            )
        if cfg.methods.get("parametric", True):
            rows.append(
                {
                    "method": "parametric",
                    "confidence": conf,
                    "var": parametric_var(returns, conf),
                }
            )
        if cfg.methods.get("monte_carlo", True):
            engine = MonteCarloVaREngine.fit(returns, seed=cfg.seed)
            rows.append(
                {
                    "method": "monte_carlo",
                    "confidence": conf,
                    "var": engine.var(conf, cfg.n_simulations),
                }
            )
    return pl.DataFrame(rows)


def compute_cvar_table(returns: pl.Series, cfg: VarRiskConfig) -> pl.DataFrame:
    """CVaR for historical, parametric, and Monte Carlo."""
    rows = []
    for conf in cfg.confidence_levels:
        rows.append(
            {
                "method": "historical",
                "confidence": conf,
                "cvar": historical_cvar(returns, conf),
            }
        )
        rows.append(
            {
                "method": "parametric",
                "confidence": conf,
                "cvar": parametric_cvar(returns, conf),
            }
        )
        if cfg.methods.get("monte_carlo", True):
            engine = MonteCarloVaREngine.fit(returns, seed=cfg.seed)
            rows.append(
                {
                    "method": "monte_carlo",
                    "confidence": conf,
                    "cvar": engine.cvar(conf, cfg.n_simulations),
                }
            )
    return pl.DataFrame(rows)
