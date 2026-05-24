"""GARCH volatility (arch package boundary)."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.utils.config import VolatilityRiskConfig


def fit_garch_volatility(
    returns: pl.Series,
    cfg: VolatilityRiskConfig,
) -> pl.Series:
    """Fit GARCH(p,q) on portfolio returns; return conditional volatility series.

    Uses arch on a collected return vector (approved non-Polars boundary).
    """
    from arch import arch_model

    r = returns.drop_nulls().to_numpy() * 100.0
    if len(r) < cfg.long_vol_window:
        return pl.Series("garch_vol", [None] * len(returns))
    model = arch_model(r, vol="Garch", p=cfg.garch_p, q=cfg.garch_q, rescale=False)
    fit = model.fit(disp="off")
    vol = fit.conditional_volatility / 100.0
    ann = cfg.annualization_factor**0.5
    vol_ann = vol * ann
    padded = [None] * (len(returns) - len(vol_ann)) + vol_ann.tolist()
    return pl.Series("garch_vol", padded)
