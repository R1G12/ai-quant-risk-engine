"""VaR / CVaR tests."""

import numpy as np
import polars as pl
from scipy import stats

from src.risk.cvar.expected_shortfall import historical_cvar
from src.risk.var.historical import historical_var
from src.risk.var.monte_carlo import MonteCarloVaREngine
from src.risk.var.parametric import parametric_var


def test_parametric_var_near_gaussian_quantile() -> None:
    rng = np.random.default_rng(42)
    r = pl.Series("r", rng.normal(0.0, 0.02, size=5000))
    var = parametric_var(r, 0.95)
    expected = float(r.mean()) + stats.norm.ppf(0.05) * float(r.std())
    assert abs(var - expected) < 1e-6


def test_historical_cvar_more_extreme_than_var() -> None:
    r = pl.Series("r", list(np.random.default_rng(0).normal(0, 0.02, 1000)))
    conf = 0.95
    var = historical_var(r, conf)
    cvar = historical_cvar(r, conf)
    assert cvar <= var + 1e-9


def test_monte_carlo_engine_seed() -> None:
    r = pl.Series("r", [0.01, -0.02, 0.005, -0.01, 0.02] * 50)
    a = MonteCarloVaREngine.fit(r, seed=1).var(0.95, 1000)
    b = MonteCarloVaREngine.fit(r, seed=1).var(0.95, 1000)
    assert a == b
