"""Stress testing tests."""

from src.simulation.monte_carlo.engine import run_gbm_simulation
from src.simulation.stress_testing.apply import apply_shock
from src.utils.config import SimulationConfig


def test_vol_multiplier_increases_sigma() -> None:
    mu, sigma = 0.05, 0.2
    shocked_mu, shocked_sigma = apply_shock(mu, sigma, {"vol_multiplier": 2.0})
    assert shocked_sigma == sigma * 2.0
    assert shocked_mu == mu


def test_stress_produces_worse_var() -> None:
    cfg = SimulationConfig(n_paths=500, horizon_days=40, seed=1)
    _, base_tail = run_gbm_simulation(0.05, 0.15, cfg)
    mu2, sigma2 = apply_shock(0.05, 0.15, {"vol_multiplier": 2.5})
    _, stress_tail = run_gbm_simulation(mu2, sigma2, cfg)
    assert stress_tail["var"][0] <= base_tail["var"][0]
