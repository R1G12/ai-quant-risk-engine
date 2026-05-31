"""GBM simulation tests."""

import numpy as np

from src.simulation.monte_carlo.engine import run_gbm_simulation
from src.simulation.stochastic_processes.gbm import simulate_gbm_paths
from src.simulation.validation.sensitivity import assert_sigma_sensitivity
from src.utils.config import SimulationConfig


def test_gbm_reproducible_with_seed() -> None:
    a = simulate_gbm_paths(0.05, 0.2, n_paths=100, n_steps=10, dt=1 / 252, seed=99)
    b = simulate_gbm_paths(0.05, 0.2, n_paths=100, n_steps=10, dt=1 / 252, seed=99)
    assert np.allclose(a, b)


def test_gbm_path_shape() -> None:
    paths = simulate_gbm_paths(0.0, 0.15, n_paths=50, n_steps=20, dt=1 / 252, seed=1)
    assert paths.shape == (50, 21)
    assert np.all(paths[:, 0] == 1.0)


def test_run_gbm_simulation_summaries() -> None:
    cfg = SimulationConfig(n_paths=200, horizon_days=30, seed=42)
    summary, tail = run_gbm_simulation(0.08, 0.2, cfg)
    assert summary.height == 200
    assert "max_drawdown" in summary.columns
    assert tail.height == 1


def test_sigma_sensitivity() -> None:
    assert_sigma_sensitivity(0.05, 0.2, seed=7)
