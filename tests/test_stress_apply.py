"""Stress shock application tests."""

from src.simulation.stress_testing.apply import apply_shock


def test_drift_shift() -> None:
    mu, sigma = apply_shock(0.1, 0.2, {"drift_shift": -0.01, "vol_multiplier": 1.0})
    assert abs(mu - 0.09) < 1e-12
    assert sigma == 0.2


def test_defaults_unchanged() -> None:
    mu, sigma = apply_shock(0.1, 0.2, {})
    assert mu == 0.1
    assert sigma == 0.2
