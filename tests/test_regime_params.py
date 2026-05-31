"""Regime parameter mapping tests."""

from src.simulation.regimes.params import load_regime_params
from src.utils.config import load_app_config


def test_load_regime_params_returns_dict() -> None:
    app = load_app_config()
    params = load_regime_params(app)
    assert isinstance(params, dict)
    assert len(params) >= 1
    for p in params.values():
        assert "mu" in p and "sigma" in p
