"""Phase 4 configuration tests."""

from src.utils.config import load_app_config


def test_research_config_loaded() -> None:
    app = load_app_config()
    assert app.research.meta.experiment_id == "baseline"
    assert app.research.simulation.n_paths >= 100
    assert "gbm" in app.research.simulation.simulation_types
    assert len(app.research.scenarios.scenarios) >= 1
    assert app.research.backtest.tc_bps > 0
