"""DVC: run_scenario_analysis."""

from __future__ import annotations

import polars as pl

from src.utils.config import load_app_config
from src.utils.experiment import log_research_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.logger import get_logger
from src.utils.paths import METRICS_DIR, RESEARCH_SCENARIOS_DIR, ensure_dir

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    exp_id = app.research.meta.experiment_id
    ensure_dir(RESEARCH_SCENARIOS_DIR)

    from src.simulation.pipeline._calibration import calibrate_gbm
    from src.simulation.monte_carlo.engine import run_gbm_simulation
    from src.simulation.stress_testing.apply import apply_shock

    base_mu, base_sigma = calibrate_gbm(app)
    cfg = app.research.simulation
    rows = []
    for name, shock in app.research.scenarios.scenarios.items():
        mu, sigma = apply_shock(base_mu, base_sigma, shock)
        _, tail = run_gbm_simulation(mu, sigma, cfg)
        rows.append(
            tail.with_columns(pl.lit(name).alias("scenario")).select(
                ["scenario", "confidence", "var", "cvar", "mean_return", "volatility"]
            )
        )

    comparison = pl.concat(rows, how="diagonal_relaxed").with_columns(
        pl.lit(exp_id).alias("experiment_id")
    )
    out = RESEARCH_SCENARIOS_DIR / f"experiment_id={exp_id}" / "scenario_comparison.parquet"
    ensure_dir(out.parent)
    sink_lazy_parquet(comparison, out, compression=app.market.compression)

    log_research_metrics(
        METRICS_DIR / "research_scenarios",
        {"n_scenarios": comparison.height},
        experiment_id=exp_id,
        stage="run_scenario_analysis",
        enable_dvc_exp=app.research.meta.enable_dvc_experiments,
    )
    LOGGER.info("Scenario analysis written", extra={"path": str(out)})


if __name__ == "__main__":
    run()
