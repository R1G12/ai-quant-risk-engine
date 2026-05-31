"""DVC: run_stress_tests."""

from __future__ import annotations

import polars as pl

from src.simulation.monte_carlo.engine import run_gbm_simulation
from src.simulation.pipeline._calibration import calibrate_gbm
from src.simulation.stress_testing.apply import apply_shock
from src.utils.config import load_app_config
from src.utils.experiment import log_research_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.logger import get_logger
from src.utils.paths import (
    EXPERIMENTS_STRESS_DIR,
    METRICS_DIR,
    RESEARCH_STRESS_DIR,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    exp_id = app.research.meta.experiment_id
    cfg = app.research.simulation
    scenarios = app.research.scenarios.scenarios
    base_mu, base_sigma = calibrate_gbm(app)
    ensure_dir(RESEARCH_STRESS_DIR)

    rows = []
    for name, shock in scenarios.items():
        mu, sigma = apply_shock(base_mu, base_sigma, shock)
        _, tail = run_gbm_simulation(mu, sigma, cfg)
        tail = tail.with_columns(
            pl.lit(name).alias("scenario"),
            pl.lit(exp_id).alias("experiment_id"),
        )
        rows.append(tail)

    combined = pl.concat(rows, how="diagonal_relaxed")
    out_dir = RESEARCH_STRESS_DIR / f"experiment_id={exp_id}"
    ensure_dir(out_dir)
    sink_lazy_parquet(combined, out_dir / "stress_metrics.parquet", compression=app.market.compression)

    worst_var = float(combined["var"].min()) if combined.height else 0.0
    log_research_metrics(
        METRICS_DIR / "research_stress",
        {"worst_stress_var": worst_var, "n_scenarios": len(scenarios)},
        manifest_path=EXPERIMENTS_STRESS_DIR / exp_id / "manifest.yaml",
        experiment_id=exp_id,
        stage="run_stress_tests",
        params={"scenarios": list(scenarios.keys())},
        outputs=[str(out_dir)],
        enable_dvc_exp=app.research.meta.enable_dvc_experiments,
    )
    LOGGER.info("Stress tests complete", extra={"worst_var": worst_var})


if __name__ == "__main__":
    run()
