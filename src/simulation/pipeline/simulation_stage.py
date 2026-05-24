"""DVC: generate_simulations."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.risk.portfolio.holdings import load_weights
from src.simulation.monte_carlo.engine import run_gbm_simulation, run_multivariate_simulation
from src.simulation.pipeline._calibration import calibrate_gbm, calibrate_multivariate
from src.simulation.portfolio_paths.wealth import weights_from_dict
from src.simulation.regimes.params import load_regime_params
from src.simulation.validation.sanity import validate_path_summaries
from src.utils.config import SimulationConfig, load_app_config
from src.utils.experiment import log_research_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.logger import get_logger
from src.utils.paths import (
    EXPERIMENTS_SIMULATIONS_DIR,
    METRICS_DIR,
    RESEARCH_SIMULATIONS_DIR,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def _write_sim_output(
    base: Path,
    exp_id: str,
    sim_type: str,
    summary: pl.DataFrame,
    tail: pl.DataFrame,
    compression: str,
) -> None:
    out_dir = base / f"experiment_id={exp_id}" / f"simulation_type={sim_type}"
    ensure_dir(out_dir)
    summary = summary.with_columns(
        pl.lit(exp_id).alias("experiment_id"),
        pl.lit(sim_type).alias("simulation_type"),
    )
    tail = tail.with_columns(
        pl.lit(exp_id).alias("experiment_id"),
        pl.lit(sim_type).alias("simulation_type"),
    )
    sink_lazy_parquet(summary, out_dir / "paths_summary.parquet", compression=compression)
    sink_lazy_parquet(tail, out_dir / "tail_metrics.parquet", compression=compression)


def run() -> None:
    app = load_app_config()
    cfg = app.research.simulation
    exp_id = app.research.meta.experiment_id
    ensure_dir(RESEARCH_SIMULATIONS_DIR)

    outputs: list[str] = []

    if "gbm" in cfg.simulation_types:
        mu, sigma = calibrate_gbm(app)
        summary, tail = run_gbm_simulation(mu, sigma, cfg)
        validate_path_summaries(summary)
        _write_sim_output(RESEARCH_SIMULATIONS_DIR, exp_id, "gbm", summary, tail, app.market.compression)
        outputs.append(str(RESEARCH_SIMULATIONS_DIR / f"experiment_id={exp_id}" / "simulation_type=gbm"))

    if "multivariate" in cfg.simulation_types:
        mu, cov, tickers = calibrate_multivariate(app)
        w = weights_from_dict(load_weights(app), tickers)
        summary, tail = run_multivariate_simulation(mu, cov, w, cfg)
        validate_path_summaries(summary)
        _write_sim_output(
            RESEARCH_SIMULATIONS_DIR, exp_id, "multivariate", summary, tail, app.market.compression
        )
        outputs.append(
            str(RESEARCH_SIMULATIONS_DIR / f"experiment_id={exp_id}" / "simulation_type=multivariate")
        )

    if "regime_gbm" in cfg.simulation_types:
        regime_params = load_regime_params(app)
        rows = []
        tail_rows = []
        for i, (label, p) in enumerate(regime_params.items()):
            sub_cfg = SimulationConfig(
                n_paths=max(cfg.n_paths // max(len(regime_params), 1), 100),
                horizon_days=cfg.horizon_days,
                seed=cfg.seed + i + 10,
                dt=cfg.dt,
                save_paths=cfg.save_paths,
                simulation_types=cfg.simulation_types,
            )
            s, t = run_gbm_simulation(p["mu"], p["sigma"], sub_cfg)
            s = s.with_columns(pl.lit(label).alias("regime"))
            t = t.with_columns(pl.lit(label).alias("regime"))
            rows.append(s)
            tail_rows.append(t)
        summary = pl.concat(rows, how="diagonal_relaxed")
        tail = pl.concat(tail_rows, how="diagonal_relaxed")
        _write_sim_output(RESEARCH_SIMULATIONS_DIR, exp_id, "regime_gbm", summary, tail, app.market.compression)
        outputs.append(
            str(RESEARCH_SIMULATIONS_DIR / f"experiment_id={exp_id}" / "simulation_type=regime_gbm")
        )

    manifest = EXPERIMENTS_SIMULATIONS_DIR / exp_id / "manifest.yaml"
    log_research_metrics(
        METRICS_DIR / "research_simulations",
        {"n_simulation_types": len(cfg.simulation_types), "n_paths": cfg.n_paths},
        manifest_path=manifest,
        experiment_id=exp_id,
        stage="generate_simulations",
        params={"n_paths": cfg.n_paths, "horizon_days": cfg.horizon_days, "seed": cfg.seed},
        outputs=outputs,
        enable_dvc_exp=app.research.meta.enable_dvc_experiments,
    )
    LOGGER.info("Simulations written", extra={"experiment_id": exp_id})


if __name__ == "__main__":
    run()
