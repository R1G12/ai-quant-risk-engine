"""DVC: evaluate_performance."""

from __future__ import annotations

import polars as pl

from src.backtesting.metrics.performance import max_drawdown, sharpe_ratio
from src.utils.config import load_app_config
from src.utils.experiment import log_research_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.logger import get_logger
from src.utils.paths import (
    METRICS_DIR,
    RESEARCH_BACKTESTS_DIR,
    RESEARCH_EVALUATION_DIR,
    RESEARCH_PERFORMANCE_PATH,
    RESEARCH_SIMULATIONS_DIR,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    exp_id = app.research.meta.experiment_id
    ensure_dir(RESEARCH_EVALUATION_DIR)

    rows: list[dict] = []

    bt_dir = RESEARCH_BACKTESTS_DIR / f"experiment_id={exp_id}"
    equity_path = bt_dir / "equity_curve.parquet"
    if equity_path.is_file():
        eq = pl.read_parquet(equity_path)
        rows.append(
            {
                "experiment_id": exp_id,
                "source": "backtest",
                "sharpe": sharpe_ratio(
                    eq["portfolio_return"], app.features.risk_free_rate, app.features.annualization_factor
                ),
                "max_drawdown": max_drawdown(eq["equity"]),
                "n_obs": eq.height,
            }
        )

    sim_base = RESEARCH_SIMULATIONS_DIR / f"experiment_id={exp_id}"
    if sim_base.is_dir():
        for sim_dir in sim_base.glob("simulation_type=*"):
            tail_path = sim_dir / "tail_metrics.parquet"
            if not tail_path.is_file():
                continue
            tail = pl.read_parquet(tail_path)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "source": f"simulation:{sim_dir.name.split('=', 1)[-1]}",
                    "var_95": float(tail.filter(pl.col("confidence") == 0.95)["var"][0])
                    if tail.height
                    else None,
                    "cvar_95": float(tail.filter(pl.col("confidence") == 0.95)["cvar"][0])
                    if tail.height
                    else None,
                    "mean_return": float(tail["mean_return"][0]) if tail.height else None,
                    "volatility": float(tail["volatility"][0]) if tail.height else None,
                }
            )

    summary = pl.DataFrame(rows) if rows else pl.DataFrame({"experiment_id": [exp_id]})
    sink_lazy_parquet(summary, RESEARCH_PERFORMANCE_PATH, compression=app.market.compression)

    log_research_metrics(
        METRICS_DIR / "research_evaluation",
        {"n_rows": summary.height},
        experiment_id=exp_id,
        stage="evaluate_performance",
        enable_dvc_exp=app.research.meta.enable_dvc_experiments,
    )
    LOGGER.info("Performance evaluation written", extra={"rows": summary.height})


if __name__ == "__main__":
    run()
