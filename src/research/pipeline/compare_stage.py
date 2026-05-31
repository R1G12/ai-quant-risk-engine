"""DVC: compare_experiments."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from src.utils.config import load_app_config
from src.utils.experiment import log_research_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.logger import get_logger
from src.utils.paths import EXPERIMENTS_DIR, METRICS_DIR, RESEARCH_COMPARISONS_PATH, ensure_dir

LOGGER = get_logger(__name__)


def _load_manifests(root: Path) -> pl.DataFrame:
    rows: list[dict] = []
    for manifest in root.rglob("manifest.yaml"):
        with manifest.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        metrics = doc.get("metrics", {})
        params = doc.get("params", {})
        flat_params = {f"param_{k}": v for k, v in params.items()}
        flat_metrics = {f"metric_{k}": v for k, v in metrics.items()}
        rows.append(
            {
                "experiment_id": doc.get("experiment_id", "unknown"),
                "stage": doc.get("stage", "unknown"),
                "git_revision": doc.get("git_revision", "unknown"),
                "manifest_path": str(manifest),
                **flat_params,
                **flat_metrics,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def run() -> None:
    app = load_app_config()
    ensure_dir(RESEARCH_COMPARISONS_PATH.parent)
    comparison = _load_manifests(EXPERIMENTS_DIR)
    sink_lazy_parquet(comparison, RESEARCH_COMPARISONS_PATH, compression=app.market.compression)

    log_research_metrics(
        METRICS_DIR / "research_compare",
        {"n_manifests": comparison.height},
        stage="compare_experiments",
        enable_dvc_exp=app.research.meta.enable_dvc_experiments,
    )
    LOGGER.info("Experiment comparison written", extra={"manifests": comparison.height})


if __name__ == "__main__":
    run()
