"""Read experiment YAML manifests for lineage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.paths import EXPERIMENTS_BACKTESTS_DIR, EXPERIMENTS_SIMULATIONS_DIR


@dataclass(frozen=True)
class ExperimentLineage:
    experiment_id: str
    stage: str
    params: dict
    outputs: list[str]
    path: Path


def _read_manifest(path: Path) -> ExperimentLineage | None:
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentLineage(
        experiment_id=str(data.get("experiment_id", path.parent.name)),
        stage=str(data.get("stage", "unknown")),
        params=dict(data.get("params", {})),
        outputs=list(data.get("outputs", [])),
        path=path,
    )


def load_experiment_lineage(experiment_id: str) -> list[ExperimentLineage]:
    rows: list[ExperimentLineage] = []
    for base in (EXPERIMENTS_SIMULATIONS_DIR, EXPERIMENTS_BACKTESTS_DIR):
        manifest = base / experiment_id / "manifest.yaml"
        item = _read_manifest(manifest)
        if item:
            rows.append(item)
    return rows
