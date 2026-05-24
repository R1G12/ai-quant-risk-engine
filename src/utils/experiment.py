"""Research experiment tracking (DVCLive + manifests)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dvclive import Live

from src.utils.paths import PROJECT_ROOT, ensure_dir


def _git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_manifest(
    manifest_path: Path,
    *,
    experiment_id: str,
    stage: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    outputs: list[str],
) -> None:
    """Write experiment manifest YAML."""
    ensure_dir(manifest_path.parent)
    doc = {
        "experiment_id": experiment_id,
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "params": params,
        "metrics": metrics,
        "outputs": outputs,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, default_flow_style=False)


def research_live(metrics_dir: Path, *, enable_dvc_exp: bool = True) -> Live:
    """DVCLive for Phase 4 research stages."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return Live(
        dir=metrics_dir,
        save_dvc_exp=enable_dvc_exp,
        dvcyaml=enable_dvc_exp,
    )


def log_research_metrics(
    metrics_dir: Path,
    metrics: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    experiment_id: str = "baseline",
    stage: str = "unknown",
    params: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
    enable_dvc_exp: bool = True,
) -> None:
    """Log metrics with optional experiment manifest."""
    with research_live(metrics_dir, enable_dvc_exp=enable_dvc_exp) as live:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                live.log_metric(key, value)
            else:
                live.log_metric(key, str(value))

    meta_path = metrics_dir / "metrics.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    if manifest_path is not None:
        write_manifest(
            manifest_path,
            experiment_id=experiment_id,
            stage=stage,
            params=params or {},
            metrics=metrics,
            outputs=outputs or [],
        )
