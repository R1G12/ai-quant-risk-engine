"""Orchestration CLI helpers (invoke DVC workflows)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from src.orchestration.workflows import WORKFLOWS
from src.utils.logger import get_logger
from src.utils.paths import PROJECT_ROOT

LOGGER = get_logger(__name__)


def run_workflow(
    name: str,
    *,
    market_source: str | None = None,
    dry_run: bool = False,
) -> int:
    if name not in WORKFLOWS:
        raise ValueError(f"Unknown workflow {name!r}; choose from {sorted(WORKFLOWS)}")
    spec = WORKFLOWS[name]
    env = os.environ.copy()
    if market_source:
        env["MARKET_SOURCE"] = market_source
    if spec.dvc_stages:
        cmd = ["dvc", "repro", *spec.dvc_stages]
    else:
        cmd = ["dvc", "repro"]
    LOGGER.info("Orchestrating workflow", extra={"workflow": name, "cmd": " ".join(cmd)})
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False).returncode
