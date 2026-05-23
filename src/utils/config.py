"""Configuration loader with YAML merge and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(file_path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict (empty dict if missing)."""
    if not file_path.is_file():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """Runtime configuration for pipeline stages."""

    batch_size: int = 32
    max_seq_length: int = 128
    seed: int = 42
    model_name: str = "ProsusAI/finbert"


def load_config() -> Config:
    """Load configuration from YAML files and environment variables.

    Order of precedence (low to high):
        1. configs/base.yaml
        2. configs/dvc_params.yaml
        3. configs/finbert.yaml
        4. PROJECT_* environment variables
    """
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    base_cfg = _load_yaml(config_dir / "base.yaml")
    dvc_cfg = _load_yaml(config_dir / "dvc_params.yaml")
    finbert_cfg = _load_yaml(config_dir / "finbert.yaml")

    merged: dict[str, Any] = {**base_cfg, **dvc_cfg, **finbert_cfg}

    batch = int(os.getenv("PROJECT_BATCH_SIZE", merged.get("batch_size", 32)))
    seq = int(os.getenv("PROJECT_MAX_SEQ_LENGTH", merged.get("max_seq_length", 128)))
    seed = int(os.getenv("PROJECT_SEED", merged.get("seed", 42)))
    model = os.getenv("PROJECT_MODEL_NAME", merged.get("model_name", "ProsusAI/finbert"))

    return Config(
        batch_size=batch,
        max_seq_length=seq,
        seed=seed,
        model_name=str(model),
    )
