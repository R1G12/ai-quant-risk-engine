'''config.py – configuration loader for the quant risk engine'''

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseSettings, Field, validator

# Helper to load a YAML file into a dict
def _load_yaml(file_path: Path) -> Dict[str, Any]:
    with file_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

class BaseConfig(BaseSettings):
    """Base configuration model.

    All fields are optional and can be overridden by environment variables
    using the ``PROJECT_`` prefix (e.g. ``PROJECT_BATCH_SIZE``).
    """

    # Example generic settings – extend as needed
    batch_size: int = Field(32, description="Batch size for model inference")
    max_seq_length: int = Field(128, description="Maximum token length for FinBERT")
    seed: int = Field(42, description="Random seed for reproducibility")

    class Config:
        env_prefix = "PROJECT_"
        case_sensitive = False

    @validator("batch_size", "max_seq_length", "seed")
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

def load_config() -> BaseConfig:
    """Load the hierarchical configuration.

    Order of precedence (low → high):
    1. ``configs/base.yaml`` – defaults for the whole project.
    2. ``configs/dvc_params.yaml`` – parameters tracked by DVC.
    3. ``configs/finbert.yaml`` – model‑specific overrides.
    4. Environment variables (``PROJECT_`` prefix).
    The later sources overwrite the earlier ones.
    Returns a validated ``BaseConfig`` instance.
    """
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    # Load YAML files if they exist; missing files are ignored.
    base_cfg = _load_yaml(config_dir / "base.yaml")
    dvc_cfg = _load_yaml(config_dir / "dvc_params.yaml")
    finbert_cfg = _load_yaml(config_dir / "finbert.yaml")

    # Merge dictionaries – later dict overwrites earlier keys
    merged: Dict[str, Any] = {**base_cfg, **dvc_cfg, **finbert_cfg}

    # Create the pydantic settings model – environment vars are applied automatically
    return BaseConfig(**merged)
