'''config.py – simple configuration loader for the quant risk engine

We avoid pydantic here because the current environment has a pydantic version that is
incompatible with the previously‑written ``BaseSettings`` approach.  A lightweight
`dataclass` provides the same functionality: default values, environment variable
overrides, and hierarchical YAML merging.
'''

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

# Helper to load a YAML file into a dict (empty dict if file missing)
def _load_yaml(file_path: Path) -> Dict[str, Any]:
    if not file_path.is_file():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@dataclass
class Config:
    """Configuration values for the project.

    Values are taken from environment variables (``PROJECT_`` prefix) with sensible
    defaults.  Types are validated on instantiation – if an env var cannot be
    converted to ``int`` a ``ValueError`` is raised.
    """

    batch_size: int = int(os.getenv("PROJECT_BATCH_SIZE", 32))
    max_seq_length: int = int(os.getenv("PROJECT_MAX_SEQ_LENGTH", 128))
    seed: int = int(os.getenv("PROJECT_SEED", 42))

def load_config() -> Config:
    """Load configuration from the three yaml files and environment.

    Order of precedence (low → high):
        1. ``configs/base.yaml`` – default values.
        2. ``configs/dvc_params.yaml`` – DVC‑tracked parameters.
        3. ``configs/finbert.yaml`` – model‑specific overrides.
        4. Environment variables with ``PROJECT_`` prefix.
    """
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    base_cfg = _load_yaml(config_dir / "base.yaml")
    dvc_cfg = _load_yaml(config_dir / "dvc_params.yaml")
    finbert_cfg = _load_yaml(config_dir / "finbert.yaml")

    merged: Dict[str, Any] = {**base_cfg, **dvc_cfg, **finbert_cfg}

    # Override merged values with env vars if they exist
    batch = int(os.getenv("PROJECT_BATCH_SIZE", merged.get("batch_size", 32)))
    seq = int(os.getenv("PROJECT_MAX_SEQ_LENGTH", merged.get("max_seq_length", 128)))
    seed = int(os.getenv("PROJECT_SEED", merged.get("seed", 42)))

    return Config(batch_size=batch, max_seq_length=seq, seed=seed)
