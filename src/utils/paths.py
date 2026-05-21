'''paths.py – central Path definitions and helper utilities'''

from __future__ import annotations

import pathlib
from typing import Union

# Resolve the repository root (two levels up from this file)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Standard directories – they will be created lazily when needed
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
CONFIGS_DIR = PROJECT_ROOT / "configs"

def ensure_dir(path: Union[pathlib.Path, str]) -> pathlib.Path:
    """Create *path* (including parents) if it does not exist.

    Returns the resolved ``Path`` object for convenient chaining.
    """
    p = pathlib.Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
