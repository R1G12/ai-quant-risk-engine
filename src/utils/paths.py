"""Central path definitions and partition helpers."""

from __future__ import annotations

import pathlib
from typing import Union

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
FEATURES_DIR = DATA_DIR / "features"
ANALYTICS_DIR = DATA_DIR / "analytics"

RAW_NEWS_PATH = RAW_DATA_DIR / "news.csv"
RAW_MARKET_DIR = RAW_DATA_DIR / "market"
PROCESSED_NEWS_PATH = PROCESSED_DATA_DIR / "news.parquet"
PROCESSED_MARKET_DIR = PROCESSED_DATA_DIR / "market"
PROCESSED_SENTIMENT_PATH = PROCESSED_DATA_DIR / "sentiment.parquet"

EXTERNAL_SAMPLE_MARKET_DIR = EXTERNAL_DATA_DIR / "sample_market"

FEATURES_RETURNS_DIR = FEATURES_DIR / "returns"
FEATURES_VOLATILITY_DIR = FEATURES_DIR / "volatility"
FEATURES_TECHNICAL_DIR = FEATURES_DIR / "technical"
FEATURES_SENTIMENT_AGG_DIR = FEATURES_DIR / "sentiment_agg"
FEATURES_MERGED_DIR = FEATURES_DIR / "merged"
RISK_DATASET_PATH = FEATURES_MERGED_DIR / "risk_dataset.parquet"
RISK_DATASET_METADATA_PATH = FEATURES_MERGED_DIR / "_metadata.json"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_dir(path: Union[pathlib.Path, str]) -> pathlib.Path:
    """Create path (including parents) if missing."""
    p = pathlib.Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def market_processed_glob() -> str:
    """Glob for partitioned cleaned market parquet files."""
    return str(PROCESSED_MARKET_DIR / "year=*" / "month=*" / "*.parquet")


def market_partition_path(year: int, month: int) -> pathlib.Path:
    """Hive-style partition directory for market data."""
    return PROCESSED_MARKET_DIR / f"year={year}" / f"month={month:02d}"
