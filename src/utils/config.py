"""Configuration loader with YAML merge and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT


def _load_yaml(file_path: Path) -> dict[str, Any]:
    if not file_path.is_file():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """FinBERT / Phase 1 configuration."""

    batch_size: int = 32
    max_seq_length: int = 128
    seed: int = 42
    model_name: str = "ProsusAI/finbert"


@dataclass
class MarketConfig:
    """Market data ingestion configuration."""

    source: str = "sample"
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT"])
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    partition_freq: str = "month"
    compression: str = "zstd"
    default_sentiment_ticker: str = "MARKET"


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    volatility_window: int = 21
    sharpe_window: int = 63
    risk_free_rate: float = 0.05
    momentum_window: int = 12
    sma_windows: list[int] = field(default_factory=lambda: [10, 20])
    correlation_window: int = 30
    annualization_factor: int = 252


@dataclass
class AppConfig:
    """Full application configuration."""

    finbert: Config
    market: MarketConfig
    features: FeatureConfig
    sentiment_map: dict[str, Any]


def load_config() -> Config:
    """Load FinBERT configuration (Phase 1 compatibility)."""
    return load_app_config().finbert


def load_app_config() -> AppConfig:
    """Load merged configuration for all pipeline stages."""
    base_cfg = _load_yaml(CONFIGS_DIR / "base.yaml")
    dvc_cfg = _load_yaml(CONFIGS_DIR / "dvc_params.yaml")
    finbert_cfg = _load_yaml(CONFIGS_DIR / "finbert.yaml")
    market_cfg = _load_yaml(CONFIGS_DIR / "market.yaml")
    features_cfg = _load_yaml(CONFIGS_DIR / "features.yaml")
    sentiment_map = _load_yaml(CONFIGS_DIR / "sentiment_map.yaml")

    params = _load_yaml(PROJECT_ROOT / "params.yaml")
    market_params = params.get("market", {})
    feature_params = params.get("features", {})

    merged_finbert: dict[str, Any] = {**base_cfg, **dvc_cfg, **finbert_cfg}
    merged_market: dict[str, Any] = {**market_cfg, **market_params}
    merged_features: dict[str, Any] = {**features_cfg, **feature_params}

    source = os.getenv("MARKET_SOURCE", merged_market.get("source", "sample"))

    finbert = Config(
        batch_size=int(os.getenv("PROJECT_BATCH_SIZE", merged_finbert.get("batch_size", 32))),
        max_seq_length=int(
            os.getenv("PROJECT_MAX_SEQ_LENGTH", merged_finbert.get("max_seq_length", 128))
        ),
        seed=int(os.getenv("PROJECT_SEED", merged_finbert.get("seed", 42))),
        model_name=str(
            os.getenv("PROJECT_MODEL_NAME", merged_finbert.get("model_name", "ProsusAI/finbert"))
        ),
    )

    market = MarketConfig(
        source=str(source),
        tickers=list(merged_market.get("tickers", ["AAPL", "MSFT"])),
        start_date=str(merged_market.get("start_date", "2024-01-01")),
        end_date=str(merged_market.get("end_date", "2024-03-31")),
        partition_freq=str(merged_market.get("partition_freq", "month")),
        compression=str(merged_market.get("compression", "zstd")),
        default_sentiment_ticker=str(
            merged_market.get("default_sentiment_ticker", "MARKET")
        ),
    )

    features = FeatureConfig(
        volatility_window=int(merged_features.get("volatility_window", 21)),
        sharpe_window=int(merged_features.get("sharpe_window", 63)),
        risk_free_rate=float(merged_features.get("risk_free_rate", 0.05)),
        momentum_window=int(merged_features.get("momentum_window", 12)),
        sma_windows=[int(x) for x in merged_features.get("sma_windows", [10, 20])],
        correlation_window=int(merged_features.get("correlation_window", 30)),
        annualization_factor=int(merged_features.get("annualization_factor", 252)),
    )

    return AppConfig(
        finbert=finbert,
        market=market,
        features=features,
        sentiment_map=sentiment_map,
    )
