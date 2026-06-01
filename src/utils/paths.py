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
ANALYTICS_RISK_DIR = ANALYTICS_DIR / "risk"

RAW_PORTFOLIO_DIR = RAW_DATA_DIR / "portfolio"
HOLDINGS_PATH = RAW_PORTFOLIO_DIR / "holdings.parquet"

INPUT_PORTFOLIO_DIR = DATA_DIR / "input" / "portfolio"
PROCESSED_PORTFOLIO_DIR = PROCESSED_DATA_DIR / "portfolio"
TRACKER_TRADES_PATH = PROCESSED_PORTFOLIO_DIR / "trades.parquet"
TRACKER_METADATA_PATH = PROCESSED_PORTFOLIO_DIR / "_tracker_metadata.json"

RISK_DIR = DATA_DIR / "risk"
RISK_VOLATILITY_DIR = RISK_DIR / "volatility"
RISK_VAR_DIR = RISK_DIR / "var"
RISK_CVAR_DIR = RISK_DIR / "cvar"
RISK_CORRELATIONS_DIR = RISK_DIR / "correlations"
RISK_PORTFOLIO_DIR = RISK_DIR / "portfolio"
RISK_OPTIMIZATION_DIR = RISK_DIR / "optimization"
RISK_FRONTIER_DIR = RISK_DIR / "frontier"
RISK_FRONTIER_PATH = RISK_FRONTIER_DIR / "frontier.parquet"
RISK_OPT_WEIGHTS_PATH = RISK_OPTIMIZATION_DIR / "optimal_weights.parquet"
RISK_OPT_METADATA_PATH = RISK_OPTIMIZATION_DIR / "_metadata.json"
RISK_PORTFOLIO_RETURNS_PATH = RISK_PORTFOLIO_DIR / "portfolio_returns.parquet"
RISK_PORTFOLIO_METRICS_PATH = RISK_PORTFOLIO_DIR / "portfolio_metrics.parquet"
RISK_REGIMES_PATH = RISK_PORTFOLIO_DIR / "regimes.parquet"

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

RESEARCH_DIR = DATA_DIR / "research"
RESEARCH_SIMULATIONS_DIR = RESEARCH_DIR / "simulations"
RESEARCH_BACKTESTS_DIR = RESEARCH_DIR / "backtests"
RESEARCH_STRESS_DIR = RESEARCH_DIR / "stress"
RESEARCH_SCENARIOS_DIR = RESEARCH_DIR / "scenarios"
RESEARCH_EVALUATION_DIR = RESEARCH_DIR / "evaluation"
RESEARCH_COMPARISONS_DIR = RESEARCH_DIR / "comparisons"
RESEARCH_REPORTS_DIR = RESEARCH_DIR / "reports"
RESEARCH_COMPARISONS_PATH = RESEARCH_COMPARISONS_DIR / "comparisons.parquet"
RESEARCH_PERFORMANCE_PATH = RESEARCH_EVALUATION_DIR / "performance_summary.parquet"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_SIMULATIONS_DIR = EXPERIMENTS_DIR / "simulations"
EXPERIMENTS_BACKTESTS_DIR = EXPERIMENTS_DIR / "backtests"
EXPERIMENTS_STRESS_DIR = EXPERIMENTS_DIR / "stress_tests"

ANALYTICS_RESEARCH_DIR = ANALYTICS_DIR / "research"
ANALYTICS_DASHBOARD_DIR = ANALYTICS_DIR / "dashboard"
MARKET_LIVE_CACHE_DIR = DATA_DIR / "cache" / "market_live"
MARKET_LIVE_RAW_PATH = MARKET_LIVE_CACHE_DIR / "market_raw.parquet"
RESEARCH_REPORT_MD_PATH = RESEARCH_REPORTS_DIR / "REPORT.md"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_PORTFOLIO_DIR = REPORTS_DIR / "portfolio"
REPORTS_RISK_DIR = REPORTS_DIR / "risk"
REPORTS_SIMULATIONS_DIR = REPORTS_DIR / "simulations"
REPORTS_EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
REPORTS_GOVERNANCE_DIR = REPORTS_DIR / "governance"

PLATFORM_METRICS_DIR = METRICS_DIR / "platform"


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
