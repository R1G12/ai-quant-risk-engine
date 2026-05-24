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
class VolatilityRiskConfig:
    """Phase 3 volatility modeling parameters."""

    ewma_span: int = 21
    rolling_vol_window: int = 21
    long_vol_window: int = 63
    annualization_factor: int = 252
    regime_zscore_threshold: float = 1.5
    garch_p: int = 1
    garch_q: int = 1


@dataclass
class VarRiskConfig:
    """Phase 3 VaR parameters."""

    confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99])
    n_simulations: int = 10_000
    seed: int = 42
    save_simulations: bool = False
    methods: dict[str, bool] = field(
        default_factory=lambda: {
            "historical": True,
            "parametric": True,
            "monte_carlo": True,
        }
    )


@dataclass
class PortfolioRiskConfig:
    """Phase 3 portfolio analytics parameters."""

    holdings_path: str = "data/raw/portfolio/holdings.parquet"
    weight_mode: str = "equal"
    benchmark_ticker: str = "AAPL"
    rolling_metrics_window: int = 63
    sentiment_return_scale: float = 0.1


@dataclass
class OptimizationRiskConfig:
    """Phase 3 portfolio optimization parameters."""

    long_only: bool = True
    max_weight: float = 0.4
    leverage_cap: float = 1.0
    shrinkage: str = "ledoit_wolf"
    frontier_points: int = 25
    min_weight: float = 0.0
    use_sentiment_adjustment: bool = True


@dataclass
class RiskConfig:
    """Aggregated Phase 3 risk configuration."""

    volatility: VolatilityRiskConfig
    var: VarRiskConfig
    portfolio: PortfolioRiskConfig
    optimization: OptimizationRiskConfig
    hmm_n_regimes: int = 3
    hmm_n_iter: int = 200


@dataclass
class SimulationConfig:
    """Phase 4 Monte Carlo simulation parameters."""

    n_paths: int = 5000
    horizon_days: int = 63
    seed: int = 42
    dt: float = 1.0 / 252.0
    save_paths: bool = False
    plot_n_paths: int = 50
    simulation_types: list[str] = field(default_factory=lambda: ["gbm", "multivariate", "regime_gbm"])


@dataclass
class BacktestConfig:
    """Phase 4 backtesting parameters."""

    rebalance_freq: str = "monthly"
    tc_bps: float = 10.0
    slippage_bps: float = 5.0
    walk_forward_train_days: int = 42
    walk_forward_test_days: int = 21
    weight_source: str = "max_sharpe"
    benchmark_ticker: str = "AAPL"


@dataclass
class ScenarioConfig:
    """Phase 4 stress/scenario shock definitions."""

    scenarios: dict[str, dict[str, float]]
    sentiment_return_scale: float = 0.15


@dataclass
class ResearchMetaConfig:
    """Phase 4 experiment metadata."""

    experiment_id: str = "baseline"
    enable_dvc_experiments: bool = True


@dataclass
class ResearchConfig:
    """Aggregated Phase 4 research configuration."""

    simulation: SimulationConfig
    backtest: BacktestConfig
    scenarios: ScenarioConfig
    meta: ResearchMetaConfig


@dataclass
class AppConfig:
    """Full application configuration."""

    finbert: Config
    market: MarketConfig
    features: FeatureConfig
    sentiment_map: dict[str, Any]
    risk: RiskConfig
    research: ResearchConfig


def load_config() -> Config:
    """Load FinBERT configuration (Phase 1 compatibility)."""
    return load_app_config().finbert


def _merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out


def load_app_config() -> AppConfig:
    """Load merged configuration for all pipeline stages."""
    base_cfg = _load_yaml(CONFIGS_DIR / "base.yaml")
    dvc_cfg = _load_yaml(CONFIGS_DIR / "dvc_params.yaml")
    finbert_cfg = _load_yaml(CONFIGS_DIR / "finbert.yaml")
    market_cfg = _load_yaml(CONFIGS_DIR / "market.yaml")
    features_cfg = _load_yaml(CONFIGS_DIR / "features.yaml")
    sentiment_map = _load_yaml(CONFIGS_DIR / "sentiment_map.yaml")

    risk_vol_cfg = _load_yaml(CONFIGS_DIR / "risk" / "volatility.yaml")
    risk_var_cfg = _load_yaml(CONFIGS_DIR / "risk" / "var.yaml")
    risk_port_cfg = _load_yaml(CONFIGS_DIR / "risk" / "portfolio.yaml")
    risk_opt_cfg = _load_yaml(CONFIGS_DIR / "risk" / "optimization.yaml")

    params = _load_yaml(PROJECT_ROOT / "params.yaml")
    market_params = params.get("market", {})
    feature_params = params.get("features", {})
    risk_params = params.get("risk", {})
    research_params = params.get("research", {})

    sim_cfg = _load_yaml(CONFIGS_DIR / "simulation" / "monte_carlo.yaml")
    bt_cfg = _load_yaml(CONFIGS_DIR / "backtesting" / "engine.yaml")
    scen_cfg = _load_yaml(CONFIGS_DIR / "scenarios" / "stress.yaml")
    res_cfg = _load_yaml(CONFIGS_DIR / "research.yaml")

    merged_sim = _merge_dicts(sim_cfg, research_params.get("simulation", {}))
    merged_bt = _merge_dicts(bt_cfg, research_params.get("backtest", {}))
    merged_scen = _merge_dicts(scen_cfg, research_params.get("scenarios", {}))
    merged_res = _merge_dicts(res_cfg, research_params.get("meta", {}))

    merged_finbert: dict[str, Any] = {**base_cfg, **dvc_cfg, **finbert_cfg}
    merged_market: dict[str, Any] = {**market_cfg, **market_params}
    merged_features: dict[str, Any] = {**features_cfg, **feature_params}
    merged_risk_vol = _merge_dicts(risk_vol_cfg, risk_params.get("volatility", {}))
    merged_risk_var = _merge_dicts(risk_var_cfg, risk_params.get("var", {}))
    merged_risk_port = _merge_dicts(risk_port_cfg, risk_params.get("portfolio", {}))
    merged_risk_opt = _merge_dicts(risk_opt_cfg, risk_params.get("optimization", {}))

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

    risk = RiskConfig(
        volatility=VolatilityRiskConfig(
            ewma_span=int(merged_risk_vol.get("ewma_span", 21)),
            rolling_vol_window=int(merged_risk_vol.get("rolling_vol_window", 21)),
            long_vol_window=int(merged_risk_vol.get("long_vol_window", 63)),
            annualization_factor=int(merged_risk_vol.get("annualization_factor", 252)),
            regime_zscore_threshold=float(merged_risk_vol.get("regime_zscore_threshold", 1.5)),
            garch_p=int(merged_risk_vol.get("garch_p", 1)),
            garch_q=int(merged_risk_vol.get("garch_q", 1)),
        ),
        var=VarRiskConfig(
            confidence_levels=[float(x) for x in merged_risk_var.get("confidence_levels", [0.95, 0.99])],
            n_simulations=int(merged_risk_var.get("n_simulations", 10_000)),
            seed=int(merged_risk_var.get("seed", 42)),
            save_simulations=bool(merged_risk_var.get("save_simulations", False)),
            methods=dict(merged_risk_var.get("methods", {})) or {
                "historical": True,
                "parametric": True,
                "monte_carlo": True,
            },
        ),
        portfolio=PortfolioRiskConfig(
            holdings_path=str(merged_risk_port.get("holdings_path", "data/raw/portfolio/holdings.parquet")),
            weight_mode=str(merged_risk_port.get("weight_mode", "equal")),
            benchmark_ticker=str(merged_risk_port.get("benchmark_ticker", "AAPL")),
            rolling_metrics_window=int(merged_risk_port.get("rolling_metrics_window", 63)),
            sentiment_return_scale=float(merged_risk_port.get("sentiment_return_scale", 0.1)),
        ),
        optimization=OptimizationRiskConfig(
            long_only=bool(merged_risk_opt.get("long_only", True)),
            max_weight=float(merged_risk_opt.get("max_weight", 0.4)),
            leverage_cap=float(merged_risk_opt.get("leverage_cap", 1.0)),
            shrinkage=str(merged_risk_opt.get("shrinkage", "ledoit_wolf")),
            frontier_points=int(merged_risk_opt.get("frontier_points", 25)),
            min_weight=float(merged_risk_opt.get("min_weight", 0.0)),
            use_sentiment_adjustment=bool(merged_risk_opt.get("use_sentiment_adjustment", True)),
        ),
        hmm_n_regimes=int(risk_params.get("hmm_n_regimes", 3)),
        hmm_n_iter=int(risk_params.get("hmm_n_iter", 200)),
    )

    research = ResearchConfig(
        simulation=SimulationConfig(
            n_paths=int(merged_sim.get("n_paths", 5000)),
            horizon_days=int(merged_sim.get("horizon_days", 63)),
            seed=int(merged_sim.get("seed", 42)),
            dt=float(merged_sim.get("dt", 1.0 / 252.0)),
            save_paths=bool(merged_sim.get("save_paths", False)),
            plot_n_paths=int(merged_sim.get("plot_n_paths", 50)),
            simulation_types=list(merged_sim.get("simulation_types", ["gbm", "multivariate", "regime_gbm"])),
        ),
        backtest=BacktestConfig(
            rebalance_freq=str(merged_bt.get("rebalance_freq", "monthly")),
            tc_bps=float(merged_bt.get("tc_bps", 10.0)),
            slippage_bps=float(merged_bt.get("slippage_bps", 5.0)),
            walk_forward_train_days=int(merged_bt.get("walk_forward_train_days", 42)),
            walk_forward_test_days=int(merged_bt.get("walk_forward_test_days", 21)),
            weight_source=str(merged_bt.get("weight_source", "max_sharpe")),
            benchmark_ticker=str(merged_bt.get("benchmark_ticker", "AAPL")),
        ),
        scenarios=ScenarioConfig(
            scenarios=dict(merged_scen.get("scenarios", {})) or {},
            sentiment_return_scale=float(merged_scen.get("sentiment_return_scale", 0.15)),
        ),
        meta=ResearchMetaConfig(
            experiment_id=str(merged_res.get("experiment_id", "baseline")),
            enable_dvc_experiments=bool(merged_res.get("enable_dvc_experiments", True)),
        ),
    )

    return AppConfig(
        finbert=finbert,
        market=market,
        features=features,
        sentiment_map=sentiment_map,
        risk=risk,
        research=research,
    )
