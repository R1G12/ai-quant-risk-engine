"""Configuration loader with YAML merge and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
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
class IngestionConfig:
    """News ingestion parameters (Phase 1)."""

    source: str = "sample"
    seed: int = 42
    max_headlines_per_ticker: int = 10


@dataclass
class MarketConfig:
    """Market data ingestion configuration."""

    source: str = "sample"
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT"])
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    use_rolling_window: bool = True
    rolling_days: int = 365
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
    allow_shorts: bool = False
    max_gross_per_ticker: float = 0.5
    weighting: str = "equal"


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
class PortfolioRunConfig:
    """Notebook-style portfolio options from configs/run.yaml."""

    weighting: str = "equal"
    allow_shorts: bool = True
    max_gross_per_ticker: float = 0.5
    position_sides: dict[str, str] = field(default_factory=dict)
    manual_weights: dict[str, float] = field(default_factory=dict)
    anchor_weights: dict[str, float] = field(default_factory=dict)
    risk_free: float = 0.05
    min_gross_divisor: float = 5.0
    sentiment_position_sides: bool = True
    sentiment_sides_window_days: int = 30
    sentiment_mu_blend: float = 0.3
    sentiment_mu_mode: str = "vol_scaled"
    sentiment_mu_scale: float = 0.5
    sentiment_mu_window_days: int = 30
    sentiment_mu_hist_window_days: int | None = None
    sentiment_magnitude_tilt: bool = True
    sentiment_tilt_beta: float = 0.2
    sentiment_tilt_cap: float = 2.0
    regime_sentiment_mix: dict[str, float] = field(
        default_factory=lambda: {"low": 0.4, "mid": 0.75, "high": 1.0}
    )
    use_legacy_bullish_mu: bool = False


@dataclass
class RunMarketOverrides:
    tickers: list[str] = field(default_factory=list)
    use_rolling_window: bool | None = None
    rolling_days: int | None = None


@dataclass
class RunResearchOverrides:
    backtest_weight_source: str = "max_sharpe"


@dataclass
class RunConfig:
    """User-facing run profile (configs/run.yaml)."""

    mode: str = "demo"
    profile_path: Path | None = None
    market: RunMarketOverrides = field(default_factory=RunMarketOverrides)
    portfolio: PortfolioRunConfig = field(default_factory=PortfolioRunConfig)
    research: RunResearchOverrides = field(default_factory=RunResearchOverrides)


@dataclass
class AppConfig:
    """Full application configuration."""

    finbert: Config
    ingestion: IngestionConfig
    market: MarketConfig
    features: FeatureConfig
    sentiment_map: dict[str, Any]
    risk: RiskConfig
    research: ResearchConfig
    run: RunConfig | None = None


def load_config() -> Config:
    """Load FinBERT configuration (Phase 1 compatibility)."""
    return load_app_config().finbert


def resolve_market_dates(
    merged_market: dict[str, Any],
    *,
    today: date | None = None,
) -> tuple[str, str]:
    """Resolve start/end dates; rolling 1Y window when enabled and not pinned."""
    use_rolling = bool(merged_market.get("use_rolling_window", True))
    rolling_days = int(merged_market.get("rolling_days", 365))
    pinned_start = merged_market.get("start_date")
    pinned_end = merged_market.get("end_date")

    if use_rolling and not os.getenv("MARKET_PIN_DATES"):
        end = today or date.today()
        start = end - timedelta(days=rolling_days)
        return start.isoformat(), end.isoformat()

    return (
        str(pinned_start or (date.today() - timedelta(days=rolling_days)).isoformat()),
        str(pinned_end or date.today().isoformat()),
    )


def _merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out


def _default_run_profile_path() -> Path:
    env_path = os.getenv("RUN_PROFILE")
    if env_path:
        return Path(env_path)
    if os.getenv("CI", "").lower() in ("1", "true", "yes"):
        ci_path = CONFIGS_DIR / "run.ci.yaml"
        if ci_path.is_file():
            return ci_path
    return CONFIGS_DIR / "run.yaml"


def load_run_config(path: Path | None = None) -> RunConfig | None:
    """Load configs/run.yaml (or RUN_PROFILE path). Returns None if file missing."""
    profile_path = path or _default_run_profile_path()
    if not profile_path.is_file():
        return None
    raw = _load_yaml(profile_path)
    market_raw = raw.get("market", {}) or {}
    port_raw = raw.get("portfolio", {}) or {}
    research_raw = raw.get("research", {}) or {}
    position_sides = {str(k): str(v) for k, v in (port_raw.get("position_sides") or {}).items()}
    regime_mix_raw = port_raw.get("regime_sentiment_mix") or {}
    regime_sentiment_mix = (
        {str(k): float(v) for k, v in regime_mix_raw.items()}
        if regime_mix_raw
        else {"low": 0.4, "mid": 0.75, "high": 1.0}
    )
    hist_window = port_raw.get("sentiment_mu_hist_window_days")
    return RunConfig(
        mode=str(raw.get("mode", "demo")).lower(),
        profile_path=profile_path.resolve(),
        market=RunMarketOverrides(
            tickers=[str(t) for t in market_raw.get("tickers", [])],
            use_rolling_window=market_raw.get("use_rolling_window"),
            rolling_days=market_raw.get("rolling_days"),
        ),
        portfolio=PortfolioRunConfig(
            weighting=str(port_raw.get("weighting", "equal")),
            allow_shorts=bool(port_raw.get("allow_shorts", True)),
            max_gross_per_ticker=float(port_raw.get("max_gross_per_ticker", 0.5)),
            position_sides=position_sides,
            manual_weights={str(k): float(v) for k, v in (port_raw.get("manual_weights") or {}).items()},
            anchor_weights={str(k): float(v) for k, v in (port_raw.get("anchor_weights") or {}).items()},
            risk_free=float(port_raw.get("risk_free", 0.05)),
            min_gross_divisor=float(port_raw.get("min_gross_divisor", 5.0)),
            sentiment_position_sides=bool(port_raw.get("sentiment_position_sides", True)),
            sentiment_sides_window_days=int(port_raw.get("sentiment_sides_window_days", 30)),
            sentiment_mu_blend=float(port_raw.get("sentiment_mu_blend", 0.3)),
            sentiment_mu_mode=str(port_raw.get("sentiment_mu_mode", "vol_scaled")),
            sentiment_mu_scale=float(port_raw.get("sentiment_mu_scale", 0.5)),
            sentiment_mu_window_days=int(port_raw.get("sentiment_mu_window_days", 30)),
            sentiment_mu_hist_window_days=int(hist_window) if hist_window is not None else None,
            sentiment_magnitude_tilt=bool(port_raw.get("sentiment_magnitude_tilt", True)),
            sentiment_tilt_beta=float(port_raw.get("sentiment_tilt_beta", 0.2)),
            sentiment_tilt_cap=float(port_raw.get("sentiment_tilt_cap", 2.0)),
            regime_sentiment_mix=regime_sentiment_mix,
            use_legacy_bullish_mu=bool(port_raw.get("use_legacy_bullish_mu", False)),
        ),
        research=RunResearchOverrides(
            backtest_weight_source=str(research_raw.get("backtest_weight_source", "max_sharpe")),
        ),
    )


def market_source_for_run_mode(mode: str) -> str:
    """Map run profile mode to market adapter source."""
    return "yfinance" if mode == "live" else "sample"


def news_source_for_run_mode(mode: str) -> str:
    """Map run profile mode to news adapter source."""
    return "yfinance" if mode == "live" else "sample"


def apply_run_profile_env(run: RunConfig, *, force: bool = False) -> dict[str, str]:
    """Apply run-profile env vars to ``os.environ`` (returns the updates)."""
    env = run_profile_env(run, force=force)
    os.environ.update(env)
    return env


def run_profile_env(run: RunConfig, *, force: bool = False) -> dict[str, str]:
    """Environment overrides derived from run.mode.

    When ``force`` is True (``aqre run profile``), always set sources from the
    run profile so a leftover ``MARKET_SOURCE=sample`` in the shell cannot
    desync market data from ``configs/run.yaml`` tickers.
    """
    env: dict[str, str] = {}
    if force or not os.getenv("MARKET_SOURCE"):
        env["MARKET_SOURCE"] = market_source_for_run_mode(run.mode)
    if force or not os.getenv("NEWS_SOURCE"):
        env["NEWS_SOURCE"] = news_source_for_run_mode(run.mode)
    return env


def _apply_run_to_merged(
    run: RunConfig,
    merged_ingestion: dict[str, Any],
    merged_market: dict[str, Any],
    merged_features: dict[str, Any],
    merged_risk_port: dict[str, Any],
    merged_risk_opt: dict[str, Any],
    merged_bt: dict[str, Any],
) -> None:
    if run.market.tickers:
        merged_market["tickers"] = run.market.tickers
    if run.market.use_rolling_window is not None:
        merged_market["use_rolling_window"] = run.market.use_rolling_window
    if run.market.rolling_days is not None:
        merged_market["rolling_days"] = run.market.rolling_days
    if not os.getenv("MARKET_SOURCE"):
        merged_market["source"] = market_source_for_run_mode(run.mode)
    if not os.getenv("NEWS_SOURCE"):
        merged_ingestion["source"] = news_source_for_run_mode(run.mode)
    merged_features["risk_free_rate"] = run.portfolio.risk_free
    merged_risk_port["weight_mode"] = run.portfolio.weighting
    merged_risk_opt["long_only"] = not run.portfolio.allow_shorts
    merged_risk_opt["max_weight"] = run.portfolio.max_gross_per_ticker
    merged_risk_opt["allow_shorts"] = run.portfolio.allow_shorts
    merged_risk_opt["max_gross_per_ticker"] = run.portfolio.max_gross_per_ticker
    merged_risk_opt["weighting"] = run.portfolio.weighting
    merged_bt["weight_source"] = run.research.backtest_weight_source


def load_app_config(run_profile: Path | None = None) -> AppConfig:
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
    ingestion_params = params.get("ingestion", {})
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
    merged_ingestion: dict[str, Any] = {**ingestion_params}
    merged_market: dict[str, Any] = {**market_cfg, **market_params}
    merged_features: dict[str, Any] = {**features_cfg, **feature_params}
    merged_risk_vol = _merge_dicts(risk_vol_cfg, risk_params.get("volatility", {}))
    merged_risk_var = _merge_dicts(risk_var_cfg, risk_params.get("var", {}))
    merged_risk_port = _merge_dicts(risk_port_cfg, risk_params.get("portfolio", {}))
    merged_risk_opt = _merge_dicts(risk_opt_cfg, risk_params.get("optimization", {}))

    run = load_run_config(run_profile)
    if run is not None:
        _apply_run_to_merged(
            run,
            merged_ingestion,
            merged_market,
            merged_features,
            merged_risk_port,
            merged_risk_opt,
            merged_bt,
        )

    news_source = os.getenv("NEWS_SOURCE", merged_ingestion.get("source", "sample"))
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

    ingestion = IngestionConfig(
        source=str(news_source),
        seed=int(merged_ingestion.get("seed", 42)),
        max_headlines_per_ticker=int(merged_ingestion.get("max_headlines_per_ticker", 10)),
    )

    start_date, end_date = resolve_market_dates(merged_market)
    market = MarketConfig(
        source=str(source),
        tickers=list(merged_market.get("tickers", ["AAPL", "MSFT"])),
        start_date=start_date,
        end_date=end_date,
        use_rolling_window=bool(merged_market.get("use_rolling_window", True)),
        rolling_days=int(merged_market.get("rolling_days", 365)),
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
            allow_shorts=bool(merged_risk_opt.get("allow_shorts", False)),
            max_gross_per_ticker=float(merged_risk_opt.get("max_gross_per_ticker", 0.5)),
            weighting=str(merged_risk_opt.get("weighting", "equal")),
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
        ingestion=ingestion,
        market=market,
        features=features,
        sentiment_map=sentiment_map,
        risk=risk,
        research=research,
        run=run,
    )
