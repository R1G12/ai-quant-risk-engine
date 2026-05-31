"""Synthetic parquet artifacts for Phase 5 integration tests."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

# Align with configs/run.ci.yaml sample tickers used in CI.
CI_SAMPLE_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "XOM", "GS", "JPM")


def write_minimal_platform_artifacts(
    root: Path,
    *,
    experiment_id: str = "baseline",
    tickers: tuple[str, ...] = CI_SAMPLE_TICKERS,
) -> dict[str, Path]:
    """Write minimal risk/research artifacts under a temp tree."""
    paths: dict[str, Path] = {}
    start = date(2024, 1, 2)
    rows: list[dict] = []
    for i in range(60):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for j, ticker in enumerate(tickers):
            ret = 0.001 * ((i + j) % 5 - 2)
            rows.append(
                {
                    "timestamp": d.isoformat(),
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0 + i,
                    "volume": 1_000_000.0,
                    "returns": ret,
                    "bullish_ratio": 0.45 + 0.1 * (j % 3),
                }
            )

    merged_dir = root / "data" / "features" / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    risk_path = merged_dir / "risk_dataset.parquet"
    pl.DataFrame(rows).with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC")
    ).write_parquet(risk_path)
    paths["risk_dataset"] = risk_path

    var_dir = root / "data" / "risk" / "var"
    var_dir.mkdir(parents=True, exist_ok=True)
    var_path = var_dir / "var_metrics.parquet"
    pl.DataFrame(
        {
            "method": ["historical", "parametric"],
            "confidence": [0.95, 0.95],
            "var": [-0.02, -0.018],
        }
    ).write_parquet(var_path)
    paths["var"] = var_path

    corr_dir = root / "data" / "risk" / "correlations"
    corr_dir.mkdir(parents=True, exist_ok=True)
    corr_path = corr_dir / "correlations_latest.parquet"
    pl.DataFrame(
        {
            "asset_i": ["AAPL", "AAPL", "MSFT"],
            "asset_j": ["AAPL", "MSFT", "MSFT"],
            "metric": ["corr", "corr", "corr"],
            "value": [1.0, 0.35, 1.0],
        }
    ).write_parquet(corr_path)
    paths["correlations"] = corr_path

    port_dir = root / "data" / "risk" / "portfolio"
    port_dir.mkdir(parents=True, exist_ok=True)
    regimes_path = port_dir / "regimes.parquet"
    pl.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in rows[:30]],
            "regime_label": ["low_vol"] * 15 + ["high_vol"] * 15,
        }
    ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC")).write_parquet(
        regimes_path
    )
    paths["regimes"] = regimes_path

    metrics_path = port_dir / "portfolio_metrics.parquet"
    pl.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in rows[:20]],
            "ewma_vol": [0.12 + 0.01 * (i % 3) for i in range(20)],
            "drawdown": [-0.01 * i for i in range(20)],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC")).write_parquet(
        metrics_path
    )
    paths["portfolio_metrics"] = metrics_path

    port_returns_path = port_dir / "portfolio_returns.parquet"
    ts = sorted({r["timestamp"] for r in rows})[:40]
    pl.DataFrame(
        {
            "timestamp": ts,
            "portfolio_return": [0.001] * len(ts),
        }
    ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC")).write_parquet(
        port_returns_path
    )
    paths["portfolio_returns"] = port_returns_path

    bt_dir = root / "data" / "research" / "backtests" / f"experiment_id={experiment_id}"
    bt_dir.mkdir(parents=True, exist_ok=True)
    eq_path = bt_dir / "equity_curve.parquet"
    pl.DataFrame(
        {
            "timestamp": ts,
            "portfolio_return": [0.001] * len(ts),
            "equity": [1.0 + 0.001 * i for i in range(len(ts))],
            "drawdown": [0.0] * max(0, len(ts) - 5)
            + [-0.05, -0.08, -0.06, -0.04, -0.02][-min(5, len(ts)) :],
            "is_rebalance": [False] * len(ts),
        }
    ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC")).write_parquet(
        eq_path
    )
    paths["equity"] = eq_path

    holdings_dir = root / "data" / "raw" / "portfolio"
    holdings_dir.mkdir(parents=True, exist_ok=True)
    holdings_path = holdings_dir / "holdings.parquet"
    w = 1.0 / len(tickers)
    pl.DataFrame({"asset": list(tickers), "weight": [w] * len(tickers)}).write_parquet(
        holdings_path
    )
    paths["holdings"] = holdings_path

    return paths


def _sync_path(monkeypatch, module: str, name: str, value: Path) -> None:
    """Re-bind a path constant imported at module level in tests."""
    monkeypatch.setattr(f"{module}.{name}", value, raising=False)


def patch_paths_to_root(monkeypatch, root: Path) -> dict[str, float]:
    """Point path constants at a temp artifact tree (utils.paths + importers)."""
    from src.utils import paths as p

    weights = {t: 1.0 / len(CI_SAMPLE_TICKERS) for t in CI_SAMPLE_TICKERS}

    def _load_weights(_app):
        return dict(weights)

    for target in (
        "src.risk.portfolio.holdings.load_weights",
        "src.copilot.context.builder.load_weights",
        "src.risk.portfolio.exposures.load_weights",
    ):
        monkeypatch.setattr(target, _load_weights)

    path_map = {
        "RISK_DATASET_PATH": root / "data/features/merged/risk_dataset.parquet",
        "RISK_VAR_DIR": root / "data/risk/var",
        "RISK_CORRELATIONS_DIR": root / "data/risk/correlations",
        "RISK_REGIMES_PATH": root / "data/risk/portfolio/regimes.parquet",
        "RISK_PORTFOLIO_METRICS_PATH": root / "data/risk/portfolio/portfolio_metrics.parquet",
        "RISK_PORTFOLIO_RETURNS_PATH": root / "data/risk/portfolio/portfolio_returns.parquet",
        "HOLDINGS_PATH": root / "data/raw/portfolio/holdings.parquet",
        "RESEARCH_BACKTESTS_DIR": root / "data/research/backtests",
        "REPORTS_PORTFOLIO_DIR": root / "reports/portfolio",
        "REPORTS_RISK_DIR": root / "reports/risk",
        "REPORTS_GOVERNANCE_DIR": root / "reports/governance",
        "REPORTS_SIMULATIONS_DIR": root / "reports/simulations",
    }

    for name, value in path_map.items():
        monkeypatch.setattr(p, name, value)

    modules_using_paths = (
        "src.copilot.context.builder",
        "src.copilot.explanations.drawdown",
        "src.copilot.attribution.portfolio",
        "src.monitoring.quality.validators",
        "src.monitoring.drift.baseline",
        "src.monitoring.health.checks",
        "src.features.wide_returns",
        "src.simulation.pipeline._calibration",
        "src.analytics.dashboard_kpis",
    )
    for mod in modules_using_paths:
        for name, value in path_map.items():
            _sync_path(monkeypatch, mod, name, value)

    return weights
