"""Monte Carlo orchestration and path summaries."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.simulation.stochastic_processes.gbm import simulate_gbm_paths
from src.simulation.stochastic_processes.multivariate import simulate_multivariate_gbm
from src.utils.config import SimulationConfig

_EPS = 1e-12


def _terminal_returns(paths: np.ndarray) -> np.ndarray:
    """Simple return per path; NaN when starting wealth is ~0."""
    start = paths[:, 0]
    end = paths[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(start) > _EPS, end / start - 1.0, np.nan)


def _max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Max drawdown per path; safe when wealth or running max is ~0."""
    running_max = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_max > _EPS, paths / running_max - 1.0, 0.0)
    return dd.min(axis=1)


def path_summary_stats(paths: np.ndarray) -> pl.DataFrame:
    """Per-path summary: terminal return, max drawdown."""
    terminal = _terminal_returns(paths)
    max_dd = _max_drawdowns(paths)
    return pl.DataFrame(
        {
            "path_id": list(range(paths.shape[0])),
            "terminal_return": terminal.tolist(),
            "max_drawdown": max_dd.tolist(),
        }
    )


def tail_metrics_from_paths(paths: np.ndarray, confidence: float = 0.95) -> pl.DataFrame:
    """VaR/CVaR style metrics on terminal simple returns."""
    rets = _terminal_returns(paths)
    finite = rets[np.isfinite(rets)]
    if finite.size == 0:
        return pl.DataFrame(
            {
                "confidence": [confidence],
                "var": [float("nan")],
                "cvar": [float("nan")],
                "mean_return": [float("nan")],
                "volatility": [float("nan")],
            }
        )
    var = float(np.quantile(finite, 1.0 - confidence))
    tail = finite[finite <= var]
    cvar = float(tail.mean()) if len(tail) else var
    return pl.DataFrame(
        {
            "confidence": [confidence],
            "var": [var],
            "cvar": [cvar],
            "mean_return": [float(finite.mean())],
            "volatility": [float(finite.std())],
        }
    )


def sample_paths_for_plot(
    mu: float,
    sigma: float,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Small path sample for fan charts (not persisted to parquet)."""
    n = min(cfg.plot_n_paths, cfg.n_paths)
    return simulate_gbm_paths(
        mu,
        sigma,
        n_paths=n,
        n_steps=cfg.horizon_days,
        dt=cfg.dt,
        seed=cfg.seed + 100,
    )


def run_gbm_simulation(
    mu: float,
    sigma: float,
    cfg: SimulationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run 1D GBM and return path summaries + tail metrics."""
    paths = simulate_gbm_paths(
        mu,
        sigma,
        n_paths=cfg.n_paths,
        n_steps=cfg.horizon_days,
        dt=cfg.dt,
        seed=cfg.seed,
    )
    return path_summary_stats(paths), tail_metrics_from_paths(paths)


def run_multivariate_simulation(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    cfg: SimulationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Multivariate asset paths -> portfolio wealth paths."""
    asset_paths = simulate_multivariate_gbm(
        mu,
        cov,
        n_paths=cfg.n_paths,
        n_steps=cfg.horizon_days,
        dt=cfg.dt,
        seed=cfg.seed + 1,
    )
    port_paths = np.einsum("pat,a->pt", asset_paths, weights)
    return path_summary_stats(port_paths), tail_metrics_from_paths(port_paths)
