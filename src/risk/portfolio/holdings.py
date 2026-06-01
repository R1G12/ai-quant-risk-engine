"""Portfolio holdings helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.paths import HOLDINGS_PATH, PROJECT_ROOT, RISK_OPT_WEIGHTS_PATH, ensure_dir

LOGGER = get_logger(__name__)
MANIFEST_PATH = PROJECT_ROOT / "data" / "run_manifest.json"


def _holdings_path(app: AppConfig) -> Path:
    path = Path(app.risk.portfolio.holdings_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalized_tickers(tickers: list[str]) -> list[str]:
    return sorted(str(t).strip().upper() for t in tickers if str(t).strip())


def _manifest_tickers(manifest_path: Path) -> list[str] | None:
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    tickers = data.get("tickers") or data.get("tickers_requested")
    if not tickers:
        return None
    return _normalized_tickers(list(tickers))


def _holdings_file_tickers(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    try:
        assets = pl.read_parquet(path)["asset"].cast(pl.Utf8).to_list()
    except Exception:
        return None
    return _normalized_tickers(assets)


def _holdings_match_config(app: AppConfig, path: Path) -> bool:
    expected = _normalized_tickers(app.market.tickers)
    if not expected:
        return False
    file_tickers = _holdings_file_tickers(path)
    if file_tickers == expected:
        return True
    manifest_tickers = _manifest_tickers(MANIFEST_PATH)
    return manifest_tickers == expected if manifest_tickers is not None else False


def ensure_holdings(app: AppConfig) -> Path:
    """Ensure holdings parquet exists and matches the active market tickers."""
    path = _holdings_path(app)
    if path.is_file() and _holdings_match_config(app, path):
        return path

    if path.is_file():
        LOGGER.warning(
            "Refreshing stale holdings at %s (tickers no longer match active run profile)",
            path,
        )

    ensure_dir(path.parent)
    tickers = app.market.tickers
    n = len(tickers)
    weight = 1.0 / n if n else 0.0
    df = pl.DataFrame(
        {
            "asset": tickers,
            "weight": [weight] * n,
        }
    )
    df.write_parquet(path)
    return path


def portfolio_weighting_mode(app: AppConfig) -> str:
    """Active weighting mode from run profile or risk config."""
    if app.run is not None:
        return str(app.run.portfolio.weighting)
    return str(app.risk.optimization.weighting or app.risk.portfolio.weight_mode)


def read_holdings_weights(app: AppConfig) -> dict[str, float]:
    """Load weights stored in holdings.parquet (may be equal placeholder before optimization)."""
    path = ensure_holdings(app)
    df = pl.read_parquet(path)
    return dict(zip(df["asset"].to_list(), df["weight"].to_list(), strict=False))


def _load_optimization_weights(app: AppConfig) -> dict[str, float] | None:
    """Weights from ``optimize_portfolios`` (max_sharpe or backtest weight_source label)."""
    if not RISK_OPT_WEIGHTS_PATH.is_file():
        return None
    df = pl.read_parquet(RISK_OPT_WEIGHTS_PATH)
    if df.is_empty() or "portfolio" not in df.columns:
        return None
    label = app.research.backtest.weight_source or "max_sharpe"
    sub = df.filter(pl.col("portfolio") == label)
    if sub.is_empty():
        sub = df.filter(pl.col("portfolio") == "max_sharpe")
    if sub.is_empty():
        return None
    return dict(zip(sub["asset"].to_list(), sub["weight"].to_list(), strict=False))


def load_portfolio_weights(app: AppConfig) -> dict[str, float]:
    """Effective portfolio weights for UI and analytics.

  - ``equal`` / ``manual``: from holdings (written by ``aqre prepare``).
  - ``optimised`` / ``partial``: from ``data/risk/optimization/optimal_weights.parquet``
    when present; otherwise holdings with a warning.
    """
    mode = portfolio_weighting_mode(app)
    holdings = read_holdings_weights(app)

    if mode in ("optimised", "partial"):
        opt = _load_optimization_weights(app)
        if opt:
            return opt
        LOGGER.warning(
            "Weighting mode %r but %s missing; using holdings weights. "
            "Run: dvc repro optimize_portfolios",
            mode,
            RISK_OPT_WEIGHTS_PATH.name,
        )
        return holdings

    return holdings


def load_weights(app: AppConfig) -> dict[str, float]:
    """Load portfolio weights (respects run-profile weighting mode)."""
    return load_portfolio_weights(app)
