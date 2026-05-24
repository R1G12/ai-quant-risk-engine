"""Portfolio holdings helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.config import AppConfig
from src.utils.paths import HOLDINGS_PATH, PROJECT_ROOT, ensure_dir


def ensure_holdings(app: AppConfig) -> Path:
    """Ensure holdings parquet exists (equal-weight sample if missing)."""
    path = Path(app.risk.portfolio.holdings_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return path

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


def load_weights(app: AppConfig) -> dict[str, float]:
    """Load static weights from holdings file."""
    path = ensure_holdings(app)
    df = pl.read_parquet(path)
    return dict(zip(df["asset"].to_list(), df["weight"].to_list(), strict=False))
