"""Data quality validators on parquet artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.utils.config import AppConfig
from src.utils.paths import RISK_DATASET_PATH


@dataclass(frozen=True)
class QualityCheck:
    name: str
    ok: bool
    message: str


def run_quality_checks(app: AppConfig) -> list[QualityCheck]:
    checks: list[QualityCheck] = []
    if not RISK_DATASET_PATH.is_file():
        checks.append(
            QualityCheck("risk_dataset_rows", False, "risk_dataset.parquet missing")
        )
        return checks

    df = pl.scan_parquet(RISK_DATASET_PATH)
    n = df.select(pl.len()).collect().item()
    null_returns = (
        df.filter(pl.col("returns").is_null()).select(pl.len()).collect().item()
    )
    tickers = df.select("ticker").unique().collect().height

    checks.append(
        QualityCheck(
            "risk_dataset_rows",
            n > 0,
            f"{n} rows across {tickers} tickers",
        )
    )
    checks.append(
        QualityCheck(
            "returns_null_rate",
            null_returns / max(n, 1) < 0.5,
            f"{null_returns} null returns ({null_returns / max(n, 1):.1%})",
        )
    )
    expected = set(app.market.tickers)
    present = set(df.select("ticker").unique().collect()["ticker"].to_list())
    missing = expected - present
    checks.append(
        QualityCheck(
            "ticker_coverage",
            not missing,
            "All config tickers present" if not missing else f"Missing: {sorted(missing)}",
        )
    )
    return checks
