"""Risk metrics endpoints."""

from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException

from src.utils.paths import RISK_VAR_DIR, RISK_CORRELATIONS_DIR

router = APIRouter()


@router.get("/var")
def var_metrics() -> dict:
    path = RISK_VAR_DIR / "var_metrics.parquet"
    if not path.is_file():
        raise HTTPException(404, "VaR metrics not found; run Phase 3 pipeline.")
    df = pl.read_parquet(path)
    return {"rows": df.to_dicts()}


@router.get("/correlations")
def correlations() -> dict:
    path = RISK_CORRELATIONS_DIR / "correlations_latest.parquet"
    if not path.is_file():
        raise HTTPException(404, "Correlations not found.")
    df = pl.read_parquet(path)
    return {"rows": df.to_dicts()}
