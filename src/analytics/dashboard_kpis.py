"""KPI cards for Streamlit dashboard (filtered window)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.analytics.charts.context import DateRange, filter_by_date_range
from src.backtesting.metrics.performance import max_drawdown, sharpe_ratio
from src.utils.config import AppConfig
from src.utils.paths import RESEARCH_BACKTESTS_DIR, RESEARCH_PERFORMANCE_PATH, RISK_VAR_DIR


@dataclass(frozen=True)
class WindowKpis:
    sharpe: float | None
    max_drawdown: float | None
    var_95: float | None


def _load_equity(app: AppConfig, dr: DateRange | None) -> pl.DataFrame | None:
    exp = app.research.meta.experiment_id
    path = RESEARCH_BACKTESTS_DIR / f"experiment_id={exp}" / "equity_curve.parquet"
    if not path.is_file():
        return None
    eq = pl.read_parquet(path).sort("timestamp")
    return filter_by_date_range(eq, "timestamp", dr)


def compute_window_kpis(app: AppConfig, dr: DateRange | None) -> WindowKpis:
    """Sharpe / max DD from filtered equity; VaR95 from pipeline summary."""
    sharpe_val: float | None = None
    mdd_val: float | None = None
    var_val: float | None = None

    eq = _load_equity(app, dr)
    if eq is not None and eq.height > 2 and "portfolio_return" in eq.columns:
        rets = eq["portfolio_return"]
        sharpe_val = sharpe_ratio(rets, app.features.risk_free_rate, app.features.annualization_factor)
        if "equity" in eq.columns:
            mdd_val = max_drawdown(eq["equity"])

    if RESEARCH_PERFORMANCE_PATH.is_file():
        perf = pl.read_parquet(RESEARCH_PERFORMANCE_PATH)
        if perf.height and "var_95" in perf.columns:
            v = perf["var_95"][0]
            var_val = float(v) if v is not None else None

    if var_val is None:
        var_path = RISK_VAR_DIR / "var_metrics.parquet"
        if var_path.is_file():
            var_df = pl.read_parquet(var_path).filter(pl.col("confidence") == 0.95)
            if var_df.height:
                var_val = float(var_df["var"][0])

    return WindowKpis(sharpe=sharpe_val, max_drawdown=mdd_val, var_95=var_val)


def equity_csv_bytes(app: AppConfig, dr: DateRange | None) -> bytes | None:
    eq = _load_equity(app, dr)
    if eq is None or eq.height == 0:
        return None
    return eq.write_csv().encode("utf-8")
