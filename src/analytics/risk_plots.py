"""Phase 3 Plotly risk visualizations."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.logger import get_logger
from src.utils.paths import (
    ANALYTICS_RISK_DIR,
    RISK_CORRELATIONS_DIR,
    RISK_FRONTIER_PATH,
    RISK_PORTFOLIO_METRICS_PATH,
    RISK_VAR_DIR,
    RISK_VOLATILITY_DIR,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def write_risk_dashboard(output_dir: Path | None = None) -> None:
    """Write HTML plots: frontier, vol, drawdown, VaR, correlation heatmap."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        LOGGER.warning("Plotly not installed; skipping risk dashboard")
        return

    out = output_dir or ANALYTICS_RISK_DIR
    ensure_dir(out)

    if RISK_FRONTIER_PATH.is_file():
        fr = pl.read_parquet(RISK_FRONTIER_PATH)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=fr["volatility"], y=fr["expected_return"], mode="lines+markers", name="frontier")
        )
        fig.update_layout(title="Efficient Frontier", template="plotly_dark")
        fig.write_html(str(out / "efficient_frontier.html"), include_plotlyjs="cdn")

    vol_glob = list(RISK_VOLATILITY_DIR.glob("year=*/month=*/*.parquet"))
    if vol_glob:
        vol = pl.scan_parquet(vol_glob).filter(pl.col("scope") == "portfolio").collect()
        if vol.height:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=vol["timestamp"], y=vol["ewma_vol"], name="EWMA vol"))
            fig2.add_trace(go.Scatter(x=vol["timestamp"], y=vol["rolling_vol"], name="Rolling vol"))
            fig2.update_layout(title="Rolling vs EWMA Volatility", template="plotly_dark")
            fig2.write_html(str(out / "rolling_volatility.html"), include_plotlyjs="cdn")

    if RISK_PORTFOLIO_METRICS_PATH.is_file():
        pm = pl.read_parquet(RISK_PORTFOLIO_METRICS_PATH)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=pm["timestamp"], y=pm["drawdown"], fill="tozeroy", name="drawdown"))
        fig3.update_layout(title="Portfolio Drawdown", template="plotly_dark")
        fig3.write_html(str(out / "drawdown.html"), include_plotlyjs="cdn")

    var_path = RISK_VAR_DIR / "var_metrics.parquet"
    if var_path.is_file():
        var_df = pl.read_parquet(var_path)
        fig4 = go.Figure()
        fig4.add_trace(
            go.Bar(
                x=[f"{m}_{c}" for m, c in zip(var_df["method"], var_df["confidence"], strict=False)],
                y=var_df["var"],
            )
        )
        fig4.update_layout(title="VaR by Method", template="plotly_dark")
        fig4.write_html(str(out / "var_distribution.html"), include_plotlyjs="cdn")

    corr_path = RISK_CORRELATIONS_DIR / "correlations_latest.parquet"
    if corr_path.is_file():
        corr = pl.read_parquet(corr_path).filter(pl.col("metric") == "corr")
        if corr.height:
            assets = sorted(set(corr["asset_i"].to_list()))
            import numpy as np

            mat = np.eye(len(assets))
            idx = {a: i for i, a in enumerate(assets)}
            for row in corr.iter_rows(named=True):
                i, j = idx[row["asset_i"]], idx[row["asset_j"]]
                mat[i, j] = row["value"] if row["value"] is not None else 0.0
            fig5 = go.Figure(data=go.Heatmap(z=mat, x=assets, y=assets, colorscale="RdBu"))
            fig5.update_layout(title="Correlation Heatmap", template="plotly_dark")
            fig5.write_html(str(out / "correlation_heatmap.html"), include_plotlyjs="cdn")

    LOGGER.info("Risk dashboard written", extra={"dir": str(out)})
