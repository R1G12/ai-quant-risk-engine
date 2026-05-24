"""Chart registry: build all Phase 2–4 Plotly figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.analytics.charts.base import ChartSpec
from src.analytics.validation import assert_equity_sane
from src.simulation.monte_carlo.engine import sample_paths_for_plot
from src.simulation.pipeline._calibration import calibrate_gbm, calibrate_multivariate
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.paths import (
    RESEARCH_BACKTESTS_DIR,
    RESEARCH_SCENARIOS_DIR,
    RESEARCH_SIMULATIONS_DIR,
    RESEARCH_STRESS_DIR,
    RISK_CORRELATIONS_DIR,
    RISK_FRONTIER_PATH,
    RISK_PORTFOLIO_METRICS_PATH,
    RISK_REGIMES_PATH,
    RISK_VAR_DIR,
    RISK_VOLATILITY_DIR,
)

LOGGER = get_logger(__name__)

VAR_NOTE = (
    "VaR/CVaR are reported on terminal simple returns over the simulation horizon. "
    "More negative values indicate worse tail losses."
)


def _go():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


def _build_regime_transition(app: AppConfig):
    go, _ = _go()
    if not RISK_REGIMES_PATH.is_file():
        return None
    reg = pl.read_parquet(RISK_REGIMES_PATH).sort("timestamp")
    if reg.height < 2:
        return None
    labels = reg["regime_label"].to_list()
    states = sorted(set(labels))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    mat = np.zeros((n, n))
    for a, b in zip(labels[:-1], labels[1:], strict=False):
        mat[idx[a], idx[b]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prob = mat / row_sums
    fig = go.Figure(data=go.Heatmap(z=prob, x=states, y=states, colorscale="Blues"))
    fig.update_layout(
        title="Regime transition probabilities (empirical)",
        template="plotly_dark",
        xaxis_title="To",
        yaxis_title="From",
    )
    return fig


def _build_equity(app: AppConfig):
    go, _ = _go()
    exp = app.research.meta.experiment_id
    path = RESEARCH_BACKTESTS_DIR / f"experiment_id={exp}" / "equity_curve.parquet"
    if not path.is_file():
        return None
    eq = pl.read_parquet(path).sort("timestamp")
    try:
        assert_equity_sane(eq)
    except ValueError as exc:
        LOGGER.warning("Equity validation: %s", exc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["timestamp"], y=eq["equity"], name="Equity", mode="lines"))
    if "drawdown" in eq.columns:
        fig.add_trace(
            go.Scatter(
                x=eq["timestamp"],
                y=eq["drawdown"],
                name="Drawdown",
                mode="lines",
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )
        fig.update_layout(
            yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".1%"),
        )
    fig.update_layout(title="Backtest equity curve", template="plotly_dark")
    return fig


def _build_rolling_sharpe(app: AppConfig):
    go, _ = _go()
    exp = app.research.meta.experiment_id
    path = RESEARCH_BACKTESTS_DIR / f"experiment_id={exp}" / "rolling_metrics.parquet"
    if not path.is_file():
        return None
    rm = pl.read_parquet(path).sort("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rm["timestamp"], y=rm["rolling_sharpe"], name="Rolling Sharpe"))
    fig.update_layout(title="Rolling Sharpe (21d)", template="plotly_dark")
    return fig


def _build_mc_fan(app: AppConfig):
    go, _ = _go()
    mu, sigma = calibrate_gbm(app)
    paths = sample_paths_for_plot(mu, sigma, app.research.simulation)
    t = np.arange(paths.shape[1])
    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=0)
    fig = go.Figure()
    n_show = min(20, paths.shape[0])
    for i in range(n_show):
        fig.add_trace(
            go.Scatter(x=t, y=paths[i], mode="lines", line=dict(width=0.5, color="gray"), showlegend=False)
        )
    fig.add_trace(go.Scatter(x=t, y=p50, name="Median", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=t, y=p95, name="P95", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=p5, name="P5", line=dict(dash="dash"), fill="tonexty"))
    fig.update_layout(title="Monte Carlo fan chart (GBM)", template="plotly_dark", xaxis_title="Step")
    return fig


def _build_mc_hist(sim_type: str):
    def _inner(app: AppConfig):
        go, make_subplots = _go()
        exp = app.research.meta.experiment_id
        p = (
            RESEARCH_SIMULATIONS_DIR
            / f"experiment_id={exp}"
            / f"simulation_type={sim_type}"
            / "paths_summary.parquet"
        )
        if not p.is_file():
            return None
        df = pl.read_parquet(p)
        if sim_type == "regime_gbm" and "regime" in df.columns:
            regimes = df["regime"].unique().to_list()
            fig = make_subplots(rows=1, cols=len(regimes), subplot_titles=[str(r) for r in regimes])
            for i, r in enumerate(regimes, start=1):
                sub = df.filter(pl.col("regime") == r)
                fig.add_trace(
                    go.Histogram(x=sub["terminal_return"], name=str(r), showlegend=False),
                    row=1,
                    col=i,
                )
            fig.update_layout(title="Regime GBM — terminal returns by regime", template="plotly_dark")
            return fig
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Terminal return", "Max drawdown"))
        fig.add_trace(go.Histogram(x=df["terminal_return"], name="terminal"), row=1, col=1)
        fig.add_trace(go.Histogram(x=df["max_drawdown"], name="drawdown"), row=1, col=2)
        fig.update_layout(title=f"Simulation — {sim_type}", template="plotly_dark")
        return fig

    return _inner


def _build_stress(app: AppConfig):
    go, _ = _go()
    exp = app.research.meta.experiment_id
    path = RESEARCH_STRESS_DIR / f"experiment_id={exp}" / "stress_metrics.parquet"
    if not path.is_file():
        return None
    stress = pl.read_parquet(path).sort("scenario")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stress["scenario"], y=stress["var"], name="VaR"))
    fig.update_layout(title="Stress VaR by scenario", template="plotly_dark")
    return fig


def _build_scenario(app: AppConfig):
    go, _ = _go()
    exp = app.research.meta.experiment_id
    path = RESEARCH_SCENARIOS_DIR / f"experiment_id={exp}" / "scenario_comparison.parquet"
    if not path.is_file():
        return None
    scen = pl.read_parquet(path).sort("scenario")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scen["scenario"], y=scen["var"], name="VaR"))
    fig.add_trace(go.Bar(x=scen["scenario"], y=scen["cvar"], name="CVaR"))
    fig.update_layout(title="Scenario comparison", template="plotly_dark", barmode="group")
    return fig


def _build_frontier(_app: AppConfig):
    go, _ = _go()
    if not RISK_FRONTIER_PATH.is_file():
        return None
    fr = pl.read_parquet(RISK_FRONTIER_PATH)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fr["volatility"], y=fr["expected_return"], mode="lines+markers", name="Frontier")
    )
    fig.update_layout(title="Efficient frontier", template="plotly_dark")
    return fig


def _build_vol(_app: AppConfig):
    go, _ = _go()
    vol_glob = list(RISK_VOLATILITY_DIR.glob("year=*/month=*/*.parquet"))
    if not vol_glob:
        return None
    vol = pl.scan_parquet(vol_glob).filter(pl.col("scope") == "portfolio").collect()
    if not vol.height:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol["timestamp"], y=vol["ewma_vol"], name="EWMA"))
    fig.add_trace(go.Scatter(x=vol["timestamp"], y=vol["rolling_vol"], name="Rolling"))
    fig.update_layout(title="Portfolio volatility", template="plotly_dark")
    return fig


def _build_risk_drawdown(_app: AppConfig):
    go, _ = _go()
    if not RISK_PORTFOLIO_METRICS_PATH.is_file():
        return None
    pm = pl.read_parquet(RISK_PORTFOLIO_METRICS_PATH)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pm["timestamp"], y=pm["drawdown"], fill="tozeroy"))
    fig.update_layout(title="Historical portfolio drawdown", template="plotly_dark")
    return fig


def _build_var(_app: AppConfig):
    go, _ = _go()
    var_path = RISK_VAR_DIR / "var_metrics.parquet"
    if not var_path.is_file():
        return None
    var_df = pl.read_parquet(var_path)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"{m}_{c}" for m, c in zip(var_df["method"], var_df["confidence"], strict=False)],
            y=var_df["var"],
        )
    )
    fig.update_layout(title="VaR by method", template="plotly_dark")
    return fig


def _build_corr(_app: AppConfig):
    go, _ = _go()
    corr_path = RISK_CORRELATIONS_DIR / "correlations_latest.parquet"
    if not corr_path.is_file():
        return None
    corr = pl.read_parquet(corr_path).filter(pl.col("metric") == "corr")
    if not corr.height:
        return None
    assets = sorted(set(corr["asset_i"].to_list()))
    mat = np.eye(len(assets))
    idx = {a: i for i, a in enumerate(assets)}
    for row in corr.iter_rows(named=True):
        i, j = idx[row["asset_i"]], idx[row["asset_j"]]
        mat[i, j] = row["value"] if row["value"] is not None else 0.0
    fig = go.Figure(data=go.Heatmap(z=mat, x=assets, y=assets, colorscale="RdBu", zmid=0))
    fig.update_layout(title="Asset correlation heatmap", template="plotly_dark")
    return fig


def build_chart_registry() -> list[ChartSpec]:
    """All dashboard charts in display order."""
    return [
        ChartSpec(
            "equity_curve",
            "Backtest equity curve",
            "Historical performance using fixed Phase 3 weights. "
            "Costs apply only on rebalance days (monthly). Walk-forward window IDs are in the data.",
            "Backtest",
            _build_equity,
            "equity_curve.html",
        ),
        ChartSpec(
            "rolling_sharpe",
            "Rolling Sharpe",
            "21-day rolling annualized Sharpe of backtest daily returns.",
            "Backtest",
            _build_rolling_sharpe,
            "rolling_sharpe.html",
        ),
        ChartSpec(
            "mc_fan_gbm",
            "Monte Carlo fan chart",
            "Sample simulated wealth paths with median and 5th/95th percentile bands. "
            "Synthetic forward-looking scenarios, not a forecast.",
            "Simulation",
            _build_mc_fan,
            "mc_fan_gbm.html",
        ),
        ChartSpec(
            "mc_paths_gbm",
            "GBM simulation distribution",
            "Distribution of terminal returns and max drawdown across simulated paths (1D GBM).",
            "Simulation",
            _build_mc_hist("gbm"),
            "mc_paths_gbm.html",
        ),
        ChartSpec(
            "mc_paths_multivariate",
            "Multivariate simulation distribution",
            "Portfolio paths with correlated asset shocks from the estimated covariance matrix.",
            "Simulation",
            _build_mc_hist("multivariate"),
            "mc_paths_multivariate.html",
        ),
        ChartSpec(
            "mc_paths_regime_gbm",
            "Regime-conditioned simulation",
            "Terminal return distributions per HMM regime label.",
            "Simulation",
            _build_mc_hist("regime_gbm"),
            "mc_paths_regime_gbm.html",
        ),
        ChartSpec(
            "regime_transitions",
            "Regime transitions",
            "Empirical probability of moving between HMM regime labels on consecutive days.",
            "Simulation",
            _build_regime_transition,
            "regime_transitions.html",
        ),
        ChartSpec(
            "stress_dashboard",
            "Stress test VaR",
            "Tail risk under shocked vol/drift parameters. " + VAR_NOTE,
            "Stress",
            _build_stress,
            "stress_dashboard.html",
        ),
        ChartSpec(
            "scenario_comparison",
            "Scenario comparison",
            "Side-by-side VaR and CVaR across YAML-defined scenarios. " + VAR_NOTE,
            "Stress",
            _build_scenario,
            "scenario_comparison.html",
        ),
        ChartSpec(
            "efficient_frontier",
            "Efficient frontier",
            "Risk/return trade-off from Markowitz optimization (Phase 3).",
            "Risk",
            _build_frontier,
            "efficient_frontier.html",
        ),
        ChartSpec(
            "rolling_volatility",
            "Rolling volatility",
            "EWMA vs rolling portfolio volatility from Phase 3 metrics.",
            "Risk",
            _build_vol,
            "rolling_volatility.html",
        ),
        ChartSpec(
            "drawdown",
            "Portfolio drawdown",
            "Historical drawdown series on the optimized portfolio.",
            "Risk",
            _build_risk_drawdown,
            "drawdown.html",
        ),
        ChartSpec(
            "var_distribution",
            "VaR by method",
            "Value-at-Risk estimates from historical, parametric, and Monte Carlo engines.",
            "Risk",
            _build_var,
            "var_distribution.html",
        ),
        ChartSpec(
            "correlation_heatmap",
            "Correlation heatmap",
            "Pairwise return correlations between holdings.",
            "Risk",
            _build_corr,
            "correlation_heatmap.html",
        ),
    ]


def write_standalone_charts(
    registry: list[ChartSpec],
    app: AppConfig,
    research_dir: Path,
    risk_dir: Path,
) -> dict[str, object]:
    """Write per-chart HTML files; return built figures for dashboard."""
    figures: dict[str, object] = {}
    for spec in registry:
        try:
            fig = spec.builder(app)
        except Exception as exc:
            LOGGER.warning("Chart %s failed: %s", spec.id, exc)
            continue
        if fig is None:
            continue
        figures[spec.id] = fig
        if spec.standalone_name:
            out = risk_dir if spec.group == "Risk" else research_dir
            out.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out / spec.standalone_name), include_plotlyjs="cdn")
    return figures
