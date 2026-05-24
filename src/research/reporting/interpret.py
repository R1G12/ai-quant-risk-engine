"""Generate interpreted Markdown research report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.utils.config import AppConfig
from src.utils.paths import (
    RESEARCH_BACKTESTS_DIR,
    RESEARCH_PERFORMANCE_PATH,
    RESEARCH_SCENARIOS_DIR,
    RESEARCH_SIMULATIONS_DIR,
    RESEARCH_STRESS_DIR,
)


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def write_interpreted_report(app: AppConfig, output_path: Path) -> None:
    """Write REPORT.md with plain-language metric interpretation."""
    exp = app.research.meta.experiment_id
    lines: list[str] = [
        f"# Research Report — `{exp}`",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Executive summary",
        "",
    ]

    if RESEARCH_PERFORMANCE_PATH.is_file():
        perf = pl.read_parquet(RESEARCH_PERFORMANCE_PATH)
        bt = perf.filter(pl.col("source") == "backtest")
        if bt.height:
            row = bt.row(0, named=True)
            sharpe = row.get("sharpe")
            mdd = row.get("max_drawdown")
            if sharpe is not None:
                lines.append(f"- **Backtest Sharpe** (annualized proxy): {sharpe:.3f}")
            else:
                lines.append("- **Backtest Sharpe**: n/a")
            lines.append(
                f"- **Backtest max drawdown**: {_pct(mdd)}" if mdd is not None else "- Max drawdown: n/a"
            )
        for sim in perf.filter(pl.col("source").str.starts_with("simulation:")).iter_rows(named=True):
            src = sim["source"].replace("simulation:", "")
            lines.append(
                f"- **MC {src}** VaR95: {_pct(sim.get('var_95'))}, CVaR95: {_pct(sim.get('cvar_95'))}"
            )

    stress_path = RESEARCH_STRESS_DIR / f"experiment_id={exp}" / "stress_metrics.parquet"
    if stress_path.is_file():
        stress = pl.read_parquet(stress_path)
        if stress.height:
            worst = stress.sort("var").row(0, named=True)
            lines.append(
                f"- **Worst stress scenario**: `{worst.get('scenario')}` with VaR95 {_pct(worst.get('var'))}"
            )

    lines.extend(
        [
            "",
            "## Backtest",
            "",
            "Historical replay using **fixed Phase 3 optimal weights** (no re-optimization inside the backtest).",
            f"- Rebalance: **{app.research.backtest.rebalance_freq}**; costs **{app.research.backtest.tc_bps} bps** + "
            f"**{app.research.backtest.slippage_bps} bps** slippage on turnover.",
            f"- Walk-forward labeling: **{app.research.backtest.walk_forward_train_days}** train warmup days, "
            f"**{app.research.backtest.walk_forward_test_days}**-day test blocks.",
            "",
            "Interpretation: Sharpe above 0 suggests risk-adjusted gains over the sample; compare max drawdown to your risk budget.",
            "",
            "## Simulations",
            "",
            f"Monte Carlo with **{app.research.simulation.n_paths}** paths, **{app.research.simulation.horizon_days}**-day horizon.",
            "These are **model scenarios**, not forecasts. VaR/CVaR describe the left tail of terminal returns.",
            "",
        ]
    )

    sim_base = RESEARCH_SIMULATIONS_DIR / f"experiment_id={exp}"
    if sim_base.is_dir():
        for d in sorted(sim_base.glob("simulation_type=*")):
            tail = d / "tail_metrics.parquet"
            if tail.is_file():
                t = pl.read_parquet(tail)
                if t.height:
                    r = t.row(0, named=True)
                    st = d.name.split("=", 1)[-1]
                    lines.append(
                        f"- `{st}`: mean terminal return {_pct(r.get('mean_return'))}, "
                        f"vol {_pct(r.get('volatility'))}, VaR95 {_pct(r.get('var'))}"
                    )

    lines.extend(["", "## Stress and scenarios", ""])
    scen_path = RESEARCH_SCENARIOS_DIR / f"experiment_id={exp}" / "scenario_comparison.parquet"
    if scen_path.is_file():
        scen = pl.read_parquet(scen_path).sort("var")
        lines.append("| Scenario | VaR95 | CVaR95 |")
        lines.append("|----------|-------|--------|")
        for r in scen.iter_rows(named=True):
            lines.append(f"| {r['scenario']} | {_pct(r['var'])} | {_pct(r['cvar'])} |")

    eq_path = RESEARCH_BACKTESTS_DIR / f"experiment_id={exp}" / "equity_curve.parquet"
    lines.extend(["", "## Data quality", ""])
    if eq_path.is_file():
        eq = pl.read_parquet(eq_path)
        dupes = eq.group_by("timestamp").len().filter(pl.col("len") > 1).height
        lines.append(f"- Equity curve rows: **{eq.height}**; duplicate timestamps: **{dupes}**")
        if dupes:
            lines.append("- ⚠ Duplicate timestamps were detected — review upstream feature merge.")
        else:
            lines.append("- ✓ Timestamps are unique in the backtest output.")
    else:
        lines.append("- Backtest equity file not found.")

    lines.extend(
        [
            "",
            "## How to explore",
            "",
            "Open the interactive dashboard:",
            "",
            "`data/analytics/dashboard/index.html`",
            "",
            "Individual Plotly files remain under `data/analytics/research/` and `data/analytics/risk/`.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
