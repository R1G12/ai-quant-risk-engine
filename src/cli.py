"""Unified CLI for pipeline phase bundles and dashboard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from src.utils.config import apply_run_profile_env, load_app_config, load_run_config, run_profile_env

app = typer.Typer(
    name="aqre",
    help="AI Quant Risk Engine — run DVC phase bundles, show config, launch dashboard.",
    no_args_is_help=True,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE_STAGES: dict[str, list[str]] = {
    "phase1": ["ingest", "preprocess", "sentiment"],
    "phase2": [
        "ingest_market_data",
        "clean_market_data",
        "generate_returns",
        "generate_volatility_features",
        "generate_technical_features",
        "generate_sentiment_features",
        "merge_features",
    ],
    "phase3": [
        "generate_volatility_metrics",
        "generate_var_metrics",
        "generate_cvar_metrics",
        "generate_correlations",
        "generate_portfolio_metrics",
        "optimize_portfolios",
        "generate_efficient_frontier",
    ],
    "phase4": [
        "generate_simulations",
        "run_backtests",
        "run_stress_tests",
        "run_scenario_analysis",
        "evaluate_performance",
        "compare_experiments",
        "generate_research_reports",
    ],
}


def _run_dvc(stages: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    cmd = ["dvc", "repro", *stages]
    typer.echo(" ".join(cmd))
    if dry_run:
        return
    run_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=run_env, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


config_app = typer.Typer(help="Inspect resolved configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    profile: str | None = typer.Option(None, "--profile", help="Path to configs/run.yaml"),
) -> None:
    """Print resolved configuration (tickers, dates, experiment id)."""
    cfg = load_app_config(Path(profile) if profile else None)
    typer.echo(f"experiment_id:     {cfg.research.meta.experiment_id}")
    typer.echo(f"market.source:     {cfg.market.source}")
    typer.echo(f"market.tickers:    {', '.join(cfg.market.tickers)}")
    typer.echo(f"market.dates:      {cfg.market.start_date} -> {cfg.market.end_date}")
    typer.echo(f"rolling_window:    {cfg.market.use_rolling_window} ({cfg.market.rolling_days}d)")
    news_source = os.getenv("NEWS_SOURCE")
    if not news_source:
        from src.ingestion.adapters import get_news_source

        news_source = get_news_source()
    typer.echo(f"ingestion.source:  {news_source}")
    if cfg.run is not None:
        typer.echo(f"run.mode:          {cfg.run.mode}")
        typer.echo(f"run.profile:       {cfg.run.profile_path}")
        typer.echo(f"run.weighting:     {cfg.run.portfolio.weighting}")
        typer.echo(f"run.allow_shorts:  {cfg.run.portfolio.allow_shorts}")
        typer.echo(f"run.anchors:       {cfg.run.portfolio.anchor_weights}")


run_app = typer.Typer(help="Run DVC stage bundles.")
app.add_typer(run_app, name="run")


def _profile_path_option(profile: str | None) -> Path | None:
    return Path(profile) if profile else None


@run_app.command("profile")
def run_profile(
    profile: str | None = typer.Option(None, "--profile", help="Path to configs/run.yaml"),
    skip_repro: bool = typer.Option(False, "--skip-repro", help="Skip dvc repro"),
    pin_dates: bool = typer.Option(False, "--pin-dates", help="Set MARKET_PIN_DATES=1"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands only"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Launch dashboard after pipeline"),
    legacy: bool = typer.Option(False, "--legacy", help="Use legacy Phase 4 dashboard"),
) -> None:
    """Prepare run profile, run full pipeline, optionally launch dashboard."""
    profile_path = _profile_path_option(profile)
    run_cfg = load_run_config(profile_path)
    if run_cfg is None:
        typer.echo("Run profile not found. Create configs/run.yaml or pass --profile.", err=True)
        raise typer.Exit(1)
    env = run_profile_env(run_cfg, force=True)
    if not dry_run:
        from src.portfolio.prepare import materialize_run

        apply_run_profile_env(run_cfg, force=True)
        app = load_app_config(profile_path)
        holdings = materialize_run(app)
        typer.echo(f"Prepared holdings: {holdings}")
    if pin_dates:
        env["MARKET_PIN_DATES"] = "1"
    if not skip_repro:
        run_env = {**os.environ, **env}
        if not dry_run and run_cfg.mode == "live":
            from src.ingestion.news_refresh import holdings_missing_from_news

            tickers = [str(t) for t in run_cfg.market.tickers]
            if holdings_missing_from_news(tickers):
                typer.echo(
                    "News/sentiment cache has no articles for current holdings; "
                    "forcing ingest → preprocess → sentiment (NEWS_SOURCE=yfinance).",
                    err=True,
                )
                force_cmd = ["dvc", "repro", "-f", "ingest", "preprocess", "sentiment"]
                typer.echo(" ".join(force_cmd))
                force = subprocess.run(force_cmd, cwd=PROJECT_ROOT, env=run_env, check=False)
                if force.returncode != 0:
                    raise typer.Exit(force.returncode)
        cmd = ["dvc", "repro"]
        typer.echo(" ".join(cmd))
        if not dry_run:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=run_env, check=False)
            if result.returncode != 0:
                raise typer.Exit(result.returncode)
    if dashboard and not dry_run:
        dash_args = ["dashboard", "--legacy"] if legacy else ["dashboard"]
        raise typer.Exit(
            subprocess.run(
                [sys.executable, "-m", "src.cli", *dash_args],
                cwd=PROJECT_ROOT,
                check=False,
            ).returncode
        )


@app.command("prepare")
def prepare_cmd(
    profile: str | None = typer.Option(None, "--profile", help="Path to configs/run.yaml"),
) -> None:
    """Materialize holdings and run manifest from configs/run.yaml."""
    from src.portfolio.prepare import materialize_run

    profile_path = _profile_path_option(profile)
    run_cfg = load_run_config(profile_path)
    if run_cfg is None:
        typer.echo("Run profile not found. Create configs/run.yaml or pass --profile.", err=True)
        raise typer.Exit(1)
    apply_run_profile_env(run_cfg, force=True)
    app = load_app_config(profile_path)
    path = materialize_run(app)
    typer.echo(f"Holdings written: {path}")
    try:
        from src.portfolio.tracker.ingest import ingest_trades

        trades_path = ingest_trades(app)
        typer.echo(f"Tracker trades: {trades_path}")
    except FileNotFoundError as exc:
        typer.echo(f"Tracker ingest skipped: {exc}", err=True)
    manifest_path = PROJECT_ROOT / "data" / "run_manifest.json"
    typer.echo(f"Manifest: {manifest_path}")
    if manifest_path.is_file():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        skipped = manifest.get("skipped_tickers") or []
        if skipped:
            typer.echo(f"Skipped {len(skipped)} unavailable ticker(s): {', '.join(skipped)}")


tracker_app = typer.Typer(help="Portfolio trade ledger (Excel → parquet).")
app.add_typer(tracker_app, name="tracker")


@tracker_app.command("ingest")
def tracker_ingest(
    force: bool = typer.Option(False, "--force", help="Re-ingest even if parquet is newer"),
) -> None:
    """Read input_trades.xlsx (or dummy_portfolio.xlsx) into trades.parquet."""
    from src.portfolio.tracker.ingest import ingest_trades
    from src.utils.config import load_app_config

    out = ingest_trades(load_app_config(), force=force)
    typer.echo(f"Trades written: {out}")


@run_app.command("all")
def run_all(
    market_source: str | None = typer.Option(None, "--market-source", help="sample or yfinance"),
    pin_dates: bool = typer.Option(False, "--pin-dates", help="Set MARKET_PIN_DATES=1"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Run full pipeline (dvc repro)."""
    env: dict[str, str] = {}
    if market_source:
        env["MARKET_SOURCE"] = market_source
    if pin_dates:
        env["MARKET_PIN_DATES"] = "1"
    cmd = ["dvc", "repro"]
    typer.echo(" ".join(cmd))
    if dry_run:
        return
    run_env = {**os.environ, **env}
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=run_env, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


def _run_phase(
    phase: str,
    *,
    market_source: str | None,
    pin_dates: bool,
    dry_run: bool,
) -> None:
    if phase not in PHASE_STAGES:
        typer.echo(f"Unknown phase: {phase}", err=True)
        raise typer.Exit(1)
    env: dict[str, str] = {}
    if market_source:
        env["MARKET_SOURCE"] = market_source
    if pin_dates:
        env["MARKET_PIN_DATES"] = "1"
    _run_dvc(PHASE_STAGES[phase], dry_run=dry_run, env=env)


@run_app.command("phase1")
def run_phase1(
    market_source: str | None = typer.Option(None, "--market-source", help="sample or yfinance"),
    pin_dates: bool = typer.Option(False, "--pin-dates", help="Set MARKET_PIN_DATES=1"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Run Phase 1 (ingest, preprocess, sentiment)."""
    _run_phase("phase1", market_source=market_source, pin_dates=pin_dates, dry_run=dry_run)


@run_app.command("phase2")
def run_phase2(
    market_source: str | None = typer.Option(None, "--market-source", help="sample or yfinance"),
    pin_dates: bool = typer.Option(False, "--pin-dates", help="Set MARKET_PIN_DATES=1"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Run Phase 2 (market data and features)."""
    _run_phase("phase2", market_source=market_source, pin_dates=pin_dates, dry_run=dry_run)


@run_app.command("phase3")
def run_phase3(
    market_source: str | None = typer.Option(None, "--market-source", help="sample or yfinance"),
    pin_dates: bool = typer.Option(False, "--pin-dates", help="Set MARKET_PIN_DATES=1"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Run Phase 3 (risk engine and optimization)."""
    _run_phase("phase3", market_source=market_source, pin_dates=pin_dates, dry_run=dry_run)


@run_app.command("phase4")
def run_phase4(
    market_source: str | None = typer.Option(None, "--market-source", help="sample or yfinance"),
    pin_dates: bool = typer.Option(False, "--pin-dates", help="Set MARKET_PIN_DATES=1"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Run Phase 4 (simulation, backtest, research reports)."""
    _run_phase("phase4", market_source=market_source, pin_dates=pin_dates, dry_run=dry_run)


@run_app.command("stage")
def run_stage(
    name: str = typer.Argument(..., help="DVC stage name"),
    market_source: str | None = typer.Option(None, "--market-source"),
    pin_dates: bool = typer.Option(False, "--pin-dates"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run a single DVC stage."""
    env: dict[str, str] = {}
    if market_source:
        env["MARKET_SOURCE"] = market_source
    if pin_dates:
        env["MARKET_PIN_DATES"] = "1"
    _run_dvc([name], dry_run=dry_run, env=env)


@app.command("dashboard")
def dashboard(
    legacy: bool = typer.Option(False, "--legacy", help="Use Phase 4 chart explorer only"),
) -> None:
    """Launch Streamlit dashboard (platform or legacy)."""
    script = (
        PROJECT_ROOT / "src" / "analytics" / "streamlit_dashboard.py"
        if legacy
        else PROJECT_ROOT / "src" / "dashboards" / "app.py"
    )
    typer.echo(f"streamlit run {script}")
    raise typer.Exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(script)],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
    )


workflow_app = typer.Typer(help="Orchestrated DVC workflows.")
app.add_typer(workflow_app, name="workflow")


@workflow_app.command("list")
def workflow_list() -> None:
    from src.orchestration.workflows import WORKFLOWS

    for name, spec in WORKFLOWS.items():
        typer.echo(f"{name}: {spec.description}")


@workflow_app.command("run")
def workflow_run(
    name: str = typer.Argument(..., help="Workflow name"),
    market_source: str | None = typer.Option(None, "--market-source"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    from src.orchestration.scheduler import run_workflow

    code = run_workflow(name, market_source=market_source, dry_run=dry_run)
    raise typer.Exit(code)


api_app = typer.Typer(help="Platform API server.")
app.add_typer(api_app, name="api")


@api_app.command("serve")
def api_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start FastAPI (requires .[platform])."""
    args = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", host, "--port", str(port)]
    if reload:
        args.append("--reload")
    raise typer.Exit(subprocess.run(args, cwd=PROJECT_ROOT, check=False).returncode)


copilot_app = typer.Typer(help="Portfolio copilot CLI.")
app.add_typer(copilot_app, name="copilot")


@copilot_app.command("ask")
def copilot_ask(question: str = typer.Argument(..., help="Question for the copilot")) -> None:
    from src.copilot.reasoning.engine import CopilotEngine

    resp = CopilotEngine().ask(question)
    typer.echo(f"## {resp.title} (confidence: {resp.confidence})\n")
    typer.echo(resp.answer)
    if resp.requires_human_review:
        typer.echo("\n[Human validation required]")


if __name__ == "__main__":
    app()
