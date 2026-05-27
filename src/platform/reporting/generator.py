"""Institutional report generation to reports/."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.copilot.context.builder import build_portfolio_context
from src.copilot.reasoning.engine import CopilotEngine
from src.copilot.summarization.report import generate_portfolio_summary
from src.monitoring.alerts.rules import evaluate_alerts
from src.monitoring.health.checks import run_health_checks
from src.research.reporting.interpret import write_interpreted_report
from src.utils.config import AppConfig, load_app_config
from src.utils.logger import get_logger
from src.utils.paths import (
    REPORTS_EXPERIMENTS_DIR,
    REPORTS_GOVERNANCE_DIR,
    REPORTS_PORTFOLIO_DIR,
    REPORTS_RISK_DIR,
    REPORTS_SIMULATIONS_DIR,
    RESEARCH_REPORT_MD_PATH,
    ensure_dir,
)

LOGGER = get_logger(__name__)

COPILOT_QUESTIONS = [
    "Why did portfolio risk increase?",
    "Which assets contribute most to VaR?",
    "What is the current volatility regime?",
]


def _write_copilot_brief(app: AppConfig, out: Path) -> Path:
    engine = CopilotEngine(app)
    ctx = build_portfolio_context(app)
    lines = [
        f"# Copilot risk brief — `{app.research.meta.experiment_id}`",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        generate_portfolio_summary(ctx),
        "",
    ]
    for q in COPILOT_QUESTIONS:
        resp = engine.ask(q, ctx)
        lines.extend([f"## {resp.title}", "", resp.answer, ""])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _write_governance_report(app: AppConfig, out: Path) -> Path:
    health = run_health_checks(app)
    alerts = evaluate_alerts(app)
    lines = [
        "# Governance & observability snapshot",
        "",
        "## Health",
        "",
    ]
    for h in health:
        lines.append(f"- {'[OK]' if h.ok else '[WARN]'} {h.name}: {h.message}")
    lines.extend(["", "## Alerts", ""])
    if alerts:
        for a in alerts:
            lines.append(f"- **{a.severity}**: {a.message}")
    else:
        lines.append("- No active alerts.")
    lines.append("\n_Human validation required for all automated outputs._")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_all_reports(app: AppConfig | None = None) -> dict[str, Path]:
    """Write portfolio, risk, governance reports under reports/."""
    app = app or load_app_config()
    ensure_dir(REPORTS_PORTFOLIO_DIR)
    ensure_dir(REPORTS_RISK_DIR)
    ensure_dir(REPORTS_SIMULATIONS_DIR)
    ensure_dir(REPORTS_EXPERIMENTS_DIR)
    ensure_dir(REPORTS_GOVERNANCE_DIR)

    paths: dict[str, Path] = {}

    portfolio_path = REPORTS_PORTFOLIO_DIR / f"portfolio_{app.research.meta.experiment_id}.md"
    ctx = build_portfolio_context(app)
    portfolio_path.write_text(
        f"# Portfolio report\n\n{generate_portfolio_summary(ctx)}\n",
        encoding="utf-8",
    )
    paths["portfolio"] = portfolio_path

    risk_path = REPORTS_RISK_DIR / f"copilot_brief_{app.research.meta.experiment_id}.md"
    paths["risk"] = _write_copilot_brief(app, risk_path)

    gov_path = REPORTS_GOVERNANCE_DIR / f"governance_{app.research.meta.experiment_id}.md"
    paths["governance"] = _write_governance_report(app, gov_path)

    if RESEARCH_REPORT_MD_PATH.parent.exists():
        sim_path = REPORTS_SIMULATIONS_DIR / f"research_{app.research.meta.experiment_id}.md"
        if RESEARCH_REPORT_MD_PATH.is_file():
            sim_path.write_text(RESEARCH_REPORT_MD_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            write_interpreted_report(app, sim_path)
        paths["simulations"] = sim_path

    LOGGER.info("Platform reports generated", extra={"paths": {k: str(v) for k, v in paths.items()}})
    return paths
