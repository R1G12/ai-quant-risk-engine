"""End-to-end Phase 5 smoke tests with synthetic artifacts."""

from __future__ import annotations

import pytest

from tests.helpers.platform_fixtures import patch_paths_to_root, write_minimal_platform_artifacts

# Required copilot questions from Phase 5 spec
COPILOT_SPEC_QUESTIONS = [
    "Why did portfolio risk increase?",
    "Which assets contribute most to VaR?",
    "What drove today's drawdown?",
    "Which sectors dominate exposure?",
    "Which positions are sentiment-sensitive?",
    "What changed in the correlation structure?",
    "What is the current volatility regime?",
]


@pytest.fixture
def platform_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    write_minimal_platform_artifacts(root)
    weights = patch_paths_to_root(monkeypatch, root)
    (root / "reports/portfolio").mkdir(parents=True)
    (root / "reports/risk").mkdir(parents=True)
    (root / "reports/governance").mkdir(parents=True)
    (root / "reports/simulations").mkdir(parents=True)
    return root, weights


def test_copilot_answers_all_spec_questions(platform_artifacts) -> None:
    _root, _weights = platform_artifacts
    from src.copilot.reasoning.engine import CopilotEngine

    engine = CopilotEngine()
    for q in COPILOT_SPEC_QUESTIONS:
        resp = engine.ask(q)
        assert resp.answer.strip(), f"Empty answer for: {q}"
        assert resp.title
        assert resp.confidence in ("low", "medium", "high")
        assert isinstance(resp.evidence, dict)


def test_portfolio_context_loads_from_artifacts(platform_artifacts) -> None:
    _root, weights = platform_artifacts
    from src.copilot.context.builder import build_portfolio_context

    ctx = build_portfolio_context()
    assert set(ctx.tickers) == set(weights.keys())
    assert abs(sum(ctx.weights.values()) - 1.0) < 0.01
    assert ctx.var_table is not None and ctx.var_table.height >= 1
    assert ctx.correlations is not None
    assert ctx.regimes is not None


def test_platform_reports_generated(platform_artifacts) -> None:
    from src.platform.reporting.generator import generate_all_reports

    paths = generate_all_reports()
    assert paths["portfolio"].read_text(encoding="utf-8")
    assert "VaR" in paths["risk"].read_text(encoding="utf-8")
    assert "Health" in paths["governance"].read_text(encoding="utf-8")


def test_monitoring_passes_on_synthetic_data(platform_artifacts) -> None:
    from src.monitoring.alerts.rules import evaluate_alerts
    from src.monitoring.health.checks import run_health_checks
    from src.monitoring.quality.validators import run_quality_checks
    from src.utils.config import load_app_config

    app = load_app_config()
    health = run_health_checks(app)
    quality = run_quality_checks(app)
    assert any(c.ok for c in quality if c.name == "risk_dataset_rows")
    assert isinstance(evaluate_alerts(app), list)
    assert len(health) >= 3


def test_api_full_smoke(platform_artifacts) -> None:
    _root, weights = platform_artifacts
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    client = TestClient(create_app())

    r = client.get("/portfolio/summary")
    assert r.status_code == 200
    assert "summary" in r.json()

    r = client.get("/portfolio/kpis")
    assert r.status_code == 200

    r = client.get("/portfolio/exposures")
    assert r.status_code == 200
    assert len(r.json()["rows"]) == len(weights)

    r = client.get("/risk/var")
    assert r.status_code == 200
    assert len(r.json()["rows"]) >= 1

    r = client.get("/risk/correlations")
    assert r.status_code == 200

    r = client.post(
        "/inference/copilot",
        json={"question": "Which assets contribute most to VaR?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["evidence"]


def test_drift_detection_runs(platform_artifacts) -> None:
    from src.monitoring.drift.baseline import compute_feature_drift

    reports = compute_feature_drift("returns")
    assert len(reports) == 1
    assert hasattr(reports[0], "flagged")
