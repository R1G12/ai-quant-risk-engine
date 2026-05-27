"""Tests for FastAPI platform API."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/monitoring/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "checks" in body


def test_copilot_inference(client: TestClient) -> None:
    r = client.post("/inference/copilot", json={"question": "What is the current volatility regime?"})
    assert r.status_code == 200
    data = r.json()
    assert data["title"]
    assert data["requires_human_review"] is True


def test_portfolio_summary(client: TestClient) -> None:
    r = client.get("/portfolio/summary")
    assert r.status_code == 200
    body = r.json()
    assert "experiment_id" in body
    assert "summary" in body


def test_portfolio_exposures(client: TestClient) -> None:
    r = client.get("/portfolio/exposures")
    assert r.status_code == 200
    assert "rows" in r.json()


def test_quality_endpoint(client: TestClient) -> None:
    r = client.get("/monitoring/quality")
    assert r.status_code == 200
    assert "checks" in r.json()


def test_alerts_endpoint(client: TestClient) -> None:
    r = client.get("/monitoring/alerts")
    assert r.status_code == 200
    assert "alerts" in r.json()


def test_risk_var_endpoint(client: TestClient) -> None:
    r = client.get("/risk/var")
    # 200 if Phase 3 artifacts exist, else 404
    assert r.status_code in (200, 404)


def test_copilot_rejects_short_question(client: TestClient) -> None:
    r = client.post("/inference/copilot", json={"question": "?"})
    assert r.status_code == 422
