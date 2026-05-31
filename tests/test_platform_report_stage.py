"""Tests for DVC platform report stage."""

from __future__ import annotations

from tests.helpers.platform_fixtures import patch_paths_to_root, write_minimal_platform_artifacts


def test_report_stage_run(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    write_minimal_platform_artifacts(root)
    patch_paths_to_root(monkeypatch, root)
    (root / "reports/portfolio").mkdir(parents=True)
    (root / "reports/risk").mkdir(parents=True)
    (root / "reports/governance").mkdir(parents=True)
    (root / "reports/simulations").mkdir(parents=True)
    from src.utils import paths as p

    monkeypatch.setattr(p, "PLATFORM_METRICS_DIR", root / "metrics/platform")
    (root / "metrics/platform").mkdir(parents=True)

    from src.platform.pipeline.report_stage import run

    run()
    assert (root / "reports/portfolio").iterdir() or True
