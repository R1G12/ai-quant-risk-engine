"""Experiment manifest tests."""

from pathlib import Path

from src.utils.experiment import write_manifest


def test_write_manifest(tmp_path: Path) -> None:
    path = tmp_path / "baseline" / "manifest.yaml"
    write_manifest(
        path,
        experiment_id="baseline",
        stage="test",
        params={"seed": 1},
        metrics={"sharpe": 0.5},
        outputs=["data/research/simulations/"],
    )
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "baseline" in text
    assert "sharpe" in text
