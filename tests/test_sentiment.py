"""Tests for FinBERT sentiment (mocked pipeline)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl

from src.preprocessing.preprocess import TEXT_COLUMN
from src.sentiment.finbert import run_sentiment


def _sample_processed(tmp_path: Path) -> Path:
    path = tmp_path / "news.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": ["2026-01-01"],
            "source": ["Bloomberg"],
            "title": ["Markets rise"],
            TEXT_COLUMN: ["Equities rallied on strong earnings."],
        }
    ).write_parquet(path)
    return path


@patch("src.sentiment.finbert._init_pipeline")
def test_run_sentiment_writes_schema(mock_init: MagicMock, tmp_path: Path) -> None:
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "positive", "score": 0.92}]
    mock_init.return_value = mock_pipe

    inp = _sample_processed(tmp_path)
    out = tmp_path / "sentiment.parquet"
    metrics_dir = tmp_path / "metrics"

    run_sentiment(input_path=inp, output_path=out, metrics_path=metrics_dir)

    df = pl.read_parquet(out)
    assert "sentiment_label" in df.columns
    assert "sentiment_score" in df.columns
    assert df.height == 1
    assert (metrics_dir / "metrics.json").is_file()
