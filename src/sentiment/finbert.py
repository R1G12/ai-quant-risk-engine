"""Sentiment inference using Hugging Face FinBERT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import polars as pl
import torch
from dvclive import Live
from transformers import Pipeline, pipeline

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import METRICS_DIR, PROCESSED_DATA_DIR

LOGGER = get_logger(__name__)

TEXT_COLUMN_PRIORITY = ("text", "clean_text", "content")


def _resolve_text_column(df: pl.DataFrame) -> str:
    """Return the first available canonical text column name."""
    for col in TEXT_COLUMN_PRIORITY:
        if col in df.columns:
            return col
    raise KeyError(f"No text column found; expected one of {TEXT_COLUMN_PRIORITY}")


def _resolve_device() -> int | str:
    """Use GPU when available, otherwise CPU."""
    return 0 if torch.cuda.is_available() else -1


def _load_processed_news(input_path: Path) -> pl.DataFrame:
    """Load the processed news parquet file."""
    LOGGER.debug("Reading processed news parquet", extra={"path": str(input_path)})
    return pl.read_parquet(input_path)


def _init_pipeline(model_name: str, max_seq_length: int) -> Pipeline:
    """Create a Hugging Face sentiment pipeline for FinBERT."""
    LOGGER.info("Initializing FinBERT sentiment pipeline", extra={"model": model_name})
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=max_seq_length,
        device=_resolve_device(),
    )


def _run_batch(p: Pipeline, texts: List[str]) -> List[dict]:
    """Run the pipeline on a list of texts."""
    return p(texts)


def _add_sentiment_columns(df: pl.DataFrame, results: List[dict]) -> pl.DataFrame:
    """Append sentiment_label and sentiment_score columns."""
    labels = [r["label"] for r in results]
    scores = [float(r["score"]) for r in results]
    return df.with_columns(
        pl.Series("sentiment_label", labels),
        pl.Series("sentiment_score", scores),
    )


def _close_file_handlers() -> None:
    """Flush and close file log handlers before DVCLive writes metrics."""
    for handler in list(LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            LOGGER.removeHandler(handler)


def run_sentiment(
    input_path: Path | None = None,
    output_path: Path | None = None,
    metrics_path: Path | None = None,
) -> Path:
    """Run FinBERT sentiment analysis on processed news."""
    cfg = load_config()
    LOGGER.info(
        "Starting sentiment pipeline",
        extra={
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "model_name": cfg.model_name,
        },
    )

    inp = input_path or PROCESSED_DATA_DIR / "news.parquet"
    out = output_path or PROCESSED_DATA_DIR / "sentiment.parquet"
    metrics_dir = metrics_path or METRICS_DIR / "sentiment"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = _load_processed_news(inp)
    if df.is_empty():
        LOGGER.warning("Processed news file is empty; exiting sentiment step")
        return out

    text_col = _resolve_text_column(df)
    texts = df[text_col].to_list()

    sentiment_pipe = _init_pipeline(cfg.model_name, cfg.max_seq_length)

    batch_size = cfg.batch_size
    results: List[dict] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_res = _run_batch(sentiment_pipe, batch_texts)
        results.extend(batch_res)
        LOGGER.debug(
            "Processed batch",
            extra={"batch_start": i, "batch_end": i + len(batch_texts)},
        )

    df_sent = _add_sentiment_columns(df, results)

    out.parent.mkdir(parents=True, exist_ok=True)
    df_sent.write_parquet(out)
    LOGGER.info("Sentiment results written", extra={"rows": df_sent.height, "path": str(out)})

    pos_scores = [r["score"] for r in results if r["label"].lower() == "positive"]
    avg_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0

    _close_file_handlers()
    with Live(dir=metrics_dir, save_dvc_exp=False, dvcyaml=False) as live:
        live.log_metric("avg_positive_score", avg_pos)
        live.log_metric("total_rows", df_sent.height)
    LOGGER.info("DVCLive metrics logged", extra={"path": str(metrics_dir)})

    return out


if __name__ == "__main__":
    out_path = run_sentiment()
    LOGGER.info("Sentiment pipeline completed", extra={"output": str(out_path)})
