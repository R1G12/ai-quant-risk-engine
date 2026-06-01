"""Preprocess raw financial news into cleaned Parquet for downstream inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import PROCESSED_NEWS_PATH, RAW_NEWS_PATH

LOGGER = get_logger(__name__)

TEXT_COLUMN = "text"
SOURCE_TEXT_COLUMNS = ("content", "text", "clean_text")


def _strip_utf8(column: str) -> pl.Expr:
    """Trim whitespace on a UTF-8 column."""
    return (
        pl.col(column)
        .map_elements(lambda s: s.strip() if isinstance(s, str) else s, return_dtype=pl.Utf8)
        .alias(column)
    )


def _read_raw_news(path: Path) -> pl.DataFrame:
    """Read raw news from Parquet (preferred) or legacy CSV."""
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        LOGGER.warning(
            "Reading legacy news CSV at %s; re-run ingest to produce data/raw/news.parquet",
            path,
        )
        return pl.read_csv(path)
    if path.with_suffix(".parquet").is_file():
        return pl.read_parquet(path.with_suffix(".parquet"))
    return pl.read_parquet(path)


def _clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize text, filter empty rows, trim strings, sort columns."""
    source_col = next((c for c in SOURCE_TEXT_COLUMNS if c in df.columns), None)
    if source_col is None:
        raise KeyError(f"No text column found; expected one of {SOURCE_TEXT_COLUMNS}")

    df = df.with_columns(pl.col(source_col).alias(TEXT_COLUMN))
    df = df.with_columns(_strip_utf8(TEXT_COLUMN))
    df = df.filter(pl.col(TEXT_COLUMN).is_not_null() & (pl.col(TEXT_COLUMN) != ""))

    string_cols = [
        name
        for name, dtype in df.schema.items()
        if dtype == pl.Utf8 and name != TEXT_COLUMN
    ]
    if string_cols:
        df = df.with_columns([_strip_utf8(col) for col in string_cols])

    return df.select(sorted(df.columns))


def preprocess_news(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Read raw news, clean it, and write a Parquet file."""
    cfg = load_config()
    LOGGER.info("Starting preprocessing", extra={"seed": cfg.seed})

    inp = input_path or RAW_NEWS_PATH
    out = output_path or PROCESSED_NEWS_PATH

    df = _read_raw_news(inp)
    df = _clean_dataframe(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    LOGGER.info("Preprocessing completed", extra={"output_path": str(out), "rows": df.height})
    return out


if __name__ == "__main__":
    processed_path = preprocess_news()
    LOGGER.info("Processed file written", extra={"path": str(processed_path)})
