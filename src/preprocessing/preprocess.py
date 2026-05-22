import polars as pl
from pathlib import Path
from typing import Optional

from src.utils.paths import RAW_NEWS_PATH, PROCESSED_NEWS_PATH, LOGS_DIR
from src.utils.logger import get_logger
from src.utils.config import load_config

LOGGER = get_logger(__name__)

def _clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Apply basic cleaning steps.

    * Determine the text column ("content" or "text").
    * Trim whitespace on the text column using .apply (pure Python).
    * Filter out rows where the text column is null or empty after trimming.
    * Trim whitespace on all other string columns.
    * Reorder columns alphabetically for reproducibility.
    """
    # Choose the appropriate text column name
    text_col = "content" if "content" in df.columns else "text"
    # Trim whitespace on the text column
    df = df.with_columns([
        pl.col(text_col).map_elements(lambda s: s.strip() if isinstance(s, str) else s).alias(text_col)
    ])
    # Filter rows: text column not null and not empty string
    df = df.filter(
        pl.col(text_col).is_not_null() & (pl.col(text_col) != "")
    )
    # Trim whitespace on all other string columns
    string_cols = [name for name, dtype in df.schema.items() if dtype == pl.Utf8 and name != text_col]
    for col in string_cols:
        df = df.with_columns([
            pl.col(col).map_elements(lambda s: s.strip() if isinstance(s, str) else s).alias(col)
        ])
    # Reorder columns alphabetically for reproducibility
    df = df.select(sorted(df.columns))
    return df

def preprocess_news(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Read raw news CSV lazily, clean it, and write a Parquet file.

    Parameters
    ----------
    input_path: Path | None
        Path to the raw CSV file. If ``None`` the default ``RAW_NEWS_PATH`` is used.
    output_path: Path | None
        Destination for the processed Parquet file. If ``None`` the default
        ``PROCESSED_NEWS_PATH`` is used.

    Returns
    -------
    Path
        Path to the generated Parquet file.
    """
    cfg = load_config()
    LOGGER.info("Starting preprocessing", extra={"batch_size": cfg.batch_size, "seed": cfg.seed})

    inp = input_path or RAW_NEWS_PATH
    out = output_path or PROCESSED_NEWS_PATH

    # Read CSV eagerly (no lazy frame)
    df = pl.read_csv(inp)

    # Create a unified column "clean_text" using available content column
    df = df.with_columns(
        pl.col("content").alias("clean_text")
    )

    # Trim whitespace on the unified text column
    df = df.with_columns(
        pl.col("clean_text").map_elements(lambda s: s.strip() if isinstance(s, str) else s).alias("clean_text")
    )

    # Filter out rows where the text column is null or empty after stripping
    df = df.filter(pl.col("clean_text").is_not_null() & (pl.col("clean_text") != ""))

    # Trim whitespace on all other string columns (Utf8 dtype) except the unified text column
    string_cols = [name for name, dtype in df.schema.items() if dtype == pl.Utf8 and name not in {"clean_text", "content"}]
    if string_cols:
        df = df.with_columns([
            pl.col(col).map_elements(lambda s: s.strip() if isinstance(s, str) else s).alias(col) for col in string_cols
        ])

    # Optional: reorder columns alphabetically for reproducibility
    df = df.select(sorted(df.columns))

    # Ensure output directory exists
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write Parquet
    df.write_parquet(out)
    LOGGER.info("Preprocessing completed", extra={"output_path": str(out)})
    return out

if __name__ == "__main__":
    # Simple CLI usage: python -m src.preprocessing.preprocess
    processed_path = preprocess_news()
    LOGGER.info("Processed file written", extra={"path": str(processed_path)})
