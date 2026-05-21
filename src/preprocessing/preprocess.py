import polars as pl
from pathlib import Path
from typing import Optional

from src.utils.paths import RAW_NEWS_PATH, PROCESSED_NEWS_PATH, LOGS_DIR
from src.utils.logger import get_logger
from src.utils.config import load_config

LOGGER = get_logger(__name__)

def _clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Apply basic cleaning steps.

    * Drop rows where the ``text`` column is null or empty.
    * Trim whitespace from string columns.
    * Ensure a deterministic column order.
    """
    # Drop rows with null/empty text
    df = df.filter(pl.col("text").is_not_null() & (pl.col("text").str.strip() != ""))
    # Strip whitespace from all string columns
    for name, dtype in df.schema.items():
        if dtype == pl.Utf8:
            df = df.with_column(pl.col(name).str.strip().alias(name))
    # Optional: reorder columns alphabetically for reproducibility
    df = df.select(sorted(df.columns))
    return df

def preprocess_news(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Read raw news CSV, clean it, and write a Parquet file.

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

    LOGGER.debug("Reading raw data", extra={"path": str(inp)})
    df = pl.read_csv(inp)
    LOGGER.info("Raw data read", extra={"rows": df.height, "columns": df.width})

    df_clean = _clean_dataframe(df)
    LOGGER.info("Data cleaned", extra={"rows_before": df.height, "rows_after": df_clean.height})

    # Ensure output directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Writing processed data", extra={"path": str(out)})
    df_clean.write_parquet(out)
    LOGGER.info("Preprocessing completed", extra={"output_path": str(out)})
    return out

if __name__ == "__main__":
    # Simple CLI usage: python -m src.preprocessing.preprocess
    processed_path = preprocess_news()
    LOGGER.info("Processed file written", extra={"path": str(processed_path)})
