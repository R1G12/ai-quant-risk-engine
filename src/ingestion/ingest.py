"""News ingestion stage for the AI Quant Risk Engine."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.adapters import get_news_source, load_news
from src.ingestion.news_schema import RAW_NEWS_COLUMNS
from src.utils.logger import get_logger
from src.utils.paths import RAW_NEWS_PATH

LOGGER = get_logger(__name__)


def fetch_news(output_path: Path | None = None, source: str | None = None) -> Path:
    """Ingest financial news and write Parquet to the raw landing zone.

    Parameters
    ----------
    output_path:
        Destination file. Defaults to ``data/raw/news.parquet``.
    source:
        Adapter name (e.g. ``sample``, ``yfinance``). Uses ``NEWS_SOURCE`` env when omitted.
    """
    if output_path is None:
        output_path = RAW_NEWS_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = source or get_news_source()
    LOGGER.info("Ingesting news", extra={"source": resolved, "path": str(output_path)})

    df = load_news(resolved)
    missing = [c for c in RAW_NEWS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"News adapter missing columns: {missing}")

    df = df.select(RAW_NEWS_COLUMNS)
    if output_path.suffix.lower() == ".csv":
        df.write_csv(str(output_path))
    else:
        df.write_parquet(output_path, compression="zstd")

    LOGGER.info(
        "News raw file written",
        extra={"rows": df.height, "path": str(output_path), "source": resolved},
    )
    return output_path


if __name__ == "__main__":
    try:
        out_path = fetch_news()
        LOGGER.info("Ingestion completed successfully", extra={"path": str(out_path)})
    except Exception as exc:
        LOGGER.error("Ingestion failed", exc_info=exc)
        raise
