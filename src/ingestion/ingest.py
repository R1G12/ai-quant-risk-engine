"""News ingestion stage for the AI Quant Risk Engine."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.adapters import get_news_source, load_news
from src.utils.logger import get_logger
from src.utils.paths import RAW_DATA_DIR

LOGGER = get_logger(__name__)


def fetch_news(output_path: Path | None = None, source: str | None = None) -> Path:
    """Ingest financial news and write CSV to the raw landing zone.

    Parameters
    ----------
    output_path:
        Destination file. Defaults to ``data/raw/news.csv``.
    source:
        Adapter name (e.g. ``sample``). Uses ``NEWS_SOURCE`` env or ``params.yaml`` when omitted.
    """
    if output_path is None:
        output_path = RAW_DATA_DIR / "news.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = source or get_news_source()
    LOGGER.info("Ingesting news", extra={"source": resolved, "path": str(output_path)})

    df = load_news(resolved)
    df.write_csv(str(output_path))

    LOGGER.info("News CSV written", extra={"rows": df.height, "path": str(output_path)})
    return output_path


if __name__ == "__main__":
    try:
        csv_path = fetch_news()
        LOGGER.info("Ingestion completed successfully", extra={"path": str(csv_path)})
    except Exception as exc:
        LOGGER.error("Ingestion failed", exc_info=exc)
        raise
