'''ingest.py
Ingestion utilities for the AI Quant Risk Engine.

This module provides a minimal, reproducible way to generate a **sample**
financial‑news dataset. In a production setting the function could be expanded
to call external APIs (e.g. Bloomberg, Reuters) or to download raw files.

The implementation deliberately uses **Polars** for CSV handling – this satisfies the
user's preference for Polars over pandas and demonstrates lazy, Arrow‑based I/O.

Typical usage::

    from src.ingestion.ingest import fetch_news
    fetch_news()

Running the module as a script will also invoke ``fetch_news`` via ``python -m``.
''' 

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Mapping

import polars as pl

from src.utils.paths import RAW_DATA_DIR, LOGS_DIR
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _sample_news() -> List[Mapping[str, str]]:
    """Return a small hard‑coded list of news items.

    Each record contains:
        - ``date``: ISO‑8601 date string
        - ``source``: news source name
        - ``title``: headline
        - ``content``: short article body
    """
    return [
        {
            "date": "2026-01-01",
            "source": "Bloomberg",
            "title": "Tech stocks rally as earnings beat expectations",
            "content": "Major technology companies reported better‑than‑expected earnings, driving a broad market rally.",
        },
        {
            "date": "2026-01-02",
            "source": "Reuters",
            "title": "Oil prices dip after OPEC announces production increase",
            "content": "OPEC's decision to raise output eased concerns over supply shortages, pulling oil futures lower.",
        },
        {
            "date": "2026-01-03",
            "source": "Financial Times",
            "title": "Central bank hints at rate cuts later this year",
            "content": "The central bank's recent statement suggested a more accommodative monetary stance, potentially lowering borrowing costs.",
        },
    ]


def fetch_news(output_path: Path | None = None) -> Path:
    """Generate a CSV of sample financial news.

    Parameters
    ----------
    output_path:
        Destination file. If ``None`` the default **data/raw/news.csv** inside the
        repository is used.

    Returns
    -------
    Path
        The path to the written CSV file.
    """
    if output_path is None:
        output_path = RAW_DATA_DIR / "news.csv"

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating sample news dataset", extra={"path": str(output_path)})

    data = _sample_news()
    df = pl.DataFrame(data)
    # Polars writes with UTF‑8 and includes header by default
    df.write_csv(str(output_path))

    LOGGER.info("Sample news CSV written", extra={"rows": df.height, "path": str(output_path)})
    return output_path


if __name__ == "__main__":
    # Simple CLI entry point – useful for quick repro via ``python -m src.ingestion.ingest``
    try:
        csv_path = fetch_news()
        LOGGER.info("Ingestion completed successfully", extra={"path": str(csv_path)})
    except Exception as exc:  # pragma: no cover – defensive logging
        LOGGER.error("Ingestion failed", exc_info=exc)
        raise
