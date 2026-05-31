"""Sample / synthetic financial news adapter for CI and offline runs."""

from __future__ import annotations

from typing import Mapping

import polars as pl


def _sample_news() -> list[Mapping[str, str]]:
    """Return a small hard-coded list of news items."""
    return [
        {
            "date": "2026-01-01",
            "source": "Bloomberg",
            "title": "Tech stocks rally as earnings beat expectations",
            "content": "Major technology companies reported better-than-expected earnings, driving a broad market rally.",
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


def load_sample_news() -> pl.DataFrame:
    """Build a Polars DataFrame of sample financial news."""
    return pl.DataFrame(_sample_news())
