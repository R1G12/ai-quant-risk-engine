"""Historical FX rates for tracker P&L (quote → display currency)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.market.adapters.yfinance import (
    download_symbol_history,
    yfinance_close_series,
    yfinance_scalar_float,
)
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class FxRateTable:
    """Daily FX rates with forward-fill lookup (weekends/holidays use prior close)."""

    pair: str
    rates: dict[date, float]

    @classmethod
    def identity(cls, *, pair: str = "NONE") -> FxRateTable:
        return cls(pair=pair, rates={})

    @classmethod
    def from_dict(cls, pair: str, rates: dict[date, float]) -> FxRateTable:
        return cls(pair=pair, rates=dict(rates))

    @classmethod
    def load(cls, pair: str, start: date, end: date) -> FxRateTable:
        """Load daily closes for ``pair`` (e.g. ``USDSGD=X``) from yfinance."""
        if start > end:
            start, end = end, start

        hist = download_symbol_history(pair, start, end)
        series = yfinance_close_series(hist)
        if series is None or series.empty:
            LOGGER.warning("No FX data for %s in [%s, %s]", pair, start, end)
            return cls.identity(pair=pair)

        rates: dict[date, float] = {}
        for idx, val in series.items():
            d = idx.date() if hasattr(idx, "date") else idx
            rates[d] = yfinance_scalar_float(val)

        LOGGER.info("FX loaded %s: %d observations [%s → %s]", pair, len(rates), start, end)
        return cls(pair=pair, rates=rates)

    def rate_on(self, d: date, *, default: float = 1.0) -> float:
        """Rate on ``d``; forward-fill from the latest observation on or before ``d``."""
        if not self.rates:
            return default
        if d in self.rates:
            return self.rates[d]
        prior = [day for day in self.rates if day <= d]
        if prior:
            return self.rates[max(prior)]
        # Before first observation: use earliest available
        return self.rates[min(self.rates)]

    def latest_date(self) -> date | None:
        if not self.rates:
            return None
        return max(self.rates)

    def panel(self) -> pl.DataFrame:
        if not self.rates:
            return pl.DataFrame(schema={"date": pl.Date, "rate": pl.Float64})
        rows = sorted(self.rates.items())
        return pl.DataFrame({"date": [r[0] for r in rows], "rate": [r[1] for r in rows]})
