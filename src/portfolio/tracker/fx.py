"""Historical FX rates for tracker P&L (quote → display currency)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from src.market.adapters.yfinance import yfinance_download_strategies
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
        try:
            import yfinance as yf
        except ImportError:
            LOGGER.warning("yfinance not installed; FX rates default to 1.0")
            return cls.identity(pair=pair)

        strategies = yfinance_download_strategies()
        # Prefer the SSL-bypass strategy first; it is the one validated for other
        # tickers in live mode in this project.
        strategies = sorted(strategies, key=lambda s: 0 if s[0] == "curl_cffi_no_verify" else 1)

        hist = None
        last_exc: Exception | None = None
        for label, session in strategies:
            try:
                kwargs = dict(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    progress=False,
                    auto_adjust=True,
                )
                if session is not None:
                    kwargs["session"] = session
                hist = yf.download(pair, **kwargs)
            except Exception as exc:
                last_exc = exc
                continue
            if hist is not None and not hist.empty:
                break

        if hist is None or hist.empty:
            LOGGER.warning(
                "No FX data for %s in [%s, %s] (last error: %s)",
                pair,
                start,
                end,
                last_exc,
            )
            return cls.identity(pair=pair)

        close_col = "Close" if "Close" in hist.columns else hist.columns[-1]
        series = hist[close_col].dropna()
        if series.empty:
            return cls.identity(pair=pair)

        rates: dict[date, float] = {}
        for idx, val in series.items():
            d = idx.date() if hasattr(idx, "date") else idx
            rates[d] = float(val)

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
