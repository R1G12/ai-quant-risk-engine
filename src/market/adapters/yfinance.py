"""yfinance market data adapter (pandas boundary – convert immediately to Polars)."""

from __future__ import annotations

import polars as pl

from src.utils.config import MarketConfig
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def load_yfinance_market(cfg: MarketConfig) -> pl.LazyFrame:
    """Download OHLCV via yfinance and return a Polars LazyFrame.

    This is the only approved pandas touchpoint: yfinance returns pandas,
    which is converted immediately via ``pl.from_pandas``.
    """
    import yfinance as yf

    LOGGER.info(
        "Downloading market data via yfinance",
        extra={"tickers": cfg.tickers, "start": cfg.start_date, "end": cfg.end_date},
    )
    pdf = yf.download(
        tickers=cfg.tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    if pdf.empty:
        raise ValueError("yfinance returned no data")

    frames: list[pl.DataFrame] = []
    if len(cfg.tickers) == 1:
        ticker = cfg.tickers[0]
        part = pdf.reset_index()
        part.columns = [str(c).lower() for c in part.columns]
        part = part.rename(columns={"date": "timestamp", "adj close": "adj_close"})
        part = part.with_columns(pl.lit(ticker).alias("ticker"))
        frames.append(pl.from_pandas(part))
    else:
        for ticker in cfg.tickers:
            if ticker not in pdf.columns.get_level_values(0):
                continue
            part = pdf[ticker].reset_index()
            part.columns = [str(c).lower() for c in part.columns]
            part = part.rename(columns={"date": "timestamp", "adj close": "adj_close"})
            part = part.with_columns(pl.lit(ticker).alias("ticker"))
            frames.append(pl.from_pandas(part))

    if not frames:
        raise ValueError("No ticker data parsed from yfinance response")

    df = pl.concat(frames, how="vertical_relaxed")
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    for src, dst in rename_map.items():
        if src in df.columns and src != dst:
            df = df.rename({src: dst})

    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")),
        pl.lit("yfinance").alias("source"),
    )
    select_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "source"]
    if "adj_close" in df.columns:
        select_cols.insert(6, "adj_close")
    return df.select([c for c in select_cols if c in df.columns]).lazy()
