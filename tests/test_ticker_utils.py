"""Ticker input parsing tests."""

from src.analytics.ticker_utils import parse_ticker_input


def test_parse_ticker_input() -> None:
    assert parse_ticker_input("aapl, msft") == ["AAPL", "MSFT"]
    assert parse_ticker_input("TSLA  NVDA; AMD") == ["TSLA", "NVDA", "AMD"]
    assert parse_ticker_input("") == []
