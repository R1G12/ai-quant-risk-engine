"""yfinance period helper tests."""

from src.market.adapters.yfinance import period_for_rolling_days


def test_period_for_rolling_days() -> None:
    assert period_for_rolling_days(5) == "5d"
    assert period_for_rolling_days(30) == "1mo"
    assert period_for_rolling_days(365) == "1y"
    assert period_for_rolling_days(500) == "2y"
