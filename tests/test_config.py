"""Configuration loading tests."""

from src.utils.config import load_app_config, load_config


def test_load_config_reads_yaml_defaults() -> None:
    cfg = load_config()
    assert cfg.batch_size == 32
    assert cfg.model_name == "ProsusAI/finbert"


def test_load_app_config_market_sample() -> None:
    app = load_app_config()
    assert app.market.source in {"sample", "yfinance"}
    assert len(app.market.tickers) >= 2
    assert app.features.volatility_window == 21
    assert app.risk.volatility.ewma_span == 21
    assert 0.95 in app.risk.var.confidence_levels
    assert app.research.simulation.seed == 42
    assert app.tracker.excel_primary == "input_trades.xlsx"
    assert app.tracker.trades_parquet.endswith("trades.parquet")
    assert app.tracker.quote_currency == "USD"
    assert app.tracker.display_currency == "SGD"
    assert app.tracker.fx_pair == "USDSGD=X"
