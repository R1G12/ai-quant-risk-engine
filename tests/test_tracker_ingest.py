"""Tests for Excel trade ledger ingest."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.portfolio.tracker.ingest import ingest_trades, read_trades_excel, resolve_excel_path
from src.portfolio.tracker.schema import TRADE_COLUMNS, validate_and_normalize
from src.utils.config import AppConfig, PortfolioTrackerConfig, load_app_config


def _write_xlsx(path: Path, rows: list[dict]) -> None:
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "trades"
    ws.append(TRADE_COLUMNS)
    for row in rows:
        ws.append([row.get(c) for c in TRADE_COLUMNS])
    wb.save(path)


def _tracker_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setattr("src.portfolio.tracker.ingest.PROJECT_ROOT", tmp_path)
    input_dir = tmp_path / "data" / "input" / "portfolio"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = tmp_path / "data" / "processed" / "portfolio" / "trades.parquet"
    cfg = PortfolioTrackerConfig(
        input_dir="data/input/portfolio",
        excel_primary="input_trades.xlsx",
        excel_fallback="dummy_portfolio.xlsx",
        trades_parquet=str(out_parquet.relative_to(tmp_path)),
        metadata_json="data/processed/portfolio/_tracker_metadata.json",
    )
    app = load_app_config()
    app.tracker = cfg
    return app


@pytest.fixture
def sample_rows() -> list[dict]:
    return [
        {
            "trade_id": "T1",
            "ticker": "AAPL",
            "trade_date": "2025-01-10",
            "side": "long",
            "action": "buy",
            "quantity": 5,
            "price": 100.0,
            "fees": 0.0,
            "lot_id": "AAPL_L1",
            "rolled_from_lot_id": "",
            "notes": "",
        },
        {
            "trade_id": "T2",
            "ticker": "AAPL",
            "trade_date": "2025-02-01",
            "side": "long",
            "action": "sell",
            "quantity": 5,
            "price": 110.0,
            "fees": 1.0,
            "lot_id": "AAPL_L1",
            "rolled_from_lot_id": "",
            "notes": "",
        },
    ]


def test_validate_and_normalize(sample_rows: list[dict]) -> None:
    df = pl.DataFrame(sample_rows).select(TRADE_COLUMNS)
    out = validate_and_normalize(df)
    assert out.height == 2
    assert out["ticker"][0] == "AAPL"
    assert out["trade_date"].dtype == pl.Date


def test_ingest_writes_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_rows: list[dict],
) -> None:
    app = _tracker_app(tmp_path, monkeypatch)
    xlsx = tmp_path / "data" / "input" / "portfolio" / "input_trades.xlsx"
    _write_xlsx(xlsx, sample_rows)
    out = ingest_trades(app, force=True)
    assert out.is_file()
    trades = pl.read_parquet(out)
    assert trades.height == 2


def test_resolve_fallback_dummy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_rows: list[dict],
) -> None:
    app = _tracker_app(tmp_path, monkeypatch)
    fallback = tmp_path / "data" / "input" / "portfolio" / "dummy_portfolio.xlsx"
    _write_xlsx(fallback, sample_rows)
    path, label = resolve_excel_path(app)
    assert label == "fallback"
    assert path.name == "dummy_portfolio.xlsx"


def test_read_trades_excel(
    tmp_path: Path,
    sample_rows: list[dict],
) -> None:
    xlsx = tmp_path / "trades.xlsx"
    _write_xlsx(xlsx, sample_rows)
    df = read_trades_excel(xlsx, sheet_name="trades")
    assert "ticker" in df.columns
