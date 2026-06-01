# Portfolio position tracker

Track **actual** trades separately from model weights (`holdings.parquet` / `optimal_weights.parquet`).

## Quick start

1. Copy the template:
   ```powershell
   Copy-Item data\input\portfolio\dummy_portfolio.xlsx data\input\portfolio\input_trades.xlsx
   ```
2. Edit `input_trades.xlsx` (sheet **`trades`**).
3. Ingest to parquet:
   ```powershell
   aqre tracker ingest
   ```
4. Open the dashboard → **Tracker** page (`src/dashboards/pages/8_Tracker.py`).

`input_trades.xlsx` is gitignored; `dummy_portfolio.xlsx` is committed as the reference template.

## Excel columns (one row per trade)

| Column | Required | Description |
|--------|----------|-------------|
| `trade_id` | No | Auto UUID if empty |
| `ticker` | Yes | e.g. `NVDA` |
| `trade_date` | Yes | `YYYY-MM-DD` |
| `side` | Yes | `long` or `short` |
| `action` | Yes | `buy` or `sell` |
| `quantity` | Yes | Number of shares (> 0) |
| `price` | Yes | Execution price per share |
| `fees` | No | Default 0 |
| `lot_id` | Yes | Groups trades into one position leg |
| `rolled_from_lot_id` | No | Previous lot when rolling |
| `notes` | No | Free text |

**Conventions**

- **Long**: `buy` adds shares; `sell` reduces/closes.
- **Short**: `sell` opens/adds; `buy` covers/closes.
- **Roll**: close old lot (`sell`/`buy` as appropriate) then open new lot with `rolled_from_lot_id` set.

## Paths

| Path | Role |
|------|------|
| `data/input/portfolio/dummy_portfolio.xlsx` | Committed template |
| `data/input/portfolio/input_trades.xlsx` | Your live book (local) |
| `data/processed/portfolio/trades.parquet` | Ingested ledger |
| `configs/portfolio_tracker.yaml` | Paths and defaults |

## Commands

```powershell
aqre tracker ingest
aqre tracker ingest --force
aqre prepare   # also runs tracker ingest when Excel is present
dvc repro ingest_portfolio_trades
```

## Dashboard

The **Tracker** page shows:

- Open positions (quantity, avg cost, unrealized P&L)
- Closed history (realized P&L, holding days, rolls)
- Full trade log
- Price charts with buy/sell markers
- **vs model** — weight table + performance chart (see below)

### Marks and prices

Prices prefer **`data/processed/market/`** parquet. If a ticker is missing there, a short **yfinance** window ending **yesterday** is used (today’s bar is not requested). The UI shows a warning when marks may be **outdated** or sourced from yfinance.

### vs model tab

1. **Weight table** — model side uses **`optimal_weights.parquet`** (`max_sharpe` by default, same as backtests), not equal placeholder `holdings.parquet`. Actual side uses open-position notionals vs an assumed portfolio value ($).
2. **Performance comparison** — indexed equity curves (base 100 at window start):
   - **Model**: static optimized weights × daily returns from `risk_dataset`.
   - **Actual**: daily NAV rebuilt from your trade ledger (cash + positions marked from the close panel).
3. **Period** — presets (3M / 6M / 1Y / All) plus start/end date inputs, clamped to pipeline and trade dates.

Run `dvc repro optimize_portfolios` (or full profile) before expecting a non-trivial model curve.

Rebuild the template file:

```powershell
python scripts/build_dummy_portfolio_xlsx.py
```
