# AI Quant Risk Engine

Institutional-style **financial sentiment + risk modeling** platform for a hedge-fund class project.

- **Phase 1:** FinBERT sentiment pipeline (news → preprocess → sentiment)
- **Phase 2:** Market data lake + lazy Polars feature engineering → `risk_dataset.parquet`

## Features

### Phase 1
- DVC pipeline: `ingest` → `preprocess` → `sentiment`
- FinBERT via `transformers.pipeline`
- Polars + Parquet I/O

### Phase 2
- Market ingestion (`sample` | `yfinance`)
- Hive-partitioned OHLCV lake: `data/processed/market/year=YYYY/month=MM/`
- Feature modules: returns, volatility, technical, sentiment aggregation
- DVC stages through `merge_features` → `data/features/merged/risk_dataset.parquet`

## Project structure

```
ai-quant-risk-engine/
├── configs/              # YAML + schemas/
├── data/
│   ├── raw/market/       # market_raw.parquet
│   ├── processed/market/ # partitioned OHLCV
│   ├── features/         # returns, volatility, technical, merged
│   └── external/         # sample_market (generated)
├── docs/                 # architecture, data_architecture, polars_guidelines, feature_store
├── src/
│   ├── ingestion/        # news
│   ├── market/           # OHLCV ingest + clean
│   ├── features/         # feature engineering + pipeline stages
│   ├── sentiment/
│   └── utils/
├── dvc.yaml
└── params.yaml
```

## Setup

```bash
cd ai-quant-risk-engine
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,market]"
copy .env.example .env
```

## Run pipelines

Activate venv first, then:

```bash
# Full project (Phase 1 + Phase 2)
dvc repro

# Phase 2 only (sample mode, no yfinance)
set MARKET_SOURCE=sample
dvc repro ingest_market_data clean_market_data generate_returns
dvc repro generate_volatility_features generate_technical_features
dvc repro generate_sentiment_features merge_features

# Live market data (requires pip install -e ".[market]")
set MARKET_SOURCE=yfinance
dvc repro ingest_market_data clean_market_data
```

## Phase 2 pipeline

| Stage | Output |
|-------|--------|
| `ingest_market_data` | `data/raw/market/` |
| `clean_market_data` | `data/processed/market/` (partitioned) |
| `generate_returns` | `data/features/returns/` |
| `generate_volatility_features` | `data/features/volatility/` |
| `generate_technical_features` | `data/features/technical/` |
| `generate_sentiment_features` | `data/features/sentiment_agg/` |
| `merge_features` | `data/features/merged/risk_dataset.parquet` |

## Testing

```bash
pytest
```

CI uses `MARKET_SOURCE=sample` (no network for market stages).

## Documentation

- [Architecture](docs/architecture.md)
- [Data architecture](docs/data_architecture.md)
- [Polars guidelines](docs/polars_guidelines.md)
- [Feature store](docs/feature_store.md)
- [MLOps](docs/mlops.md)
- [Agents guide](docs/agents.md)
- [Roadmap](docs/roadmap.md)

## Roadmap

- **Phase 2b:** VaR, GARCH, portfolio optimization in `src/risk` / `src/portfolio`
- **Phase 3:** Private-markets Monte Carlo / quantum research
