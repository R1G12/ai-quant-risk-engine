# Architecture

## Overview

The AI Quant Risk Engine separates **exploration** (notebooks) from **production** (`src/`). Phase 1 covers financial news sentiment; Phase 2 adds a scalable market data and feature engineering platform.

```mermaid
flowchart TB
  subgraph phase1 [Phase1_Sentiment]
    ingest[ingestion.ingest]
    preprocess[preprocessing.preprocess]
    sentiment[sentiment.finbert]
    ingest --> preprocess --> sentiment
  end
  subgraph phase2 [Phase2_MarketFeatures]
    mIngest[market.ingest]
    mClean[market.clean]
    returns[features.returns]
    volFeat[features.volatility]
    techFeat[features.technical]
    sentFeat[features.sentiment_agg]
    merge[features.merge]
    mIngest --> mClean --> returns --> volFeat --> techFeat
    sentiment -.-> sentFeat
    volFeat --> merge
    techFeat --> merge
    sentFeat --> merge
  end
  merge --> riskOut[risk_dataset]
  riskOut --> riskP3[Phase3_risk_engine]
```

## Modules

| Package | Responsibility |
|---------|----------------|
| `src.ingestion` | Sample / future API news ingestion |
| `src.preprocessing` | News cleaning → canonical `text` |
| `src.sentiment` | FinBERT inference |
| `src.market` | OHLCV ingest, clean, partitioned parquet lake |
| `src.features` | Returns, volatility, technical, sentiment agg, merge |
| `src.schemas` | Column contracts |
| `src.validation` | Schema checks |
| `src.risk` | Phase 3 – VaR, vol, regimes, optimization |
| `src.portfolio` | Re-exports `src.risk.portfolio` (deprecated path) |

## Data schemas (summary)

| Dataset | Key columns |
|---------|-------------|
| Market (clean) | `timestamp`, `ticker`, `open`, `high`, `low`, `close`, `volume` |
| Market (features) | + `returns`, `log_returns`, `volatility`, `rolling_sharpe`, … |
| Sentiment agg | `timestamp`, `ticker`, `confidence`, `bullish_ratio` |
| Risk dataset | Join on `timestamp`, `ticker` |

Full definitions: [`configs/schemas/`](../configs/schemas/) and [data_architecture.md](data_architecture.md).

## Partitioning

`data/processed/market/year=YYYY/month=MM/*.parquet`

## Technology

- **Polars** LazyFrame-first ([polars_guidelines.md](polars_guidelines.md))
- **Parquet** + zstd compression
- **DVC** reproducible stages
- **Hugging Face** FinBERT (Phase 1)
- **yfinance** optional market source (pandas boundary in adapter only)

## Phase boundaries

| Phase | Scope |
|-------|--------|
| 1 | News sentiment |
| 2 | Market lake + feature store + `risk_dataset` |
| 3 | Risk models + portfolio optimization |
| 4 | Private-markets Monte Carlo research |
