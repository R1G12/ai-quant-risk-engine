# Roadmap

## Phase 1 – Financial Sentiment Engine (complete)

- [x] DVC pipeline: ingest → preprocess → sentiment
- [x] FinBERT + DVCLive
- [x] Polars + Parquet
- [x] Tests and CI

## Phase 2 – Market data and feature engineering (complete)

- [x] Partitioned market parquet lake (`year=YYYY/month=MM`)
- [x] Dual ingestion: `sample` (CI) + `yfinance` (local)
- [x] Feature modules: returns, volatility, Sharpe, technical, liquidity, correlation
- [x] Sentiment aggregation with `configs/sentiment_map.yaml`
- [x] DVC stages through `merge_features`
- [x] `risk_dataset.parquet` + metadata
- [x] Docs: `data_architecture.md`, `polars_guidelines.md`, `feature_store.md`
- [x] Schema validation + expanded tests

## Phase 2b – Risk and portfolio (planned)

- Port notebook risk logic into `src/risk` and `src/portfolio`
- VaR / CVaR, GARCH, HMM regimes
- Portfolio optimization consuming `risk_dataset`
- Populate portfolio schema

## Phase 3 – Private markets (research)

- Sparse fundamentals → Monte Carlo at scale
- Quantum Amplitude Estimation research track

## Stretch / backlog

- Migrate Phase 1 news from CSV → `data/raw/news/*.parquet`
- `sink_parquet(partition_by)` when Polars version supports it uniformly
- HF embeddings (`embedding_id` column)
- DVC remote storage
