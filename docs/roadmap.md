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

## Phase 3 – Quantitative risk engine (complete)

- [x] Volatility: rolling, EWMA, regime flags, GARCH
- [x] VaR / CVaR: historical, parametric, Monte Carlo
- [x] Correlation/covariance with shrinkage
- [x] HMM regimes, portfolio analytics
- [x] Markowitz optimization + efficient frontier
- [x] DVC stages + Plotly analytics + tests

## Phase 4 – Private markets (research)

- Sparse fundamentals → Monte Carlo at scale
- Quantum Amplitude Estimation research track

## Stretch / backlog

- Migrate Phase 1 news from CSV → `data/raw/news/*.parquet`
- ~~`sink_parquet(PartitionBy)`~~ — done (Polars >= 1.20, `src/utils/io.py`, `src/market/partitions.py`)
- HF embeddings (`embedding_id` column)
- ~~DVC remote storage~~ — configured (`DVC_test_quant`)
