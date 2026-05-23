# Roadmap

## Phase 1 – Financial Sentiment Engine (complete)

- [x] Project scaffold and `src/` modules
- [x] DVC pipeline: ingest → preprocess → sentiment
- [x] FinBERT via Hugging Face `transformers.pipeline`
- [x] Polars + Parquet data layer
- [x] DVCLive metrics
- [x] Config YAML + `params.yaml`
- [x] Tests and CI (pytest + fast DVC stages)
- [x] Documentation (README, docs/)

## Phase 2 – Risk and portfolio (planned)

- Move notebook prototypes from `notebooks/trading_risk_manager.ipynb` into `src/risk` and `src/portfolio`
- Market data ingestion (e.g. yfinance or vendor API)
- VaR / CVaR, volatility (GARCH), regime detection (HMM)
- Portfolio optimization (mean-variance, constraints)
- DVC stages for risk and portfolio artifacts
- Polars-native refactor of notebook pandas code

## Phase 3 – Private markets and simulation (research)

- Sparse quarterly/annual fundamentals → large Monte Carlo ensembles
- Quantum Amplitude Estimation (QAE) for simulation speedup (research track)
- Integration with sentiment and risk signals for private-equity-style workflows

## Non-goals (Phase 1)

- Live trading execution
- Production Bloomberg/Reuters feeds
- Quantum hardware integration
