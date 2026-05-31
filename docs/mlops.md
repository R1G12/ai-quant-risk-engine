# MLOps

## Pipelines (DVC)

Defined in [`dvc.yaml`](../dvc.yaml).

### Phase 1 – Sentiment

| Stage | Command | Outputs |
|-------|---------|---------|
| `ingest` | `python -m src.ingestion.ingest` | `data/raw/news.csv` |
| `preprocess` | `python -m src.preprocessing.preprocess` | `data/processed/news.parquet` |
| `sentiment` | `python -m src.sentiment.finbert` | `data/processed/sentiment.parquet` |

### Phase 2 – Market features

| Stage | Command | Outputs |
|-------|---------|---------|
| `ingest_market_data` | `python -m src.market.ingest` | `data/raw/market/` |
| `clean_market_data` | `python -m src.market.clean` | `data/processed/market/` |
| `generate_returns` | `python -m src.features.pipeline.returns_stage` | `data/features/returns/` |
| `generate_volatility_features` | `python -m src.features.pipeline.volatility_stage` | `data/features/volatility/` |
| `generate_technical_features` | `python -m src.features.pipeline.technical_stage` | `data/features/technical/` |
| `generate_sentiment_features` | `python -m src.features.pipeline.sentiment_stage` | `data/features/sentiment_agg/` |
| `merge_features` | `python -m src.features.pipeline.merge_stage` | `data/features/merged/` |

## Parameters

- [`configs/run.yaml`](../configs/run.yaml) – local run profile (tickers, `demo`/`live`, portfolio weighting); see [run_profile.md](run_profile.md)
- [`configs/run.ci.yaml`](../configs/run.ci.yaml) – CI sample profile (`RUN_PROFILE` in GitHub Actions)
- [`params.yaml`](../params.yaml) – DVC-tracked params (`market`, `features`, `sentiment`, …)
- [`configs/market.yaml`](../configs/market.yaml) – default tickers, dates, compression
- [`configs/features.yaml`](../configs/features.yaml) – windows, risk-free rate
- [`configs/risk/`](../configs/risk/) – volatility, VaR, portfolio, optimization

Environment overrides:

| Variable | Purpose |
|----------|---------|
| `RUN_PROFILE` | Path to `configs/run.yaml` or `configs/run.ci.yaml` |
| `MARKET_SOURCE` | `sample` or `yfinance` |
| `MARKET_PIN_DATES` | `1` = pinned dates (CI) |
| `PROJECT_*` | FinBERT / batch overrides |
| `HF_TOKEN` | Hugging Face Hub auth |

## Workflow

```bash
dvc repro                              # full pipeline
dvc repro merge_features               # single stage
dvc params diff
dvc metrics show
```

## CI

See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

- `pytest -q` with extras `dev,market,risk,research,dashboard`
- Env: `MARKET_SOURCE=sample`, `MARKET_PIN_DATES=1`, `RUN_PROFILE=configs/run.ci.yaml`
- `dvc repro` Phase 1 (`ingest`, `preprocess`) then full sample pipeline through `generate_research_reports`
- Verify `data/analytics/dashboard/index.html`

Do not commit generated files under `metrics/sentiment/`, `metrics/research_*/`, or `metrics/platform/` — DVC stage outputs (gitignored).

## Metrics

DVCLive outputs under `metrics/` per stage (`save_dvc_exp=False`).

## Git artifacts

Generated and gitignored: `data/processed/market/`, `data/features/`, `data/raw/market/`.

Sample market data is generated on first run under `data/external/sample_market/`.

## Python version

Use **Python 3.12** locally (`.venv312`) and in CI. `dvc.yaml` stages call `python -m ...`; that must resolve to 3.12 after activating the venv.

## DVC remote (optional)

`dvc push` uploads tracked outputs from `.dvc/cache` to a **remote**. It is **not required** for local development.

**No DVC cloud account exists** — configure any storage backend:

| Type | Example command |
|------|-----------------|
| Local folder | `localstore` → `D:\Romain\Projects\Finance\02_Risk Analysis\DVC_test_quant` (see `.dvc/config`) |
| Amazon S3 | `dvc remote add -d s3remote s3://bucket/path` |
| Google GCS | `dvc remote add -d gcs gs://bucket/path` |
| SSH server | `dvc remote add -d sshremote ssh://user@host/path` |

Then: `dvc push` / `dvc pull`.

For a class project on one machine, a **local folder remote** is enough to practice the same workflow teams use with S3, without paying for cloud storage.
