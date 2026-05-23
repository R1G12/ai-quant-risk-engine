# MLOps

## Pipeline (DVC)

Defined in [`dvc.yaml`](../dvc.yaml):

| Stage | Command | Outputs |
|-------|---------|---------|
| `ingest` | `python -m src.ingestion.ingest` | `data/raw/news.csv` |
| `preprocess` | `python -m src.preprocessing.preprocess` | `data/processed/news.parquet` |
| `sentiment` | `python -m src.sentiment.finbert` | `data/processed/sentiment.parquet`, `metrics/sentiment/` |

Dependencies between stages are explicit in `dvc.yaml` (`deps` / `outs`).

## Parameters

- [`params.yaml`](../params.yaml) – DVC-tracked hierarchical params
- [`configs/base.yaml`](../configs/base.yaml) – defaults (`batch_size`, `max_seq_length`, `seed`)
- [`configs/dvc_params.yaml`](../configs/dvc_params.yaml) – DVC overrides
- [`configs/finbert.yaml`](../configs/finbert.yaml) – `model_name`

Environment overrides (highest precedence):

- `PROJECT_BATCH_SIZE`
- `PROJECT_MAX_SEQ_LENGTH`
- `PROJECT_SEED`
- `PROJECT_MODEL_NAME`

## Reproducibility workflow

```bash
# Full pipeline
dvc repro

# After changing params.yaml
dvc repro sentiment

# Inspect parameter drift
dvc params diff

# View metrics
dvc metrics show
cat metrics/sentiment/metrics.json
```

Commit `dvc.lock` after pipeline changes so hashes stay reproducible across machines.

## Metrics (DVCLive)

The sentiment stage logs:

- `avg_positive_score` – mean confidence for positive labels
- `total_rows` – number of scored articles

DVCLive runs with `save_dvc_exp=False` to avoid automatic DVC experiment stashing (which can fail on Windows when log files are locked).

Top-level `dvc.yaml` registers:

```yaml
metrics:
  - metrics/sentiment/metrics.json
```

## CI

GitHub Actions (`.github/workflows/ci.yml`):

1. `pip install -e ".[dev]"`
2. `pytest`
3. `dvc repro ingest preprocess` (no FinBERT download)

Run full `dvc repro` locally before releases.

## Artifacts and git

- `data/raw/news.csv` – small sample; tracked in git
- `data/processed/*` – generated; listed in `.gitignore`
- `logs/` – JSON logs per module; gitignored
