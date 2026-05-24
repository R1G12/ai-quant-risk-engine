# Experimentation

## Run individual stages

```bash
# Activate .venv312 first (Python 3.12)
python -m src.ingestion.ingest
python -m src.preprocessing.preprocess
python -m src.sentiment.finbert
```

Or via DVC:

```bash
dvc repro ingest
dvc repro preprocess
dvc repro sentiment
```

## Change parameters

Edit [`params.yaml`](../params.yaml) or [`configs/`](../configs/):

```yaml
sentiment:
  batch_size: 16
  max_seq_length: 256
  model_name: ProsusAI/finbert
```

Then re-run affected stages:

```bash
dvc repro sentiment
dvc params diff
```

Environment overrides (no file edit):

```bash
set PROJECT_BATCH_SIZE=8
dvc repro sentiment
```

## Compare metrics

After sentiment:

```bash
dvc metrics show
type metrics\sentiment\metrics.json
```

Key fields:

- `avg_positive_score`
- `total_rows`

## Inspect outputs with Polars

```python
import polars as pl

df = pl.read_parquet("data/processed/sentiment.parquet")
print(df.select("text", "sentiment_label", "sentiment_score"))
```

## Notebooks

Use `notebooks/` for ad-hoc charts and Phase 2 prototypes. Promote stable logic into `src/` and add DVC stages when outputs must be reproducible.

## Troubleshooting

| Issue | Mitigation |
|-------|------------|
| FinBERT download slow | Set `HF_TOKEN`; cache model locally |
| DVCLive / git lock on Windows | Already using `Live(save_dvc_exp=False)` |
| CUDA OOM | Lower `batch_size` in params |
| Empty processed file | Re-run `dvc repro ingest preprocess` |
