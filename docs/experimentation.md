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

**Run profile (tickers, mode, portfolio):** edit [`configs/run.yaml`](../configs/run.yaml) and run `aqre prepare` — see [run_profile.md](run_profile.md).

**DVC / model params:** edit [`params.yaml`](../params.yaml) or [`configs/`](../configs/):

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

## Phase 4 research experiments

Baseline reproducible run:

```bash
dvc repro generate_simulations run_backtests generate_research_reports
```

Parameter sweeps (requires prior `dvc repro` through Phase 3):

```bash
dvc exp run generate_simulations -S research.simulation.n_paths=2000 -S research.simulation.seed=7
dvc exp show
dvc repro compare_experiments
```

See also [`experiment_tracking.md`](experiment_tracking.md) and [`research_framework.md`](research_framework.md).

## Troubleshooting

| Issue | Mitigation |
|-------|------------|
| FinBERT download slow | Set `HF_TOKEN`; cache model locally |
| `metrics/...` tracked by SCM | `git rm -r --cached metrics/sentiment metrics/research_*` — see [quickstart_nontechnical.md](quickstart_nontechnical.md) |
| DVCLive / git lock on Windows | Sentiment stage uses `save_dvc_exp=False` where configured; research metrics stay gitignored |
| CUDA OOM | Lower `batch_size` in params |
| Empty processed file | Re-run `dvc repro ingest preprocess` |
| CI holdings mismatch | Run `aqre prepare` or let risk stages refresh stale `holdings.parquet` — [run_profile.md](run_profile.md) |
