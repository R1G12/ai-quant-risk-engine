# Experiment Tracking

## Manifests

Each research stage writes `experiments/<type>/<experiment_id>/manifest.yaml` with:

- `params`, `metrics`, `git_revision`, `outputs`, `timestamp`

## DVCLive

`ResearchLive` (`src/utils/experiment.py`) logs to `metrics/research_*` with `save_dvc_exp=True`.

These directories are **DVC pipeline outputs** and are gitignored. Do not `git add metrics/sentiment/` or `metrics/research_*` — see [mlops.md](mlops.md) and [quickstart_nontechnical.md](quickstart_nontechnical.md) if Git reports “already tracked by SCM”.

## Comparison

```bash
dvc exp run generate_simulations -S research.simulation.n_paths=2000 -S research.simulation.seed=1
dvc exp show
dvc exp diff
dvc repro compare_experiments
```

Aggregated manifests: `data/research/comparisons/comparisons.parquet`.
