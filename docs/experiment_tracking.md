# Experiment Tracking

## Manifests

Each research stage writes `experiments/<type>/<experiment_id>/manifest.yaml` with:

- `params`, `metrics`, `git_revision`, `outputs`, `timestamp`

## DVCLive

`ResearchLive` (`src/utils/experiment.py`) logs to `metrics/research_*` with `save_dvc_exp=True`.

## Comparison

```bash
dvc exp run generate_simulations -S research.simulation.n_paths=2000 -S research.simulation.seed=1
dvc exp show
dvc exp diff
dvc repro compare_experiments
```

Aggregated manifests: `data/research/comparisons/comparisons.parquet`.
