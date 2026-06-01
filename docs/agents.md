# Agent development guide

Instructions for AI coding agents working on this repository.

## Architecture philosophy

- Production logic lives in `src/`, not notebooks
- Notebooks are for exploration and prototyping only
- Prefer small, focused modules over monolithic scripts
- Configuration via YAML + environment variables, not hardcoded constants

## Coding standards

- **Python 3.12.x only** (`.venv312`; `requires-python >=3.12,<3.13`) with type hints on public functions
- **Docstrings** on modules and public APIs
- **`pathlib.Path`** for all file paths (see `src.utils.paths`)
- **Structured logging** via `src.utils.logger.get_logger`
- **Dataclasses** for config objects (`src.utils.config.Config`)

## Data stack rules

| Context | Library | Format |
|---------|---------|--------|
| `src/**/*.py` | **Polars only** | Parquet for processed data |
| `notebooks/**` | pandas allowed temporarily | varies |

Do not add pandas imports to `src/` unless explicitly migrating a notebook module into production (then convert to Polars).

## Naming conventions

- Modules: lowercase (`ingest.py`, `finbert.py`)
- Functions: `snake_case` verbs (`fetch_news`, `preprocess_news`, `run_sentiment`)
- Canonical text column: `text` after preprocessing
- Sentiment outputs: `sentiment_label`, `sentiment_score`
- Market join keys: `timestamp`, `ticker` (UTC timestamps)
- Partition columns: `year`, `month` on market data

## Phase 2+ rules

- Use `scan_parquet` for reads, lazy pipelines for transforms
- Feature logic in `src/features/`; DVC entrypoints in `src/features/pipeline/`
- Market sources in `src/market/adapters/` only; ticker validation in `src/market/ticker_validation.py` (no synthetic fill for missing symbols in sample mode)
- **Run profile:** `configs/run.yaml` / `configs/run.ci.yaml` merged via `load_app_config()`; `RUN_PROFILE` env for CI
- `MARKET_SOURCE` env overrides `params.yaml` → `market.source` (except `aqre run profile` / `aqre prepare`, which set source from `mode`)
- Never commit `data/features/`, `data/processed/market/`, or generated `metrics/sentiment/`, `metrics/research_*/`, `metrics/platform/` (gitignored; DVC owns them)
- Portfolio weights: `src/portfolio/prepare.py`, `src/portfolio/weights.py`, `src/risk/portfolio/holdings.py`
- Trailing stop policy + Signals UI: `src/portfolio/stops.py`, `src/dashboards/pages/7_Signals.py` — [signals_dashboard.md](signals_dashboard.md)

## Modularity requirements

- Each pipeline stage is runnable: `python -m src.<package>.<module>`
- Stages must be wired in `dvc.yaml` when they affect reproducible artifacts
- Shared code goes in `src.utils`, not duplicated across stages

## Reproducibility principles

1. Any new stage needs `dvc.yaml` entry with `deps`, `outs`, and relevant `params`
2. Update `params.yaml` and `configs/` together
3. Log metrics with DVCLive where applicable
4. Avoid non-deterministic defaults without `seed` documentation

## MLOps conventions

- Run `dvc repro` after changing pipeline code or params
- Commit `dvc.lock` when outputs change
- Do not commit `.venv`, `.venv312`, `logs/`, `__pycache__/`, `*.pyc`, `data/processed/`, or DVCLive folders under `metrics/` (see [mlops.md](mlops.md))
- CI: `.github/workflows/ci.yml` — `pytest`, full sample `dvc repro`, `RUN_PROFILE=configs/run.ci.yaml`
- Activate `.venv312` before `dvc repro` / `pytest` so `python` on PATH is 3.12
- Use `.env` for secrets (never commit); `.env.example` for templates
- Tests that call `materialize_run()` must patch `PROJECT_ROOT` to `tmp_path` so they do not write holdings into the real repo (breaks CI)

## Preferred libraries

- **polars** – tabular data in `src/`
- **transformers** + **torch** – FinBERT inference
- **dvc** + **dvclive** – pipeline and metrics
- **pyyaml** – configuration
- **pytest** – tests (mock HF pipeline in unit tests)

## Signals / portfolio — do not regress

Restored features are tracked in [reimplementation_backlog.md](reimplementation_backlog.md).

- **Single source** for 30d FinBERT scores: `src/portfolio/sentiment_sides.py` (dashboard loaders wrap it).
- Avoid full-file `Write` on `signals_loaders.py` or `optimization_stage.py` without running the backlog regression pytest bundle.
- After edits to Signals or optimization, run:

```powershell
.\.venv312\Scripts\pytest.exe tests/test_signals_regime_chart.py tests/test_min_gross_weights.py tests/test_sentiment_position_sides.py tests/test_expected_returns.py tests/test_regime_policy.py tests/test_sentiment_tilt.py tests/test_signals_loaders.py tests/test_optimization_run_profile.py -q
```

## Rules for agents

1. Read `docs/architecture.md` before large refactors
2. Minimize diff scope; match existing style
3. Add tests for new `src/` behavior
4. Update README and relevant `docs/` when changing pipeline contracts
5. Do not remove DVC stages without user approval
6. Ask before adding heavy dependencies to core `[project.dependencies]`
