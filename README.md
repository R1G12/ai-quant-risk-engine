# AI Quant Risk Engine

Institutional-style **financial sentiment + risk modeling** platform for a hedge-fund class project. Phase 1 delivers a reproducible MLOps pipeline: ingest financial news, preprocess with Polars, score sentiment with Hugging Face FinBERT, and log metrics with DVCLive.

## Features (Phase 1)

- Modular `src/` architecture (ingestion, preprocessing, sentiment)
- **DVC** pipeline: `ingest` → `preprocess` → `sentiment`
- **FinBERT** inference via `transformers.pipeline`
- **Polars + Parquet** for efficient tabular I/O (no pandas in production code)
- YAML configuration + `params.yaml` for DVC parameter tracking
- Structured JSON logging
- DVCLive experiment metrics

## Project structure

```
ai-quant-risk-engine/
├── configs/           # YAML configuration (base, dvc_params, finbert)
├── data/
│   ├── raw/           # news.csv (ingest output)
│   ├── processed/     # news.parquet, sentiment.parquet (gitignored)
│   └── external/      # future external datasets
├── docs/              # architecture, MLOps, agents, roadmap
├── metrics/           # DVCLive metrics (sentiment)
├── notebooks/         # exploration only (Phase 2+ prototypes)
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── sentiment/
│   ├── risk/          # Phase 2 placeholder
│   ├── portfolio/     # Phase 2 placeholder
│   └── utils/
├── tests/
├── dvc.yaml           # pipeline definition
└── params.yaml        # DVC-tracked parameters
```

## Setup

```bash
cd ai-quant-risk-engine
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
# Optional: exploration notebook dependencies
pip install -e ".[notebook]"
```

Copy environment template:

```bash
copy .env.example .env
```

## Run the pipeline

Activate the virtual environment first so DVC uses the project interpreter (not a system Python):

```bash
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
```

End-to-end reproducible run:

```bash
dvc repro
```

Individual stages:

```bash
python -m src.ingestion.ingest
python -m src.preprocessing.preprocess
python -m src.sentiment.finbert
```

Fast CI subset (no model download):

```bash
dvc repro ingest preprocess
```

Full sentiment requires downloading `ProsusAI/finbert` from Hugging Face on first run.

## DVC usage

| Command | Purpose |
|---------|---------|
| `dvc repro` | Run all stages |
| `dvc repro preprocess` | Run up to preprocess |
| `dvc params diff` | Compare parameter changes |
| `dvc metrics show` | View sentiment metrics |

Parameters live in [`params.yaml`](params.yaml) and [`configs/`](configs/).

## Hugging Face / FinBERT

- Model: `ProsusAI/finbert` (configurable in `configs/finbert.yaml`)
- Inference: `transformers.pipeline("sentiment-analysis", ...)`
- Device: GPU if available, otherwise CPU
- Set `HF_TOKEN` in `.env` for higher Hub rate limits (optional)

## Data conventions

| Stage | Input | Output |
|-------|-------|--------|
| ingest | sample generator | `data/raw/news.csv` |
| preprocess | raw CSV | `data/processed/news.parquet` (canonical `text` column) |
| sentiment | processed parquet | `data/processed/sentiment.parquet` + `metrics/sentiment/` |

Production code uses **Polars** only. Pandas may appear in [`notebooks/`](notebooks/) for exploration until Phase 2 migration.

## Testing

```bash
pytest
```

Sentiment tests mock the Hugging Face pipeline to avoid model downloads in CI.

## Documentation

- [Architecture](docs/architecture.md)
- [MLOps](docs/mlops.md)
- [Agents guide](docs/agents.md)
- [Roadmap](docs/roadmap.md)
- [Experimentation](docs/experimentation.md)

## Roadmap

- **Phase 1** (current): sentiment engine + DVC/DVCLive foundation
- **Phase 2**: VaR, regime detection, portfolio optimization in `src/risk` and `src/portfolio`
- **Phase 3**: private-markets Monte Carlo / quantum-inspired simulation research

See [docs/roadmap.md](docs/roadmap.md) for details.
