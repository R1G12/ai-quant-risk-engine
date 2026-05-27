# Model serving

## API stack

- **FastAPI** (`src/api/main.py`)
- Optional install: `pip install -e ".[platform]"`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/monitoring/health` | Artifact health |
| GET | `/monitoring/quality` | Data quality checks |
| GET | `/monitoring/alerts` | Threshold alerts |
| GET | `/portfolio/summary` | NL + structured summary |
| GET | `/portfolio/exposures` | Holdings table |
| GET | `/portfolio/kpis` | Sharpe, DD, VaR |
| GET | `/risk/var` | VaR metrics parquet |
| GET | `/risk/correlations` | Correlation artifact |
| POST | `/inference/copilot` | Q&A body `{ "question": "..." }` |

## Run locally

```powershell
.venv312\Scripts\activate
pip install -e ".[platform,dashboard]"
aqre api serve
# or: uvicorn src.api.main:app --reload
```

## Inference caching

`src/api/inference/service.py` provides `@lru_cache` for repeated copilot questions (dev only; disable in production).

## FinBERT serving

Sentiment inference remains a **batch DVC stage** (`sentiment`). Online FinBERT serving is a future Phase 5b extension.
