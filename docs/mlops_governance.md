# MLOps governance (Phase 5)

## Controls

| Control | Mechanism |
|---------|-----------|
| Reproducibility | DVC lock + params.yaml |
| Experiment trace | `experiments/*/manifest.yaml` |
| Human review | Copilot `requires_human_review` default |
| Audit trail | Copilot `evidence` JSON in API + UI |
| Data quality | `monitoring/quality` gates |
| Report archive | `reports/` (gitignored outputs) |

## DVC stage

`generate_platform_reports` — depends on risk + research artifacts; writes `reports/{portfolio,risk,governance}/`.

## Workflows

Named bundles in `src/orchestration/workflows.py`:

- `refresh_market`
- `risk_refresh`
- `research_refresh`
- `platform_reports`
- `full_refresh`

Run via: `aqre workflow run refresh_market`

## Promotion checklist

1. `pytest` green on Python 3.12
2. `dvc repro` through required phases
3. Monitoring health OK
4. Risk committee sign-off on copilot brief
5. Commit `dvc.lock` + params changes

## Secrets

- `.env` for `HF_TOKEN`, never committed
- API has no auth in MVP — add API keys before external deployment
