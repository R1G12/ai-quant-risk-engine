# Observability

## Modules

| Module | Path | Purpose |
|--------|------|---------|
| Health | `monitoring/health` | Artifact presence & freshness |
| Quality | `monitoring/quality` | Schema / null rates / ticker coverage |
| Drift | `monitoring/drift` | Z-score vs historical feature baseline |
| Lineage | `monitoring/lineage` | Experiment YAML manifests |
| Alerts | `monitoring/alerts` | Compose health + drift into alerts |

## Dashboard integration

Streamlit page **Monitoring** (`dashboards/pages/5_Monitoring.py`) surfaces checks interactively.

## API integration

`GET /monitoring/health`, `/quality`, `/alerts` expose the same logic for automation.

## Metrics

DVC stage `generate_platform_reports` logs to `metrics/platform/metrics.json` (gitignored; do not commit — [mlops.md](mlops.md)).

## Future work

- OpenTelemetry traces for API latency
- Prometheus exporters
- Feature store drift vs training baseline partitions
