# Documentation index

Entry point for all project docs. The root [README](../README.md) covers setup, CI, and CLI; this page lists deeper guides by phase.

## Getting started

| Doc | Topics |
|-----|--------|
| [quickstart_nontechnical.md](quickstart_nontechnical.md) | One-command setup scripts, dashboard, troubleshooting |
| [run_profile.md](run_profile.md) | `configs/run.yaml`, `aqre prepare`, weighting modes, CI profile |
| [roadmap.md](roadmap.md) | Phase status and backlog |

## Architecture & data

| Doc | Topics |
|-----|--------|
| [architecture.md](architecture.md) | Module map, phase boundaries, schemas |
| [data_architecture.md](data_architecture.md) | Parquet layout, partitioning, ingestion modes |
| [feature_store.md](feature_store.md) | Feature modules, `risk_dataset`, sentiment bridge |
| [polars_guidelines.md](polars_guidelines.md) | Lazy I/O, streaming writes, conventions |

## Risk & portfolio (Phase 3)

| Doc | Topics |
|-----|--------|
| [risk_models.md](risk_models.md) | VaR, vol, GARCH, HMM, optimization |
| [portfolio_theory.md](portfolio_theory.md) | Markowitz, weighting modes, constraints |
| [quantitative_methods.md](quantitative_methods.md) | Formulas and approved numeric boundaries |
| [optimization_pipeline.md](optimization_pipeline.md) | DVC stage order, outputs, run commands |

## Research (Phase 4)

| Doc | Topics |
|-----|--------|
| [research_framework.md](research_framework.md) | Simulations, backtests, dashboards, latency |
| [simulation_architecture.md](simulation_architecture.md) | GBM, multivariate, regime, stress processes |
| [backtesting_methodology.md](backtesting_methodology.md) | Costs, bias controls, experiment sweeps |
| [experiment_tracking.md](experiment_tracking.md) | Manifests, DVCLive, `dvc exp` |
| [experimentation.md](experimentation.md) | Stage-by-stage how-to, troubleshooting |

## Platform (Phase 5)

| Doc | Topics |
|-----|--------|
| [system_architecture.md](system_architecture.md) | Layers, data flow, deployment |
| [copilot_architecture.md](copilot_architecture.md) | Q&A routing, explainers, evidence |
| [portfolio_dashboard.md](portfolio_dashboard.md) | Portfolio page — weights by `weighting` mode, exposures, attribution |
| [signals_dashboard.md](signals_dashboard.md) | FinBERT 30d, trailing stops, HMM regime, VaR 95% (Signals page) |
| [decision_engine.md](decision_engine.md) | Human-in-the-loop workflow |
| [model_serving.md](model_serving.md) | FastAPI endpoints, local serve |
| [observability.md](observability.md) | Health, quality, drift, alerts |
| [mlops_governance.md](mlops_governance.md) | Controls, workflows, promotion checklist |

## Engineering

| Doc | Topics |
|-----|--------|
| [mlops.md](mlops.md) | DVC stages, CI, remotes, metrics gitignore |
| [agents.md](agents.md) | Conventions for AI agents and contributors |
