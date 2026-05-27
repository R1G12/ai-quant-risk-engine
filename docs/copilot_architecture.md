# Copilot architecture

## Layout

```
src/copilot/
├── context/       # PortfolioContext from DVC artifacts
├── explanations/  # VaR, drawdown, regime explainers
├── attribution/   # Weight × return proxies
├── summarization/ # Executive summaries
├── reasoning/     # Question router (CopilotEngine)
└── prompts/       # YAML templates for optional LLM
```

## Question routing

`CopilotEngine.ask()` pattern-matches user questions and returns:

- `answer` (markdown)
- `evidence` (JSON-serializable audit dict)
- `confidence` (`low` | `medium` | `high`)
- `requires_human_review` (default `True`)

## Supported questions (MVP)

- VaR contributors
- Risk increase drivers
- Drawdown context
- Volatility regime
- Sentiment sensitivity
- Correlation structure
- Exposure / concentration
- Portfolio summary

## Extensibility

1. Add explainers under `explanations/`.
2. Register patterns in `reasoning/engine.py`.
3. Optionally render `prompts/templates.yaml` with an LLM — **evidence JSON must still be attached**.
