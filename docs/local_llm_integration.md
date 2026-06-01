# Local LLM integration (optional)

The platform copilot is **rule-based** today ([`src/copilot/`](../src/copilot/)). This guide explains how you could add a **local** LLM later without cloud API cost.

## When to use a local LLM

- Rephrase rule-based answers into natural language
- Summarize long evidence JSON from `CopilotEngine.ask()`
- Optional “chat” layer on top of existing explainers

Keep **evidence JSON** attached for auditability even when an LLM rewrites prose.

## Lightweight options (2026)

| Stack | Pros | Cons |
|-------|------|------|
| **Ollama** + small instruct model (e.g. Phi, Llama 3.x 8B) | Easy install, HTTP API | RAM/VRAM; quality vs size tradeoff |
| **LM Studio** | GUI, local OpenAI-compatible server | Manual model management |
| **llama.cpp** / **GPT4All** | CPU-friendly | More integration work |

For finance Q&A grounded in your parquet artifacts, prefer **small instruct models** with short context and strict prompts that only use provided evidence.

## Suggested hook points

```text
src/copilot/reasoning/engine.py   # after pattern match, before return
src/copilot/prompts/templates.yaml  # already exists for optional LLM
src/api/inference/                # new POST /inference/chat if serving online
```

Flow:

```mermaid
flowchart LR
  Q[User question] --> Engine[CopilotEngine rule router]
  Engine --> Evidence[Evidence dict]
  Evidence --> OptionalLLM[Optional local LLM rewrite]
  OptionalLLM --> Answer[Markdown answer]
```

## Implementation sketch

1. Add config flag `platform.copilot.use_local_llm: false` in [`configs/platform.yaml`](../configs/platform.yaml).
2. Create `src/copilot/llm/local_client.py` wrapping Ollama HTTP (`http://localhost:11434/api/generate`).
3. Load prompt from `src/copilot/prompts/templates.yaml`; inject `evidence` as JSON.
4. Fall back to rule-based markdown if LLM unreachable.
5. Do **not** let the LLM invent numbers — prompt must say “only use evidence block.”

## Compute expectations

- **7B–8B** quantized: ~4–8 GB RAM minimum; GPU optional but faster
- Latency: ~1–5 s per short answer on CPU after warm-up
- FinBERT remains separate (batch DVC); do not replace FinBERT with an LLM for headline scoring

## Related docs

- [copilot_architecture.md](copilot_architecture.md)
- [model_serving.md](model_serving.md) — online FinBERT is a separate future step
