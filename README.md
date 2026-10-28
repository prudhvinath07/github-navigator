---
title: Github Navigator 3000
emoji: 🌍
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# GitHub Navigator 3000

Paste a public GitHub repository URL and chat with an AI about the codebase.
Repository content is extracted with [gitingest](https://github.com/coderamp-labs/gitingest)
and sent to any LLM provider through [LiteLLM](https://docs.litellm.ai/docs/providers).

## Configuration

Set these as **Secrets / Variables** in the Space settings (or in a local `.env`, see `.env.example`):

| Variable | Purpose |
|---|---|
| `LLM_MODEL` | LiteLLM model string, e.g. `openrouter/nvidia/nemotron-3.5-lightning:free`, `openrouter/google/gemini-2.5-flash` (default), `gemini/gemini-2.5-flash`, `openai/gpt-4o` |
| `LLM_REASONING` | Thinking: `on` / `off`, or an effort level (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Unset = model default. The model's reasoning is shown in a collapsible "Thinking" section |
| provider key | The provider's standard variable: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ... |
| `LLM_API_KEY` / `LLM_API_BASE` | Optional generic key and base URL, for OpenAI-compatible endpoints (e.g. Requesty: `LLM_MODEL=openai/google/gemini-2.5-flash`, `LLM_API_BASE=https://router.requesty.ai/v1`) |
| `LLM_CONTEXT_TOKENS` | Optional override of the model's context window used for truncation |
| `GITHUB_TOKEN` | Optional, for private repositories and higher GitHub rate limits |

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your model and key
python app.py          # http://localhost:8080
```
