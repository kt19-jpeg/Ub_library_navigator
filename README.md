# UB Library Navigator

Role-aware library triage agent for University at Buffalo Libraries.

## Run the chat agent (Layer 2, slice 1)

```bash
cd Ub_library_navigator
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then set OPENAI_API_KEY (or see LLM_* for other providers)
.venv/bin/python -m agent.cli --debug
```

Providers: `LLM_PROVIDER=openai|anthropic|local`, `LLM_MODEL=...`. For `anthropic` also
`pip install -r requirements-optional.txt` (langchain-anthropic); `local` uses Ollama
(`langchain-ollama`) or any OpenAI-compatible server via `LLM_BASE_URL`.

Example with Ollama: `LLM_PROVIDER=local LLM_MODEL=qwen2.5:14b .venv/bin/python -m agent.cli --debug`

## Tests

```bash
.venv/bin/pytest
```
