# UB Library Navigator

Role-aware library triage agent for University at Buffalo Libraries.

See `docs/PROJECT_OVERVIEW.md` for the full module map and end-to-end pipeline explanation.

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

## Refresh the library-guide data (optional, for RAG)

```bash
.venv/bin/python -m pipeline.refresh          # fetch, parse, snapshot, embed
.venv/bin/python -m agent.rag_live_check      # spot-check retrieval
```

Needs `OPENAI_API_KEY` for embeddings. Without it, `pipeline/refresh.py` still updates
`data/processed/records.json` and just skips the embedding step; the chat agent runs fine with
no library-guide data at all, it just won't have a "Subject librarian" / "Helpful pages" /
"Browse databases" footer on its answers.

## Tests

```bash
.venv/bin/pytest
```
