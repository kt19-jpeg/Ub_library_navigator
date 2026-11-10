# UB Library Navigator

Role-aware library triage agent for University at Buffalo Libraries.

See `docs/PROJECT_OVERVIEW.md` for the full module map and end-to-end pipeline explanation.

## Run in GitHub Codespaces (no local setup)

1. On GitHub, click **Code → Codespaces → Create codespace** (on whichever branch you want to
   demo). The container builds automatically from `.devcontainer/devcontainer.json`: Python
   dependencies and the headless Chromium build OneSearch needs are installed for you — this
   takes a few minutes the first time, with no steps to run by hand.
2. Once it's ready, open a terminal in the Codespace and set your API key for that session:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
   For a demo with multiple people each creating their own Codespace, set `OPENAI_API_KEY` once
   as a **Codespaces secret** instead (repo Settings → Secrets and variables → Codespaces), so
   every new Codespace already has it — nobody has to paste a key in.
3. Start the app:
   ```bash
   streamlit run streamlit_app/Home.py --server.port 8501
   ```
4. Codespaces detects port 8501 and pops up a toast to open it in the browser (or use the
   **Ports** tab). That's the full chat UI, including the staff dashboard page in the sidebar —
   set `STAFF_DASHBOARD_AUTH_METHOD=password` and `STAFF_DASHBOARD_PASSWORD=...` the same way as
   `OPENAI_API_KEY` if you want that page reachable too.

## Deploy to the cloud (Render)

Makes the app reachable at a public URL instead of only `localhost`/Codespaces.

1. Push this repo to GitHub if it isn't already (it is).
2. On [render.com](https://render.com), **New + → Blueprint**, connect this GitHub repo. Render
   reads `render.yaml` and pre-fills a Docker-based web service (`Dockerfile` at the repo root
   installs Python deps + headless Chromium for OneSearch, same recipe as the devcontainer).
3. Render will prompt for the env vars marked `sync: false` in `render.yaml` — fill in:
   - `OPENAI_API_KEY` — required.
   - `APP_ACCESS_PASSWORD` — **set this.** The chat app has no login by default; once it's on a
     public URL, every message spends your real API key, so `APP_ACCESS_METHOD=password` is
     already set in the blueprint and this is the password that gates the whole app (prompted
     before the role picker, same pattern as the staff dashboard below).
   - `STAFF_DASHBOARD_PASSWORD` — for the staff dashboard page, same idea, separate password.
4. Deploy. Render builds the Docker image and gives you a URL like
   `https://ub-library-navigator.onrender.com`.

**Known limitation: the free plan's disk is ephemeral.** `analytics/sessions.db` (anonymized
usage logging, read by the staff dashboard) is wiped on every redeploy and on the free plan's
idle-spin-down/restart cycle. Fine for a demo; for logging that needs to persist, either attach a
[Render persistent disk](https://render.com/docs/disks) (needs a paid instance type — mount it at
`/app/analytics`) or point `DATABASE_URL` at an external Postgres instead of the default SQLite
file.

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
.venv/bin/python -m pipeline.refresh          # fetch, parse, snapshot, embed (applies data/subject_keywords.json)
.venv/bin/python -m pipeline.subject_keywords # only if new subjects appear: generate their keywords
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
