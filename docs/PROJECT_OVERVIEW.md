# UB Library Navigator — Project Overview

One doc to re-orient on this project. Replaces the per-phase log files that used to live in
`docs/` (deleted here — their useful content is folded into this file and into the per-slice
sections already in `CLAUDE.md`). Design rationale for each slice stays in `CLAUDE.md` and in
`docs/superpowers/specs/*`; this file is the map, not the rationale.

## What this is

A role-aware chat agent for University at Buffalo Libraries: a patron picks a role
(undergraduate/graduate/faculty), asks a question in plain language, and the agent searches UB's
OneSearch catalog + a small library-guide knowledge base, picks the best 1-3 results, and composes
an answer with real links/call numbers/locations — never LLM-invented ones. It escalates to a human
librarian when it's not confident or when the user just asks for a person.

Two layers, built in that order:
- **Layer 1** (`onesearch/`, `scraper/`, `pipeline/`): talks to the real world — UB's OneSearch API
  and UB's library-guide web pages. No LLM calls here.
- **Layer 2** (`agent/`): the LangGraph orchestration — planning, ranking, composing, escalating.
  Plus `analytics/` (usage logging + staff dashboard) and `streamlit_app/` (the UI), which sit on
  top of Layer 2.

## Module map

### Layer 1 — `onesearch/` (talks to UB's real OneSearch catalog)
- `session.py` — mints a OneSearch session via headless Playwright (bare `requests` gets blocked
  by bot mitigation; loading the real page lets its own JS pass the check). Cookies are reused for
  15 minutes (`SESSION_WINDOW_SECONDS`), not re-minted per search.
- `client.py` — `OneSearchClient.search()` / `search_with_fallback()` (9s timeout, degrades to a
  direct results-page link on timeout or error) / `fetch_detail()` (batched, only ever called for
  the small number of results Layer 2 decided to show — never looped over a full result set).
  `search_with_fallback` also silently retries once with all filters dropped if a filtered search
  comes back empty.
- `cache.py` — short-TTL (`QueryCache`, default 300s) in-memory cache keyed by normalized query+filters.
- `parser.py` — turns OneSearch's raw JSON into `SearchCandidate`/`ResourceDetail`.
- `models.py` — `SearchCandidate`, `ResourceDetail`, `SearchResult` dataclasses.

### Layer 1 — `scraper/` + `pipeline/` (library-guide data, feeds RAG)
- `scraper/fetch.py` — polite fetcher: obeys `robots.txt`, 10s minimum gap between requests to the
  same host, identifying User-Agent, no retries.
- `scraper/parse_*.py` — HTML → structured records for three source pages: subject librarians, help
  topics, database-by-subject browse pages.
- `pipeline/sources.py` — the 3 URLs scraped.
- `pipeline/records.py` — the shared `Record` dataclass (`id`, `type`, `text`, `metadata`,
  content-hash for change detection).
- `pipeline/snapshot.py` — JSON snapshot of all records on disk (`data/processed/records.json`).
- `pipeline/embed.py` — Chroma collection (`vectorstore/`), OpenAI `text-embedding-3-small`
  embeddings, `sync_collection()` (only re-embeds changed records, by content hash).
- `pipeline/refresh.py` — orchestrates one full refresh: fetch each source independently, **keep
  the old data for any source that fails or whose parsed count drops suspiciously** (possible
  layout change) rather than wiping it, save the merged snapshot, sync to Chroma. Runs biweekly via
  `.github/workflows/weekly-refresh.yml`, which opens a PR with the updated snapshot rather than
  pushing straight to main.

### Layer 2 — `agent/` (the orchestration graph)
- `models.py` — `AgentState` (the graph's state dict), `QueryPlan` (planner output, incl.
  `topic_category` for logging), `Selection`/`SelectionItem` (compose output), `RagContext`.
- `router.py` — `is_human_request()`: a small, deliberately conservative regex, no LLM call. The
  explicit-human-request fast path.
- `escalation.py` — `build_escalation_block()`: UB's live-chat link + a copy-paste context summary.
  No API exists to pre-fill the chat widget (confirmed live), so the user pastes it in themselves.
- `planner.py` — one structured-output LLM call: message → `QueryPlan` (keywords, resource_type,
  peer_reviewed, needs_clarification, topic_category). Falls back to an unfiltered raw-message
  search if the model call fails or returns something unusable.
- `onesearch_tool.py` — `run_onesearch()`: the only place Layer 2 talks to Layer 1. Wraps
  `search_with_fallback`, adds the empty-result filter-relaxation retry and user-facing notes.
- `rag.py` — `query_rag()`: queries the Chroma collection for the 3 record types, each with its own
  tuned distance cutoff (see comments in the file — short/formulaic records need a looser cutoff
  than natural-language help-topic titles). Never raises; degrades to an empty `RagContext`.
- `compose.py` — one structured-output LLM call (`select_results`) to pick 1-3 candidates +
  confidence; `validate_selection()` guards against the LLM inventing record ids; `render_answer()`
  / `render_rag_sections()` build the actual answer text **in code**, from real data, so links,
  call numbers, and locations are never LLM output. Low confidence or zero results → escalation
  block instead of a forced answer.
- `graph.py` — wires it together: `entry → plan → {search, rag in parallel} → compose`. `entry` and
  a clarifying `plan` result are both deterministic/early-exit paths (`END`), matching CLAUDE.md's
  "don't burn an LLM+search round trip on every trivial turn" goal. Every node is wrapped in
  `_timed()` so `state["timings"]` always has a per-node breakdown.
- `llm.py` — `get_chat_model()`: provider switch (`openai`/`anthropic`/`local`), raises
  `LLMConfigError` with a clear message rather than a bare import/key error.
- `cli.py` — terminal chat loop (`python -m agent.cli --debug`); also `build_default_graph()`,
  reused by the Streamlit app and the evaluation runner.
- `citation_tool.py` — deterministic APA/MLA/Chicago formatter, built on `feature/citation-tool`
  (not yet merged into this branch's lineage — see "Branches & merge order" below).

### `analytics/` (usage logging + staff dashboard, sits on top of the graph)
- `db.py` — SQLAlchemy `LogEntry` table (role, course, topic_category, resource_type, escalated,
  had_results). No query text, no session/user id — each row is independent and anonymous by
  construction.
- `log_writer.py` — `log_turn(role, state)`: called by both UIs right after `graph.invoke()`. Never
  raises — a logging failure must never break a chat turn that already succeeded.
- `auth.py` — `check_staff_access()`: a swappable auth-method registry for the staff dashboard
  (today: one method, `"password"`, checked against `STAFF_DASHBOARD_PASSWORD`). Fails closed if
  unset/unrecognized.
- `dashboard_queries.py` — aggregated, small-count-suppressed queries (`MIN_GROUP_COUNT = 5`) over
  `LogEntry`: volume by role, topic breakdown, one aggregate escalation rate (deliberately never
  cross-tabulated with role/topic — that combination would be more identifying than either alone).

### `streamlit_app/` (the UI)
- `Home.py` — the patron-facing chat UI. Role picker, chat history, debug toggle, UB theme
  (`.streamlit/config.toml` + inline CSS for what the theme file can't reach).
- `pages/1_Staff_Dashboard.py` — auto-appears as a nav item (Streamlit multipage convention).
  Password gate, then the three aggregate views from `dashboard_queries.py`.

### `evaluation/` (offline-ish quality check, not wired into CI yet)
- `run_eval.py` — runs every question in `evaluation/golden_test_set.json` through a real
  `graph.invoke()`, scores planning/clarification/RAG-hit/escalation accuracy, judges faithfulness
  of the compose step's stated reasons via a second LLM call. Writes `evaluation/last_run.json`
  (gitignored). Deliberately **not** added to `weekly-refresh.yml`: real OneSearch + LLM calls cost
  time and money per run; revisit once the golden set and metrics are proven out further across
  more runs.
- `metrics.py` / `faithfulness.py` / `langsmith_config.py` — scoring logic, the faithfulness judge
  prompt, and optional LangSmith tracing config.

## End-to-end flow

**Chat turn:** UI (CLI or Streamlit) → `graph.invoke({"role", "message", "history"})` →
`entry` (regex human-request check; escalates immediately if matched) → `plan` (LLM: message →
`QueryPlan`; ends the turn with a clarifying question if too ambiguous) → `search` (OneSearch, via
Layer 1) and `rag` (Chroma library-guide lookup) run in parallel → `compose` (LLM picks results,
code renders the final answer with real links/locations) → UI prints `state["answer"]` and calls
`log_turn()` to write one anonymized row.

**Data refresh (biweekly, automated):** `pipeline.refresh.refresh_all()` → fetch the 3 library-guide
source pages (politely, via `scraper.fetch`) → parse each → compare counts against the last
snapshot (keep old data if a source failed or its count suspiciously dropped) → save
`data/processed/records.json` → re-embed only changed records into the Chroma collection in
`vectorstore/`. GitHub Actions opens a PR with the updated snapshot; nothing pushes straight to
`main`.

## How to run things

```bash
cp .env.example .env            # set OPENAI_API_KEY at minimum
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m agent.cli --debug          # terminal chat
.venv/bin/streamlit run streamlit_app/Home.py  # web UI (chat + staff dashboard page)

.venv/bin/python -m pipeline.refresh           # (re)build the library-guide RAG data
.venv/bin/python -m evaluation.run_eval        # score against the golden test set

.venv/bin/pytest                                # full test suite (327 tests across every branch below once merged)
```

Staff dashboard needs `STAFF_DASHBOARD_AUTH_METHOD=password` and `STAFF_DASHBOARD_PASSWORD` set in
`.env` — unset by default, so the page shows a setup message instead of unlocking.

## Branches & merge order (none of these are merged yet — all awaiting review)

Built as a stack, each PR targeting the branch below it, so you can review and merge bottom-up:

1. `feature/human-request-fast-path` → `main` (PR, deterministic human-escalation fast path)
2. `feature/anon-logging` → `feature/human-request-fast-path` (anonymized usage logging)
3. `feature/staff-dashboard` → `feature/anon-logging` (this project's PR #9: the staff dashboard)
4. `feature/final-docs` → `feature/staff-dashboard` (this doc + log cleanup)

Independent of that stack, also awaiting review:

5. `feature/citation-tool` → `main` (deterministic APA/MLA/Chicago citation formatting)

**Merging:** take the stack 1→4 in order, then `feature/citation-tool` separately — it touches
`agent/cli.py`, `agent/models.py`, `CLAUDE.md`, and `tests/test_agent_cli.py`, so merging it last
will produce conflicts in those files. A full merge simulation (scratch clone, stack + citation-tool
merged together) was run and confirmed the conflicts are small and mechanical: two `run_chat()`
keyword-argument additions that combine cleanly (`log_fn` + `citations`), and additive `CLAUDE.md`
sections. The merged result passes the full suite (251 tests). No code logic conflicts anywhere.

## Known limitations / deliberately deferred

- **Staff dashboard auth is a shared password, not real SSO** — the brief said the real method is
  still pending library confirmation; built behind a swappable registry (`analytics/auth.py`) so
  swapping in SSO later doesn't require touching the dashboard page.
- **Evaluation suite isn't CI-wired** — real OneSearch + LLM calls per run cost time and money; run
  it manually (`python -m evaluation.run_eval`) until the golden set is proven out further.
- **Citation formatter simplifications** — author-count cutoffs (1-2 full, 3+ "et al.") aren't each
  style guide's real cutoff; organizational authors (not a person) can render oddly in APA. Cosmetic,
  not a crash or missing citation. See `CLAUDE.md`'s citation-tool section for specifics.
- **No Streamlit Cloud deployment set up** — runs locally / wherever `streamlit run` is invoked;
  deploying it anywhere specific hasn't been decided.
- **`course` is always logged as `None`** — the `LogEntry` column exists, but no UI currently asks
  the patron for it.

## Where the rest of the detail lives

- `CLAUDE.md` — the running design log: one section per build slice, including real bugs found
  live and why each non-obvious decision was made. Read this when you need the *why*, not just
  the *what*.
- `docs/superpowers/specs/*.md` — full design docs for each architectural slice.
- `docs/superpowers/plans/*.md` — the task-by-task implementation plans those specs turned into.
- `docs/design/ub-ai-assistant-reference.png` — the original UI mockup the Streamlit theme matches.
