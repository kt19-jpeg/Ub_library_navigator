# Layer 2, Slice 1: Agent Core (plan → search → compose)

Date: 2026-09-24. Status: approved in conversation, pending written-spec review.

## Goal

The thinnest end-to-end path from a user's natural-language question to a grounded answer using Layer 1 (`onesearch/`): turn the sentence into keywords + filters, search OneSearch, pick the 1-3 best results, fetch their access details, and answer. Runs from a CLI. Model provider is switchable (OpenAI, Anthropic, local).

## Scope

In: model factory, query planner, OneSearch step (with one refinement), compose step with deterministic answer rendering, LangGraph wiring, CLI, offline tests, manual live check.

Out (later slices): fast paths (human request, citation follow-up), escalation, citation tool, real RAG enrichment (a stub returns nothing), scrapers/Chroma/embeddings, logging, Streamlit UI/dashboard, evaluation. The empty stubs `agent/router.py`, `agent/escalation.py`, `agent/citation_tool.py` stay untouched.

## Layer 1 changes this slice depends on

Found while building the pipeline against live UB (2026-09-24); done as Task 1 of the plan:

- `OneSearchClient.fallback_link(query)` and `record_link(record_id)` helpers (used by search degradation and answer rendering).
- `search_with_fallback` now mints/reuses the session before starting its timeout clock and returns at the timeout instead of waiting for the slow worker (a `with ThreadPoolExecutor` block made a 9s timeout take ~34s).
- Session mint waits for the page's own `/pnxs` response instead of `networkidle`, which now times out at 30s (response wait: 2.4-4.2s, 4/4 runs).

## Model layer: `agent/llm.py`

`get_chat_model(provider=None, model=None) -> BaseChatModel`, driven by env vars `LLM_PROVIDER` (`openai` | `anthropic` | `local`, default `openai`), `LLM_MODEL`, `LLM_BASE_URL` (local only), plus `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

- `openai`: `langchain-openai` `ChatOpenAI`.
- `anthropic`: `langchain-anthropic` `ChatAnthropic` (optional extra).
- `local`: Ollama via `langchain-ollama` (optional extra), or any OpenAI-compatible server by setting `LLM_BASE_URL` (uses `ChatOpenAI(base_url=...)`, no extra package).
- A missing package or key raises an error naming exactly what to install or set.
- CLI flags `--provider` / `--model` override env vars.
- Embeddings are out of scope here and stay on OpenAI in a later slice.

## Data model: `agent/models.py`

- `QueryPlan` (pydantic): `keywords: str`, `resource_type: Literal[books, articles, journals, dissertations, videos] | None`, `peer_reviewed: bool`, `needs_clarification: str | None`. `resource_type` values mirror `onesearch.client.RESOURCE_TYPES`.
- `Selection` (pydantic): `items: list[{record_id, reason}]` (1-3), `confidence: Literal[high, medium, low]`.
- `AgentState` (TypedDict): `role`, `message`, `history` (last 3 turns), `plan`, `search_result`, `rag_context` (always `[]` in this slice), `selection`, `details`, `answer`, `notes`.

## Graph (`agent/graph.py`, LangGraph)

```
plan → search → rag (stub) → compose → END
   ↘ needs_clarification → END (returns the question)
```

### plan: `agent/planner.py` (LLM call 1)

`plan_query(model, role, message, history) -> QueryPlan` via structured output. Strips filler ("find me", "peer reviewed", "journals" are filters, not keywords). Role sets defaults (undergraduate + broad topic leans to books/overview; graduate/faculty leans to peer-reviewed articles); explicit user wording always overrides. On any exception or validation failure (bad JSON, unknown `resource_type`) it returns `QueryPlan(keywords=<raw message>, resource_type=None, peer_reviewed=False)`, which keeps weaker local models usable. Example: "find me peer reviewed journals on mental health" → `keywords="mental health"`, `resource_type="journals"`, `peer_reviewed=True`. "Peer reviewed journals" is ambiguous (journal titles vs. articles in peer-reviewed journals); the planner may set `needs_clarification` instead of guessing.

### search: `agent/onesearch_tool.py` (no LLM)

`run_onesearch(plan, client) -> (SearchResult, notes)`. Calls `client.search_with_fallback(plan.keywords, resource_type=..., peer_reviewed=...)`. If the result has no candidates and any filter was set, it refines exactly once (two searches total, per the original brief): same keywords, all filters dropped, and records in `notes` which filters were relaxed so the answer can say so. A timed-out result passes its `fallback_link` through unchanged. This is the only place in Layer 2 that talks to `OneSearchClient`.

### rag: stub

Returns `[]`. Kept as a real node so the graph shape does not change when RAG lands.

### compose: `agent/compose.py` (LLM call 2 + deterministic rendering)

1. The LLM receives compact candidate fields (record_id, title, author, date, type, subject, truncated description, delivery category), the role and `rag_context`, and returns a `Selection`.
2. Code validates it: every `record_id` must be in the candidate list, at most 3 are kept, duplicates removed. Invalid ids are dropped, so the LLM can never widen `fetch_detail()`.
3. Code calls `client.fetch_detail(selected_ids)` once (batched).
4. The final answer is rendered by code, not the LLM: the LLM supplies only each item's one-line reason; links, call numbers, library/sub-location and availability come from the real `ResourceDetail`. This prevents invented URLs or shelf locations.
5. `confidence == low` or an empty valid selection returns an honest "I'm not confident" message plus the OneSearch fallback link (real escalation arrives in a later slice).

## Error handling

- Planner LLM error → fallback plan (raw message, no filters).
- Compose LLM error → deterministic fallback: top 3 search results, no reasoning text, plus the fallback link.
- Layer 1 timeout or HTTP error → apologetic message plus the direct OneSearch link (built by Layer 1's `search_with_fallback`).
- Missing key/package → clear startup error.

## CLI: `agent/cli.py` (`python -m agent.cli`)

Asks for a role (undergraduate / graduate / faculty), then loops: read message → run graph → print answer. Keeps the last 3 turns as history. `/role` changes the role mid-conversation, `/quit` exits. `--provider`, `--model`, and `--debug` (prints plan, filters used, per-step timings).

## Latency

Two LLM calls, one search (plus at most one refinement), one batched detail call. Target within the 3-6s warm budget with a hosted model; local-model speed depends on hardware.

## Testing

All offline: a fake chat model returns scripted structured output and Layer 1 is mocked.

- `llm.py`: provider selection, missing-key/package messages.
- planner: example query → `mental health` + `journals` + peer-reviewed; invalid output falls back; clarification is passed through.
- search step: refines exactly once when empty and reports relaxed filters; no refinement when the first pass has results or no filters were set; timeout fallback link passes through.
- compose: ids not in the candidates dropped, capped at 3, `fetch_detail` called only with chosen ids; rendered answer contains real links/locations; low confidence and compose-error fallbacks.
- graph: end to end, plus the clarification short-circuit.
- Manual live check (not CI): the example query against real UB and a chosen model.

## Files

```
agent/llm.py  agent/models.py  agent/planner.py  agent/onesearch_tool.py
agent/compose.py  agent/graph.py  agent/cli.py
tests/test_agent_*.py
```

Dependencies: `langchain-anthropic` and `langchain-ollama` are optional extras; `langgraph`, `langchain`, `langchain-openai`, `pydantic`, `python-dotenv` are core.

## Decisions carried into CLAUDE.md

The refinement rule is "one refinement, all filters dropped, keywords kept" (two searches total), replacing the earlier wording that implied dropping filters one at a time.
