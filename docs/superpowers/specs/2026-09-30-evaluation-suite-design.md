# Evaluation Suite (Layer 2, Slice 4)

Date: 2026-09-30. Status: approved in conversation, pending written-spec review.

## Goal

A manual, locally-run script that scores the agent's real behavior — planning correctness,
escalation accuracy, RAG directory-lookup accuracy, and answer faithfulness — against a small
golden set of (role, question) pairs, plus zero-code-change LangSmith tracing for debugging a
specific bad answer. Not CI-wired in this slice.

## A note on scope narrowing from the original brief

The original brief ("RAGAS scores retrieval precision, recall, and answer faithfulness against a
golden test set... tagged by role") assumed two things that don't hold for this system:

1. **A static corpus.** OneSearch is live and external; the same query returns different results
   as UB's catalog changes over time. A golden set of *expected titles* would go stale. Instead,
   the golden set labels *expected behavior* (which filters the planner should pick, whether the
   turn should escalate) — stable regardless of what OneSearch returns on a given day.
2. **A free-text generated answer to score end-to-end.** This system's answer is mostly
   code-rendered from real data (links, call numbers, locations never come from the LLM — established
   in every prior slice). The only LLM-generated text worth checking for faithfulness is the
   one-line "Why" reason in `agent/compose.py`'s `Selection`. Full-answer RAGAS metrics (context
   precision/recall, answer relevancy) don't fit either of these; this slice scores what's
   actually generated, not what a typical RAG chatbot generates.

## The `ragas` package is not used — a real, currently-open upstream bug

Confirmed live 2026-09-30: every `ragas` release from 0.3.9 through the current 0.4.3
unconditionally imports `langchain_community.chat_models.vertexai.ChatVertexAI` at module load
time (`ragas/llms/base.py`), which no longer exists in current `langchain-community` (Vertex AI
chat support moved to a separate `langchain-google-vertexai` package). `import ragas` fails
outright for anyone not using Google Vertex AI, us included. Tracked upstream in
[ragas#2741](https://github.com/vibrantlabsai/ragas/issues/2741),
[#2753](https://github.com/vibrantlabsai/ragas/issues/2753),
[#2995](https://github.com/vibrantlabsai/ragas/issues/2995), and
[#2745](https://github.com/vibrantlabsai/ragas/issues/2745); the fix
([PR #3017](https://github.com/vibrantlabsai/ragas/pull/3017)) is open and unmerged. Downgrading
`ragas` does not help — the break is in `langchain-community`, which every `ragas` version relies
on the same way.

**Decision:** don't depend on `ragas`. `requirements.txt`'s `ragas` line is removed. The one
metric that would have used it (Faithfulness) is implemented directly as a small LLM-judge
function using this project's own `agent.llm.get_chat_model()` — narrower in scope than RAGAS's
general-purpose metric, but it only needs to judge one line of text against one candidate's real
fields, and it stays testable offline with the existing `FakeChatModel` pattern. `langsmith`
(tracing) is unaffected — it is a separate package with no such bug, already installed as a
transitive dependency and confirmed working.

## Metrics

Four metrics, all computed from a single `graph.invoke(...)` call per golden question (not
fragmented per-metric calls — this reflects the real turn a user gets, including the "search and
rag both always run" limitation already documented in the RAG pipeline spec):

| Metric | What it checks | How |
|---|---|---|
| Planning accuracy | Did `plan_query()` pick the expected `resource_type` and `peer_reviewed`? | Deterministic comparison of `state["plan"]` to the golden question's expected fields. No LLM judge. |
| Escalation accuracy | Did the turn escalate exactly when it should have? | Deterministic: `agent.escalation.LIVE_CHAT_URL in state["answer"]`, compared to the golden question's `should_escalate`. Scored as a confusion matrix (false escalations vs. missed escalations), matching the original brief's explicit call-out that both directions matter. |
| RAG hit accuracy | For directory-style questions ("who's the chemistry librarian"), did the expected RAG type get a hit? | Deterministic: for questions with `expected_rag_type` set, check `state["rag_context"]` has a non-empty list for that type. Questions with `expected_rag_type: null` are skipped for this metric. |
| Faithfulness | Does each selected item's one-line "Why" reason stay grounded in that candidate's real title/subject/description, with no invented claims? | LLM-judge: for each item in `state["selection"]`, look up its real candidate from `state["search_result"].candidates` by `record_id`, and ask the configured model (same `LLM_PROVIDER` as the rest of the app) for a structured `grounded: bool` verdict. Skipped when there is no selection (nothing to judge). |

## Golden test set

`evaluation/golden_test_set.json`, ~15 entries (currently `[]`, a placeholder committed with no
content), covering: all 3 roles, one clear-topic question per role, one ambiguous question
(expects `needs_clarification`, so planning/RAG/escalation metrics are skipped for it — it never
reaches search), one deliberately unanswerable question (`should_escalate: true`), and 2-3
RAG-only directory questions (`expected_rag_type` set).

```json
{
  "role": "graduate",
  "message": "find me peer reviewed journals on mental health",
  "expected_resource_type": "journals",
  "expected_peer_reviewed": true,
  "should_escalate": false,
  "expected_rag_type": null
}
```

`expected_resource_type`/`expected_peer_reviewed` are `null`/omitted for the ambiguous and
directory-only questions where planning isn't the point being tested.

## Running it

`python -m evaluation.run_eval` (renamed from the originally-planned `run_ragas.py` — misleading
to keep that name once the `ragas` package is dropped; `CLAUDE.md`'s repo-structure section is
updated to match). Manual only in this slice, real OneSearch + real LLM calls, not wired into
`weekly-refresh.yml` (a fast follow once the golden set and metrics are proven out, per the
decision to keep this cheap while it's still being tuned).

Output: a summary table to stdout (per-metric pass rate, escalation confusion matrix) and a
single overwritten `evaluation/last_run.json` with full per-question results, so two runs can be
diffed by hand when tuning a prompt. No historical archive of past runs in this slice (YAGNI —
add timestamped reports later if actually needed).

## LangSmith tracing

Separate concern from the metrics above — this is for inspecting *why* one specific run did what
it did, not for scoring. Confirmed live 2026-09-30 against the installed `langsmith` 0.14.0: its
`tracing_is_enabled()` reads `LANGSMITH_TRACING_V2` (falling back to `LANGCHAIN_TRACING_V2`) and
similarly-namespaced `_API_KEY`/`_PROJECT` vars purely from the environment, through LangChain's
global callback manager — **zero code changes needed** to `agent/graph.py` or anywhere else; any
compiled LangGraph graph is traced automatically once these are set. `evaluation/langsmith_config.py`
is a small helper, `tracing_configured() -> bool`, that checks the vars are set and gives a clear
message when they're not (same pattern as `agent.llm.LLMConfigError`) — `run_eval.py` prints a
one-line note if tracing isn't configured, but still runs without it. `.env.example` gains
`LANGSMITH_TRACING_V2=` alongside the existing `LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` entries.

## Files

```
evaluation/golden_test_set.json   ~15 golden questions (placeholder [] today)
evaluation/metrics.py             the 4 scoring functions above
evaluation/faithfulness.py        the LLM-judge function (own module: it's the one metric
                                   that needs a model + FakeChatModel-based tests, kept
                                   separate from the 3 purely-deterministic metrics)
evaluation/run_eval.py            orchestration: load golden set, run graph, score, report
evaluation/langsmith_config.py    tracing_configured() helper
tests/test_evaluation_metrics.py
tests/test_evaluation_faithfulness.py
tests/test_evaluation_run_eval.py
```

Dependencies: none added. `ragas` line removed from `requirements.txt`.

## Testing

All offline except the real script run itself:

- `metrics.py`: planning/escalation/RAG-hit scoring against constructed `AgentState`-shaped
  dicts, no network, no model.
- `faithfulness.py`: `FakeChatModel` (existing test helper) returns scripted grounded/ungrounded
  verdicts; verifies the prompt includes the real candidate fields and the stated reason.
- `run_eval.py`: a fake graph (matching the `FakeGraph` pattern already used in
  `tests/test_agent_cli.py`) drives the orchestration logic — loading the golden set, calling the
  graph once per question, aggregating metrics, writing the report — without real network/API
  calls.
- Manual, not CI: `python -m evaluation.run_eval` run for real once the golden set has real
  content, output saved to `docs/` for the record, same pattern as every other slice's live check.
