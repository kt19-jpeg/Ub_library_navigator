# Evaluation Suite (Layer 2, Slice 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A manual script that scores the real agent's behavior (planning correctness,
escalation accuracy, RAG directory-lookup accuracy, answer faithfulness) against a golden set of
15 (role, question) pairs, plus zero-code LangSmith tracing for debugging a specific bad answer.

**Architecture:** Three purely-deterministic metrics (planning, escalation, RAG hit) compare the
real graph's output to each golden question's expected *behavior*, not expected search results
(OneSearch is live and would make results-based ground truth go stale). The fourth metric
(faithfulness of the LLM's one-line "Why" reasons) is a small LLM-judge function using the
project's own `agent.llm.get_chat_model()` — the `ragas` package is not used (see Global
Constraints). One real `graph.invoke()` per golden question feeds all four metrics; LangSmith
tracing needs no code changes, only env vars.

**Tech Stack:** Same as prior slices. No new packages. `ragas` is removed from `requirements.txt`.

**Spec:** `docs/superpowers/specs/2026-09-30-evaluation-suite-design.md`

## Global Constraints

- Run every command from `Ub_library_navigator/` with the project venv: `.venv/bin/pytest`.
- **Do not add `ragas` as a dependency.** Confirmed live 2026-09-30: every release from 0.3.9
  through 0.4.3 unconditionally imports `langchain_community.chat_models.vertexai.ChatVertexAI`,
  which no longer exists in current `langchain-community` — `import ragas` fails outright,
  tracked upstream in ragas#2741, #2753, #2995, #2745 with an unmerged fix (PR #3017). The
  `ragas` line is removed from `requirements.txt` in Task 4.
- The golden set labels *expected behavior* (filters, should-escalate, expected RAG type), never
  expected OneSearch titles/results — those drift as UB's live catalog changes.
- All four metrics are computed from a single `graph.invoke(...)` call per golden question, not
  fragmented per-metric calls — this reflects the real turn a user gets.
- This suite is manual in this slice (`python -m evaluation.run_eval`), not wired into
  `weekly-refresh.yml` — real OneSearch + LLM calls cost time and money per run.
- LangSmith tracing needs zero code changes anywhere in `agent/`, `onesearch/`, or `pipeline/` —
  confirmed live against the installed `langsmith` 0.14.0: it reads `LANGSMITH_TRACING_V2`
  (falling back to `LANGCHAIN_TRACING_V2`) and matching `_API_KEY`/`_PROJECT` vars purely from the
  environment via LangChain's global callback manager.
- Commit messages are `type: short description`. **Never add a Co-Authored-By or "Generated with
  Claude Code" line.** Commit after each task; push once after the whole slice is reviewed.
- A faithfulness or escalation judgment calling a real model must never raise out of the eval
  run — a model failure fails closed (counts as ungrounded / not-escalated), never crashes the run.

## File Structure

| File | Responsibility |
|---|---|
| `evaluation/metrics.py` | 3 deterministic metrics: planning, escalation, RAG hit, plus `summarize()` |
| `evaluation/faithfulness.py` | LLM-judge for the "Why" reasons |
| `evaluation/langsmith_config.py` | `tracing_configured()` — env-var check only |
| `evaluation/golden_test_set.json` | 15 golden questions, real content (currently `[]`) |
| `evaluation/run_eval.py` | Orchestration: load set, run graph once per question, score, report |
| `requirements.txt` (modify) | Remove `ragas` |
| `.env.example` (modify) | Add `LANGSMITH_TRACING_V2=` |
| `CLAUDE.md` (modify) | Record the ragas finding and the live baseline results |

---

### Task 1: Deterministic metrics (`evaluation/metrics.py`)

**Files:**
- Create: `evaluation/metrics.py`
- Test: `tests/test_evaluation_metrics.py`

**Interfaces:**
- Consumes: `agent.escalation.LIVE_CHAT_URL`; `pipeline.records.LIBRARIAN_SUBJECT`/`HELP_TOPIC`/
  `DATABASE_SUBJECT`; a real graph-state dict with `state["plan"]` (`QueryPlan`), `state["answer"]`
  (`str`), `state["rag_context"]` (`RagContext`, may be absent/`None`).
- Produces: `QuestionResult` (frozen dataclass: `id`, `clarification_ok: bool | None`,
  `planning_ok: bool | None`, `escalation_expected: bool`, `escalation_actual: bool`,
  `rag_hit: bool | None`); `score_question(question: dict, state: dict) -> QuestionResult`;
  `Summary` (frozen dataclass: `planning_accuracy`, `clarification_accuracy`, `rag_hit_accuracy`
  — each `float | None` — plus the 4 escalation confusion-matrix counts); `summarize(results:
  list[QuestionResult]) -> Summary`. A golden question dict may set `check_planning: false`
  (planning target isn't well-defined, e.g. directory/nonsense questions) or
  `expect_clarification: true` (the turn should ask back, not search) to opt out of the metrics
  that don't apply to it.


- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluation_metrics.py`:

```python
from agent.escalation import LIVE_CHAT_URL
from agent.models import QueryPlan, RagContext
from evaluation.metrics import QuestionResult, score_question, summarize
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record


def _state(plan, answer, rag_context=None):
    return {"plan": plan, "answer": answer, "rag_context": rag_context or RagContext()}


def test_scores_correct_planning_and_no_escalation():
    question = {
        "id": "q1", "expected_resource_type": "journals", "expected_peer_reviewed": True,
        "should_escalate": False,
    }
    plan = QueryPlan(keywords="x", resource_type="journals", peer_reviewed=True)
    result = score_question(question, _state(plan, "Here is what I found..."))

    assert result.planning_ok is True
    assert result.escalation_expected is False
    assert result.escalation_actual is False
    assert result.clarification_ok is None
    assert result.rag_hit is None


def test_scores_wrong_resource_type_as_not_ok():
    question = {"id": "q2", "expected_resource_type": "books", "expected_peer_reviewed": False}
    plan = QueryPlan(keywords="x", resource_type="articles", peer_reviewed=False)
    result = score_question(question, _state(plan, "answer"))
    assert result.planning_ok is False


def test_detects_escalation_from_the_live_chat_link():
    question = {"id": "q3", "expected_resource_type": None, "expected_peer_reviewed": False, "should_escalate": True}
    plan = QueryPlan(keywords="x")
    answer = f"I am not confident...\nTalk to a librarian: {LIVE_CHAT_URL}"
    result = score_question(question, _state(plan, answer))
    assert result.escalation_actual is True
    assert result.escalation_expected is True


def test_clarification_question_only_scores_clarification():
    question = {"id": "q4", "expect_clarification": True}
    plan = QueryPlan(keywords="", needs_clarification="What topic are you researching?")
    result = score_question(question, _state(plan, "What topic are you researching?"))

    assert result.clarification_ok is True
    assert result.planning_ok is None
    assert result.rag_hit is None


def test_clarification_question_fails_if_plan_did_not_actually_ask():
    question = {"id": "q5", "expect_clarification": True}
    plan = QueryPlan(keywords="x")  # no needs_clarification set
    result = score_question(question, _state(plan, "Here is what I found..."))
    assert result.clarification_ok is False


def test_rag_hit_true_when_the_expected_type_has_a_hit():
    question = {"id": "q6", "expected_resource_type": None, "expected_peer_reviewed": False, "expected_rag_type": LIBRARIAN_SUBJECT}
    plan = QueryPlan(keywords="chemistry")
    rag = RagContext(librarians=[Record(id="librarian:chemistry", type=LIBRARIAN_SUBJECT, text="", metadata={})])
    result = score_question(question, _state(plan, "answer", rag))
    assert result.rag_hit is True


def test_rag_hit_false_when_the_expected_type_has_no_hit():
    question = {"id": "q7", "expected_resource_type": None, "expected_peer_reviewed": False, "expected_rag_type": HELP_TOPIC}
    plan = QueryPlan(keywords="x")
    result = score_question(question, _state(plan, "answer", RagContext()))
    assert result.rag_hit is False


def test_rag_hit_is_none_when_not_expected():
    question = {"id": "q8", "expected_resource_type": "books", "expected_peer_reviewed": False}
    plan = QueryPlan(keywords="x", resource_type="books")
    result = score_question(question, _state(plan, "answer"))
    assert result.rag_hit is None


def test_summarize_computes_rates_and_confusion_matrix():
    results = [
        QuestionResult("a", None, True, False, False, None),
        QuestionResult("b", None, False, False, False, None),
        QuestionResult("c", True, None, False, False, None),
        QuestionResult("d", None, None, True, True, None),   # escalation TP
        QuestionResult("e", None, None, False, True, None),  # escalation FP
        QuestionResult("f", None, None, True, False, None),  # escalation FN
        QuestionResult("g", None, True, False, False, True),
        QuestionResult("h", None, True, False, False, False),
    ]
    summary = summarize(results)

    assert summary.planning_accuracy == 0.75  # a, g, h True; b False -> 3 of 4
    assert summary.clarification_accuracy == 1.0
    assert summary.rag_hit_accuracy == 0.5
    assert (summary.escalation_true_positives, summary.escalation_false_positives,
            summary.escalation_true_negatives, summary.escalation_false_negatives) == (1, 1, 5, 1)


def test_summarize_handles_no_applicable_questions_for_a_metric():
    results = [QuestionResult("a", None, True, False, False, None)]
    summary = summarize(results)
    assert summary.clarification_accuracy is None
    assert summary.rag_hit_accuracy is None


def test_check_planning_false_skips_the_planning_metric():
    question = {"id": "q9", "check_planning": False, "expected_resource_type": "books"}
    plan = QueryPlan(keywords="x", resource_type="articles")  # would otherwise be "wrong"
    result = score_question(question, _state(plan, "answer"))
    assert result.planning_ok is None
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_evaluation_metrics.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'evaluation.metrics'` (the existing `evaluation/` package only has empty stub files).

- [ ] **Step 3: Write `evaluation/metrics.py`**

```python
from dataclasses import dataclass

from agent.escalation import LIVE_CHAT_URL
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT

RAG_TYPE_FIELDS = {
    LIBRARIAN_SUBJECT: "librarians",
    HELP_TOPIC: "help_topics",
    DATABASE_SUBJECT: "databases",
}


@dataclass(frozen=True)
class QuestionResult:
    id: str
    clarification_ok: bool | None  # None unless the question expects clarification
    planning_ok: bool | None  # None only for a clarification question (planning never ran)
    escalation_expected: bool
    escalation_actual: bool
    rag_hit: bool | None  # None when the question has no expected_rag_type


def score_question(question: dict, state: dict) -> QuestionResult:
    """Scores one golden question against the real graph state it produced."""
    plan = state["plan"]
    escalation_actual = LIVE_CHAT_URL in state["answer"]

    if question.get("expect_clarification"):
        clarification_ok = bool(plan.needs_clarification) and state["answer"] == plan.needs_clarification
        return QuestionResult(
            id=question["id"],
            clarification_ok=clarification_ok,
            planning_ok=None,
            escalation_expected=False,
            escalation_actual=escalation_actual,
            rag_hit=None,
        )

    planning_ok = None
    if question.get("check_planning", True):
        planning_ok = (
            plan.resource_type == question.get("expected_resource_type")
            and plan.peer_reviewed == question.get("expected_peer_reviewed", False)
        )

    rag_hit = None
    expected_rag_type = question.get("expected_rag_type")
    if expected_rag_type:
        rag_context = state.get("rag_context")
        field = RAG_TYPE_FIELDS[expected_rag_type]
        rag_hit = bool(rag_context is not None and getattr(rag_context, field))

    return QuestionResult(
        id=question["id"],
        clarification_ok=None,
        planning_ok=planning_ok,
        escalation_expected=question.get("should_escalate", False),
        escalation_actual=escalation_actual,
        rag_hit=rag_hit,
    )


@dataclass(frozen=True)
class Summary:
    planning_accuracy: float | None
    clarification_accuracy: float | None
    rag_hit_accuracy: float | None
    escalation_true_positives: int
    escalation_false_positives: int
    escalation_true_negatives: int
    escalation_false_negatives: int


def _rate(values) -> float | None:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def summarize(results: list[QuestionResult]) -> Summary:
    return Summary(
        planning_accuracy=_rate(r.planning_ok for r in results),
        clarification_accuracy=_rate(r.clarification_ok for r in results),
        rag_hit_accuracy=_rate(r.rag_hit for r in results),
        escalation_true_positives=sum(1 for r in results if r.escalation_expected and r.escalation_actual),
        escalation_false_positives=sum(1 for r in results if not r.escalation_expected and r.escalation_actual),
        escalation_true_negatives=sum(1 for r in results if not r.escalation_expected and not r.escalation_actual),
        escalation_false_negatives=sum(1 for r in results if r.escalation_expected and not r.escalation_actual),
    )
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_evaluation_metrics.py -v
```Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_evaluation_metrics.py
git commit -m "feat: add deterministic planning/escalation/RAG-hit evaluation metrics"
```

### Task 2: Faithfulness LLM-judge (`evaluation/faithfulness.py`)

**Files:**
- Create: `evaluation/faithfulness.py`
- Test: `tests/test_evaluation_faithfulness.py`

**Interfaces:**
- Consumes: a model exposing `with_structured_output(schema).invoke(messages)` (a `BaseChatModel`
  or `tests.fakes.FakeChatModel`).
- Produces: `FaithfulnessVerdict` (pydantic: `grounded: bool`); `judge_faithfulness(model,
  candidate_view: dict, reason: str) -> bool` (never raises — a model failure or unparseable
  response returns `False`, i.e. fails closed, not a crash).


- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluation_faithfulness.py`:

```python
from evaluation.faithfulness import FaithfulnessVerdict, judge_faithfulness
from tests.fakes import FakeChatModel


def test_returns_true_for_a_grounded_verdict():
    model = FakeChatModel(FaithfulnessVerdict(grounded=True))
    result = judge_faithfulness(model, {"title": "Chemistry Basics", "subject": "Chemistry"}, "Covers chemistry.")
    assert result is True


def test_returns_false_for_an_ungrounded_verdict():
    model = FakeChatModel(FaithfulnessVerdict(grounded=False))
    result = judge_faithfulness(model, {"title": "X"}, "Proves a specific theorem not mentioned anywhere.")
    assert result is False


def test_prompt_includes_the_real_candidate_fields_and_reason():
    model = FakeChatModel(FaithfulnessVerdict(grounded=True))
    judge_faithfulness(model, {"title": "Chemistry Basics", "subject": "Chemistry"}, "Covers chemistry.")

    human = model.calls[0][1][1]
    assert "Chemistry Basics" in human
    assert "Covers chemistry." in human
    assert model.schemas == [FaithfulnessVerdict]


def test_dict_response_is_validated():
    model = FakeChatModel({"grounded": True})
    assert judge_faithfulness(model, {"title": "X"}, "reason") is True


def test_model_failure_fails_closed_as_ungrounded():
    model = FakeChatModel(RuntimeError("provider down"))
    assert judge_faithfulness(model, {"title": "X"}, "reason") is False


def test_unparseable_response_fails_closed():
    model = FakeChatModel(None)
    assert judge_faithfulness(model, {"title": "X"}, "reason") is False
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_evaluation_faithfulness.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'evaluation.faithfulness'`.

- [ ] **Step 3: Write `evaluation/faithfulness.py`**

```python
from pydantic import BaseModel, Field

FAITHFULNESS_SYSTEM = """You check whether a one-line explanation about a library search result is
grounded in that result's real metadata, or whether it invents something not actually present.

Given the candidate's real title, author, subject, and description, and the stated reason for
recommending it, return grounded=true only if every claim in the reason is actually supported by
those fields. A reason that stays general or reasonably infers from the topic is fine; a reason
that states something as fact which isn't in the given fields (a specific finding, a date, a claim
about content not in the description) is not grounded."""


class FaithfulnessVerdict(BaseModel):
    grounded: bool = Field(
        description="True only if every claim in the reason is supported by the candidate's real fields."
    )


def judge_faithfulness(model, candidate_view: dict, reason: str) -> bool:
    """LLM-judge: is `reason` grounded in `candidate_view`'s real fields? Never raises: a model
    failure counts as ungrounded (fails closed) rather than crashing the whole eval run."""
    try:
        structured = model.with_structured_output(FaithfulnessVerdict)
        verdict = structured.invoke(
            [
                ("system", FAITHFULNESS_SYSTEM),
                ("human", f"candidate: {candidate_view}\nreason: {reason}"),
            ]
        )
        if isinstance(verdict, dict):
            verdict = FaithfulnessVerdict.model_validate(verdict)
        if not isinstance(verdict, FaithfulnessVerdict):
            return False
        return verdict.grounded
    except Exception:
        return False
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_evaluation_faithfulness.py -v
```Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/faithfulness.py tests/test_evaluation_faithfulness.py
git commit -m "feat: add LLM-judge faithfulness check for selection reasons"
```

### Task 3: LangSmith tracing check (`evaluation/langsmith_config.py`)

**Files:**
- Create: `evaluation/langsmith_config.py`
- Test: `tests/test_evaluation_langsmith_config.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `tracing_configured() -> bool`.


- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluation_langsmith_config.py`:

```python
from evaluation.langsmith_config import tracing_configured


def test_false_when_nothing_is_set(monkeypatch):
    for name in ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert tracing_configured() is False


def test_true_when_langsmith_vars_are_set(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    assert tracing_configured() is True


def test_true_with_the_older_langchain_names(monkeypatch):
    for name in ("LANGSMITH_TRACING_V2", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-test")
    assert tracing_configured() is True


def test_false_when_enabled_but_no_api_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert tracing_configured() is False


def test_false_when_tracing_flag_is_not_literally_true(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    assert tracing_configured() is False
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_evaluation_langsmith_config.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'evaluation.langsmith_config'`.

- [ ] **Step 3: Write `evaluation/langsmith_config.py`**

```python
import os


def tracing_configured() -> bool:
    """True if LangSmith tracing is actually switched on. Tracing itself needs no code
    changes anywhere else: LangChain/LangGraph trace automatically through a global
    callback manager once these env vars are set (confirmed live 2026-09-30 against the
    installed langsmith 0.14.0, which checks LANGSMITH_* first, falling back to the
    older LANGCHAIN_* names) - this function only decides whether to print a heads-up
    that tracing is off, not whether tracing can work."""
    enabled = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return bool(enabled and enabled.lower() == "true" and api_key)
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_evaluation_langsmith_config.py -v
```Expected: 5 passed.

- [ ] **Step 5: Add the tracing env var to `.env.example`**

Insert a new line right before the existing `LANGSMITH_API_KEY=` line:

```text
LANGSMITH_TRACING_V2=
```

- [ ] **Step 6: Commit**

```bash
git add evaluation/langsmith_config.py tests/test_evaluation_langsmith_config.py .env.example
git commit -m "feat: add LangSmith tracing configuration check"
```

### Task 4: Golden test set and orchestration (`evaluation/run_eval.py`)

**Files:**
- Modify: `evaluation/golden_test_set.json` (currently `[]`)
- Create: `evaluation/run_eval.py`
- Test: `tests/test_evaluation_run_eval.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `score_question`/`summarize` (Task 1); `judge_faithfulness` (Task 2);
  `tracing_configured` (Task 3); `agent.cli.build_default_graph`; `agent.llm.get_chat_model`/
  `LLMConfigError`.
- Produces: `GOLDEN_SET_PATH`, `REPORT_PATH`; `load_golden_set(path=GOLDEN_SET_PATH) ->
  list[dict]`; `evaluate(graph, model, questions: list[dict]) -> dict` (a JSON-serializable report:
  `summary`, `faithfulness_accuracy`, `questions` — each question entry includes its resolved
  `plan` for debugging a failure); `print_summary(report: dict, write=print) -> None`; `main() ->
  int`, the `python -m evaluation.run_eval` entrypoint.

The golden set content below reflects real, confirmed-correct behavior from live runs against
real UB data and a real model (2026-09-30), not assumptions: 5 directory/nonsense questions have
`check_planning: false` because the planner has no way to know they aren't literal topic searches
(a known limitation, already documented in the RAG pipeline spec) — expecting a specific
`resource_type` for "who is the subject librarian for chemistry" was the wrong expectation to
encode, confirmed by running it. One question (`dissertations-faculty`) also has
`check_planning: false`: its `peer_reviewed` flipped between two live runs on the same unchanged
input — a genuinely probabilistic role-default nudge, not a bug, so pinning an exact expected
value here would be a flaky test, not a real contract.


- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluation_run_eval.py`:

```python
import json

from agent.escalation import LIVE_CHAT_URL
from agent.models import QueryPlan, RagContext, SelectionItem
from evaluation.faithfulness import FaithfulnessVerdict
from evaluation.run_eval import evaluate, print_summary
from onesearch.models import SearchCandidate, SearchResult
from pipeline.records import LIBRARIAN_SUBJECT, Record
from tests.fakes import FakeChatModel


def _candidate(record_id, **overrides):
    fields = dict(
        record_id=record_id, title="A Title", author="Doe, Jane", resource_type="book",
        format=None, subject="Topic", description="About the topic.", date="2020",
        source="Alma", delivery_category="Alma-E", online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )
    fields.update(overrides)
    return SearchCandidate(**fields)


class FakeGraph:
    """Scripted per-question states, matched by message so test order doesn't matter."""

    def __init__(self, states_by_message):
        self._states = states_by_message
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        return self._states[state["message"]]


def _answer_state(plan, candidates, selection, rag_context=None, answer="Here is what I found..."):
    return {
        "plan": plan,
        "answer": answer,
        "search_result": SearchResult(candidates=candidates, timed_out=False, fallback_link=None),
        "selection": selection,
        "rag_context": rag_context or RagContext(),
    }


def test_evaluate_scores_every_question_and_reports_faithfulness():
    questions = [
        {
            "id": "q1", "role": "graduate", "message": "chemistry resources",
            "expected_resource_type": "articles", "expected_peer_reviewed": True,
            "should_escalate": False,
        },
        {
            "id": "q2", "role": "graduate", "message": "nonsense query",
            "expected_resource_type": None, "expected_peer_reviewed": False,
            "should_escalate": True,
        },
    ]
    candidate = _candidate("c1")
    states = {
        "chemistry resources": _answer_state(
            QueryPlan(keywords="chemistry", resource_type="articles", peer_reviewed=True),
            [candidate], [SelectionItem(record_id="c1", reason="Covers the topic.")],
        ),
        "nonsense query": _answer_state(
            QueryPlan(keywords="nonsense"), [], [],
            answer=f"Not confident.\nTalk to a librarian: {LIVE_CHAT_URL}",
        ),
    }
    graph = FakeGraph(states)
    model = FakeChatModel(FaithfulnessVerdict(grounded=True))

    report = evaluate(graph, model, questions)

    assert len(graph.calls) == 2
    assert report["summary"]["planning_accuracy"] == 1.0
    assert report["summary"]["escalation_true_positives"] == 1
    assert report["faithfulness_accuracy"] == 1.0
    assert report["questions"][0]["faithfulness"] == [True]
    assert report["questions"][0]["plan"]["resource_type"] == "articles"
    assert json.dumps(report)  # the whole report must be JSON-serializable


def test_evaluate_skips_faithfulness_for_clarification_questions():
    questions = [{"id": "q1", "role": "graduate", "message": "help", "expect_clarification": True}]
    states = {
        "help": _answer_state(
            QueryPlan(keywords="", needs_clarification="What topic?"), [], [],
            answer="What topic?",
        )
    }
    report = evaluate(FakeGraph(states), FakeChatModel(), questions)

    assert report["questions"][0]["faithfulness"] == []
    assert report["faithfulness_accuracy"] is None


def test_evaluate_includes_rag_hits_in_the_report():
    questions = [{
        "id": "q1", "role": "graduate", "message": "chemistry librarian",
        "expected_resource_type": None, "expected_peer_reviewed": False,
        "expected_rag_type": LIBRARIAN_SUBJECT,
    }]
    rag_context = RagContext(librarians=[Record(id="librarian:chemistry", type=LIBRARIAN_SUBJECT, text="", metadata={})])
    states = {"chemistry librarian": _answer_state(QueryPlan(keywords="chemistry"), [], [], rag_context=rag_context)}

    report = evaluate(FakeGraph(states), FakeChatModel(), questions)

    assert report["summary"]["rag_hit_accuracy"] == 1.0


def test_print_summary_prints_every_metric():
    report = {
        "summary": {
            "planning_accuracy": 0.8, "clarification_accuracy": 1.0, "rag_hit_accuracy": 0.5,
            "escalation_true_positives": 1, "escalation_false_positives": 0,
            "escalation_true_negatives": 5, "escalation_false_negatives": 0,
        },
        "faithfulness_accuracy": 0.9,
    }
    lines = []
    print_summary(report, write=lines.append)
    output = "\n".join(lines)
    assert "0.8" in output and "0.9" in output and "TP=1" in output
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_evaluation_run_eval.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'evaluation.run_eval'`.

- [ ] **Step 3: Replace `evaluation/golden_test_set.json`'s content**

```json
[
  {
    "id": "buffalo-history-undergrad",
    "role": "undergraduate",
    "message": "books about the history of Buffalo",
    "expected_resource_type": "books",
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": null
  },
  {
    "id": "citing-sources-help",
    "role": "undergraduate",
    "message": "where can I find help citing my sources",
    "expected_resource_type": null,
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": "help_topic",
    "check_planning": false
  },
  {
    "id": "mental-health-undergrad",
    "role": "undergraduate",
    "message": "mental health resources",
    "expected_resource_type": "books",
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": null
  },
  {
    "id": "videos-undergrad",
    "role": "undergraduate",
    "message": "videos about ancient Rome for my class",
    "expected_resource_type": "videos",
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": null
  },
  {
    "id": "printing-help-undergrad",
    "role": "undergraduate",
    "message": "how do I print on campus",
    "expect_clarification": true
  },
  {
    "id": "mental-health-journals-grad",
    "role": "graduate",
    "message": "find me peer reviewed journals on mental health",
    "expected_resource_type": "journals",
    "expected_peer_reviewed": true,
    "should_escalate": false,
    "expected_rag_type": null
  },
  {
    "id": "chemistry-articles-grad",
    "role": "graduate",
    "message": "chemistry resources",
    "expected_resource_type": "articles",
    "expected_peer_reviewed": true,
    "should_escalate": false,
    "expected_rag_type": null
  },
  {
    "id": "chemistry-librarian-grad",
    "role": "graduate",
    "message": "who is the subject librarian for chemistry",
    "expected_resource_type": null,
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": "librarian_subject",
    "check_planning": false
  },
  {
    "id": "archaeology-books-grad",
    "role": "graduate",
    "message": "books on archaeology methods",
    "expected_resource_type": "books",
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": null
  },
  {
    "id": "ambiguous-help-grad",
    "role": "graduate",
    "message": "can you help me with something",
    "expect_clarification": true
  },
  {
    "id": "nonsense-escalate-undergrad",
    "role": "undergraduate",
    "message": "xylophone repair instructions for my pet hamster",
    "expected_resource_type": null,
    "expected_peer_reviewed": false,
    "should_escalate": true,
    "expected_rag_type": null,
    "check_planning": false
  },
  {
    "id": "cs-databases-faculty",
    "role": "faculty",
    "message": "what databases are available for computer science research",
    "expected_resource_type": null,
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": "database_subject",
    "check_planning": false
  },
  {
    "id": "anatomy-librarian-faculty",
    "role": "faculty",
    "message": "anatomy subject librarian contact",
    "expected_resource_type": null,
    "expected_peer_reviewed": false,
    "should_escalate": false,
    "expected_rag_type": "librarian_subject",
    "check_planning": false
  },
  {
    "id": "buffalo-history-peer-reviewed-faculty",
    "role": "faculty",
    "message": "scholarly articles on the history of Buffalo",
    "expected_resource_type": "articles",
    "expected_peer_reviewed": true,
    "should_escalate": false,
    "expected_rag_type": null
  },
  {
    "id": "dissertations-faculty",
    "role": "faculty",
    "message": "recent dissertations on climate change",
    "expected_resource_type": "dissertations",
    "expected_peer_reviewed": true,
    "should_escalate": false,
    "expected_rag_type": null,
    "check_planning": false,
    "_note": "peer_reviewed flipped between live runs for this role-default nudge (confirmed 2026-09-30); inherently probabilistic, not a bug."
  }
]
```

- [ ] **Step 4: Write `evaluation/run_eval.py`**

```python
import json
import sys
from pathlib import Path

from evaluation.faithfulness import judge_faithfulness
from evaluation.metrics import score_question, summarize

GOLDEN_SET_PATH = Path("evaluation/golden_test_set.json")
REPORT_PATH = Path("evaluation/last_run.json")


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _judge_selection_faithfulness(model, state: dict) -> list[bool]:
    selection = state.get("selection") or []
    if not selection:
        return []
    candidates_by_id = {c.record_id: c for c in state["search_result"].candidates}
    judgments = []
    for item in selection:
        candidate = candidates_by_id.get(item.record_id)
        if candidate is None:
            continue
        view = {
            "title": candidate.title,
            "author": candidate.author,
            "subject": candidate.subject,
            "description": candidate.description,
        }
        judgments.append(judge_faithfulness(model, view, item.reason))
    return judgments


def evaluate(graph, model, questions: list[dict]) -> dict:
    """Runs one real graph.invoke() per golden question, scores it, and judges faithfulness
    for whatever it selected. Returns a JSON-serializable report."""
    results = []
    all_faithfulness: list[bool] = []
    per_question = []

    for question in questions:
        state = graph.invoke({"role": question["role"], "message": question["message"], "history": []})
        result = score_question(question, state)
        results.append(result)

        faithfulness = [] if question.get("expect_clarification") else _judge_selection_faithfulness(model, state)
        all_faithfulness.extend(faithfulness)

        per_question.append(
            {
                "id": question["id"],
                "role": question["role"],
                "message": question["message"],
                "plan": state["plan"].model_dump(),
                "answer": state.get("answer"),
                "result": vars(result),
                "faithfulness": faithfulness,
            }
        )

    summary = summarize(results)
    faithfulness_accuracy = (sum(all_faithfulness) / len(all_faithfulness)) if all_faithfulness else None
    return {
        "summary": vars(summary),
        "faithfulness_accuracy": faithfulness_accuracy,
        "questions": per_question,
    }


def print_summary(report: dict, write=print) -> None:
    s = report["summary"]
    write(f"Planning accuracy:      {s['planning_accuracy']}")
    write(f"Clarification accuracy: {s['clarification_accuracy']}")
    write(f"RAG hit accuracy:       {s['rag_hit_accuracy']}")
    write(f"Faithfulness accuracy:  {report['faithfulness_accuracy']}")
    write(
        "Escalation: "
        f"TP={s['escalation_true_positives']} FP={s['escalation_false_positives']} "
        f"TN={s['escalation_true_negatives']} FN={s['escalation_false_negatives']}"
    )


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()

    from agent.cli import build_default_graph
    from agent.llm import LLMConfigError, get_chat_model
    from evaluation.langsmith_config import tracing_configured

    if not tracing_configured():
        print("Note: LangSmith tracing is not configured; continuing without it.")

    try:
        graph = build_default_graph()
        model = get_chat_model()
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2

    report = evaluate(graph, model, load_golden_set())
    print_summary(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Full report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
.venv/bin/pytest tests/test_evaluation_run_eval.py -v
```Expected: 4 passed.

- [ ] **Step 6: Remove `ragas` from `requirements.txt`**

Delete the line that just contains `ragas`. `langsmith` stays (no bug in that package; already installed as a transitive dependency and used by Task 3).

- [ ] **Step 7: Run the whole offline suite**

```bash
.venv/bin/pytest -q
```Expected: all pass (roughly 177).

- [ ] **Step 8: Commit**

```bash
git add evaluation/golden_test_set.json evaluation/run_eval.py tests/test_evaluation_run_eval.py requirements.txt
git commit -m "feat: add golden test set and evaluation orchestration, drop broken ragas dependency"
```

### Task 5: Live run and docs

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none new; this task runs Tasks 1-4 for real and records the baseline.


- [ ] **Step 1: Build the Chroma store if it isn't already current (manual, needs `OPENAI_API_KEY` and network)**

```bash
.venv/bin/python -m pipeline.refresh
```Expected: counts around `{'librarian_subject': ~130, 'help_topic': ~155, 'database_subject': ~80}`, matching the RAG pipeline slice's confirmed live numbers.

- [ ] **Step 2: Run the evaluation suite for real**

```bash
.venv/bin/python -m evaluation.run_eval 2>&1 | tee docs/evaluation_run_output.txt
```
Expected (confirmed live 2026-09-30, with the golden set as corrected in Task 4): planning
accuracy 1.0, clarification accuracy 1.0, RAG hit accuracy 1.0, faithfulness accuracy around
0.9-0.95 (occasionally flags one "Why" reason as not fully grounded — that is the check working,
not a bug to chase), escalation TP=1 FP=0 TN=14 FN=0. Small variation run to run is expected and
fine (real LLM calls, not perfectly deterministic even at temperature 0) — a result in the same
neighborhood confirms the suite works; it does not need to match exactly.

- [ ] **Step 3: Update `CLAUDE.md`**

Add a new subsection after the "Layer 2, slice 2: RAG data pipeline" section (or after whichever
is now the last Layer 2 slice section), documenting: the `ragas` upstream bug and the decision to
drop it (link the 4 GitHub issues); the 4 metrics and what each one actually checks; that the
golden set encodes expected behavior, not expected OneSearch results; the 2 real findings from
tuning the golden set against live data (directory/nonsense questions can't have a well-defined
planning target; the one role-default nudge that is genuinely probabilistic); and the confirmed
live baseline numbers from this task's run.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/evaluation_run_output.txt
git commit -m "docs: record evaluation suite findings and live baseline"
```
(If the live run in Steps 1-2 could not be done, e.g. no `OPENAI_API_KEY` yet, omit
`docs/evaluation_run_output.txt` from `git add` and say so in the handoff, but still update
`CLAUDE.md` with the design findings that don't need a live run to state.)


---

## Slice acceptance (run after Task 5)

- [ ] `.venv/bin/pytest -q` passes with no network and no API key.
- [ ] `python -m evaluation.run_eval` runs against real UB data and a real model, produces
      `evaluation/last_run.json`, and prints a summary matching the expected neighborhood in
      Task 5.
- [ ] `pip show ragas` (or `grep ragas requirements.txt`) shows it is not a dependency.
- [ ] `git log --format=%B` on this slice's commits contains no `Co-Authored-By` or "Generated
      with" lines.
- [ ] Review, then `git push` once for the whole slice.
