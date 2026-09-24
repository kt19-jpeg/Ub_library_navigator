# Layer 2, Slice 1: Agent Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a patron's natural-language question plus role into a grounded answer using Layer 1 (`onesearch/`), through a plan → search → compose LangGraph graph run from a CLI, with a switchable model provider (OpenAI, Anthropic, local).

**Architecture:** An LLM planner converts the sentence into keywords + filters (`QueryPlan`); a non-LLM step searches OneSearch through Layer 1 and refines once if a filtered search is empty; an LLM compose step picks 1-3 results, code fetches their access details in one batched call and renders the answer (links and locations never come from the LLM). Models are built by one factory from env vars and are only ever used through LangChain's `with_structured_output`, so tests run offline with a tiny fake model.

**Tech Stack:** Python 3.12, LangGraph, LangChain (`langchain-core`, `langchain-openai`; optional `langchain-anthropic`, `langchain-ollama`), pydantic v2, pytest. Layer 1: requests + Playwright.

**Spec:** `docs/superpowers/specs/2026-09-24-layer2-slice1-agent-core-design.md`

## Global Constraints

- Run every command from `Ub_library_navigator/` with the project venv: `.venv/bin/python -m pytest` (or bare `.venv/bin/pytest` after Task 2 adds `pytest.ini`).
- Nothing outside `onesearch/` may call `search.lib.buffalo.edu` directly or import `playwright`; Layer 2 only uses `OneSearchClient`. (CLAUDE.md, "adapter module")
- `fetch_detail()` is called only with the 1-3 ids the compose step selected, validated against the candidate list; never looped over a full result set. (CLAUDE.md, finding 5 and retrieval flow)
- The refinement rule is: refine exactly once (two searches total), same keywords, all filters dropped, and tell the user what was relaxed.
- Links, call numbers, locations and availability in the final answer are rendered by code from `ResourceDetail` / `SearchCandidate`, never written by the LLM.
- Any LLM failure (planner or compose) degrades to a deterministic answer instead of raising; a missing key or package fails fast at startup with a message naming what to set or install.
- Tests are offline: no network, no real model, no Playwright. Live behaviour is checked only by the manual scripts in Task 8.
- Commit messages are `type: short description`. **Never add a Co-Authored-By or "Generated with Claude Code" line** (`.claude/settings.json` sets attribution to empty; this repo's rule).
- Commit after each task. Do NOT push per task; push once after the whole slice is reviewed (CLAUDE.md development rules).
- `agent/router.py`, `agent/escalation.py`, `agent/citation_tool.py` and everything under `scraper/`, `pipeline/`, `analytics/`, `streamlit_app/`, `evaluation/` are out of scope and stay untouched.

## File Structure

| File | Responsibility |
|---|---|
| `onesearch/session.py` (modify) | Mint session by waiting for the page's own `/pnxs` response |
| `onesearch/client.py` (modify) | `fallback_link()`, `record_link()`, a `search_with_fallback` whose timeout really bounds latency |
| `agent/models.py` | `QueryPlan`, `Selection`, `AgentState`, `ROLES` |
| `agent/llm.py` | `get_chat_model()` provider factory + `LLMConfigError` |
| `agent/planner.py` | `plan_query()` (LLM call 1) with safe fallback |
| `agent/onesearch_tool.py` | `run_onesearch()`: search + refine once + degrade on Layer 1 errors |
| `agent/compose.py` | `compose()`: select (LLM call 2), validate, fetch detail, render answer |
| `agent/graph.py` | `build_graph()`: LangGraph wiring, per-node timings |
| `agent/cli.py` | Chat loop, `build_default_graph()`, `main()` |
| `agent/live_check.py` | Manual end-to-end check (not CI) |
| `tests/fakes.py` | `FakeChatModel` for offline tests |
| `tests/test_agent_*.py` | One test module per agent module |

---

### Task 1: Harden Layer 1 for Layer 2 (timeout, session mint, link helpers)

Layer 2 needs two link helpers and a fallback timeout that is honest. Building the real pipeline against live UB exposed two Layer 1 defects, both fixed here: (a) `search_with_fallback` used a `with ThreadPoolExecutor` block, whose exit waits for the slow worker, so a 9s timeout took 34s to return; and a cold session mint (inside the timed call) always ate the budget; (b) the session mint waited for `networkidle`, which now times out at 30s because the site's analytics never go idle. Waiting for the page's own `/pnxs` response mints in 2.4-4.2s (measured 4/4 runs; `networkidle` failed 2/2).

**Files:**
- Modify: `onesearch/client.py`
- Modify: `onesearch/session.py`
- Test: `tests/test_client.py` (append)

**Interfaces:**
- Produces: `OneSearchClient.fallback_link(query: str) -> str` (OneSearch results page, keywords only); `OneSearchClient.record_link(record_id: str) -> str` (OneSearch full-record page); `search_with_fallback(...)` unchanged signature but now returns at the timeout and does not count session mint against it.


- [ ] **Step 1: Append the failing tests to `tests/test_client.py`**

```python
def test_fallback_link_points_at_onesearch_results_page():
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
    )
    link = client.fallback_link("mental health")
    assert link == (
        "https://search.lib.buffalo.edu/discovery/search?"
        "query=any,contains,mental%20health&vid=01SUNY_BUF:everything"
    )


def test_search_with_fallback_uses_fallback_link_on_timeout():
    def _slow_get(*args, **kwargs):
        time.sleep(0.2)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=_slow_get,
    )
    result = client.search_with_fallback("mental health", timeout_seconds=0.05)
    assert result.fallback_link == client.fallback_link("mental health")


def test_record_link_points_at_onesearch_full_record_page():
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
    )
    assert client.record_link("alma123") == (
        "https://search.lib.buffalo.edu/discovery/fulldisplay?"
        "docid=alma123&context=L&vid=01SUNY_BUF:everything"
    )


def test_search_with_fallback_returns_promptly_at_the_timeout():
    def _slow_get(*args, **kwargs):
        time.sleep(0.5)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=_slow_get,
    )
    start = time.monotonic()
    result = client.search_with_fallback("mental health", timeout_seconds=0.05)

    assert result.timed_out is True
    assert time.monotonic() - start < 0.3  # did not wait for the slow worker


def test_search_with_fallback_does_not_count_session_mint_against_the_timeout():
    def _slow_mint():
        time.sleep(0.2)
        return SessionCookies(jsessionid="J", urm_st="1", urm_se="900001", institute="UB")

    response = MagicMock()
    response.json.return_value = {"docs": []}
    client = OneSearchClient(
        session_manager=SessionManager(mint_fn=_slow_mint),
        cache=QueryCache(ttl_seconds=60),
        http_get=MagicMock(return_value=response),
    )
    result = client.search_with_fallback("mental health", timeout_seconds=0.1)

    assert result.timed_out is False
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_client.py -v

```Expected: FAIL. `fallback_link`/`record_link` don't exist (`AttributeError`), the prompt-return test fails (elapsed ~0.5s), and the mint test reports `timed_out is True`.

- [ ] **Step 3: Add the two link helpers to `OneSearchClient` (just above `search_with_fallback`)**

```python
    def fallback_link(self, query: str) -> str:
        """Direct OneSearch results page for `query` (keywords only, no filters)."""
        return (
            "https://search.lib.buffalo.edu/discovery/search?"
            f"query=any,contains,{urllib.parse.quote(query)}&vid={self._vid}"
        )

    def record_link(self, record_id: str) -> str:
        """OneSearch full-record page for one record id."""
        return (
            "https://search.lib.buffalo.edu/discovery/fulldisplay?"
            f"docid={record_id}&context=L&vid={self._vid}"
        )

```

- [ ] **Step 4: Replace the body of `search_with_fallback`**

Keep the signature and the `_build_facets(resource_type, peer_reviewed)` validation line; replace everything after that line with:

```python
        # Mint (or reuse) the session before starting the clock: a cold mint
        # takes a few seconds by itself and is not what this timeout bounds.
        self._session_manager.get_cookies()
        # No `with` block: its exit would wait for a slow worker and defeat the timeout.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self.search, query, resource_type, peer_reviewed)
            try:
                candidates = future.result(timeout=timeout_seconds)
                return SearchResult(candidates=candidates, timed_out=False, fallback_link=None)
            except concurrent.futures.TimeoutError:
                return SearchResult(
                    candidates=[], timed_out=True, fallback_link=self.fallback_link(query)
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
```

- [ ] **Step 5: Fix the session mint in `onesearch/session.py`**

Replace the `page.goto(...)` call and the `page.wait_for_load_state("networkidle")` line (and the comment above `page.goto`) with:

```python
        # Wait for the page's own search response rather than "networkidle":
        # the site's analytics keep the network busy, so networkidle timed out
        # at 30s (confirmed live 2026-09-24) while this returns in 2-4s.
        with page.expect_response(
            lambda r: "/primaws/rest/pub/pnxs" in r.url and r.status == 200, timeout=30000
        ):
            page.goto(
                "https://search.lib.buffalo.edu/discovery/search?query=any,contains,test&vid=01SUNY_BUF:everything",
                wait_until="domcontentloaded",
            )
```

- [ ] **Step 6: Run the whole suite**

```bash
.venv/bin/python -m pytest -q

```Expected: all pass (the pre-existing 32 plus the 5 new client tests).

- [ ] **Step 7: Verify the mint live (manual, needs network)**

```bash
.venv/bin/python - <<'EOF'
import time
from onesearch.session import mint_session_via_playwright
start = time.monotonic()
cookies = mint_session_via_playwright()
print(f"mint: {time.monotonic() - start:.1f}s", bool(cookies.urm_st and cookies.jsessionid))
EOF
```Expected: `mint: 2-5s True`. Run it twice.

- [ ] **Step 8: Commit**

```bash
git add onesearch/client.py onesearch/session.py tests/test_client.py
git commit -m "fix: make fallback timeout a real bound, mint session via /pnxs response, add link helpers"
```

### Task 2: Agent data models, fake chat model, project config

**Files:**
- Create: `pytest.ini`, `agent/models.py`, `tests/fakes.py`
- Test: `tests/test_agent_models.py`
- Modify: `requirements.txt`; Create: `requirements-optional.txt`

**Interfaces:**
- Produces: `ROLES: tuple[str, ...]`, `ResourceType` (Literal), `QueryPlan(keywords, resource_type, peer_reviewed, needs_clarification)`, `SelectionItem(record_id, reason)`, `Selection(items, confidence)`, `AgentState` (TypedDict, `total=False`), and test helper `FakeChatModel(*responses)` with `.with_structured_output(schema)` and `.invoke(messages)`, recording `.schemas` and `.calls`.


- [ ] **Step 1: Add config so bare `pytest` finds the packages; add dependencies**

Create `pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
```
In `requirements.txt` add a line `pydantic` (after `langchain-openai`). Create `requirements-optional.txt`:

```text
# Only needed for the matching LLM_PROVIDER
langchain-anthropic   # LLM_PROVIDER=anthropic
langchain-ollama      # LLM_PROVIDER=local without LLM_BASE_URL
```
Install: `.venv/bin/pip install -r requirements.txt`

- [ ] **Step 2: Write the failing test**

Create `tests/test_agent_models.py`:

```python
from typing import get_args

import pytest
from pydantic import ValidationError

from agent.models import ROLES, QueryPlan, ResourceType, Selection
from onesearch.client import RESOURCE_TYPES


def test_resource_type_literal_matches_layer1_resource_types():
    assert set(get_args(ResourceType)) == set(RESOURCE_TYPES)


def test_query_plan_defaults():
    plan = QueryPlan(keywords="mental health")
    assert plan.resource_type is None
    assert plan.peer_reviewed is False
    assert plan.needs_clarification is None


def test_query_plan_rejects_unknown_resource_type():
    with pytest.raises(ValidationError):
        QueryPlan(keywords="x", resource_type="podcasts")


def test_selection_requires_valid_confidence():
    with pytest.raises(ValidationError):
        Selection(items=[], confidence="certain")


def test_roles_are_the_three_project_roles():
    assert ROLES == ("undergraduate", "graduate", "faculty")
```
Also create `tests/fakes.py` (test helper, no test of its own):

```python
class FakeChatModel:
    """Stands in for a LangChain chat model in tests.

    Only supports the one thing the agent uses: with_structured_output(schema).invoke(messages).
    Each invoke() returns (or raises, if it is an Exception) the next scripted response.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.schemas = []
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.schemas.append(schema)
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
```

- [ ] **Step 3: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_agent_models.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'agent.models'`.

- [ ] **Step 4: Write `agent/models.py`**

```python
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from onesearch.models import ResourceDetail, SearchResult

ROLES = ("undergraduate", "graduate", "faculty")

# Mirrors onesearch.client.RESOURCE_TYPES (a test enforces they stay equal).
ResourceType = Literal["books", "articles", "journals", "dissertations", "videos"]


class QueryPlan(BaseModel):
    """What the planner extracts from a natural-language message."""

    keywords: str = Field(
        description="1-6 content words to search for. Filler and filter words are removed."
    )
    resource_type: ResourceType | None = Field(
        default=None, description="Restrict to one resource type, or null for any."
    )
    peer_reviewed: bool = Field(
        default=False, description="True only if the user asked for peer-reviewed material."
    )
    needs_clarification: str | None = Field(
        default=None,
        description="A short question to ask the user if the request is genuinely ambiguous.",
    )


class SelectionItem(BaseModel):
    record_id: str = Field(description="record_id copied exactly from the candidate list.")
    reason: str = Field(description="One sentence on why this result fits the user.")


class Selection(BaseModel):
    """The compose LLM's choice of results to show."""

    items: list[SelectionItem] = Field(
        default_factory=list, description="The 1-3 best results, best first."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well these results answer the user's request."
    )


class AgentState(TypedDict, total=False):
    role: str
    message: str
    history: list[dict]  # [{"user": str, "assistant": str}], most recent last
    plan: QueryPlan
    search_result: SearchResult
    rag_context: list
    selection: list[SelectionItem]
    details: dict[str, ResourceDetail]
    answer: str
    notes: list[str]
    timings: dict[str, float]
```

- [ ] **Step 5: Run it to verify it passes**

```bash
.venv/bin/pytest -q
```Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pytest.ini requirements.txt requirements-optional.txt agent/models.py tests/fakes.py tests/test_agent_models.py
git commit -m "feat: add agent data models and fake chat model for tests"
```

### Task 3: Model factory (`agent/llm.py`)

**Files:**
- Create: `agent/llm.py`
- Test: `tests/test_agent_llm.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `get_chat_model(provider: str | None = None, model: str | None = None, base_url: str | None = None) -> BaseChatModel`; `LLMConfigError(Exception)`; `PROVIDERS = ("openai", "anthropic", "local")`; `DEFAULT_MODELS: dict[str, str]`. Arguments override env vars `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`; keys come from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, optional `LLM_API_KEY`.


- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_llm.py`:

```python
import sys
import types

import pytest

from agent.llm import DEFAULT_MODELS, LLMConfigError, get_chat_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_openai_and_needs_a_key():
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        get_chat_model()


def test_openai_uses_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = get_chat_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == DEFAULT_MODELS["openai"]


def test_env_vars_select_provider_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    assert get_chat_model().model_name == "gpt-4o"


def test_arguments_override_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    assert get_chat_model(model="from-arg").model_name == "from-arg"


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
        get_chat_model(provider="bogus")


def test_anthropic_needs_key(monkeypatch):
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        get_chat_model(provider="anthropic")


def test_anthropic_missing_package_is_explained(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)  # makes import fail
    with pytest.raises(LLMConfigError, match="pip install langchain-anthropic"):
        get_chat_model(provider="anthropic")


def test_anthropic_builds_chat_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    fake = types.ModuleType("langchain_anthropic")

    class ChatAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake.ChatAnthropic = ChatAnthropic
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake)

    model = get_chat_model(provider="anthropic")
    assert model.kwargs["model"] == DEFAULT_MODELS["anthropic"]
    assert model.kwargs["api_key"] == "k"


def test_local_with_base_url_uses_openai_compatible_client():
    model = get_chat_model(provider="local", model="qwen2.5", base_url="http://localhost:1234/v1")
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "qwen2.5"
    assert model.openai_api_base == "http://localhost:1234/v1"


def test_local_without_base_url_uses_ollama(monkeypatch):
    fake = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake.ChatOllama = ChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake)

    model = get_chat_model(provider="local")
    assert model.kwargs["model"] == DEFAULT_MODELS["local"]


def test_local_missing_ollama_package_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    with pytest.raises(LLMConfigError, match="langchain-ollama"):
        get_chat_model(provider="local")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_agent_llm.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'agent.llm'`.

- [ ] **Step 3: Write `agent/llm.py`**

```python
import os

from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("openai", "anthropic", "local")
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "local": "llama3.1",
}


class LLMConfigError(Exception):
    """The requested model cannot be built: bad provider, missing key, or missing package."""


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Build the chat model. Arguments override the LLM_PROVIDER, LLM_MODEL and
    LLM_BASE_URL env vars; the provider defaults to openai."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Unknown LLM provider {provider!r}; expected one of {list(PROVIDERS)}")
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    base_url = base_url or os.getenv("LLM_BASE_URL")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigError("OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=0)

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMConfigError("ANTHROPIC_API_KEY is not set")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise LLMConfigError(
                "The anthropic provider needs langchain-anthropic: pip install langchain-anthropic"
            ) from exc
        return ChatAnthropic(model=model, api_key=api_key, temperature=0)

    # local: any OpenAI-compatible server (LM Studio, llama.cpp, vLLM, Ollama's
    # /v1 endpoint) when LLM_BASE_URL is set, otherwise Ollama's native API.
    if base_url:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=os.getenv("LLM_API_KEY") or "not-needed",
            temperature=0,
        )
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise LLMConfigError(
            "The local provider needs langchain-ollama (pip install langchain-ollama) "
            "or set LLM_BASE_URL to an OpenAI-compatible server"
        ) from exc
    return ChatOllama(model=model, temperature=0)
```

- [ ] **Step 4: Run it to verify it passes**

```bash
.venv/bin/pytest tests/test_agent_llm.py -v
```Expected: 11 passed.

- [ ] **Step 5: Document the env vars in `.env.example`**

Append:

```text

# LLM provider: openai | anthropic | local (default openai)
LLM_PROVIDER=openai
LLM_MODEL=
# local only: OpenAI-compatible server, e.g. http://localhost:1234/v1 (LM Studio) or
# http://localhost:11434/v1 (Ollama). Leave empty to use Ollama's native API.
LLM_BASE_URL=
LLM_API_KEY=
ANTHROPIC_API_KEY=
```

- [ ] **Step 6: Commit**

```bash
git add agent/llm.py tests/test_agent_llm.py .env.example
git commit -m "feat: add switchable chat model factory (openai, anthropic, local)"
```

### Task 4: Query planner (`agent/planner.py`)

**Files:**
- Create: `agent/planner.py`
- Test: `tests/test_agent_planner.py`

**Interfaces:**
- Consumes: `QueryPlan` (Task 2); a model exposing `with_structured_output(schema).invoke(messages)` (a `BaseChatModel` or `FakeChatModel`).
- Produces: `plan_query(model, role: str, message: str, history: list[dict] | None = None) -> QueryPlan` (never raises); `fallback_plan(message: str) -> QueryPlan`; `PLANNER_SYSTEM: str`; `MAX_HISTORY_TURNS = 3`. `history` items are `{"user": str, "assistant": str}`.


- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_planner.py`:

```python
from agent.models import QueryPlan
from agent.planner import fallback_plan, plan_query
from tests.fakes import FakeChatModel


def test_returns_the_models_plan():
    plan = QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True)
    model = FakeChatModel(plan)

    result = plan_query(model, "graduate", "find me peer reviewed journals on mental health")

    assert result == plan
    assert model.schemas == [QueryPlan]


def test_prompt_contains_role_message_and_recent_history():
    model = FakeChatModel(QueryPlan(keywords="x"))
    history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(5)]

    plan_query(model, "faculty", "what about books?", history)

    system, human = model.calls[0]
    assert system[0] == "system" and human[0] == "human"
    text = human[1]
    assert "role: faculty" in text
    assert "message: what about books?" in text
    assert "user: q4" in text and "user: q2" in text
    assert "user: q1" not in text  # only the last 3 turns are sent


def test_dict_response_is_validated_into_a_plan():
    model = FakeChatModel({"keywords": "climate change", "resource_type": "books", "peer_reviewed": False})
    result = plan_query(model, "undergraduate", "books on climate change")
    assert result == QueryPlan(keywords="climate change", resource_type="books")


def test_invalid_dict_falls_back_to_raw_message():
    model = FakeChatModel({"keywords": "x", "resource_type": "podcasts"})
    assert plan_query(model, "graduate", "  some question ") == fallback_plan("some question")


def test_model_error_falls_back_to_raw_message():
    model = FakeChatModel(RuntimeError("provider down"))
    assert plan_query(model, "graduate", "some question") == QueryPlan(keywords="some question")


def test_none_response_falls_back():
    model = FakeChatModel(None)
    assert plan_query(model, "graduate", "some question") == QueryPlan(keywords="some question")


def test_empty_keywords_without_clarification_falls_back():
    model = FakeChatModel(QueryPlan(keywords="  "))
    assert plan_query(model, "graduate", "help") == QueryPlan(keywords="help")


def test_clarification_is_passed_through_even_with_empty_keywords():
    plan = QueryPlan(keywords="", needs_clarification="What topic are you researching?")
    model = FakeChatModel(plan)
    assert plan_query(model, "graduate", "find me something") == plan
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_agent_planner.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'agent.planner'`.

- [ ] **Step 3: Write `agent/planner.py`**

```python
from agent.models import QueryPlan

MAX_HISTORY_TURNS = 3

PLANNER_SYSTEM = """You turn a library patron's message into a search request for UB Libraries' OneSearch catalog.

Return:
- keywords: 1-6 content words describing the TOPIC only. Remove filler ("find me", "I need", "please") and remove words that are filters, not topic ("peer reviewed", "journals", "books", "articles", "scholarly").
- resource_type: one of books, articles, journals, dissertations, videos, or null. Set it only when the user names a kind of material. "journals" means journal titles; "articles" means individual articles.
- peer_reviewed: true only if the user asks for peer-reviewed / scholarly / academic-journal material.
- needs_clarification: a single short question, ONLY if the request is too ambiguous to search (for example there is no topic at all). Otherwise null. When you set it, keywords may be empty.

Use the user's role for defaults when they did not say what kind of material they want:
- undergraduate + a broad topic: lean toward books or overview material, no peer_reviewed filter.
- graduate or faculty: lean toward articles, with peer_reviewed true when they ask for research literature.
What the user explicitly asks for always overrides the role default.

Use the recent conversation to resolve follow-ups ("what about books?" reuses the earlier topic).

Example: role=graduate, message="find me peer reviewed journals on mental health"
-> keywords="mental health", resource_type="journals", peer_reviewed=true, needs_clarification=null
"""


def fallback_plan(message: str) -> QueryPlan:
    """Used when the model fails or returns something unusable: search the raw message unfiltered."""
    return QueryPlan(keywords=message.strip())


def _format_request(role: str, message: str, history: list[dict]) -> str:
    lines = [f"role: {role}"]
    recent = history[-MAX_HISTORY_TURNS:]
    if recent:
        lines.append("recent conversation:")
        for turn in recent:
            lines.append(f"  user: {turn['user']}")
            lines.append(f"  assistant: {turn['assistant']}")
    lines.append(f"message: {message}")
    return "\n".join(lines)


def plan_query(model, role: str, message: str, history: list[dict] | None = None) -> QueryPlan:
    try:
        structured = model.with_structured_output(QueryPlan)
        plan = structured.invoke(
            [
                ("system", PLANNER_SYSTEM),
                ("human", _format_request(role, message, history or [])),
            ]
        )
        if isinstance(plan, dict):
            plan = QueryPlan.model_validate(plan)
        if not isinstance(plan, QueryPlan):
            return fallback_plan(message)
        if not plan.keywords.strip() and not plan.needs_clarification:
            return fallback_plan(message)
        return plan
    except Exception:
        return fallback_plan(message)
```

- [ ] **Step 4: Run it to verify it passes**

```bash
.venv/bin/pytest tests/test_agent_planner.py -v
```Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/planner.py tests/test_agent_planner.py
git commit -m "feat: add LLM query planner with safe fallback"
```

### Task 5: OneSearch step (`agent/onesearch_tool.py`)

**Files:**
- Modify: `agent/onesearch_tool.py` (currently an empty stub)
- Test: `tests/test_agent_onesearch_tool.py`

**Interfaces:**
- Consumes: `QueryPlan` (Task 2); `OneSearchClient.search_with_fallback(query, timeout_seconds=9.0, resource_type=None, peer_reviewed=False) -> SearchResult` and `OneSearchClient.fallback_link(query) -> str` (Task 1).
- Produces: `SearchOutcome(result: SearchResult, notes: list[str])` (frozen dataclass); `run_onesearch(plan: QueryPlan, client: OneSearchClient) -> SearchOutcome` (never raises).


- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_onesearch_tool.py`:

```python
from unittest.mock import MagicMock

from agent.models import QueryPlan
from agent.onesearch_tool import run_onesearch
from onesearch.models import SearchCandidate, SearchResult


def _candidate(record_id="1"):
    return SearchCandidate(
        record_id=record_id, title="t", author=None, resource_type=None, format=None,
        subject=None, description=None, date=None, source=None, delivery_category=None,
        online_link=None, call_number=None, library=None, sub_location=None,
        availability_status=None,
    )


def _result(candidates=(), timed_out=False, link=None):
    return SearchResult(candidates=list(candidates), timed_out=timed_out, fallback_link=link)


def test_passes_keywords_and_filters_to_layer1():
    client = MagicMock()
    client.search_with_fallback.return_value = _result([_candidate()])
    plan = QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True)

    outcome = run_onesearch(plan, client)

    client.search_with_fallback.assert_called_once_with(
        "mental health", resource_type="journals", peer_reviewed=True
    )
    assert len(outcome.result.candidates) == 1
    assert outcome.notes == []


def test_refines_once_without_filters_when_filtered_search_is_empty():
    client = MagicMock()
    client.search_with_fallback.side_effect = [_result(), _result([_candidate()])]
    plan = QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True)

    outcome = run_onesearch(plan, client)

    assert client.search_with_fallback.call_count == 2
    assert client.search_with_fallback.call_args_list[1].args == ("mental health",)
    assert client.search_with_fallback.call_args_list[1].kwargs == {}
    assert len(outcome.result.candidates) == 1
    assert "type: journals, peer-reviewed" in outcome.notes[0]


def test_stops_after_one_refinement_even_if_still_empty():
    client = MagicMock()
    client.search_with_fallback.return_value = _result()

    outcome = run_onesearch(QueryPlan(keywords="zzz", peer_reviewed=True), client)

    assert client.search_with_fallback.call_count == 2
    assert outcome.result.candidates == []


def test_no_refinement_when_no_filters_were_set():
    client = MagicMock()
    client.search_with_fallback.return_value = _result()

    run_onesearch(QueryPlan(keywords="zzz"), client)

    assert client.search_with_fallback.call_count == 1


def test_timeout_is_not_refined_and_keeps_fallback_link():
    client = MagicMock()
    client.search_with_fallback.return_value = _result(timed_out=True, link="https://x/y")

    outcome = run_onesearch(QueryPlan(keywords="a", peer_reviewed=True), client)

    assert client.search_with_fallback.call_count == 1
    assert outcome.result.timed_out is True
    assert outcome.result.fallback_link == "https://x/y"
    assert any("slow" in n for n in outcome.notes)


def test_layer1_exception_degrades_to_direct_link():
    client = MagicMock()
    client.search_with_fallback.side_effect = RuntimeError("HTTP 500")
    client.fallback_link.return_value = "https://search.lib.buffalo.edu/discovery/search?query=any,contains,a"

    outcome = run_onesearch(QueryPlan(keywords="a"), client)

    assert outcome.result.candidates == []
    assert outcome.result.timed_out is True
    assert outcome.result.fallback_link.startswith("https://search.lib.buffalo.edu/")
    assert any("could not be reached" in n for n in outcome.notes)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_agent_onesearch_tool.py -v
```Expected: FAIL, `ImportError: cannot import name 'run_onesearch'` (the file is empty).

- [ ] **Step 3: Write `agent/onesearch_tool.py`**

```python
from dataclasses import dataclass, field

from agent.models import QueryPlan
from onesearch.client import OneSearchClient
from onesearch.models import SearchResult


@dataclass(frozen=True)
class SearchOutcome:
    result: SearchResult
    notes: list[str] = field(default_factory=list)


def _describe_filters(plan: QueryPlan) -> str:
    parts = []
    if plan.resource_type:
        parts.append(f"type: {plan.resource_type}")
    if plan.peer_reviewed:
        parts.append("peer-reviewed")
    return ", ".join(parts)


def run_onesearch(plan: QueryPlan, client: OneSearchClient) -> SearchOutcome:
    """Search OneSearch for the plan. If a filtered search finds nothing, refine
    exactly once (two searches total): same keywords, all filters dropped, and
    a note saying what was relaxed. This is the only place Layer 2 talks to Layer 1."""
    filters = _describe_filters(plan)
    try:
        result = client.search_with_fallback(
            plan.keywords,
            resource_type=plan.resource_type,
            peer_reviewed=plan.peer_reviewed,
        )
        notes: list[str] = []
        if not result.candidates and not result.timed_out and filters:
            result = client.search_with_fallback(plan.keywords)
            notes.append(f"Nothing matched with filters ({filters}), so I searched without them.")
        if result.timed_out:
            notes.append("OneSearch was slow to respond.")
        return SearchOutcome(result=result, notes=notes)
    except Exception:
        # HTTP errors and anything unexpected from Layer 1: degrade to the direct link.
        return SearchOutcome(
            result=SearchResult(
                candidates=[], timed_out=True, fallback_link=client.fallback_link(plan.keywords)
            ),
            notes=["OneSearch could not be reached."],
        )
```

- [ ] **Step 4: Run it to verify it passes**

```bash
.venv/bin/pytest tests/test_agent_onesearch_tool.py -v
```Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/onesearch_tool.py tests/test_agent_onesearch_tool.py
git commit -m "feat: add OneSearch step with single refinement and graceful degradation"
```

### Task 6: Compose step (`agent/compose.py`)

**Files:**
- Create: `agent/compose.py`
- Test: `tests/test_agent_compose.py`

**Interfaces:**
- Consumes: `AgentState`, `Selection`, `SelectionItem`, `QueryPlan` (Task 2); `SearchResult` with `.candidates: list[SearchCandidate]`; `OneSearchClient.fetch_detail(ids) -> dict[str, ResourceDetail]`, `.fallback_link(q)`, `.record_link(id)` (Task 1). `ResourceDetail` fields used: `online_link`, `online_package`, `call_number`, `library`, `sub_location`, `availability_status`.
- Produces: `compose(state: AgentState, model, client) -> dict` returning `{"selection": list[SelectionItem], "details": dict[str, ResourceDetail], "answer": str}` (never raises); `select_results(...)`, `validate_selection(items, candidates)`, `render_answer(...)`, `MAX_SHOWN = 3`, `DESCRIPTION_CHARS = 300`.


- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_compose.py`:

```python
from unittest.mock import MagicMock

from agent.compose import (
    DESCRIPTION_CHARS,
    MAX_SHOWN,
    compose,
    select_results,
    validate_selection,
)
from agent.models import QueryPlan, Selection, SelectionItem
from onesearch.models import ResourceDetail, SearchCandidate, SearchResult
from tests.fakes import FakeChatModel


def _candidate(record_id, title="A Title", **overrides):
    fields = dict(
        record_id=record_id, title=title, author="Doe, Jane", resource_type="book",
        format=None, subject="Topic", description="About the topic.", date="2020",
        source="Alma", delivery_category="Alma-E", online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )
    fields.update(overrides)
    return SearchCandidate(**fields)


def _detail(record_id, **overrides):
    fields = dict(
        record_id=record_id, online_link=None, online_package=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )
    fields.update(overrides)
    return ResourceDetail(**fields)


def _client(details=None):
    client = MagicMock()
    client.fetch_detail.return_value = details or {}
    client.record_link.side_effect = lambda rid: f"https://rec/{rid}"
    client.fallback_link.side_effect = lambda q: f"https://more/{q}"
    return client


def _state(candidates, **extra):
    state = {
        "role": "graduate",
        "message": "find me peer reviewed journals on mental health",
        "plan": QueryPlan(keywords="mental health"),
        "search_result": SearchResult(candidates=candidates, timed_out=False, fallback_link=None),
        "rag_context": [],
        "notes": [],
    }
    state.update(extra)
    return state


def test_validate_selection_drops_unknown_duplicates_and_caps():
    candidates = [_candidate(str(i)) for i in range(6)]
    items = [SelectionItem(record_id=i, reason="r") for i in ["9", "1", "1", "2", "3", "4"]]

    kept = validate_selection(items, candidates)

    assert [i.record_id for i in kept] == ["1", "2", "3"]
    assert len(kept) == MAX_SHOWN


def test_select_results_sends_truncated_candidate_json_and_role():
    long = "x" * (DESCRIPTION_CHARS + 100)
    model = FakeChatModel(Selection(items=[], confidence="low"))

    select_results(model, "faculty", "the request", [_candidate("a1", description=long)], [])

    human = model.calls[0][1][1]
    assert "role: faculty" in human and "request: the request" in human
    assert '"record_id": "a1"' in human
    assert "x" * (DESCRIPTION_CHARS + 1) not in human
    assert model.schemas == [Selection]


def test_compose_fetches_details_only_for_valid_selected_ids_and_renders_them():
    candidates = [_candidate("e1", title="Online Journal"), _candidate("p1", title="Print Book"),
                  _candidate("z9", title="Not Picked")]
    model = FakeChatModel(Selection(
        items=[SelectionItem(record_id="e1", reason="Covers mental health research."),
               SelectionItem(record_id="bogus", reason="invented"),
               SelectionItem(record_id="p1", reason="A core textbook.")],
        confidence="high"))
    client = _client({
        "e1": _detail("e1", online_link="https://search.lib.buffalo.edu/view/x", online_package="SpringerLink"),
        "p1": _detail("p1", call_number="RC454 .M46 2020", library="Lockwood Library",
                      sub_location="General Collection", availability_status="unavailable"),
    })

    out = compose(_state(candidates), model, client)

    client.fetch_detail.assert_called_once_with(["e1", "p1"])
    answer = out["answer"]
    assert "1. Online Journal - Doe, Jane, 2020 [book]" in answer
    assert "Why: Covers mental health research." in answer
    assert "Online: https://search.lib.buffalo.edu/view/x (via SpringerLink)" in answer
    assert "Lockwood Library / General Collection - call number RC454 .M46 2020 (currently unavailable)" in answer
    assert "Record: https://rec/e1" in answer
    assert "bogus" not in answer and "Not Picked" not in answer
    assert "log in with your UB credentials" in answer
    assert "More results: https://more/mental health" in answer
    assert [i.record_id for i in out["selection"]] == ["e1", "p1"]


def test_compose_low_confidence_is_honest_and_skips_detail_fetch():
    model = FakeChatModel(Selection(items=[], confidence="low"))
    client = _client()

    out = compose(_state([_candidate("a")]), model, client)

    client.fetch_detail.assert_not_called()
    assert "not confident" in out["answer"]
    assert "https://more/mental health" in out["answer"]
    assert out["selection"] == []


def test_compose_llm_failure_falls_back_to_top_three_without_reasons():
    candidates = [_candidate(str(i), title=f"T{i}") for i in range(5)]
    model = FakeChatModel(RuntimeError("provider down"))
    client = _client()

    out = compose(_state(candidates), model, client)

    client.fetch_detail.assert_called_once_with(["0", "1", "2"])
    assert "T0" in out["answer"] and "T2" in out["answer"] and "T3" not in out["answer"]
    assert "Why:" not in out["answer"]
    assert "could not rank" in out["answer"]


def test_compose_survives_detail_fetch_failure_using_search_fields():
    candidates = [_candidate("p1", call_number="QA1 .B2", library="Lockwood Library",
                             availability_status="available", delivery_category="Alma-P")]
    model = FakeChatModel(Selection(items=[SelectionItem(record_id="p1", reason="Fits.")],
                                    confidence="medium"))
    client = _client()
    client.fetch_detail.side_effect = RuntimeError("HTTP 500")

    out = compose(_state(candidates), model, client)

    assert "call number QA1 .B2 (available)" in out["answer"]
    assert "could not load the full access details" in out["answer"]


def test_compose_with_no_candidates_skips_the_model_and_gives_the_link():
    model = FakeChatModel()
    state = _state([], notes=["OneSearch was slow to respond."])
    state["search_result"] = SearchResult(candidates=[], timed_out=True, fallback_link="https://direct/link")

    out = compose(state, model, _client())

    assert model.calls == []
    assert "could not find results" in out["answer"]
    assert "https://direct/link" in out["answer"]
    assert "OneSearch was slow to respond." in out["answer"]
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_agent_compose.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'agent.compose'`.

- [ ] **Step 3: Write `agent/compose.py`**

```python
import json

from agent.models import AgentState, Selection, SelectionItem
from onesearch.client import OneSearchClient
from onesearch.models import ResourceDetail, SearchCandidate

MAX_SHOWN = 3
DESCRIPTION_CHARS = 300

STATUS_LABELS = {
    "available": "available",
    "unavailable": "currently unavailable",
    "check_holdings": "check holdings",
}

COMPOSE_SYSTEM = """You help a University at Buffalo library patron choose among OneSearch results.

You are given the patron's role, their request, and a numbered list of candidate results as JSON.
Pick the 1-3 results that best answer the request, best first. For each, give the record_id copied EXACTLY from the candidate list and ONE sentence saying why it fits this patron (mention the title's topic, not links or locations).
Then set confidence:
- high: the picks clearly match the request.
- medium: reasonable but not certain.
- low: nothing here really matches; return an empty items list.
Never invent record_ids, links, call numbers, or locations. Those are added separately."""


def _candidate_view(candidate: SearchCandidate) -> dict:
    description = candidate.description or ""
    if len(description) > DESCRIPTION_CHARS:
        description = description[:DESCRIPTION_CHARS].rstrip() + "..."
    return {
        "record_id": candidate.record_id,
        "title": candidate.title,
        "author": candidate.author,
        "date": candidate.date,
        "type": candidate.resource_type,
        "subject": candidate.subject,
        "description": description or None,
        "access": candidate.delivery_category,
    }


def select_results(
    model, role: str, message: str, candidates: list[SearchCandidate], rag_context: list
) -> Selection:
    """LLM call: choose which candidates to show. Raises if the model fails."""
    lines = [f"role: {role}", f"request: {message}"]
    if rag_context:
        lines.append("library guide context: " + json.dumps(rag_context, default=str))
    lines.append("candidates:")
    lines.extend(json.dumps(_candidate_view(c)) for c in candidates)
    structured = model.with_structured_output(Selection)
    selection = structured.invoke([("system", COMPOSE_SYSTEM), ("human", "\n".join(lines))])
    if isinstance(selection, dict):
        selection = Selection.model_validate(selection)
    if not isinstance(selection, Selection):
        raise ValueError("model did not return a Selection")
    return selection


def validate_selection(
    items: list[SelectionItem], candidates: list[SearchCandidate]
) -> list[SelectionItem]:
    """Keep only ids that really are candidates, drop duplicates, cap at MAX_SHOWN.
    This is what stops the LLM from widening fetch_detail() to arbitrary ids."""
    known = {c.record_id for c in candidates}
    kept: list[SelectionItem] = []
    seen: set[str] = set()
    for item in items:
        if item.record_id in known and item.record_id not in seen:
            kept.append(item)
            seen.add(item.record_id)
        if len(kept) == MAX_SHOWN:
            break
    return kept


def _access_lines(candidate: SearchCandidate, detail: ResourceDetail | None) -> list[str]:
    source = detail or candidate
    lines = []
    if source.online_link:
        package = getattr(source, "online_package", None)
        via = f" (via {package})" if package else ""
        lines.append(f"   Online: {source.online_link}{via}")
    if source.call_number or source.library:
        place = " / ".join(p for p in (source.library, source.sub_location) if p)
        call = f"call number {source.call_number}" if source.call_number else ""
        text = "Location: " + " - ".join(p for p in (place, call) if p)
        status = STATUS_LABELS.get(source.availability_status or "", source.availability_status)
        if status:
            text += f" ({status})"
        lines.append(f"   {text}")
    return lines


def render_answer(
    items: list[SelectionItem],
    candidates: list[SearchCandidate],
    details: dict[str, ResourceDetail],
    keywords: str,
    notes: list[str],
    client: OneSearchClient,
) -> str:
    by_id = {c.record_id: c for c in candidates}
    out = [f'Here is what I found for "{keywords}":']
    out.extend(f"Note: {n}" for n in notes)
    out.append("")
    any_online = False
    for number, item in enumerate(items, start=1):
        candidate = by_id[item.record_id]
        byline = ", ".join(p for p in (candidate.author, candidate.date) if p)
        kind = f" [{candidate.resource_type}]" if candidate.resource_type else ""
        out.append(f"{number}. {candidate.title}{' - ' + byline if byline else ''}{kind}")
        if item.reason:
            out.append(f"   Why: {item.reason}")
        access = _access_lines(candidate, details.get(item.record_id))
        any_online = any_online or any("Online:" in line for line in access)
        out.extend(access)
        out.append(f"   Record: {client.record_link(item.record_id)}")
        out.append("")
    if any_online:
        out.append("Some online resources ask you to log in with your UB credentials.")
    out.append(f"More results: {client.fallback_link(keywords)}")
    return "\n".join(out)


def _render_no_answer(headline: str, notes: list[str], more_link: str) -> str:
    out = [headline]
    out.extend(f"Note: {n}" for n in notes)
    out.append(f"You can browse the full OneSearch results here: {more_link}")
    return "\n".join(out)


def compose(state: AgentState, model, client: OneSearchClient) -> dict:
    """Pick 1-3 results (LLM), fetch their access details (bounded, batched),
    and render the answer in code so links and locations are never LLM output."""
    plan = state["plan"]
    result = state["search_result"]
    candidates = result.candidates
    notes = list(state.get("notes", []))
    more_link = result.fallback_link or client.fallback_link(plan.keywords)

    if not candidates:
        answer = _render_no_answer(
            f'I could not find results for "{plan.keywords}".', notes, more_link
        )
        return {"selection": [], "details": {}, "answer": answer}

    try:
        selection = select_results(
            model, state["role"], state["message"], candidates, state.get("rag_context", [])
        )
        items = validate_selection(selection.items, candidates)
        confidence = selection.confidence
    except Exception:
        items = [SelectionItem(record_id=c.record_id, reason="") for c in candidates[:MAX_SHOWN]]
        confidence = "medium"
        notes.append("I could not rank these results, so these are the top matches from OneSearch.")

    if confidence == "low" or not items:
        answer = _render_no_answer(
            "I am not confident I found a good match for that.", notes, more_link
        )
        return {"selection": [], "details": {}, "answer": answer}

    try:
        details = client.fetch_detail([item.record_id for item in items])
    except Exception:
        details = {}
        notes.append("I could not load the full access details for these.")

    answer = render_answer(items, candidates, details, plan.keywords, notes, client)
    return {"selection": items, "details": details, "answer": answer}
```

- [ ] **Step 4: Run it to verify it passes**

```bash
.venv/bin/pytest tests/test_agent_compose.py -v
```Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/compose.py tests/test_agent_compose.py
git commit -m "feat: add compose step with validated selection and deterministic answer rendering"
```

### Task 7: Graph wiring (`agent/graph.py`)

**Files:**
- Modify: `agent/graph.py` (currently an empty stub)
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- Consumes: `plan_query` (Task 4), `run_onesearch` (Task 5), `compose` (Task 6), `AgentState` (Task 2).
- Produces: `build_graph(model, client)` returning a compiled LangGraph whose `.invoke({"role", "message", "history"})` returns the final `AgentState`, including `answer`, `plan`, `notes` and `timings` (keys `plan`, `search`, `rag`, `compose`; a clarification ends the turn after `plan`).


- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_graph.py`:

```python
from unittest.mock import MagicMock

from agent.graph import build_graph
from agent.models import QueryPlan, Selection, SelectionItem
from onesearch.models import ResourceDetail, SearchCandidate, SearchResult
from tests.fakes import FakeChatModel


def _candidate(record_id):
    return SearchCandidate(
        record_id=record_id, title="Community Mental Health Journal", author=None,
        resource_type="journal", format=None, subject=None, description=None, date="2020",
        source="Alma", delivery_category="Alma-E", online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )


def _client():
    client = MagicMock()
    client.search_with_fallback.return_value = SearchResult(
        candidates=[_candidate("j1"), _candidate("j2")], timed_out=False, fallback_link=None
    )
    client.fetch_detail.return_value = {
        "j1": ResourceDetail(
            record_id="j1", online_link="https://search.lib.buffalo.edu/view/j1",
            online_package="SpringerLink", call_number=None, library=None,
            sub_location=None, availability_status=None,
        )
    }
    client.record_link.side_effect = lambda rid: f"https://rec/{rid}"
    client.fallback_link.side_effect = lambda q: f"https://more/{q}"
    return client


def test_full_turn_plans_searches_and_composes():
    model = FakeChatModel(
        QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True),
        Selection(items=[SelectionItem(record_id="j1", reason="Core journal in the field.")],
                  confidence="high"),
    )
    client = _client()
    graph = build_graph(model, client)

    out = graph.invoke({
        "role": "graduate",
        "message": "find me peer reviewed journals on mental health",
        "history": [],
    })

    client.search_with_fallback.assert_called_once_with(
        "mental health", resource_type="journals", peer_reviewed=True
    )
    client.fetch_detail.assert_called_once_with(["j1"])
    assert "https://search.lib.buffalo.edu/view/j1" in out["answer"]
    assert "Core journal in the field." in out["answer"]
    assert out["plan"].resource_type == "journals"
    assert set(out["timings"]) == {"plan", "search", "rag", "compose"}
    assert out["rag_context"] == []


def test_clarifying_question_ends_the_turn_before_searching():
    model = FakeChatModel(QueryPlan(keywords="", needs_clarification="What topic are you researching?"))
    client = _client()
    graph = build_graph(model, client)

    out = graph.invoke({"role": "graduate", "message": "find me something", "history": []})

    assert out["answer"] == "What topic are you researching?"
    client.search_with_fallback.assert_not_called()
    client.fetch_detail.assert_not_called()


def test_planner_failure_still_produces_an_answer():
    model = FakeChatModel(
        RuntimeError("provider down"),
        Selection(items=[SelectionItem(record_id="j2", reason="Relevant.")], confidence="medium"),
    )
    client = _client()
    graph = build_graph(model, client)

    out = graph.invoke({"role": "undergraduate", "message": "mental health help", "history": []})

    client.search_with_fallback.assert_called_once_with(
        "mental health help", resource_type=None, peer_reviewed=False
    )
    assert "Community Mental Health Journal" in out["answer"]
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_agent_graph.py -v
```Expected: FAIL, `ImportError: cannot import name 'build_graph'`.

- [ ] **Step 3: Write `agent/graph.py`**

```python
import time

from langgraph.graph import END, StateGraph

from agent.compose import compose
from agent.models import AgentState
from agent.onesearch_tool import run_onesearch
from agent.planner import plan_query


def _timed(name: str, fn):
    """Wrap a node so its wall time lands in state["timings"] (shown by the CLI's --debug)."""

    def run(state: AgentState) -> dict:
        start = time.monotonic()
        update = dict(fn(state))
        update["timings"] = {
            **state.get("timings", {}),
            name: round(time.monotonic() - start, 3),
        }
        return update

    return run


def build_graph(model, client):
    """plan -> search -> rag (stub) -> compose. A clarifying question from the
    planner ends the turn early with that question as the answer."""

    def plan_node(state: AgentState) -> dict:
        plan = plan_query(model, state["role"], state["message"], state.get("history", []))
        return {"plan": plan, "answer": plan.needs_clarification}

    def search_node(state: AgentState) -> dict:
        outcome = run_onesearch(state["plan"], client)
        return {"search_result": outcome.result, "notes": outcome.notes}

    def rag_node(state: AgentState) -> dict:
        # Slice 1 stub: real LibGuides / librarian-directory retrieval arrives with the data pipeline.
        return {"rag_context": []}

    def compose_node(state: AgentState) -> dict:
        return compose(state, model, client)

    graph = StateGraph(AgentState)
    graph.add_node("plan", _timed("plan", plan_node))
    graph.add_node("search", _timed("search", search_node))
    graph.add_node("rag", _timed("rag", rag_node))
    graph.add_node("compose", _timed("compose", compose_node))
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        lambda state: "clarify" if state.get("answer") else "search",
        {"clarify": END, "search": "search"},
    )
    graph.add_edge("search", "rag")
    graph.add_edge("rag", "compose")
    graph.add_edge("compose", END)
    return graph.compile()
```

- [ ] **Step 4: Run it to verify it passes**

```bash
.venv/bin/pytest -q
```Expected: whole suite passes.

- [ ] **Step 5: Commit**

```bash
git add agent/graph.py tests/test_agent_graph.py
git commit -m "feat: wire plan, search, rag stub and compose into a LangGraph graph"
```

### Task 8: CLI, live check and docs

**Files:**
- Create: `agent/cli.py`, `agent/live_check.py`
- Test: `tests/test_agent_cli.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `build_graph` (Task 7), `get_chat_model`/`LLMConfigError` (Task 3), `ROLES` (Task 2), Layer 1 `OneSearchClient`, `SessionManager`, `QueryCache`.
- Produces: `choose_role(read, write) -> str`, `run_chat(graph, read=input, write=print, debug=False) -> None`, `build_default_graph(provider=None, model_name=None)`, `main(argv=None) -> int`; CLI `python -m agent.cli [--provider P] [--model M] [--debug]`; manual check `python -m agent.live_check`.


- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_cli.py`:

```python
from agent.cli import HISTORY_TURNS, choose_role, main, run_chat
from agent.models import QueryPlan


class FakeGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state):
        self.calls.append({**state, "history": list(state["history"])})
        return {
            "answer": f"answer to {state['message']}",
            "plan": QueryPlan(keywords=state["message"]),
            "timings": {"plan": 0.1},
            "notes": ["a note"],
        }


def _io(*inputs):
    feed = iter(inputs)
    written = []
    return (lambda prompt="": next(feed)), written.append, written


def test_choose_role_accepts_name_or_number_and_reprompts():
    read, write, written = _io("dean", "2")
    assert choose_role(read, write) == "graduate"
    assert any("Please enter one of" in line for line in written)


def test_chat_passes_role_message_and_history_to_the_graph():
    read, write, written = _io("faculty", "first question", "second question", "/quit")
    graph = FakeGraph()

    run_chat(graph, read=read, write=write)

    assert graph.calls[0] == {"role": "faculty", "message": "first question", "history": []}
    assert graph.calls[1]["history"] == [
        {"user": "first question", "assistant": "answer to first question"}
    ]
    assert "answer to second question" in written


def test_history_is_capped():
    inputs = ["undergraduate"] + [f"q{i}" for i in range(6)] + ["/quit"]
    read, write, _ = _io(*inputs)
    graph = FakeGraph()

    run_chat(graph, read=read, write=write)

    assert len(graph.calls[-1]["history"]) == HISTORY_TURNS
    assert graph.calls[-1]["history"][-1]["user"] == "q4"


def test_role_command_changes_role_mid_conversation():
    read, write, _ = _io("undergraduate", "q1", "/role", "faculty", "q2", "/quit")
    graph = FakeGraph()

    run_chat(graph, read=read, write=write)

    assert [c["role"] for c in graph.calls] == ["undergraduate", "faculty"]


def test_debug_prints_plan_timings_and_notes():
    read, write, written = _io("graduate", "q1", "/quit")

    run_chat(FakeGraph(), read=read, write=write, debug=True)

    assert any(line.startswith("[plan]") for line in written)
    assert any(line.startswith("[timings]") for line in written)
    assert "[note] a note" in written


def test_eof_ends_the_chat_cleanly():
    feed = iter(["graduate"])

    def read(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    run_chat(FakeGraph(), read=read, write=lambda _: None)


def test_main_reports_missing_key_and_exits_nonzero(monkeypatch, capsys):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    assert main([]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_agent_cli.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'agent.cli'`.

- [ ] **Step 3: Write `agent/cli.py`**

```python
import argparse
import sys

from agent.llm import LLMConfigError, get_chat_model
from agent.models import ROLES

HISTORY_TURNS = 3
HELP = "Type a question. /role changes your role, /quit exits."


def choose_role(read, write) -> str:
    """Ask for a role until the answer is a role name or its number."""
    write("Who are you? " + ", ".join(f"{i}) {r}" for i, r in enumerate(ROLES, start=1)))
    while True:
        answer = read("role> ").strip().lower()
        if answer in ROLES:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(ROLES):
            return ROLES[int(answer) - 1]
        write(f"Please enter one of: {', '.join(ROLES)} (or 1-{len(ROLES)}).")


def run_chat(graph, read=input, write=print, debug: bool = False) -> None:
    role = choose_role(read, write)
    write(f"Role set to {role}. {HELP}")
    history: list[dict] = []
    while True:
        try:
            message = read("> ").strip()
        except EOFError:
            return
        if not message:
            continue
        if message == "/quit":
            return
        if message == "/role":
            role = choose_role(read, write)
            write(f"Role set to {role}.")
            continue
        state = graph.invoke({"role": role, "message": message, "history": history})
        answer = state["answer"]
        if debug:
            write(f"[plan] {state['plan'].model_dump()}")
            write(f"[timings] {state.get('timings', {})}")
            for note in state.get("notes", []):
                write(f"[note] {note}")
        write(answer)
        history.append({"user": message, "assistant": answer})
        history = history[-HISTORY_TURNS:]


def build_default_graph(provider: str | None = None, model_name: str | None = None):
    """Real model + real Layer 1 client. Raises LLMConfigError if the model can't be built."""
    from dotenv import load_dotenv

    load_dotenv()
    model = get_chat_model(provider=provider, model=model_name)

    from agent.graph import build_graph
    from onesearch.cache import QueryCache
    from onesearch.client import OneSearchClient
    from onesearch.session import SessionManager

    client = OneSearchClient(session_manager=SessionManager(), cache=QueryCache(ttl_seconds=300))
    return build_graph(model, client)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UB Library Navigator (Layer 2, slice 1)")
    parser.add_argument("--provider", help="openai | anthropic | local (overrides LLM_PROVIDER)")
    parser.add_argument("--model", help="model name (overrides LLM_MODEL)")
    parser.add_argument("--debug", action="store_true", help="show plan and per-step timings")
    args = parser.parse_args(argv)

    try:
        graph = build_default_graph(provider=args.provider, model_name=args.model)
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    run_chat(graph, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run it to verify it passes**

```bash
.venv/bin/pytest -q
```Expected: whole suite passes (about 84 tests).

- [ ] **Step 5: Write `agent/live_check.py`**

```python
"""Manual end-to-end check against real UB OneSearch and a real model. Run locally, not in CI.

Usage: python -m agent.live_check [--provider P] [--model M] ["<question>" ...]
"""
import argparse
import sys
import time

from agent.cli import build_default_graph
from agent.llm import LLMConfigError

DEFAULT_QUESTIONS = [
    "find me peer reviewed journals on mental health",
    "books about the history of Buffalo",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--role", default="graduate")
    parser.add_argument("questions", nargs="*", default=DEFAULT_QUESTIONS)
    args = parser.parse_args(argv)

    try:
        graph = build_default_graph(provider=args.provider, model_name=args.model)
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2

    for question in args.questions:
        print("=" * 78)
        print(f"ROLE: {args.role}\nQUESTION: {question}\n")
        start = time.monotonic()
        state = graph.invoke({"role": args.role, "message": question, "history": []})
        total = time.monotonic() - start
        print(f"PLAN: {state['plan'].model_dump()}")
        print(f"NOTES: {state.get('notes', [])}")
        print(f"TIMINGS: {state.get('timings', {})} (total {total:.2f}s)\n")
        print(state["answer"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the live check (manual; needs a model and network)**

```bash
# with OpenAI:
export OPENAI_API_KEY=sk-...
.venv/bin/python -m agent.live_check 2>&1 | tee docs/layer2_live_check_output.txt

# or with a local model (LM Studio example):
LLM_PROVIDER=local LLM_BASE_URL=http://localhost:1234/v1 LLM_MODEL=<loaded-model> \
  .venv/bin/python -m agent.live_check 2>&1 | tee docs/layer2_live_check_output.txt
```
Expected for "find me peer reviewed journals on mental health": `PLAN` shows `keywords` about mental health, `resource_type: journals`, `peer_reviewed: True`; the answer lists 1-3 real journals with `Online:` resolver links and a `Record:` link each; `TIMINGS` total is within a few seconds warm (cold adds ~3s for the session mint). For "books about the history of Buffalo" expect `resource_type: books` and call numbers/locations. If the plan looks wrong with a small local model, that is expected risk: confirm the answer still comes back (fallback path) and note the model in the output file.

- [ ] **Step 7: Update docs**

In `CLAUDE.md`, finding 3, add a sentence after the "Target production design" paragraph:

> **Mint by waiting for the page's own `/pnxs` response** (`page.expect_response`), not `networkidle`: the site's analytics keep the network busy, so `networkidle` timed out at 30s (2/2 runs, 2026-09-24) while the response wait mints in 2.4-4.2s (4/4). `search_with_fallback` mints/reuses the session *before* starting its timeout clock and returns at the timeout instead of waiting for the slow worker.

In `README.md` replace the body with:

```markdown
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

## Tests

```bash
.venv/bin/pytest
```

```

- [ ] **Step 8: Commit**

```bash
git add agent/cli.py agent/live_check.py tests/test_agent_cli.py CLAUDE.md README.md docs/layer2_live_check_output.txt
git commit -m "feat: add chat CLI and live check for the agent core"
```
(If the live check could not be run yet, omit `docs/layer2_live_check_output.txt` from `git add` and say so in the handoff.)


---

## Slice acceptance (run after Task 8)

- [ ] `.venv/bin/pytest -q` passes with no network and no API key.
- [ ] `python -m agent.live_check` returns a sensible plan and answer for both default questions against real UB and a real model; output saved in `docs/`.
- [ ] `python -m agent.cli --provider local ...` (or `anthropic`) starts, or exits 2 with a message naming the missing key/package.
- [ ] `git log --format=%B` on this slice's commits contains no `Co-Authored-By` or "Generated with" lines.
- [ ] `fetch_detail` is only ever called from `agent/compose.py` with at most 3 validated ids (`grep -rn "fetch_detail" agent/`).
- [ ] Review, then `git push` once for the whole slice.
