# UB Library Navigator — Layer 1 + Layer 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the OneSearch retrieval adapter (Layer 1) exactly to the current `CLAUDE.md` spec, then build the full agentic system around it (Layer 2), preserving every capability from the pre-narrowing project brief (captured at commit `dd388a3`) but reworked to consume Layer 1's clean candidate interface instead of the old monolithic `onesearch_tool.py`.

**Architecture:** Two layers with a strict one-way dependency. Layer 1 (`onesearch/`) is a standalone package that knows nothing about LangGraph, roles, or RAG — it only turns a query into `SearchCandidate` objects and, for a bounded few, `ResourceDetail` objects. Layer 2 (`agent/`, `scraper/`, `pipeline/`, `analytics/`, `streamlit_app/`, `evaluation/`) is everything that used to be the full capstone scope: role capture, LangGraph orchestration, RAG enrichment over LibGuides/librarian directory, citation formatting, escalation, anonymized logging, staff dashboard, evaluation. Layer 2's `agent/onesearch_tool.py` is now a thin adapter over `onesearch.client.OneSearchClient`, not the retrieval logic itself. Orchestration inside Layer 2 is hybrid, per `CLAUDE.md`'s "Layer 2: Agent Orchestration" section: two cheap deterministic fast-paths (explicit human request, citation follow-up on an already-known candidate) run before anything expensive; everything else goes through a default path where OneSearch search+refine and RAG enrichment feed a single LLM Rank+Compose node, which scores confidence, selects the top 1-3 candidates, triggers Layer 1's bounded `fetch_detail()` for only those, and escalates instead of composing an answer when confidence is low.

**Tech Stack:** Playwright + requests (Layer 1 session/client), Streamlit, LangChain + LangGraph, Chroma, SQLAlchemy, OpenAI embeddings, RAGAS + LangSmith, GitHub Actions.

**Spec:**
- `CLAUDE.md` (current) — Layer 1 (OneSearch retrieval) detailed spec, confirmed technical findings, field maps, acceptance criteria.
- Prior `CLAUDE.md` at commit `dd388a3` — full 7-capability project brief; source of truth for every Layer 2 capability, repo structure, timeline, and open items pending UB Libraries confirmation. Retrieve with `git show dd388a3:CLAUDE.md`.

## Global Constraints

- Never import `requests` or `playwright` against `search.lib.buffalo.edu` from outside `onesearch/`. All access goes through `onesearch.client.OneSearchClient`. (CLAUDE.md, "adapter module")
- Real, identifying `User-Agent` on every OneSearch request: contact `kavyansht21@gmail.com`. Never spoof a browser UA at the HTTP layer. (CLAUDE.md)
- Build query strings manually with `urllib.parse.urlencode(params, safe=",", quote_via=urllib.parse.quote)` — never `requests`' `params=` dict, which breaks comma encoding. (CLAUDE.md, finding 1)
- Session mint via Playwright at most once per ~15 minutes; never launch a browser per user search. (CLAUDE.md, finding 3)
- `fetch_detail()` is only ever called for the 1-3 candidates Layer 2 chooses to show — never looped over a full result set. This is enforced by the function signature (explicit list of record IDs), not just convention.
- Hard timeout of 8-10s on the whole Layer 1 retrieval step, with graceful fallback (partial candidates + direct OneSearch link) rather than hanging. (CLAUDE.md, "Latency budget")
- No name/identifier is ever attached to a conversation log entry — role, course (if given), and topic category only. (prior CLAUDE.md, capability 7)
- Do not hardcode assumptions about the live chat platform or staff-dashboard auth method — both are unconfirmed. Build behind swappable interfaces. (prior CLAUDE.md, "Open items")
- Commit message format: `type: short description`. Never add a Co-Authored-By or "Generated with Claude Code" line (`.claude/settings.json` + `CLAUDE.md` commit conventions at `dd388a3`).
- All commits authored under the repo owner's own git identity.

---

## Phase 1 — Layer 1: OneSearch Retrieval Adapter

### Task 1: Core data structures — `onesearch/models.py` + `onesearch/cache.py`

**Files:**
- Create: `onesearch/__init__.py`
- Create: `onesearch/models.py`
- Create: `onesearch/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `SearchCandidate` dataclass (`record_id: str`, `title: str`, `author: str | None`, `resource_type: str | None`, `format: str | None`, `subject: str | None`, `description: str | None`, `date: str | None`, `source: str | None`, `delivery_category: str | None`, `online_link: str | None`, `call_number: str | None`, `library: str | None`, `sub_location: str | None`, `availability_status: str | None`, `copies_available: str | None`).
- Produces: `ResourceDetail` dataclass (`record_id: str`, `online_link: str | None`, `call_number: str | None`, `library: str | None`, `sub_location: str | None`, `availability_status: str | None`, `copies_available: str | None`, `raw: dict`).
- Produces: `QueryCache` class with `get(query: str) -> list[SearchCandidate] | None` and `set(query: str, results: list[SearchCandidate]) -> None`, TTL in seconds passed to `__init__`.

- [ ] **Step 1: Write the failing test for the cache**

```python
# tests/test_cache.py
import time
from onesearch.cache import QueryCache
from onesearch.models import SearchCandidate

def _candidate(record_id="1"):
    return SearchCandidate(
        record_id=record_id, title="t", author=None, resource_type=None,
        format=None, subject=None, description=None, date=None, source=None,
        delivery_category=None, online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
        copies_available=None,
    )

def test_cache_hit_within_ttl():
    cache = QueryCache(ttl_seconds=60)
    cache.set("machine learning", [_candidate()])
    assert cache.get("machine learning") == [_candidate()]

def test_cache_normalizes_query_text():
    cache = QueryCache(ttl_seconds=60)
    cache.set("  Machine   Learning  ", [_candidate()])
    assert cache.get("machine learning") == [_candidate()]

def test_cache_miss_after_ttl_expires():
    cache = QueryCache(ttl_seconds=0.05)
    cache.set("machine learning", [_candidate()])
    time.sleep(0.1)
    assert cache.get("machine learning") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onesearch'`

- [ ] **Step 3: Write `onesearch/models.py`**

```python
# onesearch/models.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchCandidate:
    record_id: str
    title: str
    author: str | None
    resource_type: str | None
    format: str | None
    subject: str | None
    description: str | None
    date: str | None
    source: str | None
    delivery_category: str | None
    online_link: str | None
    call_number: str | None
    library: str | None
    sub_location: str | None
    availability_status: str | None
    copies_available: str | None


@dataclass(frozen=True)
class ResourceDetail:
    record_id: str
    online_link: str | None
    call_number: str | None
    library: str | None
    sub_location: str | None
    availability_status: str | None
    copies_available: str | None
    raw: dict
```

- [ ] **Step 4: Write `onesearch/cache.py`**

```python
# onesearch/cache.py
import re
import time

from onesearch.models import SearchCandidate


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class QueryCache:
    def __init__(self, ttl_seconds: float):
        self._ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, list[SearchCandidate]]] = {}

    def get(self, query: str) -> list[SearchCandidate] | None:
        key = _normalize(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, results = entry
        if time.monotonic() - stored_at > self._ttl_seconds:
            del self._store[key]
            return None
        return results

    def set(self, query: str, results: list[SearchCandidate]) -> None:
        self._store[_normalize(query)] = (time.monotonic(), results)
```

- [ ] **Step 5: Create `onesearch/__init__.py`**

```python
# onesearch/__init__.py
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_cache.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add onesearch/__init__.py onesearch/models.py onesearch/cache.py tests/test_cache.py
git commit -m "feat: add onesearch models and query cache"
```

---

### Task 2: Response parser — `onesearch/parser.py`

**Files:**
- Create: `onesearch/parser.py`
- Create: `tests/fixtures/onesearch_brief_response.json`
- Create: `tests/fixtures/onesearch_detail_response.json`
- Test: `tests/test_parser.py`

**Interfaces:**
- Consumes: `SearchCandidate`, `ResourceDetail` from `onesearch/models.py` (Task 1).
- Produces: `parse_brief_doc(doc: dict) -> SearchCandidate`, `parse_brief_response(response_json: dict) -> list[SearchCandidate]`, `parse_detail_response(record_id: str, response_json: dict) -> ResourceDetail`.

**Note:** The reference probe scripts CLAUDE.md points to (`onesearch_probe.py`, `onesearch_probe_playwright.py`) are not present in this repo (confirmed via `find`, absent from git history). This task builds the parser directly from CLAUDE.md's confirmed field-map tables instead of factoring those scripts. Fixtures below are synthesized to match the documented field paths and quirks; **replace them with a real captured response before Phase 1 ships** (CLAUDE.md: "save at least one real captured response... once confirmed").

- [ ] **Step 1: Write the fixture files**

```json
// tests/fixtures/onesearch_brief_response.json
{
  "docs": [
    {
      "pnx": {
        "display": {
          "title": ["Machine Learning for Coastal Systems"],
          "creator": ["Wong, Kau-Fui Vincent, author.$$QWong, Kau-Fui Vincent"],
          "type": ["book"],
          "format": ["Book, Print"],
          "subject": ["Machine learning"],
          "creationdate": ["2021"],
          "source": ["Alma"]
        },
        "control": { "recordid": ["alma991234567890001"] }
      },
      "delivery": {
        "deliveryCategory": ["Alma-P"],
        "bestlocation": {
          "callNumber": "QA76.87 .W66 2021",
          "mainLocation": "Silverman Library",
          "subLocation": "Stacks",
          "itemAvailabilityStatus": "available",
          "number": "1"
        }
      }
    },
    {
      "pnx": {
        "display": {
          "title": ["Introduction to Statistical Learning"],
          "contributor": ["UB Data Science Working Group"],
          "type": ["article"],
          "format": ["Article"],
          "subject": ["Statistics"],
          "creationdate": [],
          "source": ["Alma"]
        },
        "addata": { "date": ["2019"] },
        "control": { "recordid": ["alma991234567890002"] }
      },
      "delivery": {
        "deliveryCategory": ["Alma-E"],
        "link": ["https://search.lib.buffalo.edu/discovery/openurl?id=alma991234567890002"]
      }
    }
  ]
}
```

```json
// tests/fixtures/onesearch_detail_response.json
{
  "pnx": {
    "display": {
      "title": ["Machine Learning for Coastal Systems"]
    },
    "control": { "recordid": ["alma991234567890001"] }
  },
  "delivery": {
    "deliveryCategory": ["Alma-P"],
    "bestlocation": {
      "callNumber": "QA76.87 .W66 2021",
      "mainLocation": "Silverman Library",
      "subLocation": "Stacks",
      "itemAvailabilityStatus": "available",
      "number": "1"
    }
  }
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_parser.py
import json
from pathlib import Path

from onesearch.parser import parse_brief_response, parse_detail_response

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def test_parse_brief_response_extracts_recommendation_fields():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert len(candidates) == 2
    first = candidates[0]
    assert first.record_id == "alma991234567890001"
    assert first.title == "Machine Learning for Coastal Systems"
    assert first.call_number == "QA76.87 .W66 2021"
    assert first.library == "Silverman Library"


def test_parse_brief_response_splits_creator_on_internal_separator():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[0].author == "Wong, Kau-Fui Vincent, author."


def test_parse_brief_response_falls_back_to_contributor():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[1].author == "UB Data Science Working Group"


def test_parse_brief_response_falls_back_creationdate_to_addata_date():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[1].date == "2019"


def test_parse_brief_response_online_link_present():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[1].online_link == (
        "https://search.lib.buffalo.edu/discovery/openurl?id=alma991234567890002"
    )


def test_parse_detail_response():
    detail = parse_detail_response(
        "alma991234567890001", _load("onesearch_detail_response.json")
    )
    assert detail.call_number == "QA76.87 .W66 2021"
    assert detail.library == "Silverman Library"
    assert detail.record_id == "alma991234567890001"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onesearch.parser'`

- [ ] **Step 4: Write `onesearch/parser.py`**

```python
# onesearch/parser.py
from onesearch.models import ResourceDetail, SearchCandidate


def _first(values: list | None) -> str | None:
    if not values:
        return None
    return values[0]


def _clean_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    return raw.split("$$Q")[0]


def parse_brief_doc(doc: dict) -> SearchCandidate:
    pnx = doc.get("pnx", {})
    display = pnx.get("display", {})
    addata = pnx.get("addata", {})
    control = pnx.get("control", {})
    delivery = doc.get("delivery", {})
    bestlocation = delivery.get("bestlocation", {})

    author = _clean_name(_first(display.get("creator")))
    if author is None:
        author = _clean_name(_first(display.get("contributor")))

    date = _first(display.get("creationdate")) or _first(addata.get("date"))

    return SearchCandidate(
        record_id=_first(control.get("recordid")),
        title=_first(display.get("title")),
        author=author,
        resource_type=_first(display.get("type")),
        format=_first(display.get("format")),
        subject=_first(display.get("subject")),
        description=_first(display.get("description")),
        date=date,
        source=_first(display.get("source")),
        delivery_category=_first(delivery.get("deliveryCategory")),
        online_link=_first(delivery.get("link")),
        call_number=bestlocation.get("callNumber"),
        library=bestlocation.get("mainLocation"),
        sub_location=bestlocation.get("subLocation"),
        availability_status=bestlocation.get("itemAvailabilityStatus"),
        copies_available=bestlocation.get("number"),
    )


def parse_brief_response(response_json: dict) -> list[SearchCandidate]:
    return [parse_brief_doc(doc) for doc in response_json.get("docs", [])]


def parse_detail_response(record_id: str, response_json: dict) -> ResourceDetail:
    delivery = response_json.get("delivery", {})
    bestlocation = delivery.get("bestlocation", {})
    return ResourceDetail(
        record_id=record_id,
        online_link=_first(delivery.get("link")),
        call_number=bestlocation.get("callNumber"),
        library=bestlocation.get("mainLocation"),
        sub_location=bestlocation.get("subLocation"),
        availability_status=bestlocation.get("itemAvailabilityStatus"),
        copies_available=bestlocation.get("number"),
        raw=response_json,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_parser.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add onesearch/parser.py tests/fixtures tests/test_parser.py
git commit -m "feat: add onesearch response parser with synthesized fixtures"
```

---

### Task 3: Session manager — `onesearch/session.py`

**Files:**
- Create: `onesearch/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: `SessionCookies` dataclass (`jsessionid: str`, `urm_st: str`, `urm_se: str`, `institute: str`).
- Produces: `SessionManager` class with `get_cookies() -> SessionCookies` (mints via Playwright on first call or when expired, reuses otherwise) and `_is_expired() -> bool`.
- Consumes (injected, for testability): a `mint_fn: Callable[[], SessionCookies]` — the actual Playwright driving logic lives behind this callable so the manager's reuse/expiry logic can be tested without launching a browser.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session.py
from onesearch.session import SessionCookies, SessionManager

COOKIES_A = SessionCookies(jsessionid="A", urm_st="1000", urm_se="901000", institute="UB")
COOKIES_B = SessionCookies(jsessionid="B", urm_st="2000", urm_se="902000", institute="UB")


def test_mints_session_on_first_call():
    calls = []
    manager = SessionManager(mint_fn=lambda: calls.append(1) or COOKIES_A)
    cookies = manager.get_cookies()
    assert cookies == COOKIES_A
    assert len(calls) == 1


def test_reuses_session_within_window():
    calls = [COOKIES_A, COOKIES_B]
    manager = SessionManager(mint_fn=lambda: calls.pop(0))
    first = manager.get_cookies()
    second = manager.get_cookies()
    assert first == second == COOKIES_A


def test_remints_after_window_expires():
    calls = [COOKIES_A, COOKIES_B]
    manager = SessionManager(mint_fn=lambda: calls.pop(0), session_window_seconds=0)
    first = manager.get_cookies()
    second = manager.get_cookies()
    assert first == COOKIES_A
    assert second == COOKIES_B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onesearch.session'`

- [ ] **Step 3: Write `onesearch/session.py`**

```python
# onesearch/session.py
import time
from collections.abc import Callable
from dataclasses import dataclass

SESSION_WINDOW_SECONDS = 15 * 60  # matches confirmed urm_st/urm_se gap: 900,000 ms
ONESEARCH_USER_AGENT = (
    "UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)"
)


@dataclass(frozen=True)
class SessionCookies:
    jsessionid: str
    urm_st: str
    urm_se: str
    institute: str


def mint_session_via_playwright() -> SessionCookies:
    """Loads the real OneSearch search page in headless Chromium so its own
    JS fires the request, satisfying bot-mitigation checks that a bare
    `requests` call fails (see CLAUDE.md finding 2). Route-intercepts to
    force skipDelivery=N and pcAvailability=true (finding 4), then extracts
    the session cookies the response sets."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(user_agent=ONESEARCH_USER_AGENT)

        def _force_delivery_params(route):
            request = route.request
            url = request.url.replace("skipDelivery=Y", "skipDelivery=N")
            if "pcAvailability=" not in url:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}pcAvailability=true"
            route.continue_(url=url)

        context.route("**/primaws/rest/pub/pnxs**", _force_delivery_params)
        page = context.new_page()
        page.goto("https://search.lib.buffalo.edu/discovery/search?query=any,contains,test&vid=01SUNY_BUF:UB")
        page.wait_for_load_state("networkidle")

        cookies = {c["name"]: c["value"] for c in context.cookies()}
        browser.close()

    return SessionCookies(
        jsessionid=cookies["JSESSIONID"],
        urm_st=cookies["urm_st"],
        urm_se=cookies["urm_se"],
        institute=cookies["institute"],
    )


class SessionManager:
    def __init__(
        self,
        mint_fn: Callable[[], SessionCookies] = mint_session_via_playwright,
        session_window_seconds: float = SESSION_WINDOW_SECONDS,
    ):
        self._mint_fn = mint_fn
        self._session_window_seconds = session_window_seconds
        self._cookies: SessionCookies | None = None
        self._minted_at: float | None = None

    def _is_expired(self) -> bool:
        if self._cookies is None or self._minted_at is None:
            return True
        return (time.monotonic() - self._minted_at) > self._session_window_seconds

    def get_cookies(self) -> SessionCookies:
        if self._is_expired():
            self._cookies = self._mint_fn()
            self._minted_at = time.monotonic()
        return self._cookies
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add onesearch/session.py tests/test_session.py
git commit -m "feat: add onesearch session manager with 15-min cookie reuse"
```

---

### Task 4: Search client — `onesearch/client.py` (search)

**Files:**
- Create: `onesearch/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `SessionManager.get_cookies()` (Task 3), `QueryCache` (Task 1), `parse_brief_response()` (Task 2).
- Produces: `OneSearchClient` class with `search(query: str) -> list[SearchCandidate]`. Constructor takes `session_manager: SessionManager`, `cache: QueryCache`, `http_get: Callable[..., Response] = requests.get` (injected for testability), `vid: str = "01SUNY_BUF:UB"`.
- No live network calls in this task's tests — `http_get` is mocked.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client.py
import json
from pathlib import Path
from unittest.mock import MagicMock

from onesearch.cache import QueryCache
from onesearch.client import OneSearchClient
from onesearch.session import SessionCookies, SessionManager

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_session_manager():
    cookies = SessionCookies(jsessionid="J", urm_st="1", urm_se="900001", institute="UB")
    return SessionManager(mint_fn=lambda: cookies)


def test_search_returns_parsed_candidates():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_brief_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    candidates = client.search("machine learning")

    assert len(candidates) == 2
    assert candidates[0].title == "Machine Learning for Coastal Systems"


def test_search_builds_query_string_with_literal_commas():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")

    called_url = http_get.call_args.args[0]
    assert "any,contains,machine learning" in called_url
    assert "%2C" not in called_url


def test_search_forces_skip_delivery_and_pc_availability():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")

    called_url = http_get.call_args.args[0]
    assert "skipDelivery=N" in called_url
    assert "pcAvailability=true" in called_url


def test_search_uses_identifying_user_agent():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")

    headers = http_get.call_args.kwargs["headers"]
    assert "kavyansht21@gmail.com" in headers["User-Agent"]


def test_search_hits_cache_on_second_call():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_brief_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")
    client.search("machine learning")

    assert http_get.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onesearch.client'`

- [ ] **Step 3: Write `onesearch/client.py` (search only — fetch_detail added in Task 5)**

```python
# onesearch/client.py
import urllib.parse

import requests

from onesearch.cache import QueryCache
from onesearch.models import SearchCandidate
from onesearch.parser import parse_brief_response
from onesearch.session import ONESEARCH_USER_AGENT, SessionManager

BASE_URL = "https://search.lib.buffalo.edu/primaws/rest/pub/pnxs"


def _build_search_url(query: str, vid: str) -> str:
    params = {
        "q": f"any,contains,{query}",
        "vid": vid,
        "skipDelivery": "N",
        "pcAvailability": "true",
        "scope": "MyInst_and_CI",
        "tab": "Everything",
        "offset": "0",
        "limit": "10",
    }
    # requests' params= dict percent-encodes commas (%2C), which the
    # backend rejects. Build the query string manually and keep commas
    # literal (CLAUDE.md, confirmed finding 1).
    query_string = urllib.parse.urlencode(
        params, safe=",", quote_via=urllib.parse.quote
    )
    return f"{BASE_URL}?{query_string}"


class OneSearchClient:
    def __init__(
        self,
        session_manager: SessionManager,
        cache: QueryCache,
        http_get=requests.get,
        vid: str = "01SUNY_BUF:UB",
    ):
        self._session_manager = session_manager
        self._cache = cache
        self._http_get = http_get
        self._vid = vid

    def search(self, query: str) -> list[SearchCandidate]:
        cached = self._cache.get(query)
        if cached is not None:
            return cached

        cookies = self._session_manager.get_cookies()
        url = _build_search_url(query, self._vid)
        response = self._http_get(
            url,
            headers={"User-Agent": ONESEARCH_USER_AGENT},
            cookies={
                "JSESSIONID": cookies.jsessionid,
                "urm_st": cookies.urm_st,
                "urm_se": cookies.urm_se,
                "institute": cookies.institute,
            },
            timeout=10,
        )
        response.raise_for_status()
        candidates = parse_brief_response(response.json())
        self._cache.set(query, candidates)
        return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_client.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add onesearch/client.py tests/test_client.py
git commit -m "feat: add onesearch search client with manual query encoding and caching"
```

---

### Task 5: Detail fetch + timeout/fallback — `onesearch/client.py` (fetch_detail, search_with_fallback)

**Files:**
- Modify: `onesearch/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `ResourceDetail`, `parse_detail_response()` (Task 2); `OneSearchClient` (Task 4).
- Produces: `OneSearchClient.fetch_detail(record_ids: list[str]) -> dict[str, ResourceDetail]` (bounded to the explicit list passed in — the caller, Layer 2, decides which 1-3 ids to fetch), and `OneSearchClient.search_with_fallback(query: str, timeout_seconds: float = 9.0) -> SearchResult` where `SearchResult` is a new dataclass in `onesearch/models.py`: `candidates: list[SearchCandidate]`, `timed_out: bool`, `fallback_link: str | None`.

- [ ] **Step 1: Add `SearchResult` to `onesearch/models.py`**

```python
# append to onesearch/models.py
@dataclass(frozen=True)
class SearchResult:
    candidates: list[SearchCandidate]
    timed_out: bool
    fallback_link: str | None
```

- [ ] **Step 2: Write the failing tests**

```python
# append to tests/test_client.py
import time

from onesearch.models import SearchResult


def test_fetch_detail_only_calls_for_given_record_ids():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_detail_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    details = client.fetch_detail(["alma991234567890001"])

    assert http_get.call_count == 1
    assert set(details.keys()) == {"alma991234567890001"}
    assert details["alma991234567890001"].call_number == "QA76.87 .W66 2021"


def test_fetch_detail_bounded_to_multiple_explicit_ids_only():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_detail_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.fetch_detail(["a", "b", "c"])

    assert http_get.call_count == 3


def test_search_with_fallback_returns_candidates_when_fast():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_brief_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    result = client.search_with_fallback("machine learning", timeout_seconds=5)

    assert isinstance(result, SearchResult)
    assert result.timed_out is False
    assert len(result.candidates) == 2
    assert result.fallback_link is None


def test_search_with_fallback_times_out_gracefully():
    def _slow_get(*args, **kwargs):
        time.sleep(0.2)
        raise AssertionError("should not be reached before timeout fires")

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=_slow_get,
    )
    result = client.search_with_fallback("machine learning", timeout_seconds=0.05)

    assert result.timed_out is True
    assert result.candidates == []
    assert "search.lib.buffalo.edu" in result.fallback_link
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_client.py -v`
Expected: FAIL — `fetch_detail`/`search_with_fallback` not defined

- [ ] **Step 4: Extend `onesearch/client.py`**

```python
# add to onesearch/client.py, near the top
import concurrent.futures

from onesearch.models import ResourceDetail, SearchResult
from onesearch.parser import parse_detail_response

DETAIL_URL = "https://search.lib.buffalo.edu/primaws/rest/pub/discovery/fulldisplay"


def _build_detail_url(record_id: str, vid: str) -> str:
    params = {"docid": record_id, "context": "L", "vid": vid}
    query_string = urllib.parse.urlencode(
        params, safe=",", quote_via=urllib.parse.quote
    )
    return f"{DETAIL_URL}?{query_string}"


# add these methods inside class OneSearchClient
    def _fetch_one_detail(self, record_id: str) -> ResourceDetail:
        cookies = self._session_manager.get_cookies()
        url = _build_detail_url(record_id, self._vid)
        response = self._http_get(
            url,
            headers={"User-Agent": ONESEARCH_USER_AGENT},
            cookies={
                "JSESSIONID": cookies.jsessionid,
                "urm_st": cookies.urm_st,
                "urm_se": cookies.urm_se,
                "institute": cookies.institute,
            },
            timeout=10,
        )
        response.raise_for_status()
        return parse_detail_response(record_id, response.json())

    def fetch_detail(self, record_ids: list[str]) -> dict[str, ResourceDetail]:
        # Bounded by definition: callers (Layer 2) pass only the 1-3 ids
        # they decided to show. Never call this with a full result set
        # (CLAUDE.md: "never loop fetch_detail() over every search result").
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(record_ids) or 1) as pool:
            futures = {
                pool.submit(self._fetch_one_detail, rid): rid for rid in record_ids
            }
            return {
                futures[future]: future.result() for future in futures
            }

    def search_with_fallback(
        self, query: str, timeout_seconds: float = 9.0
    ) -> SearchResult:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.search, query)
            try:
                candidates = future.result(timeout=timeout_seconds)
                return SearchResult(candidates=candidates, timed_out=False, fallback_link=None)
            except concurrent.futures.TimeoutError:
                fallback_link = (
                    "https://search.lib.buffalo.edu/discovery/search?"
                    f"query=any,contains,{urllib.parse.quote(query)}&vid={self._vid}"
                )
                return SearchResult(candidates=[], timed_out=True, fallback_link=fallback_link)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_client.py -v`
Expected: PASS (9 tests total in the file)

- [ ] **Step 6: Commit**

```bash
git add onesearch/client.py onesearch/models.py tests/test_client.py
git commit -m "feat: add bounded detail fetch and timeout/fallback to onesearch client"
```

---

### Task 6: Live integration verification (manual, gates Phase 1 acceptance)

**Files:**
- Create: `onesearch/verify_live.py` (throwaway verification script, not imported by production code)

**Interfaces:**
- Consumes: `OneSearchClient`, `SessionManager` (real `mint_session_via_playwright`), `QueryCache`.
- Produces: printed confirmation of each unmet acceptance-criteria checkbox in `CLAUDE.md`, plus the actual latency numbers CLAUDE.md asks to be documented.

This task cannot be automated in CI (it hits the live, undocumented UB endpoint) — CLAUDE.md's own test design explicitly keeps `tests/test_client.py` mocked and only wants live verification run manually. Do not add this to CI or the weekly refresh workflow.

- [ ] **Step 1: Write the verification script**

```python
# onesearch/verify_live.py
"""Manual verification against real UB data. Run locally, not in CI.

Usage: python -m onesearch.verify_live "<query>" "<physical-item-query>"
"""
import sys
import time

from onesearch.cache import QueryCache
from onesearch.client import OneSearchClient
from onesearch.session import SessionManager


def main():
    online_query = sys.argv[1] if len(sys.argv) > 1 else "machine learning"
    physical_query = sys.argv[2] if len(sys.argv) > 2 else "buffalo history"

    client = OneSearchClient(session_manager=SessionManager(), cache=QueryCache(ttl_seconds=300))

    mint_start = time.monotonic()
    client._session_manager.get_cookies()
    print(f"session mint (cold): {time.monotonic() - mint_start:.2f}s")

    search_start = time.monotonic()
    candidates = client.search(online_query)
    print(f"search call (warm session): {time.monotonic() - search_start:.2f}s, {len(candidates)} candidates")

    online_candidate = next((c for c in candidates if c.delivery_category == "Alma-E"), None)
    physical_candidates = client.search(physical_query)
    physical_candidate = next((c for c in physical_candidates if c.delivery_category == "Alma-P"), None)

    to_fetch = [c.record_id for c in (online_candidate, physical_candidate) if c]
    detail_start = time.monotonic()
    details = client.fetch_detail(to_fetch)
    print(f"detail fetch ({len(to_fetch)} items, concurrent): {time.monotonic() - detail_start:.2f}s")

    if online_candidate:
        print("ONLINE case:", details[online_candidate.record_id].online_link)
    else:
        print("ONLINE case: no Alma-E candidate found for this query, try a different one")

    if physical_candidate:
        d = details[physical_candidate.record_id]
        print("PHYSICAL case:", d.call_number, d.library, d.sub_location)
    else:
        print("PHYSICAL case: no Alma-P candidate found for this query, try a different one")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against live UB data**

Run: `python -m onesearch.verify_live "machine learning" "buffalo history"`

Expected: real HTTP 200s, a populated online link for the online case, and a real call number + library for the physical case. If the physical case prints "no Alma-P candidate found," retry with a query more likely to surface a print item until it's confirmed at least once (CLAUDE.md acceptance criteria: "physical case still needs a real test query").

- [ ] **Step 3: Record the actual numbers**

Update the "Latency budget" table in `CLAUDE.md` with the real numbers this script printed, replacing the "target" values, and check off every satisfied box under "Acceptance criteria for this phase" in `CLAUDE.md`.

- [ ] **Step 4: Commit**

```bash
git add onesearch/verify_live.py CLAUDE.md
git commit -m "docs: confirm onesearch layer 1 acceptance criteria with real latency numbers"
```

---

## Phase 2 — Layer 2: Agent Orchestration (consumes Layer 1)

### Task 7: OneSearch tool adapter — `agent/onesearch_tool.py`

**Files:**
- Modify: `agent/onesearch_tool.py` (currently empty stub)
- Test: `tests/test_onesearch_tool.py`

**Interfaces:**
- Consumes: `OneSearchClient.search_with_fallback()` (Task 5).
- Produces: `run_onesearch(query: str, client: OneSearchClient, max_refinements: int = 1) -> SearchResult` — this is where the old capability's "if the first pass looks weak, refine once and search again, two searches total" rule lives, since that's a Layer 2 orchestration decision, not something Layer 1 concerns itself with. `is_weak_result(result: SearchResult) -> bool` (few/no candidates).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onesearch_tool.py
from unittest.mock import MagicMock

from agent.onesearch_tool import is_weak_result, run_onesearch
from onesearch.models import SearchCandidate, SearchResult


def _candidate():
    return SearchCandidate(
        record_id="1", title="t", author=None, resource_type=None, format=None,
        subject=None, description=None, date=None, source=None,
        delivery_category=None, online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
        copies_available=None,
    )


def test_is_weak_result_true_when_empty():
    assert is_weak_result(SearchResult(candidates=[], timed_out=False, fallback_link=None))


def test_is_weak_result_false_with_candidates():
    result = SearchResult(candidates=[_candidate()], timed_out=False, fallback_link=None)
    assert is_weak_result(result) is False


def test_run_onesearch_refines_once_when_weak():
    client = MagicMock()
    client.search_with_fallback.side_effect = [
        SearchResult(candidates=[], timed_out=False, fallback_link=None),
        SearchResult(candidates=[_candidate()], timed_out=False, fallback_link=None),
    ]

    result = run_onesearch("vague query", client=client)

    assert client.search_with_fallback.call_count == 2
    assert len(result.candidates) == 1


def test_run_onesearch_stops_after_one_refinement_even_if_still_weak():
    client = MagicMock()
    client.search_with_fallback.return_value = SearchResult(
        candidates=[], timed_out=False, fallback_link=None
    )

    run_onesearch("very vague", client=client)

    assert client.search_with_fallback.call_count == 2


def test_run_onesearch_does_not_refine_when_first_pass_is_strong():
    client = MagicMock()
    client.search_with_fallback.return_value = SearchResult(
        candidates=[_candidate()], timed_out=False, fallback_link=None
    )

    run_onesearch("good query", client=client)

    assert client.search_with_fallback.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onesearch_tool.py -v`
Expected: FAIL — `agent.onesearch_tool` has no `run_onesearch`/`is_weak_result`

- [ ] **Step 3: Write `agent/onesearch_tool.py`**

```python
# agent/onesearch_tool.py
from onesearch.client import OneSearchClient
from onesearch.models import SearchResult

MAX_REFINEMENTS = 1
WEAK_RESULT_THRESHOLD = 1  # fewer than this many candidates counts as weak


def is_weak_result(result: SearchResult) -> bool:
    return len(result.candidates) < WEAK_RESULT_THRESHOLD


def _refine_query(query: str) -> str:
    # Simple broadening refinement: drop the most specific (last) term.
    # Layer 2's router/LLM step may replace this with an LLM-generated
    # refinement later; this keeps the tool deterministic and testable.
    terms = query.strip().split()
    if len(terms) <= 1:
        return query
    return " ".join(terms[:-1])


def run_onesearch(
    query: str, client: OneSearchClient, max_refinements: int = MAX_REFINEMENTS
) -> SearchResult:
    result = client.search_with_fallback(query)
    refinements_used = 0
    current_query = query
    while is_weak_result(result) and refinements_used < max_refinements:
        current_query = _refine_query(current_query)
        result = client.search_with_fallback(current_query)
        refinements_used += 1
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_onesearch_tool.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/onesearch_tool.py tests/test_onesearch_tool.py
git commit -m "feat: wire onesearch tool with search-refine-once orchestration rule"
```

---

### Task 8: RAG enrichment — `agent/rag_enrichment.py`

**Files:**
- Create: `agent/rag_enrichment.py` (new module — old CLAUDE.md folded this into the now-removed `retrieval.py`/merged `onesearch_tool.py`; with Layer 1 split out, RAG enrichment gets its own home)
- Test: `tests/test_rag_enrichment.py`

**Interfaces:**
- Consumes: `SearchCandidate` list (Task 7 output), a Chroma collection (injected as `vectorstore`).
- Produces: `EnrichmentResult` dataclass (`usage_notes: str | None`, `librarian_name: str | None`, `librarian_email: str | None`, `librarian_subject: str | None`), `enrich(candidates: list[SearchCandidate], query: str, vectorstore) -> EnrichmentResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_enrichment.py
from unittest.mock import MagicMock

from agent.rag_enrichment import EnrichmentResult, enrich
from onesearch.models import SearchCandidate


def _candidate(subject="Machine learning"):
    return SearchCandidate(
        record_id="1", title="t", author=None, resource_type=None, format=None,
        subject=subject, description=None, date=None, source=None,
        delivery_category=None, online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
        copies_available=None,
    )


def test_enrich_returns_usage_notes_and_librarian_from_top_chunks():
    doc_a = MagicMock(page_content="Use OneSearch filters to narrow by format.")
    doc_a.metadata = {"type": "libguide"}
    doc_b = MagicMock(page_content="Subject librarian: Jane Doe, jane.doe@buffalo.edu, Computer Science")
    doc_b.metadata = {"type": "librarian_directory", "name": "Jane Doe", "email": "jane.doe@buffalo.edu", "subject": "Computer Science"}

    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [doc_a, doc_b]

    result = enrich([_candidate()], "machine learning", vectorstore)

    assert isinstance(result, EnrichmentResult)
    assert result.librarian_name == "Jane Doe"
    assert result.librarian_email == "jane.doe@buffalo.edu"
    assert "OneSearch filters" in result.usage_notes


def test_enrich_handles_no_matches_gracefully():
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = []

    result = enrich([_candidate()], "machine learning", vectorstore)

    assert result.usage_notes is None
    assert result.librarian_name is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_enrichment.py -v`
Expected: FAIL — `agent.rag_enrichment` does not exist

- [ ] **Step 3: Write `agent/rag_enrichment.py`**

```python
# agent/rag_enrichment.py
from dataclasses import dataclass

from onesearch.models import SearchCandidate

TOP_K = 4


@dataclass(frozen=True)
class EnrichmentResult:
    usage_notes: str | None
    librarian_name: str | None
    librarian_email: str | None
    librarian_subject: str | None


def enrich(
    candidates: list[SearchCandidate], query: str, vectorstore
) -> EnrichmentResult:
    subjects = " ".join(c.subject for c in candidates if c.subject)
    search_text = f"{query} {subjects}".strip()
    docs = vectorstore.similarity_search(search_text, k=TOP_K)

    usage_notes = None
    librarian_name = librarian_email = librarian_subject = None

    for doc in docs:
        doc_type = doc.metadata.get("type")
        if doc_type == "libguide" and usage_notes is None:
            usage_notes = doc.page_content
        elif doc_type == "librarian_directory" and librarian_name is None:
            librarian_name = doc.metadata.get("name")
            librarian_email = doc.metadata.get("email")
            librarian_subject = doc.metadata.get("subject")

    return EnrichmentResult(
        usage_notes=usage_notes,
        librarian_name=librarian_name,
        librarian_email=librarian_email,
        librarian_subject=librarian_subject,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_enrichment.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/rag_enrichment.py tests/test_rag_enrichment.py
git commit -m "feat: add RAG enrichment over libguides and librarian directory"
```

---

### Task 9: Citation formatting tool — `agent/citation_tool.py`

**Files:**
- Modify: `agent/citation_tool.py` (currently empty stub)
- Test: `tests/test_citation_tool.py`

**Interfaces:**
- Consumes: `SearchCandidate` (Task 1).
- Produces: `format_citation(candidate: SearchCandidate, style: str) -> str` where `style` is one of `"apa"`, `"mla"`, `"chicago"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_citation_tool.py
import pytest

from agent.citation_tool import format_citation
from onesearch.models import SearchCandidate


def _candidate():
    return SearchCandidate(
        record_id="1", title="Machine Learning for Coastal Systems",
        author="Wong, Kau-Fui Vincent", resource_type="book", format="Book, Print",
        subject="Machine learning", description=None, date="2021", source="Alma",
        delivery_category="Alma-P", online_link=None, call_number="QA76.87 .W66 2021",
        library="Silverman Library", sub_location="Stacks", availability_status="available",
        copies_available="1",
    )


def test_format_citation_apa():
    citation = format_citation(_candidate(), "apa")
    assert citation == "Wong, Kau-Fui Vincent (2021). Machine Learning for Coastal Systems."


def test_format_citation_mla():
    citation = format_citation(_candidate(), "mla")
    assert citation == "Wong, Kau-Fui Vincent. Machine Learning for Coastal Systems. 2021."


def test_format_citation_chicago():
    citation = format_citation(_candidate(), "chicago")
    assert citation == "Wong, Kau-Fui Vincent. Machine Learning for Coastal Systems. 2021."


def test_format_citation_unknown_style_raises():
    with pytest.raises(ValueError, match="unsupported citation style"):
        format_citation(_candidate(), "harvard")


def test_format_citation_handles_missing_author():
    candidate = _candidate()
    candidate = candidate.__class__(**{**candidate.__dict__, "author": None})
    citation = format_citation(candidate, "apa")
    assert citation == "Machine Learning for Coastal Systems (2021)."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_citation_tool.py -v`
Expected: FAIL — `format_citation` not defined

- [ ] **Step 3: Write `agent/citation_tool.py`**

```python
# agent/citation_tool.py
from onesearch.models import SearchCandidate

SUPPORTED_STYLES = {"apa", "mla", "chicago"}


def format_citation(candidate: SearchCandidate, style: str) -> str:
    if style not in SUPPORTED_STYLES:
        raise ValueError(f"unsupported citation style: {style}")

    title = candidate.title
    date = candidate.date or "n.d."
    author = candidate.author

    if style == "apa":
        if author:
            return f"{author} ({date}). {title}."
        return f"{title} ({date})."

    # mla and chicago share the same simple author-title-date shape here
    if author:
        return f"{author}. {title}. {date}."
    return f"{title}. {date}."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_citation_tool.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/citation_tool.py tests/test_citation_tool.py
git commit -m "feat: add deterministic APA/MLA/Chicago citation formatter"
```

---

### Task 10: Escalation — `agent/escalation.py`

**Files:**
- Modify: `agent/escalation.py` (currently empty stub)
- Test: `tests/test_escalation.py`

**Interfaces:**
- Produces: `EscalationBackend` protocol with `handoff(context: "EscalationContext") -> bool`; `EscalationContext` dataclass (`role: str`, `query: str`, `conversation_summary: str`, `reason: str`); `ConsoleEscalationBackend` (concrete stub — live chat platform is unconfirmed per prior CLAUDE.md "Open items"); `escalate(context: EscalationContext, backend: EscalationBackend) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_escalation.py
from agent.escalation import ConsoleEscalationBackend, EscalationContext, escalate


def test_escalate_calls_backend_handoff():
    class RecordingBackend:
        def __init__(self):
            self.received = None

        def handoff(self, context):
            self.received = context
            return True

    backend = RecordingBackend()
    context = EscalationContext(
        role="undergraduate", query="how do I find primary sources",
        conversation_summary="user asked twice, low confidence match",
        reason="low_confidence",
    )

    result = escalate(context, backend)

    assert result is True
    assert backend.received is context


def test_console_backend_returns_true_and_logs(capsys):
    backend = ConsoleEscalationBackend()
    context = EscalationContext(
        role="faculty", query="grant database access",
        conversation_summary="asked about database access",
        reason="out_of_scope",
    )

    result = backend.handoff(context)

    assert result is True
    captured = capsys.readouterr()
    assert "faculty" in captured.out
    assert "out_of_scope" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_escalation.py -v`
Expected: FAIL — module has no `EscalationContext`/`escalate`/`ConsoleEscalationBackend`

- [ ] **Step 3: Write `agent/escalation.py`**

```python
# agent/escalation.py
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EscalationContext:
    role: str
    query: str
    conversation_summary: str
    reason: str  # e.g. "low_confidence", "out_of_scope", "ambiguous"


class EscalationBackend(Protocol):
    def handoff(self, context: EscalationContext) -> bool: ...


class ConsoleEscalationBackend:
    """Stub backend: the live chat platform UB Libraries actually uses is
    still unconfirmed (prior CLAUDE.md, "Open items still pending
    confirmation"). This logs the handoff instead of calling a real
    system. Swap in the real backend behind the same EscalationBackend
    protocol once the platform is confirmed — nothing else in the
    codebase should need to change."""

    def handoff(self, context: EscalationContext) -> bool:
        print(
            f"[ESCALATION] role={context.role} reason={context.reason} "
            f"query={context.query!r} summary={context.conversation_summary!r}"
        )
        return True


def escalate(context: EscalationContext, backend: EscalationBackend) -> bool:
    return backend.handoff(context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_escalation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/escalation.py tests/test_escalation.py
git commit -m "feat: add escalation interface with console stub backend"
```

---

### Task 11: Fast-path detection — `agent/router.py`

Per the hybrid design in `CLAUDE.md`'s "Layer 2: Agent Orchestration" section: two cheap, deterministic checks run before anything expensive, so a trivial follow-up never pays for a full OneSearch + RAG + LLM round trip.

**Files:**
- Modify: `agent/router.py` (currently empty stub)
- Test: `tests/test_router.py`

**Interfaces:**
- Produces: `wants_human_handoff(query: str) -> bool`; `is_followup_citation_request(query: str, has_prior_candidates: bool) -> bool` (only true when the query looks like a citation ask *and* a candidate from an earlier turn is already available — otherwise it falls through to the default path, since there's nothing yet to cite).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
from agent.router import is_followup_citation_request, wants_human_handoff


def test_wants_human_handoff_true_for_explicit_request():
    assert wants_human_handoff("talk to a real person about my thesis") is True


def test_wants_human_handoff_false_for_normal_query():
    assert wants_human_handoff("find articles about coastal erosion") is False


def test_is_followup_citation_request_true_with_prior_candidate():
    assert is_followup_citation_request("cite that in APA", has_prior_candidates=True) is True


def test_is_followup_citation_request_false_without_prior_candidate():
    # nothing to cite yet — must go through the default search path first
    assert is_followup_citation_request("cite that in APA", has_prior_candidates=False) is False


def test_is_followup_citation_request_false_for_non_citation_query():
    assert is_followup_citation_request("find articles about coastal erosion", has_prior_candidates=True) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `agent.router` has no `wants_human_handoff`/`is_followup_citation_request`

- [ ] **Step 3: Write `agent/router.py`**

```python
# agent/router.py
HUMAN_HANDOFF_KEYWORDS = {"real person", "human", "librarian help", "talk to someone"}
CITATION_KEYWORDS = {"cite", "citation", "apa", "mla", "chicago"}


def wants_human_handoff(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in HUMAN_HANDOFF_KEYWORDS)


def is_followup_citation_request(query: str, has_prior_candidates: bool) -> bool:
    if not has_prior_candidates:
        return False
    lowered = query.lower()
    return any(keyword in lowered for keyword in CITATION_KEYWORDS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_router.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/router.py tests/test_router.py
git commit -m "feat: add deterministic fast-path checks for handoff and followup citation"
```

---

### Task 12: Rank + Compose node and graph wiring — `agent/compose.py`, `agent/graph.py`

Implements the default path from `CLAUDE.md`'s "Layer 2: Agent Orchestration" section: OneSearch + RAG feed a single LLM step that scores confidence, picks the top 1-3 candidates, and only *then* triggers Layer 1's bounded `fetch_detail()` for those chosen candidates — the piece the previous version of this plan dropped.

**Files:**
- Create: `agent/compose.py`
- Modify: `agent/graph.py` (currently empty stub)
- Test: `tests/test_compose.py`, `tests/test_graph.py`

**Interfaces:**
- Consumes: `SearchCandidate`/`ResourceDetail` (Task 1), `EnrichmentResult` (Task 8), `format_citation` (Task 9), `OneSearchClient.fetch_detail` (Task 5).
- Produces (`agent/compose.py`): `RankedResult` dataclass (`selected_record_ids: list[str]`, `confidence: float`, `should_escalate: bool`, `citation_style: str | None`); `rank_and_score(query, candidates, enrichment, llm_fn) -> RankedResult`, where `llm_fn(query, candidates, enrichment) -> dict` is injected (production default calls a real LLM, tests inject a fake); `compose_answer(candidates, details, enrichment, ranked) -> str`; `CONFIDENCE_ESCALATION_THRESHOLD = 0.4`.
- Produces (`agent/graph.py`): `ConversationState` (TypedDict: `role: str`, `query: str`, `prior_candidates: list[SearchCandidate] | None`, `search_result: SearchResult | None`, `enrichment: EnrichmentResult | None`, `ranked: RankedResult | None`, `answer: str | None`, `escalated: bool`); `build_graph(client, vectorstore, escalation_backend, llm_fn=default_llm_fn) -> CompiledGraph`.

- [ ] **Step 1: Write the failing test for `agent/compose.py`**

```python
# tests/test_compose.py
from agent.compose import RankedResult, compose_answer, rank_and_score
from agent.rag_enrichment import EnrichmentResult
from onesearch.models import ResourceDetail, SearchCandidate


def _candidate(record_id="1"):
    return SearchCandidate(
        record_id=record_id, title="Machine Learning for Coastal Systems", author="Wong",
        resource_type=None, format=None, subject=None, description=None, date="2021",
        source=None, delivery_category=None, online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None, copies_available=None,
    )


def _empty_enrichment():
    return EnrichmentResult(usage_notes=None, librarian_name=None, librarian_email=None, librarian_subject=None)


def test_rank_and_score_escalates_when_no_candidates():
    result = rank_and_score("q", [], _empty_enrichment(), llm_fn=lambda **_: {})
    assert result.should_escalate is True
    assert result.selected_record_ids == []


def test_rank_and_score_uses_llm_fn_output():
    llm_fn = lambda **_: {"confidence": 0.9, "selected_record_ids": ["1"], "citation_style": None}
    result = rank_and_score("q", [_candidate()], _empty_enrichment(), llm_fn=llm_fn)
    assert result.confidence == 0.9
    assert result.selected_record_ids == ["1"]
    assert result.should_escalate is False


def test_rank_and_score_escalates_below_confidence_threshold():
    llm_fn = lambda **_: {"confidence": 0.1, "selected_record_ids": ["1"], "citation_style": None}
    result = rank_and_score("q", [_candidate()], _empty_enrichment(), llm_fn=llm_fn)
    assert result.should_escalate is True


def test_compose_answer_includes_online_link():
    details = {"1": ResourceDetail(record_id="1", online_link="https://example.edu/1", call_number=None,
                                    library=None, sub_location=None, availability_status=None,
                                    copies_available=None, raw={})}
    ranked = RankedResult(selected_record_ids=["1"], confidence=0.9, should_escalate=False, citation_style=None)
    answer = compose_answer([_candidate()], details, _empty_enrichment(), ranked)
    assert "Machine Learning for Coastal Systems" in answer
    assert "https://example.edu/1" in answer


def test_compose_answer_includes_call_number_for_physical_item():
    details = {"1": ResourceDetail(record_id="1", online_link=None, call_number="QA76.87 .W66 2021",
                                    library="Silverman Library", sub_location="Stacks",
                                    availability_status="available", copies_available="1", raw={})}
    ranked = RankedResult(selected_record_ids=["1"], confidence=0.9, should_escalate=False, citation_style=None)
    answer = compose_answer([_candidate()], details, _empty_enrichment(), ranked)
    assert "QA76.87 .W66 2021" in answer
    assert "Silverman Library" in answer


def test_compose_answer_includes_citation_when_style_set():
    details = {"1": ResourceDetail(record_id="1", online_link="https://example.edu/1", call_number=None,
                                    library=None, sub_location=None, availability_status=None,
                                    copies_available=None, raw={})}
    ranked = RankedResult(selected_record_ids=["1"], confidence=0.9, should_escalate=False, citation_style="apa")
    answer = compose_answer([_candidate()], details, _empty_enrichment(), ranked)
    assert "Wong (2021)" in answer


def test_compose_answer_includes_librarian_and_usage_notes():
    enrichment = EnrichmentResult(usage_notes="Use OneSearch filters to narrow by format.",
                                   librarian_name="Jane Doe", librarian_email="jane.doe@buffalo.edu",
                                   librarian_subject="Computer Science")
    ranked = RankedResult(selected_record_ids=["1"], confidence=0.9, should_escalate=False, citation_style=None)
    answer = compose_answer([_candidate()], {}, enrichment, ranked)
    assert "OneSearch filters" in answer
    assert "Jane Doe" in answer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compose.py -v`
Expected: FAIL — `agent.compose` does not exist

- [ ] **Step 3: Write `agent/compose.py`**

```python
# agent/compose.py
from dataclasses import dataclass

from agent.citation_tool import format_citation
from agent.rag_enrichment import EnrichmentResult
from onesearch.models import ResourceDetail, SearchCandidate

CONFIDENCE_ESCALATION_THRESHOLD = 0.4


@dataclass(frozen=True)
class RankedResult:
    selected_record_ids: list[str]
    confidence: float
    should_escalate: bool
    citation_style: str | None


def default_llm_fn(query: str, candidates: list[SearchCandidate], enrichment: EnrichmentResult) -> dict:
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    class RankingDecision(BaseModel):
        confidence: float
        selected_record_ids: list[str]
        citation_style: str | None = None

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_model = model.with_structured_output(RankingDecision)

    candidate_summary = "\n".join(
        f"- id={c.record_id} title={c.title!r} type={c.resource_type} subject={c.subject}"
        for c in candidates
    )
    prompt = (
        f"User query: {query}\n\nCandidates:\n{candidate_summary}\n\n"
        f"Library context: {enrichment.usage_notes or 'no additional context'}\n\n"
        "Score confidence 0-1 that these candidates answer the query, pick at most "
        "3 record_ids worth showing in order of relevance, and set citation_style "
        "only if the user explicitly asked for a citation format."
    )
    decision = structured_model.invoke(prompt)
    return {
        "confidence": decision.confidence,
        "selected_record_ids": decision.selected_record_ids,
        "citation_style": decision.citation_style,
    }


def rank_and_score(
    query: str, candidates: list[SearchCandidate], enrichment: EnrichmentResult, llm_fn=default_llm_fn
) -> RankedResult:
    if not candidates:
        return RankedResult(selected_record_ids=[], confidence=0.0, should_escalate=True, citation_style=None)

    response = llm_fn(query=query, candidates=candidates, enrichment=enrichment)
    confidence = response["confidence"]
    selected = response.get("selected_record_ids", [])[:3]
    citation_style = response.get("citation_style")

    return RankedResult(
        selected_record_ids=selected,
        confidence=confidence,
        should_escalate=confidence < CONFIDENCE_ESCALATION_THRESHOLD,
        citation_style=citation_style,
    )


def compose_answer(
    candidates: list[SearchCandidate],
    details: dict[str, ResourceDetail],
    enrichment: EnrichmentResult,
    ranked: RankedResult,
) -> str:
    by_id = {c.record_id: c for c in candidates}
    lines = []

    for record_id in ranked.selected_record_ids:
        candidate = by_id.get(record_id)
        if candidate is None:
            continue
        detail = details.get(record_id)
        line = candidate.title
        if candidate.author:
            line += f" — {candidate.author}"
        if detail and detail.online_link:
            line += f". Online: {detail.online_link}"
        elif detail and detail.call_number:
            line += f". Call number: {detail.call_number}, {detail.library} ({detail.sub_location})"
        if ranked.citation_style:
            line += f". {format_citation(candidate, ranked.citation_style)}"
        lines.append(line)

    if enrichment.usage_notes:
        lines.append(enrichment.usage_notes)
    if enrichment.librarian_name:
        lines.append(f"Subject librarian: {enrichment.librarian_name} ({enrichment.librarian_email})")

    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify `agent/compose.py` passes**

Run: `pytest tests/test_compose.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Write the failing test for `agent/graph.py`**

```python
# tests/test_graph.py
from unittest.mock import MagicMock

from agent.graph import ConversationState, build_graph
from onesearch.models import ResourceDetail, SearchCandidate, SearchResult


def _candidate(record_id="1"):
    return SearchCandidate(
        record_id=record_id, title="t", author="a", resource_type=None, format=None,
        subject=None, description=None, date="2021", source=None,
        delivery_category=None, online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
        copies_available=None,
    )


def _initial_state(query, role="undergraduate", prior_candidates=None):
    return ConversationState(
        role=role, query=query, prior_candidates=prior_candidates,
        search_result=None, enrichment=None, ranked=None, answer=None, escalated=False,
    )


def test_graph_escalates_immediately_on_explicit_human_request():
    client = MagicMock()
    vectorstore = MagicMock()
    escalation_backend = MagicMock()
    escalation_backend.handoff.return_value = True
    llm_fn = MagicMock()

    graph = build_graph(client, vectorstore, escalation_backend, llm_fn)
    result = graph.invoke(_initial_state("talk to a real person"))

    assert result["escalated"] is True
    client.search_with_fallback.assert_not_called()
    llm_fn.assert_not_called()


def test_graph_uses_quick_citation_path_for_known_candidate():
    client = MagicMock()
    vectorstore = MagicMock()
    escalation_backend = MagicMock()
    llm_fn = MagicMock()

    graph = build_graph(client, vectorstore, escalation_backend, llm_fn)
    result = graph.invoke(_initial_state("cite this in apa", prior_candidates=[_candidate()]))

    assert result["answer"] is not None
    client.search_with_fallback.assert_not_called()
    llm_fn.assert_not_called()


def test_graph_default_path_fetches_detail_only_for_selected_candidates():
    client = MagicMock()
    client.search_with_fallback.return_value = SearchResult(
        candidates=[_candidate("1"), _candidate("2")], timed_out=False, fallback_link=None
    )
    client.fetch_detail.return_value = {
        "1": ResourceDetail(record_id="1", online_link="https://example.edu/1", call_number=None,
                             library=None, sub_location=None, availability_status=None,
                             copies_available=None, raw={})
    }
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = []
    escalation_backend = MagicMock()
    llm_fn = MagicMock(return_value={"confidence": 0.9, "selected_record_ids": ["1"], "citation_style": None})

    graph = build_graph(client, vectorstore, escalation_backend, llm_fn)
    result = graph.invoke(_initial_state("find articles about coastal erosion"))

    client.fetch_detail.assert_called_once_with(["1"])
    assert "example.edu/1" in result["answer"]
    assert result["escalated"] is False


def test_graph_escalates_when_rank_confidence_is_low():
    client = MagicMock()
    client.search_with_fallback.return_value = SearchResult(
        candidates=[_candidate("1")], timed_out=False, fallback_link=None
    )
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = []
    escalation_backend = MagicMock()
    escalation_backend.handoff.return_value = True
    llm_fn = MagicMock(return_value={"confidence": 0.1, "selected_record_ids": ["1"], "citation_style": None})

    graph = build_graph(client, vectorstore, escalation_backend, llm_fn)
    result = graph.invoke(_initial_state("something very vague"))

    assert result["escalated"] is True
    client.fetch_detail.assert_not_called()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL — `agent.graph` has no `ConversationState`/`build_graph`

- [ ] **Step 7: Write `agent/graph.py`**

```python
# agent/graph.py
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agent.citation_tool import format_citation
from agent.compose import RankedResult, compose_answer, default_llm_fn, rank_and_score
from agent.escalation import EscalationBackend, EscalationContext, escalate
from agent.onesearch_tool import run_onesearch
from agent.rag_enrichment import EnrichmentResult, enrich
from agent.router import is_followup_citation_request, wants_human_handoff
from onesearch.client import OneSearchClient
from onesearch.models import SearchCandidate, SearchResult


class ConversationState(TypedDict):
    role: str
    query: str
    prior_candidates: list[SearchCandidate] | None
    search_result: SearchResult | None
    enrichment: EnrichmentResult | None
    ranked: RankedResult | None
    answer: str | None
    escalated: bool


def build_graph(
    client: OneSearchClient, vectorstore, escalation_backend: EscalationBackend, llm_fn=default_llm_fn,
):
    def entry_router(state: ConversationState) -> str:
        if wants_human_handoff(state["query"]):
            return "escalate"
        if is_followup_citation_request(state["query"], bool(state.get("prior_candidates"))):
            return "quick_citation"
        return "search_and_enrich"

    def quick_citation_node(state: ConversationState) -> ConversationState:
        candidate = state["prior_candidates"][0]
        style = next((s for s in ("apa", "mla", "chicago") if s in state["query"].lower()), "apa")
        return {**state, "answer": format_citation(candidate, style), "escalated": False}

    def search_and_enrich_node(state: ConversationState) -> ConversationState:
        # OneSearch narrows first; the RAG pass then enriches using those
        # results (CLAUDE.md: "once OneSearch has narrowed the field, a
        # smaller RAG pass adds institutional context"). Conceptually
        # independent of Layer 1's own latency — this repo doesn't need
        # literal thread-level parallelism for correctness.
        result = run_onesearch(state["query"], client)
        enrichment = enrich(result.candidates, state["query"], vectorstore)
        return {**state, "search_result": result, "enrichment": enrichment}

    def rank_node(state: ConversationState) -> ConversationState:
        search_result = state["search_result"]
        ranked = rank_and_score(state["query"], search_result.candidates, state["enrichment"], llm_fn)
        return {**state, "ranked": ranked}

    def rank_router(state: ConversationState) -> str:
        return "escalate" if state["ranked"].should_escalate else "detail_and_compose"

    def detail_and_compose_node(state: ConversationState) -> ConversationState:
        ranked = state["ranked"]
        # Bounded by construction: fetch_detail only ever receives the
        # handful of ids rank_and_score selected, never the full result set.
        details = client.fetch_detail(ranked.selected_record_ids)
        answer = compose_answer(state["search_result"].candidates, details, state["enrichment"], ranked)
        return {**state, "answer": answer, "escalated": False}

    def escalate_node(state: ConversationState) -> ConversationState:
        reason = "user_requested" if wants_human_handoff(state["query"]) else "low_confidence"
        context = EscalationContext(
            role=state["role"], query=state["query"], conversation_summary=state["query"], reason=reason,
        )
        escalated = escalate(context, escalation_backend)
        return {**state, "answer": None, "escalated": escalated}

    graph = StateGraph(ConversationState)
    graph.add_node("quick_citation", quick_citation_node)
    graph.add_node("search_and_enrich", search_and_enrich_node)
    graph.add_node("rank", rank_node)
    graph.add_node("detail_and_compose", detail_and_compose_node)
    graph.add_node("escalate", escalate_node)

    graph.set_conditional_entry_point(
        entry_router,
        {"escalate": "escalate", "quick_citation": "quick_citation", "search_and_enrich": "search_and_enrich"},
    )
    graph.add_edge("search_and_enrich", "rank")
    graph.add_conditional_edges(
        "rank", rank_router, {"escalate": "escalate", "detail_and_compose": "detail_and_compose"}
    )
    graph.add_edge("quick_citation", END)
    graph.add_edge("detail_and_compose", END)
    graph.add_edge("escalate", END)

    return graph.compile()
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_graph.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add agent/compose.py agent/graph.py tests/test_compose.py tests/test_graph.py
git commit -m "feat: add rank+compose node and wire hybrid fast-path/default-path graph"
```

---

## Phase 3 — Layer 2: Data Pipeline (LibGuides/FAQ/librarian directory → Chroma)

### Task 13: LibGuides + FAQ scrapers — `scraper/scrape_libguides.py`, `scraper/scrape_faqs.py`

**Files:**
- Modify: `scraper/scrape_libguides.py`, `scraper/scrape_faqs.py` (currently empty stubs)
- Test: `tests/test_scrapers.py`

**Interfaces:**
- Produces: `scrape_libguides.fetch_page_html(url: str, page_loader) -> str` and `scrape_libguides.extract_text(html: str) -> str`; same shape for `scrape_faqs`. `page_loader` is injected (a Playwright page-load callable) so tests never launch a real browser.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scrapers.py
from scraper.scrape_faqs import extract_text as extract_faq_text
from scraper.scrape_libguides import extract_text as extract_libguide_text

SAMPLE_HTML = """
<html><body>
<nav>Skip this nav</nav>
<main><h1>Citing Sources</h1><p>Use APA for psychology courses.</p></main>
<footer>Skip this footer</footer>
</body></html>
"""


def test_libguide_extract_text_drops_nav_and_footer():
    text = extract_libguide_text(SAMPLE_HTML)
    assert "Citing Sources" in text
    assert "Use APA for psychology courses." in text
    assert "Skip this nav" not in text
    assert "Skip this footer" not in text


def test_faq_extract_text_drops_nav_and_footer():
    text = extract_faq_text(SAMPLE_HTML)
    assert "Citing Sources" in text
    assert "Skip this nav" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scrapers.py -v`
Expected: FAIL — `extract_text` not defined in either module

- [ ] **Step 3: Write `scraper/scrape_libguides.py`**

```python
# scraper/scrape_libguides.py
from bs4 import BeautifulSoup

USER_AGENT = "UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)"


def fetch_page_html(url: str, page_loader) -> str:
    """page_loader(url) -> html string. Injected so production wiring can
    pass a Playwright-driven loader (LibGuides pages can render
    dynamically) while tests pass a stub."""
    return page_loader(url)


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "footer", "script", "style"]):
        tag.decompose()
    main = soup.find("main") or soup
    return " ".join(main.get_text(separator=" ", strip=True).split())
```

- [ ] **Step 4: Write `scraper/scrape_faqs.py`**

```python
# scraper/scrape_faqs.py
from bs4 import BeautifulSoup

USER_AGENT = "UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)"


def fetch_page_html(url: str, page_loader) -> str:
    return page_loader(url)


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "footer", "script", "style"]):
        tag.decompose()
    main = soup.find("main") or soup
    return " ".join(main.get_text(separator=" ", strip=True).split())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_scrapers.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add scraper/scrape_libguides.py scraper/scrape_faqs.py tests/test_scrapers.py
git commit -m "feat: add libguides and faq scrapers with shared extraction logic"
```

---

### Task 14: Librarian directory scraper — `scraper/scrape_librarian_directory.py`

**Files:**
- Modify: `scraper/scrape_librarian_directory.py` (currently empty stub)
- Test: `tests/test_librarian_directory.py`

**Interfaces:**
- Produces: `LibrarianEntry` dataclass (`name: str`, `email: str`, `subject: str`); `extract_librarians(html: str) -> list[LibrarianEntry]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_librarian_directory.py
from scraper.scrape_librarian_directory import LibrarianEntry, extract_librarians

SAMPLE_HTML = """
<table>
  <tr class="librarian-row" data-name="Jane Doe" data-email="jane.doe@buffalo.edu" data-subject="Computer Science"></tr>
  <tr class="librarian-row" data-name="John Smith" data-email="john.smith@buffalo.edu" data-subject="Biology"></tr>
</table>
"""


def test_extract_librarians_parses_all_rows():
    entries = extract_librarians(SAMPLE_HTML)
    assert entries == [
        LibrarianEntry(name="Jane Doe", email="jane.doe@buffalo.edu", subject="Computer Science"),
        LibrarianEntry(name="John Smith", email="john.smith@buffalo.edu", subject="Biology"),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_librarian_directory.py -v`
Expected: FAIL — module has no `LibrarianEntry`/`extract_librarians`

- [ ] **Step 3: Write `scraper/scrape_librarian_directory.py`**

```python
# scraper/scrape_librarian_directory.py
from dataclasses import dataclass

from bs4 import BeautifulSoup

USER_AGENT = "UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)"


@dataclass(frozen=True)
class LibrarianEntry:
    name: str
    email: str
    subject: str


def fetch_page_html(url: str, page_loader) -> str:
    return page_loader(url)


def extract_librarians(html: str) -> list[LibrarianEntry]:
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    for row in soup.select(".librarian-row"):
        entries.append(
            LibrarianEntry(
                name=row["data-name"],
                email=row["data-email"],
                subject=row["data-subject"],
            )
        )
    return entries
```

**Note:** The real UB librarian-directory page markup is unknown until UB Libraries confirms who maintains it and whether it's programmatically accessible (prior CLAUDE.md, "Open items"). `.librarian-row`/`data-*` selectors above are a placeholder shape — update `extract_librarians` to match the real page's markup once that page is identified, and add a fixture captured from it (same pattern as Task 2's OneSearch fixtures).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_librarian_directory.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add scraper/scrape_librarian_directory.py tests/test_librarian_directory.py
git commit -m "feat: add librarian directory scraper (placeholder markup, pending UB confirmation)"
```

---

### Task 15: Chunking — `pipeline/chunk.py`

**Files:**
- Modify: `pipeline/chunk.py` (currently empty stub)
- Test: `tests/test_chunk.py`

**Interfaces:**
- Produces: `Chunk` dataclass (`text: str`, `source_type: str`, `source_url: str`); `chunk_text(text: str, source_type: str, source_url: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Chunk]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunk.py
from pipeline.chunk import Chunk, chunk_text


def test_chunk_text_splits_long_text():
    text = "word " * 300
    chunks = chunk_text(text, source_type="libguide", source_url="https://example.edu", chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_type == "libguide" for c in chunks)
    assert all(c.source_url == "https://example.edu" for c in chunks)


def test_chunk_text_short_text_stays_one_chunk():
    chunks = chunk_text("short text", source_type="faq", source_url="https://example.edu", chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunk.py -v`
Expected: FAIL — `pipeline.chunk` has no `Chunk`/`chunk_text`

- [ ] **Step 3: Write `pipeline/chunk.py`**

```python
# pipeline/chunk.py
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    text: str
    source_type: str
    source_url: str


def chunk_text(
    text: str, source_type: str, source_url: str,
    chunk_size: int = 500, chunk_overlap: int = 50,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [
        Chunk(text=piece, source_type=source_type, source_url=source_url)
        for piece in splitter.split_text(text)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunk.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/chunk.py tests/test_chunk.py
git commit -m "feat: add text chunking for libguide/faq/librarian content"
```

---

### Task 16: Embedding — `pipeline/embed.py`

**Files:**
- Modify: `pipeline/embed.py` (currently empty stub)
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `Chunk` (Task 15).
- Produces: `embed_chunks(chunks: list[Chunk], vectorstore) -> int` (returns count embedded), using `metadata={"type": chunk.source_type, "source_url": chunk.source_url}` on each entry — this is the `type`/`name`/`email`/`subject` metadata shape `agent/rag_enrichment.py` (Task 8) reads back.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed.py
from unittest.mock import MagicMock

from pipeline.chunk import Chunk
from pipeline.embed import embed_chunks


def test_embed_chunks_adds_texts_with_metadata():
    vectorstore = MagicMock()
    chunks = [
        Chunk(text="a", source_type="libguide", source_url="https://a"),
        Chunk(text="b", source_type="faq", source_url="https://b"),
    ]

    count = embed_chunks(chunks, vectorstore)

    assert count == 2
    vectorstore.add_texts.assert_called_once_with(
        texts=["a", "b"],
        metadatas=[
            {"type": "libguide", "source_url": "https://a"},
            {"type": "faq", "source_url": "https://b"},
        ],
    )


def test_embed_chunks_empty_list_no_op():
    vectorstore = MagicMock()
    count = embed_chunks([], vectorstore)
    assert count == 0
    vectorstore.add_texts.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embed.py -v`
Expected: FAIL — `pipeline.embed` has no `embed_chunks`

- [ ] **Step 3: Write `pipeline/embed.py`**

```python
# pipeline/embed.py
from pipeline.chunk import Chunk


def embed_chunks(chunks: list[Chunk], vectorstore) -> int:
    if not chunks:
        return 0
    vectorstore.add_texts(
        texts=[c.text for c in chunks],
        metadatas=[{"type": c.source_type, "source_url": c.source_url} for c in chunks],
    )
    return len(chunks)


def build_vectorstore(persist_dir: str):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embed.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/embed.py tests/test_embed.py
git commit -m "feat: add embedding step writing chunks into chroma with source metadata"
```

---

### Task 17: Refresh orchestration — `pipeline/refresh.py`

**Files:**
- Modify: `pipeline/refresh.py` (currently empty stub)
- Test: `tests/test_refresh.py`

**Interfaces:**
- Consumes: scrapers (Tasks 13-14), `chunk_text` (Task 15), `embed_chunks`/`build_vectorstore` (Task 16).
- Produces: `refresh_all(sources: list[dict], vectorstore, page_loader) -> int` where each `sources` entry is `{"url": str, "type": str}`. This is the entrypoint `.github/workflows/weekly-refresh.yml` already calls as `python pipeline/refresh.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_refresh.py
from unittest.mock import MagicMock

from pipeline.refresh import refresh_all


def test_refresh_all_scrapes_chunks_and_embeds_each_source():
    vectorstore = MagicMock()
    page_loader = MagicMock(return_value="<html><main><p>Some content here about citing.</p></main></html>")
    sources = [
        {"url": "https://libguides.example.edu/cs", "type": "libguide"},
        {"url": "https://faq.example.edu/citing", "type": "faq"},
    ]

    total = refresh_all(sources, vectorstore, page_loader)

    assert total > 0
    assert vectorstore.add_texts.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_refresh.py -v`
Expected: FAIL — `pipeline.refresh` has no `refresh_all`

- [ ] **Step 3: Write `pipeline/refresh.py`**

```python
# pipeline/refresh.py
import os

from pipeline.chunk import chunk_text
from pipeline.embed import build_vectorstore, embed_chunks
from scraper.scrape_faqs import extract_text as extract_faq_text
from scraper.scrape_libguides import extract_text as extract_libguide_text

EXTRACTORS = {
    "libguide": extract_libguide_text,
    "faq": extract_faq_text,
}


def refresh_all(sources: list[dict], vectorstore, page_loader) -> int:
    total = 0
    for source in sources:
        html = page_loader(source["url"])
        extractor = EXTRACTORS[source["type"]]
        text = extractor(html)
        chunks = chunk_text(text, source_type=source["type"], source_url=source["url"])
        total += embed_chunks(chunks, vectorstore)
    return total


if __name__ == "__main__":
    from playwright.sync_api import sync_playwright

    def _playwright_page_loader(url: str) -> str:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.wait_for_load_state("networkidle")
            html = page.content()
            browser.close()
            return html

    # Source list to be filled in once UB Libraries confirms which
    # LibGuides/FAQ pages are in scope (prior CLAUDE.md, data sources).
    sources: list[dict] = []
    vs = build_vectorstore(os.environ.get("CHROMA_PERSIST_DIR", "vectorstore"))
    embedded_count = refresh_all(sources, vs, _playwright_page_loader)
    print(f"embedded {embedded_count} chunks")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_refresh.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add pipeline/refresh.py tests/test_refresh.py
git commit -m "feat: add weekly refresh orchestration entrypoint"
```

---

## Phase 4 — Layer 2: Logging, Dashboard, Evaluation

### Task 18: Anonymized logging — `analytics/db.py`, `analytics/log_writer.py`

**Files:**
- Modify: `analytics/db.py`, `analytics/log_writer.py` (currently empty stubs)
- Test: `tests/test_log_writer.py`

**Interfaces:**
- Produces: `analytics/db.py`: SQLAlchemy `Base`, `SessionLog` model (`id`, `role`, `course` nullable, `topic_category`, `escalated`, `created_at`), `get_engine(database_url: str)`, `init_db(engine)`.
- Produces: `analytics/log_writer.py`: `write_session_log(session_factory, role: str, topic_category: str, escalated: bool, course: str | None = None) -> None`. No name, query text, or any identifier is ever written (prior CLAUDE.md capability 7) — enforced by the function signature only accepting these four fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_log_writer.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from analytics.db import SessionLog, init_db
from analytics.log_writer import write_session_log


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return sessionmaker(bind=engine)


def test_write_session_log_persists_anonymized_fields():
    factory = _session_factory()
    write_session_log(factory, role="undergraduate", topic_category="citation_help", escalated=False, course="CSE 4/587")

    with factory() as session:
        rows = session.query(SessionLog).all()
        assert len(rows) == 1
        assert rows[0].role == "undergraduate"
        assert rows[0].topic_category == "citation_help"
        assert rows[0].course == "CSE 4/587"
        assert rows[0].escalated is False


def test_write_session_log_course_optional():
    factory = _session_factory()
    write_session_log(factory, role="faculty", topic_category="database_access", escalated=True)

    with factory() as session:
        row = session.query(SessionLog).one()
        assert row.course is None
        assert row.escalated is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_log_writer.py -v`
Expected: FAIL — `analytics.db` has no `SessionLog`/`init_db`

- [ ] **Step 3: Write `analytics/db.py`**

```python
# analytics/db.py
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class SessionLog(Base):
    __tablename__ = "session_logs"

    id = Column(Integer, primary_key=True)
    role = Column(String, nullable=False)
    course = Column(String, nullable=True)
    topic_category = Column(String, nullable=False)
    escalated = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def get_engine(database_url: str):
    return create_engine(database_url)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
```

- [ ] **Step 4: Write `analytics/log_writer.py`**

```python
# analytics/log_writer.py
from analytics.db import SessionLog


def write_session_log(
    session_factory, role: str, topic_category: str, escalated: bool,
    course: str | None = None,
) -> None:
    with session_factory() as session:
        session.add(
            SessionLog(role=role, course=course, topic_category=topic_category, escalated=escalated)
        )
        session.commit()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_log_writer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add analytics/db.py analytics/log_writer.py tests/test_log_writer.py
git commit -m "feat: add anonymized session logging (role, course, topic only)"
```

---

### Task 19: Streamlit chat interface — `streamlit_app/Home.py`

**Files:**
- Modify: `streamlit_app/Home.py` (currently empty stub)

**Interfaces:**
- Consumes: `build_graph`/`ConversationState` (Task 12), `write_session_log` (Task 18).
- Produces: a runnable Streamlit page — no automated test (Streamlit UI code; verify manually per Task 19 Step 3 below, consistent with how this project's own conventions treat UI).

- [ ] **Step 1: Write `streamlit_app/Home.py`**

```python
# streamlit_app/Home.py
import os

import streamlit as st

from agent.graph import ConversationState, build_graph
from analytics.db import get_engine, init_db
from analytics.log_writer import write_session_log
from onesearch.cache import QueryCache
from onesearch.client import OneSearchClient
from onesearch.session import SessionManager
from pipeline.embed import build_vectorstore
from sqlalchemy.orm import sessionmaker

st.set_page_config(page_title="UB Library Navigator")


@st.cache_resource
def get_dependencies():
    client = OneSearchClient(session_manager=SessionManager(), cache=QueryCache(ttl_seconds=300))
    vectorstore = build_vectorstore(os.environ.get("CHROMA_PERSIST_DIR", "vectorstore"))
    from agent.escalation import ConsoleEscalationBackend
    graph = build_graph(client, vectorstore, ConsoleEscalationBackend())
    engine = get_engine(os.environ.get("DATABASE_URL", "sqlite:///analytics/sessions.db"))
    init_db(engine)
    session_factory = sessionmaker(bind=engine)
    return graph, session_factory


graph, session_factory = get_dependencies()

st.title("UB Library Navigator")

if "role" not in st.session_state:
    st.session_state.role = None
if "prior_candidates" not in st.session_state:
    st.session_state.prior_candidates = None

if st.session_state.role is None:
    st.session_state.role = st.radio(
        "I am a...", ["undergraduate", "graduate", "faculty"], index=None
    )
    st.stop()

st.session_state.role = st.selectbox(
    "Role (change anytime)", ["undergraduate", "graduate", "faculty"],
    index=["undergraduate", "graduate", "faculty"].index(st.session_state.role),
)

query = st.chat_input("What are you looking for?")
if query:
    state = ConversationState(
        role=st.session_state.role, query=query,
        prior_candidates=st.session_state.prior_candidates,
        search_result=None, enrichment=None, ranked=None, answer=None, escalated=False,
    )
    result = graph.invoke(state)

    if result["escalated"]:
        st.chat_message("assistant").write("Connecting you with a librarian now.")
        topic_category = "escalation"
    else:
        st.chat_message("assistant").write(result["answer"] or "No results found.")
        topic_category = "citation_help" if state["prior_candidates"] else "resource_search"

    search_result = result.get("search_result")
    if search_result and search_result.candidates:
        # remembered so a later "cite that in MLA" turn can take the
        # quick-citation fast path instead of searching again
        st.session_state.prior_candidates = search_result.candidates

    write_session_log(
        session_factory, role=st.session_state.role,
        topic_category=topic_category, escalated=result["escalated"],
    )
```

- [ ] **Step 2: Manual verification**

Run: `streamlit run streamlit_app/Home.py`
Expected: role selector appears first; after picking a role, chat input appears; a search query returns candidate titles; a citation-style query returns a formatted citation; an escalation-phrase query shows the handoff message. Confirm role can be changed via the selectbox mid-session without losing the chat.

- [ ] **Step 3: Commit**

```bash
git add streamlit_app/Home.py
git commit -m "feat: wire streamlit chat interface to agent graph and session logging"
```

---

### Task 20: Staff dashboard — `streamlit_app/pages/1_Staff_Dashboard.py`

**Files:**
- Modify: `streamlit_app/pages/1_Staff_Dashboard.py` (currently empty stub)
- Test: `tests/test_dashboard_queries.py`

**Interfaces:**
- Produces: `AuthGate` protocol with `is_authorized() -> bool` (swappable — UB SSO/Shibboleth vs. shared staff login is unconfirmed per prior CLAUDE.md); `EnvVarAuthGate` stub reading `STAFF_DASHBOARD_AUTH_METHOD`/a shared secret from `.env`; `query_volume_by_role(session_factory) -> dict[str, int]`, `topic_breakdown(session_factory, min_count: int = 5) -> dict[str, int]` (suppresses breakdowns below `min_count`, per prior CLAUDE.md's re-identification risk mitigation), `escalation_rate(session_factory) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard_queries.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from analytics.db import init_db
from analytics.log_writer import write_session_log
from streamlit_app.pages.dashboard_queries import (
    escalation_rate,
    query_volume_by_role,
    topic_breakdown,
)


def _seeded_session_factory():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine)
    write_session_log(factory, role="undergraduate", topic_category="citation_help", escalated=False)
    write_session_log(factory, role="undergraduate", topic_category="citation_help", escalated=False)
    write_session_log(factory, role="graduate", topic_category="database_access", escalated=True)
    return factory


def test_query_volume_by_role():
    factory = _seeded_session_factory()
    assert query_volume_by_role(factory) == {"undergraduate": 2, "graduate": 1}


def test_escalation_rate():
    factory = _seeded_session_factory()
    assert escalation_rate(factory) == 1 / 3


def test_topic_breakdown_suppresses_below_min_count():
    factory = _seeded_session_factory()
    breakdown = topic_breakdown(factory, min_count=2)
    assert breakdown == {"citation_help": 2}
    assert "database_access" not in breakdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_queries.py -v`
Expected: FAIL — `streamlit_app.pages.dashboard_queries` does not exist

- [ ] **Step 3: Write `streamlit_app/pages/dashboard_queries.py`**

```python
# streamlit_app/pages/dashboard_queries.py
from collections import Counter

from analytics.db import SessionLog


def query_volume_by_role(session_factory) -> dict[str, int]:
    with session_factory() as session:
        rows = session.query(SessionLog.role).all()
    return dict(Counter(role for (role,) in rows))


def topic_breakdown(session_factory, min_count: int = 5) -> dict[str, int]:
    with session_factory() as session:
        rows = session.query(SessionLog.topic_category).all()
    counts = Counter(topic for (topic,) in rows)
    # Suppress any breakdown below the threshold: even topic-only tagging
    # can be re-identifying in a rare role/course/topic combination
    # (prior CLAUDE.md, "Re-identification in small groups").
    return {topic: count for topic, count in counts.items() if count >= min_count}


def escalation_rate(session_factory) -> float:
    with session_factory() as session:
        rows = session.query(SessionLog.escalated).all()
    if not rows:
        return 0.0
    escalated_count = sum(1 for (escalated,) in rows if escalated)
    return escalated_count / len(rows)
```

- [ ] **Step 4: Write `streamlit_app/pages/1_Staff_Dashboard.py`**

```python
# streamlit_app/pages/1_Staff_Dashboard.py
import os
from typing import Protocol

import streamlit as st
from sqlalchemy.orm import sessionmaker

from analytics.db import get_engine, init_db
from streamlit_app.pages.dashboard_queries import (
    escalation_rate,
    query_volume_by_role,
    topic_breakdown,
)


class AuthGate(Protocol):
    def is_authorized(self) -> bool: ...


class EnvVarAuthGate:
    """Stub gate: UB SSO/Shibboleth vs. a shared staff login is still
    unconfirmed (prior CLAUDE.md, 'Open items'). This checks a shared
    secret from STAFF_DASHBOARD_AUTH_METHOD/.env as a placeholder so the
    dashboard is never reachable on a bare public link. Swap in the real
    gate behind this same AuthGate protocol once the method is confirmed."""

    def is_authorized(self) -> bool:
        expected = os.environ.get("STAFF_DASHBOARD_AUTH_METHOD")
        if not expected:
            return False
        entered = st.text_input("Staff access code", type="password")
        return entered == expected


def render_dashboard(auth_gate: AuthGate, session_factory) -> None:
    st.title("Staff Usage Dashboard")

    if not auth_gate.is_authorized():
        st.warning("Staff access required.")
        st.stop()

    st.subheader("Query volume by role")
    st.bar_chart(query_volume_by_role(session_factory))

    st.subheader("Topic category breakdown")
    st.bar_chart(topic_breakdown(session_factory))

    st.subheader("Escalation rate")
    st.metric("Escalation rate", f"{escalation_rate(session_factory):.1%}")


if __name__ == "__main__":
    engine = get_engine(os.environ.get("DATABASE_URL", "sqlite:///analytics/sessions.db"))
    init_db(engine)
    render_dashboard(EnvVarAuthGate(), sessionmaker(bind=engine))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_dashboard_queries.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/pages/dashboard_queries.py streamlit_app/pages/1_Staff_Dashboard.py tests/test_dashboard_queries.py
git commit -m "feat: add staff dashboard with suppressed small-count breakdowns and swappable auth"
```

---

### Task 21: Evaluation — `evaluation/golden_test_set.json`, `evaluation/run_ragas.py`

**Files:**
- Modify: `evaluation/golden_test_set.json` (currently `[]`), `evaluation/run_ragas.py` (currently empty stub)
- Test: `tests/test_run_ragas.py`

**Interfaces:**
- Produces: golden test set entries shaped `{"role": str, "question": str, "expected_topic": str}`; `evaluation/run_ragas.py`: `load_golden_set(path: str) -> list[dict]`, `run_evaluation(golden_set: list[dict], graph, vectorstore) -> dict` (returns per-role and overall RAGAS-style scores — precision/recall/faithfulness — computed via injected `score_fn` so this task doesn't require a real RAGAS/OpenAI call to test).

- [ ] **Step 1: Seed a few golden test set entries**

```json
// evaluation/golden_test_set.json
[
  {"role": "undergraduate", "question": "How do I find articles about climate change?", "expected_topic": "resource_search"},
  {"role": "graduate", "question": "Cite this book in APA format", "expected_topic": "citation_help"},
  {"role": "faculty", "question": "I want to talk to a librarian about database access", "expected_topic": "escalation"}
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_run_ragas.py
import json
from pathlib import Path

from evaluation.run_ragas import load_golden_set, run_evaluation


def test_load_golden_set_reads_json(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{"role": "undergraduate", "question": "q", "expected_topic": "resource_search"}]))

    entries = load_golden_set(str(path))

    assert entries == [{"role": "undergraduate", "question": "q", "expected_topic": "resource_search"}]


def test_run_evaluation_scores_per_role_and_overall():
    golden_set = [
        {"role": "undergraduate", "question": "q1", "expected_topic": "resource_search"},
        {"role": "graduate", "question": "q2", "expected_topic": "citation_help"},
    ]

    def fake_score_fn(question, expected_topic, graph, vectorstore):
        return {"precision": 1.0, "recall": 1.0, "faithfulness": 1.0}

    result = run_evaluation(golden_set, graph=None, vectorstore=None, score_fn=fake_score_fn)

    assert result["overall"]["precision"] == 1.0
    assert "undergraduate" in result["by_role"]
    assert "graduate" in result["by_role"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_run_ragas.py -v`
Expected: FAIL — `evaluation.run_ragas` has no `load_golden_set`/`run_evaluation`

- [ ] **Step 4: Write `evaluation/run_ragas.py`**

```python
# evaluation/run_ragas.py
import json
from collections import defaultdict


def load_golden_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _default_score_fn(question, expected_topic, graph, vectorstore):
    from ragas import evaluate  # deferred import: not needed for tests

    raise NotImplementedError(
        "wire up real RAGAS scoring here once graph/vectorstore are live; "
        "tests inject a fake score_fn instead of calling this"
    )


def run_evaluation(golden_set: list[dict], graph, vectorstore, score_fn=_default_score_fn) -> dict:
    by_role = defaultdict(list)
    overall = []

    for entry in golden_set:
        scores = score_fn(entry["question"], entry["expected_topic"], graph, vectorstore)
        by_role[entry["role"]].append(scores)
        overall.append(scores)

    def _average(score_list):
        keys = score_list[0].keys()
        return {key: sum(s[key] for s in score_list) / len(score_list) for key in keys}

    return {
        "overall": _average(overall) if overall else {},
        "by_role": {role: _average(scores) for role, scores in by_role.items()},
    }


if __name__ == "__main__":
    golden_set = load_golden_set("evaluation/golden_test_set.json")
    results = run_evaluation(golden_set, graph=None, vectorstore=None)
    print(json.dumps(results, indent=2))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_run_ragas.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add evaluation/golden_test_set.json evaluation/run_ragas.py tests/test_run_ragas.py
git commit -m "feat: add ragas evaluation harness with per-role scoring"
```

---

### Task 22: LangSmith tracing — `evaluation/langsmith_config.py`

**Files:**
- Modify: `evaluation/langsmith_config.py` (currently empty stub)
- Modify: `agent/graph.py:build_graph` (Task 12) — enable tracing when configured
- Test: `tests/test_langsmith_config.py`

**Interfaces:**
- Produces: `configure_langsmith() -> bool` (returns whether tracing was enabled, based on `LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` env vars being present; sets `LANGCHAIN_TRACING_V2`/`LANGCHAIN_PROJECT` env vars, which is how LangGraph/LangChain auto-detect tracing — no code change needed in `graph.py` beyond calling this once at startup).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_langsmith_config.py
import os

from evaluation.langsmith_config import configure_langsmith


def test_configure_langsmith_enables_when_key_present(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "ub-library-navigator")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    enabled = configure_langsmith()

    assert enabled is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "ub-library-navigator"


def test_configure_langsmith_disabled_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    enabled = configure_langsmith()

    assert enabled is False
    assert os.environ.get("LANGCHAIN_TRACING_V2") != "true"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_langsmith_config.py -v`
Expected: FAIL — `configure_langsmith` not defined

- [ ] **Step 3: Write `evaluation/langsmith_config.py`**

```python
# evaluation/langsmith_config.py
import os


def configure_langsmith() -> bool:
    api_key = os.environ.get("LANGSMITH_API_KEY")
    project = os.environ.get("LANGSMITH_PROJECT")
    if not api_key or not project:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    return True
```

- [ ] **Step 4: Call it at startup in `streamlit_app/Home.py`**

Add near the top of `streamlit_app/Home.py`, right after the imports:

```python
from evaluation.langsmith_config import configure_langsmith

configure_langsmith()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_langsmith_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add evaluation/langsmith_config.py streamlit_app/Home.py tests/test_langsmith_config.py
git commit -m "feat: enable langsmith tracing when configured"
```

---

## Self-Review Notes

**Spec coverage against prior CLAUDE.md (`dd388a3`) capabilities:**
1. Role capture (explicit, changeable mid-conversation) → Task 19.
2. OneSearch discovery, search-refine-once → Task 7, invoked from Task 12's `search_and_enrich_node`.
3. RAG enrichment (LibGuides + librarian directory) → Task 8, fed by Tasks 13-17, invoked from Task 12's `search_and_enrich_node`.
4. Librarian recommendation → part of Task 8's `EnrichmentResult`, surfaced in Task 12's `compose_answer`.
5. Citation formatting (APA/MLA/Chicago) → Task 9, used both by Task 11's quick fast-path and folded into Task 12's `compose_answer` for the default path.
6. Escalation/handoff with conversation context → Task 10, triggered two ways in Task 12: Task 11's explicit-request fast-path, or `rank_and_score`'s low-confidence decision.
7. Anonymized conversation logging → Task 18.
8. Staff usage dashboard (role volume, topic breakdown, escalation rate) → Task 20.
Plus: two-stage retrieval architecture → Layer 1 narrows via Tasks 4-5, Layer 2 ranks/enriches/decides via Tasks 7, 8, 12 — with Task 12 now actually calling `fetch_detail()` for only the 1-3 candidates `rank_and_score` selects, closing the gap the previous version of this plan had left open; weekly refresh automation → Task 17 (existing `weekly-refresh.yml` already calls this entrypoint, unchanged); evaluation (RAGAS + LangSmith) → Tasks 21-22; known risks (session-window latency, re-identification, dashboard access control, librarian directory staleness) → addressed inline in Tasks 5, 20, 14/17.

**Placeholder scan:** `ConsoleEscalationBackend` (Task 10) and `EnvVarAuthGate` (Task 20) are real, testable, concrete implementations behind clean interfaces — not TBDs — because the live chat platform and dashboard auth method are genuinely unconfirmed by UB Libraries (prior CLAUDE.md, "Open items still pending confirmation"); this matches that document's own instruction to build these behind swappable interfaces rather than hardcode. `scraper/scrape_librarian_directory.py`'s CSS selectors are flagged explicitly as placeholder markup pending the real page. Task 12's `default_llm_fn` is a real, runnable implementation (LangChain structured output against `gpt-4o-mini`), not a stub — tests exercise `rank_and_score`/`build_graph` via an injected `llm_fn` instead, the same pattern Task 21's `run_ragas.py` already uses for `score_fn`.

**Type consistency:** `SearchCandidate`/`ResourceDetail`/`SearchResult` (Task 1/5) are used with identical field names through Tasks 7-9, 11, 12, 19. `ConversationState` fields defined in Task 12 (`prior_candidates`, `search_result`, `enrichment`, `ranked`, `answer`, `escalated`) match what Task 19's `streamlit_app/Home.py` reads and writes back into `st.session_state.prior_candidates` for the next turn's fast-path check. `RankedResult` fields from Task 12's `compose.py` (`selected_record_ids`, `confidence`, `should_escalate`, `citation_style`) are used identically by `rank_node`, `rank_router`, and `detail_and_compose_node`. `EnrichmentResult` fields from Task 8 match what Task 12's `compose_answer` and Task 19 both read (`librarian_name`, `librarian_email`, `usage_notes`). Chunk metadata keys written in Task 16 (`type`, `source_url`) match what Task 8's `enrich()` reads (`metadata.get("type")`).
