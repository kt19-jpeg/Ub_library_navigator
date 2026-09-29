# RAG Data Pipeline (Layer 2, Slice 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch UB's subject-librarian, Help A-Z, and database-subject pages into structured
records, embed them in Chroma, and add retrieval to the agent graph so an answer can point to
"who can help" and "where to look" even when OneSearch finds nothing.

**Architecture:** A polite fetcher (`scraper/fetch.py`) plus three pure HTML parsers turn the 3
source pages into `Record`s. `pipeline/refresh.py` diffs them against a committed JSON snapshot
(a failed fetch or a suspicious drop in count keeps the old data for that source) and syncs the
result into Chroma (`pipeline/embed.py`, only re-embedding changed records). `agent/rag.py` reads
that collection; the graph runs it in parallel with the OneSearch step (`plan -> {search, rag} ->
compose`), and `compose.py` appends a code-rendered library-guide footer to the answer, including
when OneSearch draws a blank.

**Tech Stack:** Same as slice 1, plus `chromadb` (already listed, first real use), `beautifulsoup4`
(already listed). No new packages.

**Spec:** `docs/superpowers/specs/2026-09-27-rag-data-pipeline-design.md`

## Global Constraints

- Run every command from `Ub_library_navigator/` with the project venv: `.venv/bin/pytest`.
- Sources are fixed to the 3 URLs in `pipeline/sources.py` (from the spec) — no sitemap crawl, no
  following links found on those pages.
- `scraper/fetch.py` is the only thing in this pipeline that makes an HTTP request; parsers are
  pure functions (HTML string in, `list[Record]` out), tested against the real saved pages in
  `tests/fixtures/pages/` (already committed).
- Every fetch obeys `robots.txt`, waits at least 10 seconds between requests to the same host, and
  sends `User-Agent: UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)` (same constant
  Layer 1 uses). **Do not use `urllib.robotparser`** for the disallow check — confirmed live
  2026-09-27, it silently drops a bare trailing "?" from a rule (turns `Disallow: /az.php?` into
  `/az.php`) and would wrongly block the one database page this pipeline needs. Task 2 implements
  the correct literal-prefix check.
- A source's fetch failing, or its parse returning under half its previous record count, must
  never wipe that source's data: the old records for that type are kept and a warning recorded
  (`pipeline/refresh.py`).
- `data/processed/records.json` is committed (sorted by id, so a refresh's diff is readable).
  `vectorstore/` stays gitignored (already is).
- RAG must degrade silently, never crash the chat: no `OPENAI_API_KEY`, an unbuilt/empty Chroma
  collection, or a query failure all result in an empty `RagContext` (`agent/rag.py`,
  `pipeline/embed.py`'s `open_rag_collection`), not an exception.
- Links, names, and locations in the RAG footer are rendered by code from real `Record` metadata,
  never written by the LLM — same rule slice 1 established for OneSearch results
  (`agent/compose.py`'s `render_rag_sections`).
- Tests are offline: no network, no real OpenAI calls. A tiny deterministic `FakeEmbeddingFunction`
  (`tests/fake_embeddings.py`) stands in for the real embedding model; `chromadb` itself runs for
  real in tests (it's a local library, not a network call) so sync/query mechanics are verified
  against its real behavior, not a mock.
- Commit messages are `type: short description`. **Never add a Co-Authored-By or "Generated with
  Claude Code" line.** Commit after each task; push once after the whole slice is reviewed.
- `scraper/scrape_libguides.py`, `scraper/scrape_faqs.py`, `scraper/scrape_librarian_directory.py`,
  and `pipeline/chunk.py` are superseded by this plan (see spec, "Why not chunking") and are
  deleted in Task 3, not built out.
- `agent/router.py`, `agent/escalation.py`, `agent/citation_tool.py`, `analytics/`,
  `streamlit_app/`, `evaluation/` stay untouched.

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/records.py` | `Record` dataclass, slugging, content hashing |
| `pipeline/sources.py` | The 3 source URLs + type |
| `pipeline/snapshot.py` | Load/save `data/processed/records.json` |
| `scraper/fetch.py` | Polite `fetch()`: robots.txt (fixed), crawl delay, identifying UA |
| `scraper/parse_librarians.py` / `parse_help.py` / `parse_databases.py` | HTML -> `list[Record]` |
| `pipeline/embed.py` | Chroma sync, `open_rag_collection()` for callers that want graceful degradation |
| `pipeline/refresh.py` | `refresh_all()`: fetch + sanity-check + snapshot + embed, per source |
| `agent/models.py` (modify) | `RagContext`, reducers for parallel-branch `notes`/`timings` |
| `agent/rag.py` | `query_rag()`: per-type top-k + distance cutoff |
| `agent/compose.py` (modify) | Appends the code-rendered library-guide footer |
| `agent/graph.py` (modify) | `plan -> {search, rag} -> compose`, parallel |
| `agent/cli.py` (modify) | Opens the Chroma collection (if any) and passes it to the graph |
| `agent/rag_live_check.py` | Manual live check (not CI) |
| `tests/fake_embeddings.py` | Deterministic offline embedding function for tests |
| `tests/fixtures/pages/*.html` | Real saved UB pages (already committed) |

---

### Task 1: Records, sources, snapshot

**Files:**
- Create: `pipeline/records.py`, `pipeline/sources.py`, `pipeline/snapshot.py`
- Test: `tests/test_pipeline_records.py`, `tests/test_pipeline_snapshot.py`

**Interfaces:**
- Produces: `Record` (frozen dataclass: `id`, `type`, `text`, `metadata`, `.content_hash`
  property, `.to_dict()`/`.from_dict()`); `LIBRARIAN_SUBJECT`, `HELP_TOPIC`, `DATABASE_SUBJECT`
  type constants; `slugify(text) -> str`; `make_unique_id(prefix, text, seen: set[str]) -> str`;
  `SOURCES: list[dict]` (`{"type": ..., "url": ...}` for the 3 sources); `load_snapshot(path) ->
  list[Record]`, `save_snapshot(records, path) -> None` (sorted by id), `DEFAULT_PATH =
  Path("data/processed/records.json")`.


- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_records.py`:

```python
from pipeline.records import Record, make_unique_id, slugify


def test_slugify_lowercases_and_collapses_punctuation():
    assert slugify("African & African-American Studies") == "african-african-american-studies"


def test_slugify_handles_empty_input():
    assert slugify("!!!") == "x"


def test_make_unique_id_adds_suffix_on_collision():
    seen: set[str] = set()
    first = make_unique_id("help", "Cafe & Hours", seen)
    second = make_unique_id("help", "Cafe: Hours", seen)  # slugifies the same
    assert first == "help:cafe-hours"
    assert second == "help:cafe-hours-2"


def test_content_hash_changes_when_text_or_metadata_changes():
    a = Record(id="x", type="help_topic", text="A", metadata={"url": "u"})
    b = Record(id="x", type="help_topic", text="B", metadata={"url": "u"})
    c = Record(id="x", type="help_topic", text="A", metadata={"url": "v"})
    assert a.content_hash != b.content_hash
    assert a.content_hash != c.content_hash


def test_content_hash_is_stable_regardless_of_metadata_key_order():
    a = Record(id="x", type="t", text="A", metadata={"a": 1, "b": 2})
    b = Record(id="x", type="t", text="A", metadata={"b": 2, "a": 1})
    assert a.content_hash == b.content_hash


def test_round_trips_through_dict():
    r = Record(id="db:x", type="database_subject", text="X (3)", metadata={"count": 3})
    assert Record.from_dict(r.to_dict()) == r
```

Create `tests/test_pipeline_snapshot.py`:

```python
import json

from pipeline.records import Record
from pipeline.snapshot import load_snapshot, save_snapshot


def test_load_missing_file_returns_empty_list(tmp_path):
    assert load_snapshot(tmp_path / "missing.json") == []


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "sub" / "records.json"
    records = [
        Record(id="b:2", type="t", text="B", metadata={}),
        Record(id="a:1", type="t", text="A", metadata={"x": 1}),
    ]

    save_snapshot(records, path)

    assert load_snapshot(path) == sorted(records, key=lambda r: r.id)


def test_save_writes_sorted_by_id_for_stable_diffs(tmp_path):
    path = tmp_path / "records.json"
    save_snapshot(
        [Record(id="z", type="t", text="Z"), Record(id="a", type="t", text="A")], path
    )

    data = json.loads(path.read_text())
    assert [item["id"] for item in data] == ["a", "z"]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_pipeline_records.py tests/test_pipeline_snapshot.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'pipeline.records'` (and `pipeline.snapshot`).

- [ ] **Step 3: Write `pipeline/records.py`**

```python
import hashlib
import json
import re
from dataclasses import dataclass, field

LIBRARIAN_SUBJECT = "librarian_subject"
HELP_TOPIC = "help_topic"
DATABASE_SUBJECT = "database_subject"


@dataclass(frozen=True)
class Record:
    id: str
    type: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = json.dumps({"text": self.text, "metadata": self.metadata}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }

    @staticmethod
    def from_dict(data: dict) -> "Record":
        return Record(id=data["id"], type=data["type"], text=data["text"], metadata=data["metadata"])


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "x"


def make_unique_id(prefix: str, text: str, seen: set[str]) -> str:
    """`prefix:slug`, with a numeric suffix if that id was already used this run
    (e.g. two subjects/titles that slugify to the same string)."""
    base = f"{prefix}:{slugify(text)}"
    candidate = base
    n = 2
    while candidate in seen:
        candidate = f"{base}-{n}"
        n += 1
    seen.add(candidate)
    return candidate
```

- [ ] **Step 4: Write `pipeline/sources.py`**

```python
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT

SOURCES = [
    {
        "type": LIBRARIAN_SUBJECT,
        "url": "https://library.buffalo.edu/help-support/askalibrarian/inperson/bysubject.html",
    },
    {
        "type": HELP_TOPIC,
        "url": "https://library.buffalo.edu/help/browse.html?title=",
    },
    {
        "type": DATABASE_SUBJECT,
        "url": "https://research.lib.buffalo.edu/az.php",
    },
]
```

- [ ] **Step 5: Write `pipeline/snapshot.py`**

```python
import json
from pathlib import Path

from pipeline.records import Record

DEFAULT_PATH = Path("data/processed/records.json")


def load_snapshot(path: Path = DEFAULT_PATH) -> list[Record]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [Record.from_dict(item) for item in data]


def save_snapshot(records: list[Record], path: Path = DEFAULT_PATH) -> None:
    """Writes sorted by id so a refresh's git diff only shows what actually changed."""
    ordered = sorted(records, key=lambda r: r.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in ordered], indent=2) + "\n")
```

- [ ] **Step 6: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_pipeline_records.py tests/test_pipeline_snapshot.py -v
```Expected: 6 + 3 passed.

- [ ] **Step 7: Commit**

```bash
git add pipeline/records.py pipeline/sources.py pipeline/snapshot.py tests/test_pipeline_records.py tests/test_pipeline_snapshot.py
git commit -m "feat: add Record model, source list, and JSON snapshot store"
```

### Task 2: Polite fetcher with a correct robots.txt check

Building and testing this against live UB data (2026-09-27) surfaced a real bug worth fixing
before it's ever relied on: `urllib.robotparser` reconstructs each `Disallow` rule through
`urlparse`/`urlunparse`, which silently drops a bare trailing `"?"`. UB's real
`research.lib.buffalo.edu/robots.txt` has `Disallow: /az.php?` (meaning: block only the
query-string form used by the site's own AJAX calls) — `urllib.robotparser` turns that into the
rule `/az.php`, which would wrongly block the bare page (`research.lib.buffalo.edu/az.php`) this
pipeline needs to fetch. Confirmed live: with the fix below, fetching that exact URL succeeds and
returns 83 real database-subject categories; without it, the fetch is wrongly refused.

**Files:**
- Create: `scraper/fetch.py`
- Test: `tests/test_scraper_fetch.py`

**Interfaces:**
- Produces: `fetch(url, get=requests.get, sleep=time.sleep, now=time.monotonic) -> str` (one
  attempt, no retry); `FetchError(Exception)`; `USER_AGENT`; `MIN_SECONDS_BETWEEN_REQUESTS = 10`.


- [ ] **Step 1: Write the failing tests**

Create `tests/test_scraper_fetch.py`:

```python
from unittest.mock import MagicMock

import pytest

from scraper.fetch import USER_AGENT, FetchError, _last_request_at, _robots_rules_cache, fetch

# Captured live from https://research.lib.buffalo.edu/robots.txt on 2026-09-27.
REAL_ROBOTS_TXT = """User-agent: *
Crawl-delay: 10
    Disallow: /az.php?
    Disallow: /az/databases?
    Disallow: /er.php
"""


@pytest.fixture(autouse=True)
def _clean_caches():
    _robots_rules_cache.clear()
    _last_request_at.clear()
    yield
    _robots_rules_cache.clear()
    _last_request_at.clear()


def _response(status=200, text="ok"):
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


def test_fetches_with_identifying_user_agent_when_no_robots_txt():
    robots_response = _response(status=404)
    page_response = _response(text="<html>hi</html>")
    get = MagicMock(side_effect=[robots_response, page_response])

    result = fetch("https://example.org/page", get=get, sleep=MagicMock(), now=lambda: 0.0)

    assert result == "<html>hi</html>"
    assert get.call_args_list[1].kwargs["headers"]["User-Agent"] == USER_AGENT


def test_a_bare_path_is_allowed_when_only_its_query_form_is_disallowed():
    # Regression test for a real bug: urllib.robotparser silently turns
    # "Disallow: /az.php?" into the rule "/az.php" (it round-trips the rule
    # through urlparse/urlunparse, which drops a bare trailing "?"), which
    # would wrongly block research.lib.buffalo.edu/az.php itself. A correct
    # robots.txt reader only blocks the querystring form.
    robots_response = _response(text=REAL_ROBOTS_TXT)
    page_response = _response(text="<html>the real page</html>")
    get = MagicMock(side_effect=[robots_response, page_response])

    result = fetch(
        "https://research.lib.buffalo.edu/az.php", get=get, sleep=MagicMock(), now=lambda: 0.0
    )

    assert result == "<html>the real page</html>"


def test_the_querystring_form_is_still_disallowed():
    robots_response = _response(text=REAL_ROBOTS_TXT)
    get = MagicMock(return_value=robots_response)

    with pytest.raises(FetchError, match="robots.txt disallows"):
        fetch(
            "https://research.lib.buffalo.edu/az.php?subject=94213",
            get=get, sleep=MagicMock(), now=lambda: 0.0,
        )


def test_a_bare_path_rule_with_no_trailing_question_mark_is_fully_disallowed():
    robots_response = _response(text=REAL_ROBOTS_TXT)
    get = MagicMock(return_value=robots_response)

    with pytest.raises(FetchError, match="robots.txt disallows"):
        fetch("https://research.lib.buffalo.edu/er.php", get=get, sleep=MagicMock(), now=lambda: 0.0)


def test_disallow_rules_only_apply_to_the_wildcard_user_agent_group():
    robots_text = "User-agent: SomeBot\nDisallow: /az.php\n\nUser-agent: *\nDisallow:\n"
    robots_response = _response(text=robots_text)
    page_response = _response(text="ok")
    get = MagicMock(side_effect=[robots_response, page_response])

    result = fetch("https://example.org/az.php", get=get, sleep=MagicMock(), now=lambda: 0.0)

    assert result == "ok"


def test_waits_at_least_ten_seconds_between_requests_to_the_same_host():
    robots_response = _response(status=404)
    page_response = _response()
    get = MagicMock(side_effect=[robots_response, page_response, page_response])
    sleep = MagicMock()
    # now() is called: once to record the first request's time (0.0), once to
    # check elapsed time before the second request (3.0), once to record the
    # second request's time (value unused by this test).
    clock = iter([0.0, 3.0, 3.0])

    fetch("https://example.org/a", get=get, sleep=sleep, now=lambda: next(clock))
    fetch("https://example.org/b", get=get, sleep=sleep, now=lambda: next(clock))

    sleep.assert_called_once()
    assert sleep.call_args.args[0] == pytest.approx(7.0)  # 10s minimum - 3s already elapsed


def test_raises_on_non_200():
    get = MagicMock(side_effect=[_response(status=404), _response(status=500)])
    with pytest.raises(FetchError, match="HTTP 500"):
        fetch("https://example.org/x", get=get, sleep=MagicMock(), now=lambda: 0.0)


def test_raises_on_network_error():
    import requests
    get = MagicMock(side_effect=[_response(status=404), requests.ConnectionError("boom")])
    with pytest.raises(FetchError, match="could not fetch"):
        fetch("https://example.org/x", get=get, sleep=MagicMock(), now=lambda: 0.0)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_scraper_fetch.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'scraper.fetch'`.

- [ ] **Step 3: Write `scraper/fetch.py`**

```python
import time
import urllib.parse

import requests

USER_AGENT = "UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)"
MIN_SECONDS_BETWEEN_REQUESTS = 10
TIMEOUT = 20

_last_request_at: dict[str, float] = {}
_robots_rules_cache: dict[str, list[str]] = {}


class FetchError(Exception):
    """A page could not be fetched (network error, non-200, or robots.txt disallows it)."""


def _parse_disallow_rules(robots_text: str) -> list[str]:
    """Returns the Disallow rule strings from the `User-agent: *` group, verbatim
    (no URL round-trip). Not using `urllib.robotparser`: it reconstructs each rule
    with `urlparse`/`urlunparse`, which silently drops a bare trailing "?" (e.g.
    "Disallow: /az.php?" becomes the rule "/az.php" instead of "/az.php?") -
    confirmed live 2026-09-27 against research.lib.buffalo.edu/robots.txt, whose
    intent is to block only the querystring form ("/az.php?...") and allow the
    bare page ("/az.php") we actually need. A plain prefix match on the rule's
    exact text (per the robots.txt convention) gets this right."""
    rules = []
    in_wildcard_group = False
    for raw_line in robots_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, _, value = line.partition(":")
        field, value = field.strip().lower(), value.strip()
        if field == "user-agent":
            in_wildcard_group = value == "*"
        elif field == "disallow" and in_wildcard_group and value:
            rules.append(value)
    return rules


def _path_and_query(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def _robots_rules_for(host: str, get=requests.get) -> list[str]:
    if host not in _robots_rules_cache:
        try:
            response = get(f"https://{host}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            _robots_rules_cache[host] = _parse_disallow_rules(response.text) if response.status_code == 200 else []
        except requests.RequestException:
            _robots_rules_cache[host] = []
    return _robots_rules_cache[host]


def _wait_for_host(host: str, sleep=time.sleep, now=time.monotonic) -> None:
    last = _last_request_at.get(host)
    if last is not None:
        elapsed = now() - last
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_at[host] = now()


def fetch(url: str, get=requests.get, sleep=time.sleep, now=time.monotonic) -> str:
    """Fetch one page: obeys robots.txt, waits at least MIN_SECONDS_BETWEEN_REQUESTS
    between requests to the same host, sends an identifying User-Agent. One attempt,
    no retry (politeness over completeness). Raises FetchError on any failure."""
    host = urllib.parse.urlparse(url).netloc
    rules = _robots_rules_for(host, get)
    if any(_path_and_query(url).startswith(rule) for rule in rules):
        raise FetchError(f"robots.txt disallows fetching {url}")
    _wait_for_host(host, sleep, now)
    try:
        response = get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise FetchError(f"could not fetch {url}: {exc}") from exc
    if response.status_code != 200:
        raise FetchError(f"{url} returned HTTP {response.status_code}")
    return response.text
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_scraper_fetch.py -v
```Expected: 8 passed, including the two robots.txt regression tests (a bare path allowed when only its query form is disallowed; the query form itself still blocked).

- [ ] **Step 5: Verify live (manual, needs network)**

```bash
.venv/bin/python - <<'EOF'
from scraper.fetch import fetch
html = fetch("https://research.lib.buffalo.edu/az.php")
print("fetched", len(html), "chars")
EOF
```Expected: a few tens of KB of real HTML, no `FetchError`. (If this raises `robots.txt disallows`, the fix in Step 3 isn't in place.)

- [ ] **Step 6: Commit**

```bash
git add scraper/fetch.py tests/test_scraper_fetch.py
git commit -m "feat: add a polite fetcher with a correct robots.txt disallow check"
```

### Task 3: HTML parsers for the three sources

**Files:**
- Create: `scraper/parse_librarians.py`, `scraper/parse_help.py`, `scraper/parse_databases.py`
- Test: `tests/test_scraper_parse_librarians.py`, `tests/test_scraper_parse_help.py`,
  `tests/test_scraper_parse_databases.py`
- Delete: `scraper/scrape_libguides.py`, `scraper/scrape_faqs.py`,
  `scraper/scrape_librarian_directory.py`, `pipeline/chunk.py` (superseded, see spec)

**Interfaces:**
- Consumes: `Record`, `LIBRARIAN_SUBJECT`/`HELP_TOPIC`/`DATABASE_SUBJECT`, `make_unique_id` (Task 1);
  the real fixture pages at `tests/fixtures/pages/{librarians,help,databases}.html` (already
  committed).
- Produces: `extract_librarian_subjects(html: str) -> list[Record]`,
  `extract_help_topics(html: str) -> list[Record]`,
  `extract_database_subjects(html: str) -> list[Record]` (also exports `BROWSE_URL`).


- [ ] **Step 1: Delete the superseded stub files**

```bash
git rm scraper/scrape_libguides.py scraper/scrape_faqs.py scraper/scrape_librarian_directory.py pipeline/chunk.py
```All four are empty (0 bytes) placeholders from the original scaffolding; this pipeline's parsers and `pipeline/refresh.py` replace them.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_scraper_parse_librarians.py`:

```python
from pathlib import Path

from pipeline.records import LIBRARIAN_SUBJECT
from scraper.parse_librarians import extract_librarian_subjects

FIXTURE = (Path(__file__).parent / "fixtures/pages/librarians.html").read_text(errors="ignore")


def test_extracts_many_real_subjects():
    records = extract_librarian_subjects(FIXTURE)
    assert len(records) > 100  # confirmed live 2026-09-25: 154 <tr> rows across 21 letter tables
    assert all(r.type == LIBRARIAN_SUBJECT for r in records)


def test_single_librarian_subject():
    records = extract_librarian_subjects(FIXTURE)
    by_subject = {r.metadata["subject"]: r for r in records}
    art = by_subject["Art"]
    assert art.metadata["librarians"] == [
        {"name": "Kicey, Michael", "profile_url": "https://library.buffalo.edu/staff/mkicey"}
    ]
    assert art.text == "Art subject librarian: Kicey, Michael"


def test_subject_with_multiple_librarians():
    records = extract_librarian_subjects(FIXTURE)
    by_subject = {r.metadata["subject"]: r for r in records}
    anatomy = by_subject["Anatomy"]
    assert [l["name"] for l in anatomy.metadata["librarians"]] == ["Aronoff, Nell", "Mueller, Laurel"]
    assert anatomy.text == "Anatomy subject librarian: Aronoff, Nell, Mueller, Laurel"


def test_ids_are_stable_and_unique():
    records = extract_librarian_subjects(FIXTURE)
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))
    assert "librarian:anatomy" in ids


def test_ignores_rows_with_no_link_or_empty_subject():
    html = "<table><tr><th>Subject Area</th><th>Librarian Name</th></tr><tr><td>Empty</td><td></td></tr></table>"
    assert extract_librarian_subjects(html) == []


def test_empty_html_gives_no_records():
    assert extract_librarian_subjects("<html></html>") == []
```

Create `tests/test_scraper_parse_help.py`:

```python
from pathlib import Path

from pipeline.records import HELP_TOPIC
from scraper.parse_help import extract_help_topics

FIXTURE = (Path(__file__).parent / "fixtures/pages/help.html").read_text(errors="ignore")


def test_extracts_many_real_topics():
    records = extract_help_topics(FIXTURE)
    assert len(records) > 80  # confirmed live 2026-09-25: 100+ topics on the page
    assert all(r.type == HELP_TOPIC for r in records)


def test_a_known_topic_has_the_right_title_and_url():
    by_title = {r.metadata["title"]: r for r in extract_help_topics(FIXTURE)}
    citing = by_title["Citing Sources"]
    assert citing.metadata["url"] == "http://research.lib.buffalo.edu/citingsources"
    assert citing.text == "Citing Sources"
    assert citing.id == "help:citing-sources"


def test_ids_are_unique():
    records = extract_help_topics(FIXTURE)
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))


def test_missing_container_gives_no_records():
    assert extract_help_topics("<html><body>nothing here</body></html>") == []


def test_link_with_no_text_is_skipped():
    html = '<div class="help-browse"><ul><li><a href="https://x"></a></li></ul></div>'
    assert extract_help_topics(html) == []
```

Create `tests/test_scraper_parse_databases.py`:

```python
from pathlib import Path

from pipeline.records import DATABASE_SUBJECT
from scraper.parse_databases import BROWSE_URL, extract_database_subjects

FIXTURE = (Path(__file__).parent / "fixtures/pages/databases.html").read_text(errors="ignore")


def test_extracts_many_real_subjects():
    records = extract_database_subjects(FIXTURE)
    assert len(records) > 50  # confirmed live 2026-09-25: 70+ subject categories
    assert all(r.type == DATABASE_SUBJECT for r in records)


def test_a_known_subject_has_the_right_count_and_text():
    by_subject = {r.metadata["subject"]: r for r in extract_database_subjects(FIXTURE)}
    chemistry = by_subject["Chemistry"]
    assert chemistry.metadata["count"] == 32
    assert chemistry.metadata["browse_url"] == BROWSE_URL
    assert chemistry.text == "Chemistry databases (32 available)"
    assert chemistry.id == "db:chemistry"


def test_all_subjects_placeholder_is_skipped():
    subjects = {r.metadata["subject"] for r in extract_database_subjects(FIXTURE)}
    assert "All Subjects" not in subjects


def test_ids_are_unique():
    records = extract_database_subjects(FIXTURE)
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))


def test_missing_select_gives_no_records():
    assert extract_database_subjects("<html></html>") == []
```

- [ ] **Step 3: Run them to verify they fail**

```bash
.venv/bin/pip install beautifulsoup4
.venv/bin/pytest tests/test_scraper_parse_librarians.py tests/test_scraper_parse_help.py tests/test_scraper_parse_databases.py -v
```Expected: FAIL, `ModuleNotFoundError` for each `scraper.parse_*` module (beautifulsoup4 is already in `requirements.txt` but wasn't yet installed in this venv).

- [ ] **Step 4: Write `scraper/parse_librarians.py`**

```python
from bs4 import BeautifulSoup

from pipeline.records import LIBRARIAN_SUBJECT, Record, make_unique_id


def extract_librarian_subjects(html: str) -> list[Record]:
    """Reads the subject -> librarian(s) table(s) on the "By Subject" ask-a-librarian
    page. A subject with more than one librarian (e.g. "Anatomy") becomes one record
    with multiple entries in metadata["librarians"]."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[Record] = []
    seen_ids: set[str] = set()
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 2:
            continue  # header rows ("Subject Area" / "Librarian Name") have <th>, not <td>
        subject = cells[0].get_text(strip=True)
        links = cells[1].find_all("a")
        if not subject or not links:
            continue
        librarians = [
            {"name": a.get_text(strip=True), "profile_url": a["href"]}
            for a in links
            if a.get_text(strip=True) and a.get("href")
        ]
        if not librarians:
            continue
        names = ", ".join(l["name"] for l in librarians)
        records.append(
            Record(
                id=make_unique_id("librarian", subject, seen_ids),
                type=LIBRARIAN_SUBJECT,
                text=f"{subject} subject librarian: {names}",
                metadata={"subject": subject, "librarians": librarians},
            )
        )
    return records
```

- [ ] **Step 5: Write `scraper/parse_help.py`**

```python
from bs4 import BeautifulSoup

from pipeline.records import HELP_TOPIC, Record, make_unique_id


def extract_help_topics(html: str) -> list[Record]:
    """Reads the Help A-Z topic index (div.help-browse > ul > li > a). Each entry is
    a title linking somewhere else on (or off) the library site; this only indexes
    the title and URL, it does not follow the link."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="help-browse")
    if container is None:
        return []
    records: list[Record] = []
    seen_ids: set[str] = set()
    for link in container.find_all("a"):
        title = link.get_text(strip=True)
        url = link.get("href")
        if not title or not url:
            continue
        records.append(
            Record(
                id=make_unique_id("help", title, seen_ids),
                type=HELP_TOPIC,
                text=title,
                metadata={"title": title, "url": url},
            )
        )
    return records
```

- [ ] **Step 6: Write `scraper/parse_databases.py`**

```python
import re

from bs4 import BeautifulSoup

from pipeline.records import DATABASE_SUBJECT, Record, make_unique_id

BROWSE_URL = "https://research.lib.buffalo.edu/az.php"
_COUNT_RE = re.compile(r"^(?P<subject>.+?)\s*\((?P<count>\d+)\)$")


def extract_database_subjects(html: str) -> list[Record]:
    """Reads the "Database Subject Filter" <select> on the A-Z Databases page:
    each <option> is "<subject> (<count>)". The individual database list itself
    is loaded by JavaScript from an endpoint robots.txt disallows, so only the
    subject categories and counts are indexed; BROWSE_URL is the one link given
    to the user to look at the actual databases."""
    soup = BeautifulSoup(html, "html.parser")
    select = soup.find("select", id="s-lg-sel-subjects")
    if select is None:
        return []
    records: list[Record] = []
    seen_ids: set[str] = set()
    for option in select.find_all("option"):
        label = option.get_text(strip=True)
        match = _COUNT_RE.match(label)
        if not match:
            continue  # e.g. "All Subjects", which has no count
        subject = match.group("subject")
        count = int(match.group("count"))
        records.append(
            Record(
                id=make_unique_id("db", subject, seen_ids),
                type=DATABASE_SUBJECT,
                text=f"{subject} databases ({count} available)",
                metadata={"subject": subject, "count": count, "browse_url": BROWSE_URL},
            )
        )
    return records
```

- [ ] **Step 7: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_scraper_parse_librarians.py tests/test_scraper_parse_help.py tests/test_scraper_parse_databases.py -v
```Expected: 6 + 5 + 5 = 16 passed, against the real saved pages.

- [ ] **Step 8: Commit**

```bash
git add -u scraper pipeline
git add scraper/parse_librarians.py scraper/parse_help.py scraper/parse_databases.py tests/test_scraper_parse_librarians.py tests/test_scraper_parse_help.py tests/test_scraper_parse_databases.py
git commit -m "feat: add HTML parsers for librarians, help topics, and database subjects"
```

### Task 4: Chroma sync (`pipeline/embed.py`)

**Files:**
- Create: `pipeline/embed.py`, `tests/fake_embeddings.py`
- Test: `tests/test_pipeline_embed.py`, `tests/test_pipeline_embed_open.py`

**Interfaces:**
- Consumes: `Record`, `LIBRARIAN_SUBJECT`/`HELP_TOPIC`/`DATABASE_SUBJECT` (Task 1).
- Produces: `COLLECTION_NAME = "records"`; `EmbeddingUnavailableError(Exception)`;
  `default_embedding_function()` (raises `EmbeddingUnavailableError` if `OPENAI_API_KEY` unset);
  `get_collection(persist_dir, embedding_function)`; `sync_collection(records, collection) ->
  SyncResult` (`added`, `updated`, `deleted`, `unchanged`); `open_rag_collection(persist_dir=
  "vectorstore")` (returns `None`, not the collection, if there's no usable key or nothing has
  been embedded yet — this is what lets the chat CLI run with no OpenAI key at all).
- `tests/fake_embeddings.py`'s `FakeEmbeddingFunction` is a deterministic, offline stand-in for
  the real embedding model, reused by every later test that needs a Chroma collection.


- [ ] **Step 1: Write the failing tests**

Create `tests/fake_embeddings.py` (test helper, no test of its own):

```python
import hashlib

from chromadb.api.types import Documents, EmbeddingFunction


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic, offline stand-in for a real embedding model: same text always
    gives the same vector, and chromadb never makes a network call in tests."""

    def __call__(self, input: Documents) -> list[list[float]]:
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            vectors.append([b / 255 for b in digest])
        return vectors

    @staticmethod
    def name() -> str:
        return "fake"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FakeEmbeddingFunction":
        return FakeEmbeddingFunction()

    @staticmethod
    def is_legacy() -> bool:
        return False
```

Create `tests/test_pipeline_embed.py`:

```python
import chromadb

from pipeline.embed import COLLECTION_NAME, sync_collection
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, Record
from tests.fake_embeddings import FakeEmbeddingFunction


def _collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=FakeEmbeddingFunction())


def test_first_sync_adds_every_record(tmp_path):
    collection = _collection(tmp_path)
    records = [Record(id="help:a", type=HELP_TOPIC, text="A", metadata={"url": "u"})]

    result = sync_collection(records, collection)

    assert (result.added, result.updated, result.deleted, result.unchanged) == (1, 0, 0, 0)
    assert collection.count() == 1


def test_second_sync_with_no_changes_touches_nothing(tmp_path):
    collection = _collection(tmp_path)
    records = [Record(id="help:a", type=HELP_TOPIC, text="A", metadata={"url": "u"})]
    sync_collection(records, collection)

    result = sync_collection(records, collection)

    assert (result.added, result.updated, result.deleted, result.unchanged) == (0, 0, 0, 1)


def test_changed_text_is_updated_not_readded(tmp_path):
    collection = _collection(tmp_path)
    sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})], collection)

    result = sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A changed", metadata={})], collection)

    assert (result.added, result.updated) == (0, 1)
    assert collection.get(ids=["help:a"])["documents"] == ["A changed"]


def test_record_missing_from_new_list_is_deleted(tmp_path):
    collection = _collection(tmp_path)
    sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})], collection)

    result = sync_collection([], collection)

    assert result.deleted == 1
    assert collection.count() == 0


def test_only_managed_types_are_touched(tmp_path):
    collection = _collection(tmp_path)
    collection.upsert(
        ids=["other:1"], documents=["unrelated"], metadatas=[{"type": "some_future_type", "content_hash": "x"}]
    )

    result = sync_collection([], collection)

    assert result.deleted == 0
    assert collection.count() == 1  # the unrelated record survives


def test_syncs_two_managed_types_independently(tmp_path):
    collection = _collection(tmp_path)
    records = [
        Record(id="help:a", type=HELP_TOPIC, text="A", metadata={}),
        Record(id="db:b", type=DATABASE_SUBJECT, text="B", metadata={}),
    ]

    result = sync_collection(records, collection)

    assert result.added == 2
    assert collection.count() == 2
```

Create `tests/test_pipeline_embed_open.py`:

```python
from pipeline.embed import COLLECTION_NAME, get_collection, open_rag_collection, sync_collection
from pipeline.records import HELP_TOPIC, Record
from tests.fake_embeddings import FakeEmbeddingFunction


def test_returns_none_when_no_openai_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert open_rag_collection(str(tmp_path)) is None


def test_returns_none_when_collection_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("pipeline.embed.default_embedding_function", lambda: FakeEmbeddingFunction())
    assert open_rag_collection(str(tmp_path)) is None


def test_returns_the_collection_once_it_has_data(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("pipeline.embed.default_embedding_function", lambda: FakeEmbeddingFunction())
    collection = get_collection(str(tmp_path), FakeEmbeddingFunction())
    sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})], collection)

    opened = open_rag_collection(str(tmp_path))

    assert opened is not None
    assert opened.name == COLLECTION_NAME
    assert opened.count() == 1
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_pipeline_embed.py tests/test_pipeline_embed_open.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'pipeline.embed'` (`chromadb` is already in `requirements.txt`; confirm it's installed: `.venv/bin/pip show chromadb`, install if not).

- [ ] **Step 3: Write `pipeline/embed.py`**

```python
import json
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record

COLLECTION_NAME = "records"
MANAGED_TYPES = (LIBRARIAN_SUBJECT, HELP_TOPIC, DATABASE_SUBJECT)


class EmbeddingUnavailableError(Exception):
    """No usable embedding backend (e.g. OPENAI_API_KEY is not set)."""


@dataclass(frozen=True)
class SyncResult:
    added: int
    updated: int
    deleted: int
    unchanged: int


def default_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingUnavailableError("OPENAI_API_KEY is not set")
    return OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")


def get_collection(persist_dir: str, embedding_function):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_function)


def open_rag_collection(persist_dir: str = "vectorstore"):
    """For callers (the CLI, the live check) that want RAG if it's available and
    silent degradation if not: returns the collection, or None if there's no
    OPENAI_API_KEY or the collection has never been populated (run refresh_all()
    first)."""
    try:
        collection = get_collection(persist_dir, default_embedding_function())
    except EmbeddingUnavailableError:
        return None
    if collection.count() == 0:
        return None
    return collection


def _to_chroma_metadata(record: Record) -> dict:
    """Chroma metadata values must be scalars; a Record's metadata can hold nested
    lists/dicts (e.g. librarian_subject's "librarians"), so those are JSON-encoded."""
    encoded = {}
    for key, value in record.metadata.items():
        encoded[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    encoded["type"] = record.type
    encoded["content_hash"] = record.content_hash
    return encoded


def sync_collection(records: list[Record], collection) -> SyncResult:
    """Adds new/changed records, deletes managed-type records no longer present,
    and skips records whose content_hash hasn't changed (no re-embedding cost)."""
    new_by_id = {r.id: r for r in records}

    existing = collection.get(where={"type": {"$in": list(MANAGED_TYPES)}})
    existing_hash = {
        id_: meta.get("content_hash")
        for id_, meta in zip(existing["ids"], existing["metadatas"])
    }

    to_delete = [id_ for id_ in existing_hash if id_ not in new_by_id]
    to_write = [r for r in records if existing_hash.get(r.id) != r.content_hash]
    unchanged = len(records) - len(to_write)
    added = sum(1 for r in to_write if r.id not in existing_hash)
    updated = len(to_write) - added

    if to_delete:
        collection.delete(ids=to_delete)
    if to_write:
        collection.upsert(
            ids=[r.id for r in to_write],
            documents=[r.text for r in to_write],
            metadatas=[_to_chroma_metadata(r) for r in to_write],
        )
    return SyncResult(added=added, updated=updated, deleted=len(to_delete), unchanged=unchanged)
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_pipeline_embed.py tests/test_pipeline_embed_open.py -v
```Expected: 6 + 3 passed. A `DeprecationWarning: ... does not implement __init__` and a `legacy embedding function config: ... 'default_space'` warning from chromadb are expected and harmless — `FakeEmbeddingFunction` is intentionally minimal; they do not fail the run.

- [ ] **Step 5: Commit**

```bash
git add pipeline/embed.py tests/fake_embeddings.py tests/test_pipeline_embed.py tests/test_pipeline_embed_open.py
git commit -m "feat: add Chroma sync with change-hash skipping and graceful no-key degradation"
```

### Task 5: Refresh orchestration (`pipeline/refresh.py`)

**Files:**
- Create: `pipeline/refresh.py`
- Test: `tests/test_pipeline_refresh.py`

**Interfaces:**
- Consumes: `SOURCES` (Task 1); `load_snapshot`/`save_snapshot`/`DEFAULT_PATH` (Task 1);
  `fetch`/`FetchError` (Task 2); `extract_librarian_subjects`/`extract_help_topics`/
  `extract_database_subjects` (Task 3); `get_collection`/`default_embedding_function`/
  `sync_collection`/`EmbeddingUnavailableError` (Task 4).
- Produces: `refresh_all(sources=SOURCES, fetch_fn=fetch, snapshot_path=DEFAULT_PATH,
  persist_dir="vectorstore", embedding_function=None) -> RefreshResult` (`counts: dict[str, int]`,
  `warnings: list[str]`). This is the entrypoint `.github/workflows/weekly-refresh.yml` calls
  (Task 9) and what `python pipeline/refresh.py` runs manually.

A note on a bug this task's tests catch if reintroduced: parsers must be looked up by name at
call time (`_parser_for`), not captured once into a module-level dict — a dict built at import
time keeps its original function references even after a test monkeypatches
`pipeline.refresh.extract_help_topics` etc., silently making that monkeypatch a no-op.


- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_refresh.py`:

```python
from pipeline.records import HELP_TOPIC, LIBRARIAN_SUBJECT, Record
from tests.fake_embeddings import FakeEmbeddingFunction
from pipeline.refresh import refresh_all
from scraper.fetch import FetchError


def _sources():
    return [
        {"type": LIBRARIAN_SUBJECT, "url": "https://a/librarians"},
        {"type": HELP_TOPIC, "url": "https://a/help"},
    ]


def _snapshot_with(path, records):
    from pipeline.snapshot import save_snapshot
    save_snapshot(records, path)


def test_first_run_with_no_prior_snapshot_writes_parsed_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.refresh.extract_librarian_subjects",
        lambda html: [Record(id="librarian:a", type=LIBRARIAN_SUBJECT, text="a", metadata={})],
    )
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:a", type=HELP_TOPIC, text="a", metadata={})],
    )
    result = refresh_all(
        sources=_sources(), fetch_fn=lambda url: "<html></html>",
        snapshot_path=tmp_path / "records.json", persist_dir=str(tmp_path / "vectorstore"),
        embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts == {LIBRARIAN_SUBJECT: 1, HELP_TOPIC: 1, "database_subject": 0}
    assert result.warnings == []


def test_fetch_failure_keeps_that_types_old_records_and_warns(tmp_path, monkeypatch):
    path = tmp_path / "records.json"
    old = [Record(id="librarian:a", type=LIBRARIAN_SUBJECT, text="a", metadata={})]
    _snapshot_with(path, old)
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:new", type=HELP_TOPIC, text="new", metadata={})],
    )

    def flaky_fetch(url):
        if "librarians" in url:
            raise FetchError("HTTP 500")
        return "<html></html>"

    result = refresh_all(
        sources=_sources(), fetch_fn=flaky_fetch, snapshot_path=path,
        persist_dir=str(tmp_path / "vectorstore"), embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts[LIBRARIAN_SUBJECT] == 1  # kept, not lost
    assert result.counts[HELP_TOPIC] == 1  # the working source still updated
    assert any("fetch failed" in w for w in result.warnings)

    from pipeline.snapshot import load_snapshot
    expected = old + [Record(id="help:new", type=HELP_TOPIC, text="new", metadata={})]
    assert sorted(load_snapshot(path), key=lambda r: r.id) == sorted(expected, key=lambda r: r.id)


def test_suspicious_drop_in_count_keeps_old_records_and_warns(tmp_path, monkeypatch):
    path = tmp_path / "records.json"
    old = [Record(id=f"help:{i}", type=HELP_TOPIC, text=str(i), metadata={}) for i in range(10)]
    _snapshot_with(path, old)
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:0", type=HELP_TOPIC, text="0", metadata={})],  # 1 of 10: looks broken
    )
    monkeypatch.setattr("pipeline.refresh.extract_librarian_subjects", lambda html: [])

    result = refresh_all(
        sources=[{"type": HELP_TOPIC, "url": "https://a/help"}],
        fetch_fn=lambda url: "<html></html>", snapshot_path=path,
        persist_dir=str(tmp_path / "vectorstore"), embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts[HELP_TOPIC] == 10  # old data kept
    assert any("possible layout change" in w for w in result.warnings)


def test_a_real_update_fully_replaces_removed_entries(tmp_path, monkeypatch):
    path = tmp_path / "records.json"
    old = [Record(id=f"help:{i}", type=HELP_TOPIC, text=str(i), metadata={}) for i in range(4)]
    _snapshot_with(path, old)
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:0", type=HELP_TOPIC, text="0", metadata={}),
                       Record(id="help:1", type=HELP_TOPIC, text="1", metadata={})],  # 2 of 4: not suspicious (>=50%)
    )
    monkeypatch.setattr("pipeline.refresh.extract_librarian_subjects", lambda html: [])

    result = refresh_all(
        sources=[{"type": HELP_TOPIC, "url": "https://a/help"}],
        fetch_fn=lambda url: "<html></html>", snapshot_path=path,
        persist_dir=str(tmp_path / "vectorstore"), embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts[HELP_TOPIC] == 2  # help:2 and help:3 really removed
    assert result.warnings == []


def test_missing_embedding_key_still_writes_the_snapshot(tmp_path, monkeypatch):
    from pipeline.embed import EmbeddingUnavailableError

    monkeypatch.setattr("pipeline.refresh.extract_librarian_subjects", lambda html: [])
    monkeypatch.setattr("pipeline.refresh.extract_help_topics", lambda html: [])
    monkeypatch.setattr(
        "pipeline.refresh.default_embedding_function",
        lambda: (_ for _ in ()).throw(EmbeddingUnavailableError("OPENAI_API_KEY is not set")),
    )

    result = refresh_all(
        sources=_sources(), fetch_fn=lambda url: "<html></html>",
        snapshot_path=tmp_path / "records.json", persist_dir=str(tmp_path / "vectorstore"),
        embedding_function=None,
    )

    assert any("embedding sync skipped" in w for w in result.warnings)
    from pipeline.snapshot import load_snapshot
    assert load_snapshot(tmp_path / "records.json") == []
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_pipeline_refresh.py -v
```Expected: FAIL, `ModuleNotFoundError: No module named 'pipeline.refresh'`.

- [ ] **Step 3: Write `pipeline/refresh.py`**

```python
from dataclasses import dataclass, field

from pipeline.embed import EmbeddingUnavailableError, default_embedding_function, get_collection, sync_collection
from pipeline.records import Record
from pipeline.snapshot import DEFAULT_PATH, load_snapshot, save_snapshot
from pipeline.sources import SOURCES
from scraper.fetch import FetchError, fetch
from scraper.parse_databases import extract_database_subjects
from scraper.parse_help import extract_help_topics
from scraper.parse_librarians import extract_librarian_subjects

MIN_KEPT_FRACTION = 0.5

def _parser_for(record_type: str):
    # Looked up by name at call time (not captured into a dict at import time) so
    # that tests can monkeypatch e.g. pipeline.refresh.extract_help_topics.
    return {
        "librarian_subject": extract_librarian_subjects,
        "help_topic": extract_help_topics,
        "database_subject": extract_database_subjects,
    }[record_type]


RECORD_TYPES = ("librarian_subject", "help_topic", "database_subject")


@dataclass(frozen=True)
class RefreshResult:
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _update_one_source(source: dict, old_by_type: dict[str, list[Record]], fetch_fn, warnings: list[str]) -> list[Record]:
    """Returns the records this source should contribute to the new snapshot: the
    freshly parsed ones if the fetch+parse looks trustworthy, otherwise the old
    ones unchanged (a failed fetch or a suspicious drop in count never wipes data)."""
    record_type = source["type"]
    old_records = old_by_type.get(record_type, [])
    try:
        html = fetch_fn(source["url"])
    except FetchError as exc:
        warnings.append(f"{record_type}: fetch failed ({exc}), keeping previous data")
        return old_records

    new_records = _parser_for(record_type)(html)
    if old_records and len(new_records) < MIN_KEPT_FRACTION * len(old_records):
        warnings.append(
            f"{record_type}: parsed {len(new_records)} records, down from {len(old_records)} "
            "(possible layout change), keeping previous data"
        )
        return old_records
    return new_records


def refresh_all(
    sources: list[dict] = SOURCES,
    fetch_fn=fetch,
    snapshot_path=DEFAULT_PATH,
    persist_dir: str = "vectorstore",
    embedding_function=None,
) -> RefreshResult:
    """Fetches every source independently, keeps old data for any source that
    fails or looks broken, writes the merged snapshot, and syncs it to Chroma.
    A missing/broken embedding backend (e.g. no OPENAI_API_KEY) still lets the
    snapshot update; only the Chroma sync step is skipped, with a warning."""
    old_records = load_snapshot(snapshot_path)
    old_by_type: dict[str, list[Record]] = {}
    for record in old_records:
        old_by_type.setdefault(record.type, []).append(record)

    warnings: list[str] = []
    new_records: list[Record] = []
    for source in sources:
        new_records.extend(_update_one_source(source, old_by_type, fetch_fn, warnings))

    save_snapshot(new_records, snapshot_path)

    try:
        collection = get_collection(
            persist_dir=persist_dir, embedding_function=embedding_function or default_embedding_function()
        )
        sync_collection(new_records, collection)
    except EmbeddingUnavailableError as exc:
        warnings.append(f"embedding sync skipped: {exc}")

    counts = {record_type: 0 for record_type in RECORD_TYPES}
    for record in new_records:
        counts[record.type] = counts.get(record.type, 0) + 1
    return RefreshResult(counts=counts, warnings=warnings)
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_pipeline_refresh.py -v
```Expected: 5 passed.

- [ ] **Step 5: Run it live (manual, needs network; takes ~11s for the 10s crawl delay)**

```bash
.venv/bin/python - <<'EOF'
from pipeline.refresh import refresh_all
result = refresh_all(embedding_function=object())  # placeholder: skip real embedding for this check
print(result.counts, result.warnings)
EOF
git status --short data/processed/records.json
wc -l data/processed/records.json
```
Expected: `counts` around `{'librarian_subject': ~130, 'help_topic': ~155, 'database_subject':
~80}` (confirmed live 2026-09-27: 132/157/83), `warnings == []`, and `data/processed/records.json`
now exists/changed. The `embedding_function=object()` placeholder will make the Chroma sync step
itself raise inside `get_collection`/`sync_collection` (chromadb validates the embedding
function's shape) — that's fine for this check, which is only confirming fetch+parse+snapshot;
Task 6's live check exercises the real embedding path with a real `OPENAI_API_KEY`. If this
raises before printing `counts`, stop and re-check Task 2's live verification first.

- [ ] **Step 6: Commit**

```bash
git add pipeline/refresh.py tests/test_pipeline_refresh.py data/processed/records.json
git commit -m "feat: add refresh orchestration with per-source fetch/sanity-check/keep-old-on-failure"
```

### Task 6: `RagContext` and retrieval (`agent/rag.py`)

**Files:**
- Modify: `agent/models.py`
- Create: `agent/rag.py`
- Test: `tests/test_agent_rag_context.py`, `tests/test_agent_rag.py`

**Interfaces:**
- Consumes: `Record` (Task 1); a Chroma collection (Task 4/5's `get_collection`/`sync_collection`).
- Produces: `RagContext` (pydantic model: `librarians`/`help_topics`/`databases: list[Record]`,
  `.is_empty()`); `AgentState.rag_context` is now typed `RagContext` (was `list`);
  `AgentState.notes`/`.timings` gain reducers so parallel graph branches (Task 8) can both write
  them in the same step without LangGraph raising a conflict (confirmed with a minimal real
  LangGraph program: without a reducer, two nodes writing the same key in one step raises
  `InvalidUpdateError`; `Annotated[list[str], operator.add]` and a small dict-merge reducer fix
  it). `query_rag(keywords: str, collection) -> tuple[RagContext, list[str]]` (never raises: a
  broken collection or missing key gives an empty context + a note); `RAG_TOP_K`,
  `RAG_DISTANCE_CUTOFF` (per-type, starting values from the spec, tune with Task 9's live check).


- [ ] **Step 1: Modify `agent/models.py`**

Four changes to the existing file:

1. Replace the first import line:

```python
from typing import Literal, TypedDict
```
with:

```python
import operator
from typing import Annotated, Literal, TypedDict
```

2. Add this import (after the existing `from onesearch.models import ...` line):

```python
from pipeline.records import Record
```

3. Add this class right before `class AgentState(TypedDict, total=False):`:

```python
class RagContext(BaseModel):
    """Library-guide hits added alongside OneSearch results. Each list is already
    filtered by distance cutoff and capped, sorted best-first."""

    model_config = {"arbitrary_types_allowed": True}

    librarians: list[Record] = Field(default_factory=list)
    help_topics: list[Record] = Field(default_factory=list)
    databases: list[Record] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.librarians or self.help_topics or self.databases)


def merge_timings(current: dict, new: dict) -> dict:
    return {**current, **new}

```

4. In `AgentState`, change these three field declarations:

```python
    rag_context: list
    ...
    notes: list[str]
    timings: dict[str, float]
```
to:

```python
    rag_context: RagContext
    ...
    notes: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], merge_timings]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_agent_rag_context.py`:

```python
from agent.models import RagContext
from pipeline.records import HELP_TOPIC, Record


def test_default_is_empty():
    assert RagContext().is_empty() is True


def test_not_empty_when_any_list_has_a_hit():
    ctx = RagContext(help_topics=[Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})])
    assert ctx.is_empty() is False
```

Create `tests/test_agent_rag.py`:

```python
import chromadb

from agent.models import RagContext
from agent.rag import query_rag
from pipeline.embed import COLLECTION_NAME, sync_collection
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record
from tests.fake_embeddings import FakeEmbeddingFunction


def _collection_with(tmp_path, records):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=FakeEmbeddingFunction())
    sync_collection(records, collection)
    return collection


def _lenient_cutoffs(monkeypatch):
    # The fake embedding function isn't semantic, so real cosine distances between
    # its vectors don't reflect real similarity. These tests check plumbing (does a
    # hit round-trip correctly, across all three types), not the cutoff's numeric
    # value, so the cutoff is raised high enough that nothing gets dropped.
    import agent.rag as rag_module
    for record_type in rag_module.RAG_DISTANCE_CUTOFF:
        monkeypatch.setitem(rag_module.RAG_DISTANCE_CUTOFF, record_type, 999.0)


def test_query_returns_hits_for_all_three_types(tmp_path, monkeypatch):
    _lenient_cutoffs(monkeypatch)
    records = [
        Record(id="librarian:anatomy", type=LIBRARIAN_SUBJECT, text="anatomy",
               metadata={"subject": "Anatomy", "librarians": [{"name": "Aronoff, Nell", "profile_url": "https://x"}]}),
        Record(id="help:citing-sources", type=HELP_TOPIC, text="Citing Sources",
               metadata={"title": "Citing Sources", "url": "https://y"}),
        Record(id="db:chemistry", type=DATABASE_SUBJECT, text="Chemistry databases (32 available)",
               metadata={"subject": "Chemistry", "count": 32, "browse_url": "https://z"}),
    ]
    collection = _collection_with(tmp_path, records)

    context, notes = query_rag("anatomy", collection)

    assert notes == []
    assert isinstance(context, RagContext)
    assert [r.id for r in context.librarians] == ["librarian:anatomy"]
    assert [r.id for r in context.help_topics] == ["help:citing-sources"]
    assert [r.id for r in context.databases] == ["db:chemistry"]


def test_hits_carry_back_their_original_metadata_including_nested_lists(tmp_path, monkeypatch):
    _lenient_cutoffs(monkeypatch)
    records = [
        Record(id="help:a", type=HELP_TOPIC, text="A", metadata={"title": "A", "url": "https://a"}),
        Record(id="librarian:anatomy", type=LIBRARIAN_SUBJECT, text="anatomy",
               metadata={"subject": "Anatomy", "librarians": [{"name": "Aronoff, Nell", "profile_url": "https://x"}]}),
    ]
    collection = _collection_with(tmp_path, records)

    context, _ = query_rag("a", collection)

    assert context.help_topics[0].metadata == {"title": "A", "url": "https://a"}
    assert context.librarians[0].metadata == {
        "subject": "Anatomy", "librarians": [{"name": "Aronoff, Nell", "profile_url": "https://x"}]
    }


def test_distant_hits_are_dropped_by_the_cutoff(tmp_path, monkeypatch):
    import agent.rag as rag_module
    monkeypatch.setitem(rag_module.RAG_DISTANCE_CUTOFF, HELP_TOPIC, -1.0)  # nothing can pass
    records = [Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})]
    collection = _collection_with(tmp_path, records)

    context, _ = query_rag("a", collection)

    assert context.help_topics == []


def test_empty_collection_gives_an_empty_context(tmp_path):
    collection = _collection_with(tmp_path, [])

    context, notes = query_rag("anything", collection)

    assert context.is_empty()
    assert notes == []


def test_query_failure_degrades_to_empty_context_with_a_note():
    class BrokenCollection:
        def query(self, **kwargs):
            raise RuntimeError("no embedding function configured")

    context, notes = query_rag("anything", BrokenCollection())

    assert context.is_empty()
    assert notes == ["Library guide lookup failed."]


def test_embedding_unavailable_gives_its_own_note():
    from pipeline.embed import EmbeddingUnavailableError

    class NoKeyCollection:
        def query(self, **kwargs):
            raise EmbeddingUnavailableError("OPENAI_API_KEY is not set")

    context, notes = query_rag("anything", NoKeyCollection())

    assert context.is_empty()
    assert notes == ["Library guide lookup was unavailable."]
```

- [ ] **Step 3: Run them to verify they fail**

```bash
.venv/bin/pytest tests/test_agent_rag_context.py tests/test_agent_rag.py -v
```Expected: FAIL — `test_agent_rag_context.py` fails on `RagContext` not existing yet; `test_agent_rag.py` fails on `ModuleNotFoundError: No module named 'agent.rag'`.

- [ ] **Step 4: Write `agent/rag.py`**

```python
import json

from agent.models import RagContext
from pipeline.embed import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, EmbeddingUnavailableError
from pipeline.records import Record

RAG_TOP_K = {
    LIBRARIAN_SUBJECT: 2,
    HELP_TOPIC: 3,
    DATABASE_SUBJECT: 2,
}
# Cosine distance from text-embedding-3-small; a starting point, tuned against real
# queries with agent/rag_live_check.py before this slice is considered done.
RAG_DISTANCE_CUTOFF = {
    LIBRARIAN_SUBJECT: 0.5,
    HELP_TOPIC: 0.5,
    DATABASE_SUBJECT: 0.5,
}


def _query_type(collection, keywords: str, record_type: str) -> list[Record]:
    result = collection.query(
        query_texts=[keywords], n_results=RAG_TOP_K[record_type], where={"type": record_type}
    )
    cutoff = RAG_DISTANCE_CUTOFF[record_type]
    hits = []
    if result["ids"]:
        for id_, metadata, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0]):
            if distance >= cutoff:
                continue
            record_metadata = {}
            for key, value in metadata.items():
                if key in ("type", "content_hash"):
                    continue
                if isinstance(value, str) and value[:1] in "[{":
                    try:
                        value = json.loads(value)
                    except ValueError:
                        pass
                record_metadata[key] = value
            hits.append(Record(id=id_, type=record_type, text="", metadata=record_metadata))
    return hits


def query_rag(keywords: str, collection) -> tuple[RagContext, list[str]]:
    """Looks up the librarian/help/database library guide for `keywords`. Never raises:
    if the collection can't be queried (e.g. no OPENAI_API_KEY for the embedding
    function), returns an empty RagContext and a note instead."""
    try:
        return (
            RagContext(
                librarians=_query_type(collection, keywords, LIBRARIAN_SUBJECT),
                help_topics=_query_type(collection, keywords, HELP_TOPIC),
                databases=_query_type(collection, keywords, DATABASE_SUBJECT),
            ),
            [],
        )
    except EmbeddingUnavailableError:
        return RagContext(), ["Library guide lookup was unavailable."]
    except Exception:
        return RagContext(), ["Library guide lookup failed."]
```

- [ ] **Step 5: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_agent_rag_context.py tests/test_agent_rag.py tests/test_agent_models.py -v
```Expected: 2 + 6 + 5 passed (`test_agent_models.py` is the slice-1 file; it must still pass unchanged after this modification).

- [ ] **Step 6: Commit**

```bash
git add agent/models.py agent/rag.py tests/test_agent_rag_context.py tests/test_agent_rag.py
git commit -m "feat: add RagContext and per-type distance-filtered retrieval"
```

### Task 7: Library-guide footer in the answer (`agent/compose.py`)

**Files:**
- Modify: `agent/compose.py` (full replacement, see below)
- Modify: `tests/test_agent_compose.py`

**Interfaces:**
- Consumes: `RagContext` (Task 6).
- Produces: `render_rag_sections(rag_context: RagContext) -> list[str]` (the "Subject librarian" /
  "Helpful pages" / "Browse databases" lines, code-rendered from real `Record` metadata, never
  from the LLM); `select_results(...)` and `render_answer(...)`/`_render_no_answer(...)` now take
  a `RagContext` argument (was a bare `list`); `compose(state, model, client)`'s behavior is
  otherwise unchanged (same return keys), except the answer text may now have a trailing
  library-guide footer, present in both the normal-answer and no-answer/low-confidence paths.


- [ ] **Step 1: Update the existing tests first**

In `tests/test_agent_compose.py`, change the import line:

```python
from agent.models import QueryPlan, Selection, SelectionItem
```
to:

```python
from agent.models import QueryPlan, RagContext, Selection, SelectionItem
```

In `_state()`, change `"rag_context": []` to `"rag_context": RagContext()`.

In `test_select_results_sends_truncated_candidate_json_and_role`, change the call

```python
select_results(model, "faculty", "the request", [_candidate("a1", description=long)], [])
```
to:

```python
select_results(model, "faculty", "the request", [_candidate("a1", description=long)], RagContext())
```

Then append these three new tests to the end of the file:

```python


def test_compose_appends_rag_sections_to_a_normal_answer():
    from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record

    candidates = [_candidate("e1")]
    model = FakeChatModel(Selection(items=[SelectionItem(record_id="e1", reason="Fits.")], confidence="high"))
    client = _client()
    rag_context = RagContext(
        librarians=[Record(id="librarian:mh", type=LIBRARIAN_SUBJECT, text="", metadata={
            "subject": "Mental Health", "librarians": [{"name": "Doe, Jane", "profile_url": "https://staff/doe"}]
        })],
        help_topics=[Record(id="help:citing", type=HELP_TOPIC, text="", metadata={
            "title": "Citing Sources", "url": "https://research/citingsources"
        })],
        databases=[Record(id="db:mh", type=DATABASE_SUBJECT, text="", metadata={
            "subject": "Psychology", "count": 12, "browse_url": "https://research/az.php"
        })],
    )

    out = compose(_state(candidates, rag_context=rag_context), model, client)

    answer = out["answer"]
    assert "Subject librarian: Doe, Jane - https://staff/doe" in answer
    assert "Helpful pages:" in answer and "- Citing Sources: https://research/citingsources" in answer
    assert "Browse databases in Psychology: https://research/az.php" in answer


def test_compose_appends_rag_sections_even_with_no_candidates():
    from pipeline.records import HELP_TOPIC, Record

    rag_context = RagContext(help_topics=[
        Record(id="help:a", type=HELP_TOPIC, text="", metadata={"title": "Ask A Librarian", "url": "https://x"})
    ])
    state = _state([], rag_context=rag_context)

    out = compose(state, FakeChatModel(), _client())

    assert "Helpful pages:" in out["answer"]
    assert "- Ask A Librarian: https://x" in out["answer"]


def test_compose_omits_rag_footer_when_context_is_empty():
    out = compose(_state([_candidate("e1")], rag_context=RagContext()),
                   FakeChatModel(Selection(items=[SelectionItem(record_id="e1", reason="ok")], confidence="high")),
                   _client())
    assert "Subject librarian" not in out["answer"]
    assert "Helpful pages" not in out["answer"]
    assert "Browse databases" not in out["answer"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_agent_compose.py -v
```Expected: FAIL — the 3 new tests fail (no RAG footer yet), and possibly the updated ones too if `compose()`'s current signature doesn't accept `RagContext`.

- [ ] **Step 3: Replace `agent/compose.py` with the full updated file**

```python
import json

from agent.models import AgentState, RagContext, Selection, SelectionItem
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


def _rag_summary(rag_context: RagContext) -> dict:
    """Compact, JSON-able view of the RAG hits for the compose prompt: titles only,
    no URLs (the LLM never writes links; render_rag_sections does that from the
    real Record objects)."""
    return {
        "subject_librarians": [r.metadata.get("subject") for r in rag_context.librarians],
        "help_pages": [r.metadata.get("title") for r in rag_context.help_topics],
        "database_subjects": [r.metadata.get("subject") for r in rag_context.databases],
    }


def select_results(
    model, role: str, message: str, candidates: list[SearchCandidate], rag_context: RagContext
) -> Selection:
    """LLM call: choose which candidates to show. Raises if the model fails."""
    lines = [f"role: {role}", f"request: {message}"]
    if not rag_context.is_empty():
        lines.append("library guide context: " + json.dumps(_rag_summary(rag_context)))
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
    rag_context: RagContext,
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
    rag_lines = render_rag_sections(rag_context)
    if rag_lines:
        out.append("")
        out.extend(rag_lines)
    return "\n".join(out)


def render_rag_sections(rag_context: RagContext) -> list[str]:
    """The library-guide footer: subject librarian, helpful pages, database browse
    link. Code-rendered from real Record metadata, same rule as the rest of this
    module - the LLM never writes a link or a name here."""
    out: list[str] = []
    if rag_context.librarians:
        top = rag_context.librarians[0]
        names = ", ".join(l["name"] for l in top.metadata["librarians"])
        urls = ", ".join(l["profile_url"] for l in top.metadata["librarians"])
        out.append(f"Subject librarian: {names} - {urls}")
    if rag_context.help_topics:
        out.append("Helpful pages:")
        for topic in rag_context.help_topics[:2]:
            out.append(f"- {topic.metadata['title']}: {topic.metadata['url']}")
    if rag_context.databases:
        top = rag_context.databases[0]
        out.append(f"Browse databases in {top.metadata['subject']}: {top.metadata['browse_url']}")
    return out


def _render_no_answer(
    headline: str, notes: list[str], more_link: str, rag_context: RagContext
) -> str:
    out = [headline]
    out.extend(f"Note: {n}" for n in notes)
    out.append(f"You can browse the full OneSearch results here: {more_link}")
    rag_lines = render_rag_sections(rag_context)
    if rag_lines:
        out.append("")
        out.extend(rag_lines)
    return "\n".join(out)


def compose(state: AgentState, model, client: OneSearchClient) -> dict:
    """Pick 1-3 results (LLM), fetch their access details (bounded, batched),
    and render the answer in code so links and locations are never LLM output."""
    plan = state["plan"]
    result = state["search_result"]
    candidates = result.candidates
    notes = list(state.get("notes", []))
    rag_context = state.get("rag_context") or RagContext()
    more_link = result.fallback_link or client.fallback_link(plan.keywords)

    if not candidates:
        answer = _render_no_answer(
            f'I could not find results for "{plan.keywords}".', notes, more_link, rag_context
        )
        return {"selection": [], "details": {}, "answer": answer}

    try:
        selection = select_results(
            model, state["role"], state["message"], candidates, rag_context
        )
        items = validate_selection(selection.items, candidates)
        confidence = selection.confidence
    except Exception:
        items = [SelectionItem(record_id=c.record_id, reason="") for c in candidates[:MAX_SHOWN]]
        confidence = "medium"
        notes.append("I could not rank these results, so these are the top matches from OneSearch.")

    if confidence == "low" or not items:
        answer = _render_no_answer(
            "I am not confident I found a good match for that.", notes, more_link, rag_context
        )
        return {"selection": [], "details": {}, "answer": answer}

    try:
        details = client.fetch_detail([item.record_id for item in items])
    except Exception:
        details = {}
        notes.append("I could not load the full access details for these.")

    answer = render_answer(items, candidates, details, plan.keywords, notes, client, rag_context)
    return {"selection": items, "details": details, "answer": answer}
```

- [ ] **Step 4: Run them to verify they pass**

```bash
.venv/bin/pytest tests/test_agent_compose.py -v
```Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/compose.py tests/test_agent_compose.py
git commit -m "feat: append a code-rendered library-guide footer to composed answers"
```

### Task 8: Wire RAG into the graph and the CLI

**Files:**
- Modify: `agent/graph.py` (full replacement, see below)
- Modify: `agent/cli.py`
- Modify: `tests/test_agent_graph.py`
- Create: `agent/rag_live_check.py`

**Interfaces:**
- Consumes: `query_rag` (Task 6); `open_rag_collection` (Task 4).
- Produces: `build_graph(model, client, collection=None)` (new `collection` parameter; `None`
  means rag always returns an empty `RagContext`, so the chat still works with no OpenAI key at
  all); graph shape is now `plan -> {search, rag} -> compose` (parallel), was
  `plan -> search -> rag -> compose` (sequential). `build_default_graph()` in `agent/cli.py` now
  opens the Chroma collection via `open_rag_collection()` and passes it through.

A note on why this needs two separate `Annotated` reducers (Task 6) rather than the sequential
graph's simple dict-merge-by-hand: `search` and `rag` run in the *same* LangGraph step, so if both
return a `"notes"`/`"timings"` key with no reducer configured, LangGraph raises
`InvalidUpdateError` (verified directly against real `langgraph` before writing this). A single
routing function returning `["run_search", "run_rag"]` (rather than two separate
`add_conditional_edges` calls on the same source node, which LangGraph rejects with `Branch with
name already exists`) is what fans out to both.


- [ ] **Step 1: Update the existing graph test**

In `tests/test_agent_graph.py`, change:

```python
    assert out["rag_context"] == []
```
to:

```python
    assert out["rag_context"] is None  # no collection passed to build_graph in this test
```

Then append these 2 new tests to the end of the file:

```python


def test_rag_hits_flow_through_to_the_final_answer(tmp_path):
    import chromadb

    from pipeline.embed import COLLECTION_NAME, sync_collection
    from pipeline.records import LIBRARIAN_SUBJECT, Record
    from tests.fake_embeddings import FakeEmbeddingFunction

    client_ = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_.get_or_create_collection(COLLECTION_NAME, embedding_function=FakeEmbeddingFunction())
    sync_collection(
        [Record(id="librarian:mental-health", type=LIBRARIAN_SUBJECT, text="mental health", metadata={
            "subject": "Mental Health", "librarians": [{"name": "Doe, Jane", "profile_url": "https://staff/doe"}]
        })],
        collection,
    )
    import agent.rag as rag_module
    for t in rag_module.RAG_DISTANCE_CUTOFF:
        rag_module.RAG_DISTANCE_CUTOFF[t] = 999.0  # fake embeddings aren't semantic; see tests/test_agent_rag.py

    model = FakeChatModel(
        QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True),
        Selection(items=[SelectionItem(record_id="j1", reason="Core journal in the field.")], confidence="high"),
    )
    graph = build_graph(model, _client(), collection=collection)

    out = graph.invoke({"role": "graduate", "message": "mental health journals", "history": []})

    assert out["rag_context"].librarians[0].metadata["subject"] == "Mental Health"
    assert "Subject librarian: Doe, Jane - https://staff/doe" in out["answer"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_agent_graph.py -v
```Expected: FAIL — `build_graph` doesn't accept `collection=` yet, and the existing `rag_context == []` assertion no longer matches once it does.

- [ ] **Step 3: Replace `agent/graph.py` with the full updated file**

```python
import time

from langgraph.graph import END, StateGraph

from agent.compose import compose
from agent.models import AgentState
from agent.onesearch_tool import run_onesearch
from agent.planner import plan_query
from agent.rag import query_rag


def _timed(name: str, fn):
    """Wrap a node so its own wall time is added to state["timings"] (merged via
    AgentState's reducer, safe even when nodes run in parallel)."""

    def run(state: AgentState) -> dict:
        start = time.monotonic()
        update = dict(fn(state))
        update["timings"] = {name: round(time.monotonic() - start, 3)}
        return update

    return run


def build_graph(model, client, collection=None):
    """plan -> {search, rag} -> compose, search and rag run in parallel. A
    clarifying question from the planner ends the turn early with that question
    as the answer. `collection` is the Chroma collection for RAG; None means "no
    library-guide data yet" (rag always returns an empty RagContext)."""

    def plan_node(state: AgentState) -> dict:
        plan = plan_query(model, state["role"], state["message"], state.get("history", []))
        return {"plan": plan, "answer": plan.needs_clarification}

    def search_node(state: AgentState) -> dict:
        outcome = run_onesearch(state["plan"], client)
        return {"search_result": outcome.result, "notes": outcome.notes}

    def rag_node(state: AgentState) -> dict:
        if collection is None:
            return {"rag_context": None, "notes": []}
        rag_context, notes = query_rag(state["plan"].keywords, collection)
        return {"rag_context": rag_context, "notes": notes}

    def compose_node(state: AgentState) -> dict:
        return compose(state, model, client)

    graph = StateGraph(AgentState)
    graph.add_node("plan", _timed("plan", plan_node))
    graph.add_node("search", _timed("search", search_node))
    graph.add_node("rag", _timed("rag", rag_node))
    graph.add_node("compose", _timed("compose", compose_node))
    graph.set_entry_point("plan")

    def after_plan(state: AgentState):
        # A clarifying question ends the turn right away (no search, no rag
        # lookup: there are no real keywords to look anything up with yet).
        # Otherwise fan out to both branches in the same step; AgentState's
        # reducers on "notes"/"timings" merge their updates.
        return "clarify" if state.get("answer") else ["run_search", "run_rag"]

    graph.add_conditional_edges(
        "plan", after_plan, {"clarify": END, "run_search": "search", "run_rag": "rag"}
    )
    graph.add_edge("search", "compose")
    graph.add_edge("rag", "compose")
    graph.add_edge("compose", END)
    return graph.compile()
```

- [ ] **Step 4: Update `agent/cli.py`'s `build_default_graph`**

Replace:

```python
    from agent.graph import build_graph
    from onesearch.cache import QueryCache
    from onesearch.client import OneSearchClient
    from onesearch.session import SessionManager

    client = OneSearchClient(session_manager=SessionManager(), cache=QueryCache(ttl_seconds=300))
    return build_graph(model, client)
```
with:

```python
    from agent.graph import build_graph
    from onesearch.cache import QueryCache
    from onesearch.client import OneSearchClient
    from onesearch.session import SessionManager
    from pipeline.embed import open_rag_collection

    client = OneSearchClient(session_manager=SessionManager(), cache=QueryCache(ttl_seconds=300))
    # None (no OPENAI_API_KEY, or `python pipeline/refresh.py` hasn't been run yet)
    # is fine: build_graph then always returns an empty RagContext for this chat.
    collection = open_rag_collection()
    return build_graph(model, client, collection=collection)
```

- [ ] **Step 5: Create `agent/rag_live_check.py`**

```python
"""Manual check of the library-guide retrieval against a real, built Chroma store.
Run `python pipeline/refresh.py` first so there is data to query. Not run in CI.

Usage: python -m agent.rag_live_check ["<query>" ...]
"""
import sys

from agent.rag import query_rag
from pipeline.embed import open_rag_collection

DEFAULT_QUERIES = [
    "mental health",
    "citing sources",
    "chemistry",
    "anatomy",
]


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    queries = argv if argv else DEFAULT_QUERIES

    collection = open_rag_collection()
    if collection is None:
        print(
            "No usable library-guide data: either OPENAI_API_KEY is not set, or "
            "`python pipeline/refresh.py` has not been run yet.",
            file=sys.stderr,
        )
        return 2

    for query in queries:
        print("=" * 78)
        print(f"QUERY: {query}\n")
        context, notes = query_rag(query, collection)
        for label, hits in (
            ("librarians", context.librarians),
            ("help_topics", context.help_topics),
            ("databases", context.databases),
        ):
            print(f"{label}:")
            if not hits:
                print("  (no hit under the current distance cutoff)")
            for hit in hits:
                print(f"  {hit.id}: {hit.metadata}")
        if notes:
            print(f"notes: {notes}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 6: Run the whole suite to verify it passes**

```bash
.venv/bin/pytest -q
```Expected: all pass (roughly 143 + 7 new = ~150).

- [ ] **Step 7: Commit**

```bash
git add agent/graph.py agent/cli.py agent/rag_live_check.py tests/test_agent_graph.py
git commit -m "feat: run OneSearch and RAG retrieval in parallel, wire RAG into the CLI"
```

### Task 9: Live verification, weekly-refresh automation, docs

**Files:**
- Modify: `.github/workflows/weekly-refresh.yml`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:** none new; this task exercises Tasks 1-8 for real and updates automation/docs to
match.


- [ ] **Step 1: Run the full pipeline for real (manual, needs `OPENAI_API_KEY` and network; ~11s for the crawl delay plus embedding time)**

```bash
export OPENAI_API_KEY=sk-...
.venv/bin/python -m pipeline.refresh
git status --short data/processed/records.json vectorstore
wc -l data/processed/records.json
```Expected: `data/processed/records.json` updates (or is created) with ~370 records total (132 librarian_subject + 157 help_topic + 83 database_subject, confirmed live 2026-09-27); `vectorstore/` is created/updated (gitignored, not committed).

- [ ] **Step 2: Run the RAG live check and save its output**

```bash
.venv/bin/python -m agent.rag_live_check 2>&1 | tee docs/rag_live_check_output.txt
```
Expected: for `"anatomy"`, a `librarians` hit for the Anatomy subject (Aronoff/Mueller); for
`"citing sources"`, a `help_topics` hit for "Citing Sources"; for `"chemistry"`, a `databases` hit
with count 32. Use these real distances to sanity-check `RAG_DISTANCE_CUTOFF` in `agent/rag.py`
(0.5 per type is the spec's starting point) — tighten or loosen per type if real hits are being
dropped or irrelevant ones are getting through, and note any change made here.

- [ ] **Step 3: Run the chat CLI once with RAG live (manual)**

```bash
.venv/bin/python -m agent.cli --debug
```Ask something like "who is the librarian for chemistry" and confirm the answer includes a `Subject librarian:` line with a real name and profile URL.

- [ ] **Step 4: Update the weekly-refresh workflow**

Read the current file, then replace it with:

```yaml
name: refresh

on:
  schedule:
    - cron: "0 6 1,15 * *"
  workflow_dispatch:

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run scrape, sanity-check, and embed pipeline
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python -m pipeline.refresh

      - name: Open a pull request with the updated snapshot
        uses: peter-evans/create-pull-request@v6
        with:
          add-paths: data/processed/records.json
          commit-message: "chore: refresh library-guide data snapshot"
          title: "Scheduled library-guide data refresh"
          branch: automated/data-refresh
```

Note what changed from the version in `main`: the schedule moves from every Monday to the 1st
and 15th of each month (per the user's "every 15 or 20 days" request); the Playwright install
step is dropped (nothing in this pipeline uses a browser); the evaluation-suite step is dropped
(`evaluation/run_ragas.py` doesn't exist yet — re-add it when that slice lands); and the refresh
now opens a pull request with the snapshot diff instead of relying on some other step to commit
straight to `main`, so a layout change that slips past the sanity check is still caught in review.

- [ ] **Step 5: Update `CLAUDE.md`**

Add a new subsection after the existing Layer 1 content (find a natural insertion point, e.g.
before "## Layer 2: Agent Orchestration") documenting: the 3 source URLs and what each yields; the
robots.txt `urllib.robotparser` bug and the fix; the fetch-failure/sanity-check keep-old-data rule;
and the confirmed live counts (132/157/83) and the RAG live-check findings from Step 2 above
(including any distance-cutoff tuning made).

- [ ] **Step 6: Update `README.md`**

Add a short section after "Run the chat agent":

```markdown
## Refresh the library-guide data (optional, for RAG)

```bash
.venv/bin/python -m pipeline.refresh          # fetch, parse, snapshot, embed
.venv/bin/python -m agent.rag_live_check      # spot-check retrieval
```

Needs `OPENAI_API_KEY` for embeddings. Without it, `pipeline/refresh.py` still updates
`data/processed/records.json` and just skips the embedding step; the chat agent runs fine with
no library-guide data at all, it just won't have a "Subject librarian" / "Helpful pages" /
"Browse databases" footer on its answers.
```

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/weekly-refresh.yml CLAUDE.md README.md docs/rag_live_check_output.txt
git commit -m "docs: update weekly refresh automation and record RAG pipeline findings"
```
(If the live checks in Steps 1-3 could not be run, e.g. no `OPENAI_API_KEY` yet, omit
`docs/rag_live_check_output.txt` from `git add` and say so in the handoff, but do not skip
updating the workflow/docs themselves.)


---

## Slice acceptance (run after Task 9)

- [ ] `.venv/bin/pytest -q` passes with no network and no API key.
- [ ] `python -m pipeline.refresh` run live updates `data/processed/records.json` with real UB
      data and no warnings; re-running it immediately is a no-op (git diff empty).
- [ ] `python -m agent.rag_live_check` returns real, correctly-typed hits for at least the 4
      default queries; output saved under `docs/`.
- [ ] The chat CLI (`python -m agent.cli`) still starts and answers with **no** `OPENAI_API_KEY`
      set (RAG silently absent) and **no** Chroma store built yet.
- [ ] A librarian-directory question through the CLI gets a real `Subject librarian:` line with a
      working profile URL.
- [ ] `git log --format=%B` on this slice's commits contains no `Co-Authored-By` or "Generated
      with" lines.
- [ ] `grep -rn "OPENAI_API_KEY\|ANTHROPIC_API_KEY" agent/ pipeline/ scraper/` shows no key ever
      logged or included in an answer.
- [ ] Review, then `git push` once for the whole slice.
