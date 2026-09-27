# RAG Data Pipeline + Retrieval (Layer 2, Slice 2)

Date: 2026-09-27. Status: approved in conversation, pending written-spec review.

## Goal

Turn three curated UB Libraries pages into a small, structured knowledge base (subject
librarians, help topics, database subject categories), embed it in Chroma, and wire retrieval
into the agent so an answer can point to "who can help" and "where to look" alongside
OneSearch results, even when OneSearch finds nothing.

## Scope

Sources are fixed to the three URLs below (not the whole site). No LibGuides content scraper
and no full database-by-database list: the original plan's `scraper/scrape_libguides.py` and
`scraper/scrape_faqs.py` are replaced by the three parsers below; `pipeline/chunk.py`'s
text-splitting approach is dropped in favor of one record per entry (see "Why not chunking").

| Source | URL | What we extract |
|---|---|---|
| Subject librarians | `https://library.buffalo.edu/help-support/askalibrarian/inperson/bysubject.html` | subject -> librarian name(s) + profile page URL (no scraping of the profile pages themselves; see "Contact details") |
| Help A-Z | `https://library.buffalo.edu/help/browse.html?title=` | topic title -> its URL (an index, not page content) |
| Databases A-Z | `https://research.lib.buffalo.edu/az.php` | subject category -> database count (the per-database list is loaded from `/az.php?...` and `/az/databases?...`, which `robots.txt` disallows; those are never fetched) |

All three are static HTML (confirmed by fetching each once on 2026-09-27); no browser/Playwright
is needed for this pipeline.

### Contact details: link out, don't store

Per-librarian contact info (email, office, phone) lives on ~40 separate staff-profile pages.
Storing it would mean fetching all of them every run and risk going stale. Instead we keep only
`name` and `profile_url` from the subject page and let the user follow the link for current
contact details. This also means no personal contact data (email/phone) is stored by this
pipeline at all.

### Why not chunking

The original plan's `pipeline/chunk.py` (500-char text splitting) fits prose pages; these three
sources are lists and tables of short entries. Splitting would cut a subject/librarian pair or a
topic/URL pair in half. Each source parser instead produces one `Record` per entry directly; there
is no separate chunking step in this slice.

## Politeness / robots.txt

Confirmed live 2026-09-27: `research.lib.buffalo.edu/robots.txt` sets `Crawl-delay: 10` and
disallows `/az.php?`, `/az/databases?`, `/er.php`, `/err.php`, `/friendly.php`, `/go.php`,
`/ld.php`, `/srch.php` for all user agents. `library.buffalo.edu` has no robots.txt (404s to the
site's generic 404 page, treated as "no restriction beyond the identifying UA and delay we
already apply everywhere").

`scraper/fetch.py` provides `fetch(url) -> str`:
- Sends `User-Agent: UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)` (same constant
  Layer 1 uses).
- Checks the exact three URLs above against each host's `robots.txt` before the first run and
  raises if any is disallowed (defense against the URL list changing later, not expected to
  trigger today).
- Waits at least 10 seconds between requests to the same host (module-level state; irrelevant
  within one run of 3 requests but keeps the rule if the source list grows).
- One attempt per page, 20s timeout. A failure is caught by the caller (`pipeline/refresh.py`),
  never retried in the same run (politeness over completeness).

## Records

One record per entry, `pipeline/records.py`:

```python
@dataclass(frozen=True)
class Record:
    id: str            # stable slug, e.g. "librarian:anatomy", "help:citing-sources", "db:chemistry"
    type: str           # "librarian_subject" | "help_topic" | "database_subject"
    text: str           # what gets embedded
    metadata: dict       # what the answer renderer reads back; JSON-serializable
```

| type | id | text | metadata |
|---|---|---|---|
| `librarian_subject` | `librarian:<subject-slug>` | `"<subject> subject librarian: <name1>, <name2>"` | `{"subject": str, "librarians": [{"name": str, "profile_url": str}, ...]}` |
| `help_topic` | `help:<slug-of-title-or-url>` | `"<title>"` | `{"title": str, "url": str}` |
| `database_subject` | `db:<subject-slug>` | `"<subject> databases (<count> available)"` | `{"subject": str, "count": int, "browse_url": "https://research.lib.buffalo.edu/az.php"}` |

Slugs: lowercase, non-alphanumeric runs collapsed to `-`, stripped. Two subjects/topics that slug
to the same id (rare, e.g. punctuation-only difference) get a numeric suffix (`-2`, `-3`, ...) in
first-seen order, so ids stay stable across runs as long as source order is stable.

## Parsers

Pure functions, HTML in, `list[Record]` out, no network inside them (`fetch` is a separate,
injectable step so tests never hit the network):

- `scraper/parse_librarians.py`: `extract_librarian_subjects(html: str) -> list[Record]`. Reads
  the subject table; a subject with multiple librarian rows (e.g. "Anatomy" -> Aronoff + Mueller,
  confirmed live) becomes one record with multiple entries in `librarians`.
- `scraper/parse_help.py`: `extract_help_topics(html: str) -> list[Record]`. Reads the Help A-Z
  link list, skipping the A-Z jump-letter links themselves (single-letter anchor text) and the
  "Search Help A-Z" control.
- `scraper/parse_databases.py`: `extract_database_subjects(html: str) -> list[Record]`. Reads the
  "Browse Databases by Subject" list, `"<name> (<count>)"` per line (confirmed live format, e.g.
  `"Chemistry (32)"`).

Test fixtures are the three pages already fetched live on 2026-09-27, saved under
`tests/fixtures/pages/{librarians,help,databases}.html`.

## Snapshot, diff, and the update-safety rule

`data/processed/records.json`: a committed, sorted-by-id JSON array of all current records plus
a `content_hash` (sha256 of `text` + sorted `metadata`) per record. Committing it means a refresh
shows up as a normal, readable PR diff.

`pipeline/refresh.py` orchestrates, per source (`type`), independently:

1. Fetch. On failure (network error, non-200): keep that type's existing records from the old
   snapshot unchanged, record a warning, move to the next source.
2. Parse into new records for that type.
3. Sanity check: if `len(new) < 0.5 * len(old)` and `len(old) > 0`, treat this as a probable
   layout change, not real data loss: keep the old records for that type, record a warning, move
   on. Otherwise, the new records **fully replace** the old ones for that type (so entries that
   really disappeared from the source are removed, not just entries that changed).
4. Write the merged snapshot (updated types + kept-unchanged types) to
   `data/processed/records.json`.
5. Sync Chroma (see below) from the merged snapshot.

`refresh_all() -> RefreshResult` (`counts` per type, `warnings: list[str]`) is the entrypoint,
replacing the old plan's `refresh_all(sources, vectorstore, page_loader)` signature (no
`page_loader`: nothing here uses a browser).

## Embedding and storage

Plain `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)` (already in `.env.example`), one
collection `ub_library_navigator`, `embedding_function=chromadb.utils.embedding_functions.
OpenAIEmbeddingFunction(api_key=..., model_name="text-embedding-3-small")`. Plain `chromadb` is
used directly rather than a `langchain-chroma` wrapper: this slice's retrieval is a handful of
`collection.query()` calls with a `where` filter on `type`, not a LangChain retriever chain, so
the wrapper would add a dependency without simplifying anything.

`pipeline/embed.py`: `sync_collection(records: list[Record], collection) -> SyncResult` (counts
`added`, `updated`, `deleted`).
- Reads back existing ids + stored `content_hash` metadata from the collection.
- Upserts only records whose `content_hash` changed or that are new (skips unchanged records:
  no re-embedding cost for a page that didn't change).
- Deletes any stored id with a `type` covered by this pipeline (`librarian_subject`,
  `help_topic`, `database_subject`) that is no longer in the new records.
- Building the Chroma store from scratch (fresh clone, or `vectorstore/` missing) is just
  `sync_collection` against an empty collection: same function, no separate "first build" path.

## Retrieval

`agent/rag.py` replaces the `rag` stub node.

`query_rag(keywords: str, collection) -> RagContext`:
- Queries the collection three times, once per type, each with `where={"type": "..."}` and
  `n_results` from `RAG_TOP_K`.
- `RAG_TOP_K = {"librarian_subject": 2, "help_topic": 3, "database_subject": 2}`.
- `RAG_DISTANCE_CUTOFF = {"librarian_subject": 0.5, "help_topic": 0.5, "database_subject": 0.5}`
  (cosine distance from `text-embedding-3-small`; starting values, tuned against real queries with
  the manual live check before this slice is called done, see Testing).
- Hits at or above the cutoff for their type are dropped. `keywords` comes from the already-built
  `QueryPlan`; if planning failed, the raw message is used (same fallback the search step gets).
- Role is not used here: subject/topic matching doesn't depend on who's asking.
- If `OPENAI_API_KEY` is unset or embedding the query fails, `query_rag` returns an empty
  `RagContext` and a note ("Library guide lookup was unavailable.") instead of raising: OneSearch
  results and the local-model chat path must keep working with no OpenAI key at all.

`RagContext` (`agent/models.py` addition): `librarians: list[Record]`, `help_topics: list[Record]`,
`databases: list[Record]` (each already filtered/capped/sorted by distance).

### Graph change

`plan -> {search, rag} -> compose` (both branches read `state["plan"]`; `search` writes
`search_result`/adds to `notes`, `rag` writes `rag_context`/adds to `notes`, no key overlap, so
LangGraph's default per-superstep state merge needs no custom reducer). The clarification
short-circuit after `plan` is unchanged. This is a design decision to verify with real LangGraph
parallel-branch code during planning (`add_edge("plan", "search")` and `add_edge("plan", "rag")`,
both `-> "compose"`); if LangGraph's join semantics don't fit this cleanly, falling back to the
current sequential `search -> rag -> compose` is acceptable and only costs the (already small)
RAG query's latency serially instead of in parallel.

## Answer rendering

Code-rendered, appended after the existing OneSearch section in `agent/compose.py`'s answer (the
LLM never sees or writes this text, same rule as the rest of `compose`):

```
Subject librarian: Aronoff, Nell - https://library.buffalo.edu/staff/naronoff
(only the single closest librarian_subject hit, if any passed the cutoff)

Helpful pages:
- Citing Sources: http://research.lib.buffalo.edu/citingsources
(up to 2 help_topic hits)

Browse databases in Chemistry: https://research.lib.buffalo.edu/az.php
(only the single closest database_subject hit, if any; the fixed browse_url, not a per-database link)
```

These sections are appended whenever `rag_context` has any hits, **including** when OneSearch
found nothing or compose's confidence was `low`: `compose()`'s `_render_no_answer` path also
gets the RAG sections appended, so a dead end still offers "here's who can help" instead of just
a link.

## Known limitation (not fixed in this slice)

A question that is really a librarian/database lookup ("who is the librarian for anatomy?") still
goes through the full plan -> search -> compose path and gets OneSearch results as the primary
answer, with the librarian hit only as a footer. Recognizing "this is a directory question, skip
OneSearch" is a planner/router change (an intent field, or a third fast path) that belongs in a
later slice, not this one.

## Testing

All offline except the two manual scripts:

- Parsers: each of the 3, against the saved real-page fixtures. Cover the multi-librarian-per-
  subject case, slug collisions, and empty/malformed input.
- `pipeline/refresh.py`: fetch failure keeps old records for that type only; sanity-check
  rejection keeps old records and warns; a real update fully replaces a type's records (including
  removal of a since-deleted entry); snapshot is written sorted and deterministic.
- `pipeline/embed.py`: unchanged-hash records are skipped; changed/new records are upserted;
  removed records are deleted; scoped to this pipeline's three `type`s only (never touches other
  `type`s in the same collection, protecting future slices that might share it).
- `agent/rag.py`: per-type top-k and cutoff filtering; missing/failing `OPENAI_API_KEY` degrades
  to an empty `RagContext` + note, not an exception.
- `agent/compose.py`: RAG sections appear in both the normal answer and the `_render_no_answer`
  path; absent entirely when `rag_context` is empty.
- Manual, not CI: `pipeline/refresh.py` run for real (writes a real snapshot + Chroma store) and
  `agent/rag_live_check.py` (new, same pattern as `onesearch/verify_live.py` and
  `agent/live_check.py`) printing real hits and distances for a handful of test queries, used to
  confirm/tune `RAG_DISTANCE_CUTOFF`.

## Files

```
scraper/fetch.py               polite fetch() + robots.txt check
scraper/parse_librarians.py
scraper/parse_help.py
scraper/parse_databases.py
pipeline/sources.py             the 3 source URLs + type
pipeline/records.py             Record dataclass, slugging, hashing
pipeline/snapshot.py            load/save data/processed/records.json
pipeline/embed.py               sync_collection()
pipeline/refresh.py             refresh_all() orchestration
agent/rag.py                    query_rag(), RagContext
agent/rag_live_check.py         manual live check
tests/fixtures/pages/*.html     saved real pages
tests/test_scraper_*.py
tests/test_pipeline_*.py
tests/test_agent_rag.py
```

Removed from scope (superseded, delete from the old plan rather than build):
`scraper/scrape_libguides.py`, `scraper/scrape_faqs.py`, `scraper/scrape_librarian_directory.py`
(as separate modules), `pipeline/chunk.py`.

Dependencies added: `chromadb` (already in `requirements.txt`, currently unused — first real use).
No new packages beyond what's already listed.

## Weekly refresh automation

`.github/workflows/weekly-refresh.yml` (kept, retimed and simplified):
- Cron changes from every Monday to the 1st and 15th of each month: `cron: "0 6 1,15 * *"`.
- Drops the `Install Playwright browsers` step (nothing in this pipeline uses a browser) and the
  `Run evaluation suite` step (`evaluation/run_ragas.py` doesn't exist yet; re-added when that
  slice lands).
- Runs `python pipeline/refresh.py`, then opens a pull request with the snapshot diff (via
  `peter-evans/create-pull-request` or equivalent) instead of committing straight to `main`, so a
  layout change that slips past the sanity check is still caught in review before merge.
