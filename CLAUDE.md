# CLAUDE.md — UB Library Navigator

Project brief for Claude Code. Read this before writing any code in this repo.

## What this project is

UB Library Navigator: a role-aware agentic RAG triage tool for University at
Buffalo Libraries. It sits in front of the existing live chat system as a
"first contact" helper: it routes users to the right resource, database, or
librarian rather than trying to answer everything itself. Capstone project
for an MS Data Science student (Buffalo), supervised by Dr. Justin Del
Vecchio, 14-week timeline (Aug 24 - Nov 29, 2026).

Full scope (seven capabilities): role capture, resource/database/item
discovery via OneSearch, librarian recommendation, citation formatting
(APA/MLA/Chicago), escalation to human librarians, anonymized usage logging,
staff dashboard. **This document only covers what to build right now: the
OneSearch retrieval layer.** Everything else is out of scope until this
layer is solid. Do not start on LangGraph orchestration, the LibGuides RAG
pass, or the librarian directory until the acceptance criteria below are met.

## Tech stack (finalized, for context, not all needed yet)

- Streamlit (presentation)
- LangChain + LangGraph (agent orchestration, Phase 2)
- Chroma (vector store, Phase 2) / SQLite or Postgres (logging)
- Playwright + BeautifulSoup (scraping, LibGuides + this layer)
- OpenAI embeddings
- RAGAS + LangSmith (evaluation, Phase 2)
- GitHub Actions (weekly content refresh)

## Current phase: the OneSearch retrieval layer ("Layer 1")

### What it needs to do

Given a single search query, return enough data to (a) recommend the right
resource/database and (b) tell the user how to get it: an online link, or a
call number and physical location for print items. This is the only job of
this layer. Ranking, reasoning about which result best answers the user's
actual question, and composing a natural-language answer all belong to
Layer 2 (the RAG/agent layer), not here. Layer 1 hands Layer 2 a clean list
of candidates; Layer 2 decides what to do with them.

### Why this isn't a normal REST API integration

`search.lib.buffalo.edu` runs Primo VE (Ex Libris). There is no official,
documented API for this data without registering a guest key through the Ex
Libris Developer Network (a real fallback, see "Fallback path" below, but
not what we're building first). Instead, this integration calls the same
internal endpoint the live OneSearch website's own frontend calls when a
person searches:

```
https://search.lib.buffalo.edu/primaws/rest/pub/pnxs
```

This is the same access level as an anonymous visitor browsing the site
with no UB login, nothing behind auth, no patron data. But it is
undocumented and unversioned. Treat it as "works today, verify
occasionally," not a stable contract. Build defensively:
- Real, identifying `User-Agent` on every request (contact:
  kavyansht21@gmail.com), never spoof a browser UA on the HTTP layer.
- Cache aggressively (see caching strategy below).
- Low request volume, no hammering. UB Libraries has already raised a
  security concern about this project once; being a quiet, well-identified,
  well-behaved caller matters here more than usual.
- Wrap every call to this endpoint behind one adapter module. Nothing else
  in the codebase should import `requests`/`playwright` and hit this host
  directly. If the endpoint shape changes, the fix should live in one file.

### Confirmed technical findings (do not re-derive these, they're already tested)

1. **Comma encoding breaks the request.** The `q` parameter
   (`any,contains,<query>`) must keep literal commas. Building the URL with
   `requests`' `params=` dict percent-encodes them to `%2C` and the backend
   rejects it. Build the query string manually with
   `urllib.parse.urlencode(params, safe=",", quote_via=urllib.parse.quote)`.

2. **A bare `requests` call gets a blank HTTP 400.** Confirmed via direct
   testing: the response comes from Ex Libris's real hosted backend
   (`*.alma.exlibrisgroup.com`, visible in response headers), not a generic
   CDN, and returns `Content-Length: 0` with security headers (CSP, HSTS,
   Report-To) but no error body. This is bot-mitigation rejecting a request
   that doesn't look like it came from a real browser session (TLS
   fingerprint, header ordering, client hints). A `Referer`/session-cookie
   warm-up alone was NOT sufficient to fix this in testing.

3. **The working fix: drive a real headless browser (Playwright) to mint a
   session, then reuse that session for fast calls.** Loading the actual
   search page in Chromium and letting its own JS fire the request works
   (confirmed, real UB data returned, HTTP 200). The response sets session
   cookies: `JSESSIONID`, `urm_st`, `urm_se`, `institute`. The gap between
   `urm_st` and `urm_se` is exactly 900,000 ms, a 15-minute session window.
   **Target production design:** mint a session via Playwright once every
   ~15 minutes (or reactively on a 400/expired check), extract those
   cookies into a `requests.Session()`, and use that lightweight session
   for actual per-query searches. Do not launch a browser per user search,
   that's the latency trap to avoid. See "Latency budget" below.

   **Mint by waiting for the page's own `/pnxs` response** (`page.expect_response`), not `networkidle`: the site's analytics keep the network busy, so `networkidle` timed out at 30s (2/2 runs, 2026-09-24) while the response wait mints in 2.4-5.4s. `search_with_fallback` mints/reuses the session *before* starting its timeout clock and returns at the timeout instead of waiting for the slow worker.

4. **`skipDelivery=Y` (the site's own default) suppresses both online
   links and physical location/call-number data.** Force
   `skipDelivery=N` and `pcAvailability=true` on outgoing requests
   (rewrite via Playwright route interception during the mint step, or set
   directly as params on the plain `requests` calls).

5. **Physical holdings come with the search; online links do not.** With
   `skipDelivery=N` the brief response carries `delivery.bestlocation`
   (call number, library, availability) for physical items, but online items
   have no usable link there: `delivery.link[]` holds thumbnails and "About
   requesting" entries, not the access link. The access link comes from a
   separate `POST /primaws/rest/pub/edelivery?vid=<vid>` with body
   `[{"recId": "<id>", "sharedDigitalCandidates": null}, ...]` (one batched
   call for all ids). Confirmed live 2026-09-24 that it works from plain
   `requests` with the minted cookies and needs only `vid` (no search-context
   replay). Response: `{"<recId>": {"electronicServices": [{"serviceUrl":
   "/view/action/uresolver.do?...", "packageName": ...}], "GetIt1": [...]}}`;
   `serviceUrl` is host-relative (prefix `https://search.lib.buffalo.edu`) and
   302s to the resource. **`fulldisplay` is not an API (404).** The site's
   `POST /pub/delivery` needs the browser's Bearer token and 500s from
   `requests`; it is not needed. Only call `/edelivery` for the 1-3 items
   Layer 2 shows, never for the full result set.

6. **Field quirks to handle in the parser:**
   - Names carry an internal separator: `"Wong, Kau-Fui Vincent,
     author.$$QWong, Kau-Fui Vincent"`. Split on `$$Q`, keep the first part.
   - Not all records have `pnx.display.creator`. Some (especially
     organizational/corporate authorship) only have
     `pnx.display.contributor`. Fall back to it when creator is empty.
   - `pnx.display.source` returns the backend name ("Alma"), not the
     human-readable label shown on the site ("Library Catalog"). Decide in
     Layer 2 whether the raw or display form matters more; don't assume
     they're the same string.

7. **Filters (resource type, peer-reviewed) are facets, not keywords.**
   `search(query, resource_type=None, peer_reviewed=False)` sends them as
   `multiFacets=facet_rtype,include,<type>|,|facet_tlevel,include,peer_reviewed`
   (literal commas, same rule as finding 1). `resource_type` is one of
   `books, articles, journals, dissertations, videos`; an unknown value does
   not error, it silently returns 0 hits, so the client raises `ValueError`
   first. Filters are part of the cache key. **`query` must be keywords
   only.** Sending a sentence like "find me peer reviewed journals on mental
   health" returns 112k hits of articles that mention "peer"; the same intent
   as `"mental health"` + `journals` + peer-reviewed returns 266 real
   journals (confirmed live 2026-09-24). Turning language into keywords +
   filters is Layer 2's job (query planning below). Note `search_with_fallback`'s
   fallback link carries the keywords only, not the filters.

### Confirmed field map (brief search-results response, `doc` object)

Recommendation fields (present in the brief list, `skipDelivery` doesn't
affect these):

| Field | Path |
|---|---|
| Title | `pnx.display.title[0]` |
| Author/creator | `pnx.display.creator[0]`, fallback `pnx.display.contributor[0]` |
| Resource type | `pnx.display.type[0]` |
| Format | `pnx.display.format[0]` |
| Subject | `pnx.display.subject[0]` |
| Description | `pnx.display.description[0]` (often absent for bare catalog records, that's normal, not a bug) |
| Date | `pnx.display.creationdate[0]`, fallback `pnx.addata.date[0]` |
| Source system | `pnx.display.source[0]` |
| Record ID (needed for detail fetch) | `pnx.control.recordid[0]` |

Delivery/location fields (need `skipDelivery=N`; online link may still
require the full-detail fetch, unconfirmed, see point 5 above):

| Field | Path |
|---|---|
| Online vs. physical | `delivery.deliveryCategory` |
| Online link | `/edelivery` → `electronicServices[0].serviceUrl` (absolutise); brief `GetIt1[].links[]` where `isLinktoOnline` when present |
| Call number | `delivery.bestlocation.callNumber` |
| Library/building | `delivery.bestlocation.mainLocation` |
| Sub-location | `delivery.bestlocation.subLocation` |
| Availability status | `delivery.bestlocation.availabilityStatus` (`available` / `unavailable` / `check_holdings`, confirmed live 2026-09-24) |

There is no copies-count field in the real `bestlocation` object.

Detail fetch = `/edelivery` for online access, merged with the physical
fields already captured from the brief search (client remembers them per
record id). Real `/edelivery` capture: `tests/fixtures/onesearch_edelivery_response.json`.

### Retrieval flow to implement

```
user query
  → get_session()            # reuse if <15 min old, else re-mint via Playwright
  → search(query)             # requests.Session, brief list, skipDelivery=N
  → return candidates to Layer 2   # STOP HERE, do not fetch detail yet
  ...
  → Layer 2 picks top 1-3 candidates
  → for each chosen candidate: fetch_detail(record_id)   # only now, only for these
  → return access info (link or location) to Layer 2 for the final answer
```

The critical rule: **never loop `fetch_detail()` over every search result.**
That's the latency trap. Detail fetching is bounded by how many results get
shown to the user (1-3), not by how many results OneSearch returned.

### Latency budget (target, document actual numbers once built)

| Stage | Target |
|---|---|
| Session reuse (warm) | 0s |
| Session mint (cold, rare, ~every 15 min) | 2-4s |
| Search call | 0.3-1s |
| Layer 2 ranking (not this repo's concern, noted for budget) | 1-3s |
| Detail fetch (1-3 items, run concurrently not sequentially) | TBD, target <3s total |
| **End-to-end target (warm session)** | **~3-6s** |

Build in a hard timeout on the whole retrieval step (8-10s) with a graceful
fallback: if it's exceeded, return whatever candidates were already
fetched plus a direct OneSearch search-results link, rather than hanging.
This is a courtesy integration against a third party's infrastructure;
assume it can have a bad moment.

### Caching

Cache search results by normalized query text for a short TTL (a few
minutes). This reduces load on UB's endpoint (see "quiet, well-behaved
caller" above) and speeds up repeated/similar searches across users, which
will be common for a shared academic topic.

### Fallback path (not built now, document only)

If this endpoint ever gets locked down: Ex Libris publishes a separate,
officially sanctioned "Primo Search API guest mode"
(`api-*.hosted.exlibrisgroup.com/primo/v1/search`), self-serve guest API
key via the Ex Libris Developer Network. Narrower and slower to set up
than what we're building, but a known, documented escape hatch. Worth a
short paragraph in the capstone write-up's limitations section either way:
this integration uses an undocumented endpoint at the same access level as
an anonymous browser visit, rate-limited and monitored, with a sanctioned
fallback identified if needed.

## Layer 2, slice 2: RAG data pipeline (built)

A small, structured knowledge base built from 3 curated UB Libraries pages, embedded in Chroma,
and queried in parallel with OneSearch so an answer can point to "who can help" and "where to
look" even when OneSearch finds nothing. Full design:
`docs/superpowers/specs/2026-09-27-rag-data-pipeline-design.md`. Implementation plan:
`docs/superpowers/plans/2026-09-27-rag-data-pipeline.md`.

**The 3 sources** (fixed list, not a site crawl, `pipeline/sources.py`):

| Source | URL | Yields |
|---|---|---|
| Subject librarians | `library.buffalo.edu/help-support/askalibrarian/inperson/bysubject.html` | subject -> librarian name(s) + profile URL (confirmed live 2026-09-27: 132 records) |
| Help A-Z | `library.buffalo.edu/help/browse.html?title=` | topic title -> URL, an index only, not page content (157 records) |
| Databases A-Z | `research.lib.buffalo.edu/az.php` | subject category -> database count; the per-database list itself is loaded by JS from an endpoint `robots.txt` disallows, so only categories are indexed, not individual databases (83 records) |

Contact details are never stored: a librarian record holds only `name` + `profile_url`, and the
user follows the link for current email/phone. This also means no personal contact data is stored
by this pipeline at all.

**A real bug in Python's standard library, found and worked around (`scraper/fetch.py`):**
`urllib.robotparser` reconstructs each `Disallow` rule through `urlparse`/`urlunparse`, which
silently drops a bare trailing `"?"`. UB's real `research.lib.buffalo.edu/robots.txt` has
`Disallow: /az.php?` (block only the query-string form used by the site's own AJAX calls);
`urllib.robotparser` turns that into the rule `/az.php` and would wrongly block the bare page this
pipeline needs. Confirmed live 2026-09-27. `scraper/fetch.py` reads the `User-agent: *` group's
`Disallow` lines itself and does a literal prefix match instead, which gets this right (fetching
the bare `az.php` succeeds; `/az.php?subject=...` is still correctly blocked). Every fetch also
waits 10s between requests to the same host and sends the same identifying User-Agent as Layer 1.

**Update safety (`pipeline/refresh.py`):** each of the 3 sources is refreshed independently. A
failed fetch, or a parse that returns under half the previous record count (a likely layout
change, not real data loss), keeps that source's old records unchanged and records a warning —
never wipes data. A real update fully replaces that source's records, so entries that actually
disappeared are removed too. The merged result is written to the committed, sorted
`data/processed/records.json` (so a refresh's diff is readable) and synced into Chroma
(`pipeline/embed.py`), which only re-embeds records whose content actually changed.

**Retrieval (`agent/rag.py`) degrades silently:** no `OPENAI_API_KEY`, no Chroma store built yet,
or a query failure all give an empty `RagContext`, never an exception — the chat agent (and its
OneSearch results) keep working with zero library-guide data. The chat model (`LLM_PROVIDER`,
e.g. `local` via Ollama) and the embedding backend (always OpenAI's `text-embedding-3-small`,
`pipeline/embed.py`) are independent: you can run planning/compose entirely on a local model and
still need `OPENAI_API_KEY` only for the embedding step.

**A real LangGraph gotcha, confirmed directly before relying on it (`agent/graph.py`):** the
graph now runs `search` and `rag` in parallel (`plan -> {search, rag} -> compose`, was strictly
sequential in slice 1). Two nodes in the same step both write `state["notes"]` and
`state["timings"]`; without a merge reducer, LangGraph raises `InvalidUpdateError` on the second
write. `AgentState.notes` gets `Annotated[list[str], operator.add]` and `.timings` gets a small
dict-merge reducer (`agent/models.py`) to fix this. Also: LangGraph rejects two separate
`add_conditional_edges` calls from the same source node (`Branch with name already exists`); one
routing function returning `["run_search", "run_rag"]` fans out to both correctly.

**Distance cutoffs are tuned from real embeddings, not the spec's starting guess
(`agent/rag.py`).** Confirmed live 2026-09-28 against the real 372-record store with 11 test
queries: `text-embedding-3-small` puts short, formulaic phrases (`"<Subject> subject librarian:
<Name>"`, `"<Subject> databases (<N> available)"`) much closer together than natural-language
titles, so one global cutoff doesn't work. Genuine `librarian_subject`/`database_subject` matches
land at 1.08-1.55 cosine distance, with nonsense queries starting around 1.6; genuine
`help_topic` matches (varied real titles) land at 0.59-0.79, with irrelevant hits at 1.26+. Final
values: `LIBRARIAN_SUBJECT: 1.6`, `HELP_TOPIC: 1.1`, `DATABASE_SUBJECT: 1.6` (the spec's original
`0.5` for all three let almost nothing through). Live check output:
`docs/rag_live_check_output.txt`.

**`pipeline/refresh.py` needed a `main()` entrypoint the plan omitted.** `python -m
pipeline.refresh` silently did nothing (no `if __name__ == "__main__":` block existed) until one
was added; confirmed the gap live before fixing it (the command produced no output and did not
touch `data/processed/records.json` or `vectorstore/`).

## Layer 2, slice 3: live chat escalation (built)

When OneSearch/RAG can't confidently help, the answer now includes a link to UB's real live chat
plus a copyable context summary (`agent/escalation.py`). Confirmed live 2026-09-29: UB's chat
widget (LibraryH3lp, `library.buffalo.edu/help-support/askalibrarian/chat/instant.html`) has no
API and no documented way to pre-fill a message — the raw page ships an unresolved `{queue}`
placeholder only filled in by UB's own JavaScript at load time, and the resulting widget is a
plain "type here to chat" box. So this links to the real page directly (never construct an h3lp
iframe URL — the queue id is an internal detail) and gives the user a paste-ready summary instead
of trying to auto-inject anything.

## Layer 2, slice 4: evaluation suite (built)

A manual script, `python -m evaluation.run_eval`, scores the real agent's behavior against a
15-question golden set tagged by role. Full design: `docs/superpowers/specs/2026-09-30-evaluation-suite-design.md`.
Plan: `docs/superpowers/plans/2026-09-30-evaluation-suite.md`.

**The `ragas` package is not used.** Confirmed live 2026-09-30: every release from 0.3.9 through
0.4.3 unconditionally imports `langchain_community.chat_models.vertexai.ChatVertexAI` at module
load time, which no longer exists in current `langchain-community` (Vertex AI chat support moved
to a separate package) — `import ragas` fails outright for anyone not using Google Vertex AI.
Tracked upstream in ragas#2741, #2753, #2995, #2745, with an unmerged fix (PR #3017). The one
metric that would have used it (faithfulness of the LLM's "Why" reasons) is a small LLM-judge
function instead (`evaluation/faithfulness.py`), using this project's own `agent.llm.get_chat_model()`.

**The golden set labels expected behavior, not expected search results** (`resource_type`/
`peer_reviewed` the planner should pick, whether the turn should escalate, which RAG type should
hit) — OneSearch is live and external, so a golden set of expected titles would go stale as UB's
catalog changes.

**Two real findings from tuning the golden set against live data, not assumptions:**
- Directory/nonsense questions ("who is the subject librarian for chemistry") have no well-defined
  planning target: the planner has no way to know these aren't literal topic searches (the same
  limitation the RAG pipeline spec already documents), so these questions set
  `check_planning: false` rather than encode an unrealistic expectation.
- One question's `peer_reviewed` flipped between two live runs on the same unchanged input — a
  genuinely probabilistic role-default nudge (faculty + research-flavored phrasing), not a bug.
  Also set `check_planning: false` with a note, rather than pin a flaky expectation.

**LangSmith tracing needs zero code changes.** Confirmed live against the installed `langsmith`
0.14.0: it reads `LANGSMITH_TRACING_V2` (falling back to `LANGCHAIN_TRACING_V2`) and matching
`_API_KEY`/`_PROJECT` vars purely from the environment, through LangChain's global callback
manager — any compiled LangGraph graph traces automatically once these are set.
`evaluation/langsmith_config.py`'s `tracing_configured()` only decides whether to print a heads-up
that tracing is off; `run_eval.py` still runs without it.

**Confirmed live baseline** (2026-10-01, real UB data + real OpenAI calls, `docs/evaluation_run_output.txt`):
planning accuracy 1.0, clarification accuracy 1.0, RAG hit accuracy 1.0, faithfulness accuracy
0.86-0.94 across runs, escalation confusion matrix TN=14 with the one true-escalation question
sometimes landing as a true positive and sometimes as a false negative run to run — real LLM
calls aren't perfectly deterministic even at temperature 0, so a result in the same neighborhood
confirms the suite works; it is not expected to match exactly every run. Not CI-wired in this
slice (real OneSearch + LLM calls cost time and money per run); add to `weekly-refresh.yml` once
the golden set and metrics are proven out further.

## Layer 2, slice 5: citation tool (built)

Deterministic APA/MLA/Chicago citation formatting, rendered inline (collapsed by default) with
every result the agent shows — `agent/citation_tool.py`. Full design:
`docs/superpowers/specs/2026-10-04-citation-tool-design.md`. Plan:
`docs/superpowers/plans/2026-10-04-citation-tool.md`.

**Not the conversational "cite that" fast path CLAUDE.md originally sketched.** That would need
persisting structured candidate data across turns (today's `history` is plain text) and resolving
which prior result a follow-up refers to. Rendering a citation for every result the moment it's
shown sidesteps both problems — deferred, not abandoned.

**The data was already free.** Confirmed live 2026-10-04: a real peer-reviewed article's
`pnx.addata` has `jtitle`, `volume`, `issue`, `pages`, `doi`, `pub`, and an author list; a real
book's has `pub`, `cop`, `isbn`, and an author list — all already present in the brief search
response Layer 1 already fetches, just never parsed before. One field-naming surprise: which key
holds the author list isn't consistent by record type (`au` on one real article with `addau`
absent; `addau` on a real book with `au` absent) — `onesearch/parser.py` checks both.

**Two real bugs found and fixed by actually running this against live data**, not left to chance —
both have regression tests: (1) the Chicago formatter interpolated a missing `volume`/`issue`
straight into an f-string, rendering the literal word "None" for journal-title records with no
article-level volume data; (2) the MLA formatter always put a comma after the year assuming page
numbers would follow, leaving a dangling comma with no closing period when pages are missing.

**Known, accepted limitations, not further "fixed":** (a) author-count handling is simplified —
1-2 authors listed in full, 3+ as "et al.", for every style, not each style's real cutoff; (b)
some OneSearch records list an issuing organization as author, not a person ("International
Center of Mental Health Policy and Economics, issuing body") — the formatter can't tell the
difference from a personal "Last, First" name, so APA output can read oddly (an "i. b." initial
from "issuing body"), and a no-comma organizational name can end up with a doubled period
("...Psychiatrists.."). Cosmetic, not a crash or a missing/garbled citation.

**A real crash found by the user manually testing the live app:** `streamlit_app/Home.py`'s
citation-rendering code unconditionally read `state["search_result"].candidates` to look up a
record's title for the "Cite: ..." expander label — but `search_result` doesn't exist in state
whenever a turn ends before `search_node` ever runs, which is true even just for an ordinary
clarifying question (confirmed live: asking "help" alone crashed with `KeyError: 'search_result'`).
Fixed with `state.get("search_result")` and an empty fallback dict when it's absent; `citations`
is itself always empty on those turns anyway (nothing was ever selected to show), so the guard
costs nothing on the normal path.

## Layer 2, slice 6: explicit human-request fast path (built)

The first of CLAUDE.md's two originally-sketched deterministic fast paths (the second, citation
formatting, shipped inline instead — see the citation tool section once that slice lands). A new
`entry` node runs before `plan`: `agent/router.py`'s `is_human_request()` is a small, deliberately
conservative regex match — no LLM call, matching CLAUDE.md's "deterministic fast path" requirement
exactly. A clear match ("talk to a human," "connect me with a librarian," a bare "human") escalates
immediately via the existing `agent/escalation.py` block, skipping `plan`/`search`/`rag`/`compose`
entirely. Kept narrow on purpose: an unusual phrasing of the same request just falls through to
the normal flow instead of a false positive, the safer failure mode for a fast path that skips
ranking.

**Real regression found and fixed by actually running this live, not just testing offline:**
`--debug` (CLI) and the Streamlit debug toggle both assumed `state["plan"]` always exists and
crashed with `KeyError: 'plan'` the moment the fast path actually fired, since `entry` ends the
turn before `plan_node` ever runs. Both now check `"plan" in state` first. Confirmed live
afterward: the fast path short-circuits cleanly (`timings` shows only `entry`), and an ordinary
query right after it in the same session is unaffected.

**Also picks up the multi-turn escalation context fix** (see "full conversation context in the
escalation paste block" below): `entry_node` now passes `state.get("history", [])` to
`build_escalation_block()`, so an explicit "talk to a human" fired after earlier turns still
quotes the whole conversation, not just that one line — and passes `include_current=False`, since
`state["message"]` here is the "talk to a human" trigger itself, not a real topic; only the prior
turns are quoted (confirmed live: with one prior turn, the paste block quotes just that topic, not
the trigger phrase; with none at all, it falls back to a generic "I would like assistance" line).

## Layer 2, slice 5: full conversation context in the escalation paste block (fixed)

A real bug found by the user manually testing the live app: the "Context to paste into the chat"
block always quoted only `state["message"]` — the current turn — even when the user had already
asked something earlier in the conversation. A librarian picking up the pasted context cold had
no idea the person had already been asking about a specific topic; they'd see only the final,
often context-free sentence ("can someone help me directly?").

**Fixed in `agent/escalation.py`.** `build_escalation_block()` gained an optional `history`
parameter (the same `[{"user", "assistant"}, ...]` list the graph already carries). When given
at least one prior turn, every question the user asked this conversation is numbered in order,
ending with the current message, instead of just the last one. With no history (the common case —
a first-turn escalation), the block is unchanged: a single line, not a numbered list of one.
`agent/compose.py`'s `_render_no_answer()` (the "could not find results" / "not confident" paths)
threads `state.get("history", [])` through. Confirmed live: a two-turn session (a real mental
health search, then a nonsense query that genuinely returns zero OneSearch results) produced a
paste block numbering both questions, not just the second.

**Follow-up, also found live by the user:** for an explicit human-request or dissatisfaction
escalation, the current `message` *is* the "please get me a human" trigger itself ("can you bring
in human in loop i want more info"), not a real topic — quoting it added a line that tells a
librarian nothing useful. `build_escalation_block()` gained `include_current: bool = True`;
callers whose `message` genuinely is the substantive question (both `_render_no_answer()` call
sites — an ordinary search that came back empty or low-confidence) leave the default, callers
whose `message` is just the human-request trigger (the `entry` fast path, `human_help`) pass
`include_current=False` so only prior real topics are quoted. With no prior turns at all (a
first-turn "talk to a human" with nothing else asked), there's no topic to show either, so the
context line falls back to a plain "I would like assistance" rather than quoting the trigger
phrase.

## Layer 2, slice 7: anonymized conversation logging (built)

CLAUDE.md capability 7, from the pre-narrowing brief: every turn writes one anonymized log row —
role, course if given, a topic category — never raw query text or an identifier. Full design:
`docs/superpowers/specs/2026-10-05-anonymized-logging-design.md`. Built autonomously; decisions
below are documented for later review, not agreed in conversation.

**Topic category comes from the planner, at zero extra LLM cost.** `QueryPlan` gained a
`topic_category` field (a fixed 10-value enum: `sciences`, `health_and_medicine`,
`engineering_and_technology`, `social_sciences`, `business_and_economics`,
`humanities_and_arts`, `law`, `education`, `library_services`, `other`), filled by the *same*
planning call that already produces `keywords`/`resource_type`/`peer_reviewed` — not a new LLM
round trip. Confirmed live: a real "chemistry resources" query logged as `sciences`.

**`course` is always `None` today** — nothing in the UI asks for it yet; the column exists for
when something does. `escalated` and `had_results` were added beyond the brief's literal field
list because the staff dashboard slice (next) needs them for "escalation rate," and neither is
identifying. No session or user identifier of any kind — every row is independent, so nothing
lets someone reconstruct one person's whole conversation even without a name attached.

**Logging happens in the caller (`agent/cli.py`, `streamlit_app/Home.py`), not in
`agent/graph.py`**, which stays free of disk I/O — both callers already have the full `state`
dict right after `graph.invoke()`. `analytics/log_writer.py`'s `log_turn()` never raises: a
logging failure must not break a chat turn that already succeeded.

**A real test-isolation bug found and fixed before this was called done:** the first version of
`agent/cli.py`'s tests didn't inject a fake logger, and running the test suite silently wrote a
real row into `analytics/sessions.db` — confirmed by literally finding the file after a test run.
Every CLI test now injects a no-op `log_fn`; `analytics/*.db` is also now gitignored as a second
layer of defense.

## Layer 2: Agent Orchestration (design only — do not build until Layer 1 acceptance criteria above are met)

This section records the agreed shape of Layer 2 so the design isn't lost between sessions. Layer 2 owns everything Layer 1 explicitly does not: ranking candidates, reasoning about which one actually answers the user's question, composing the natural-language reply, deciding when to escalate, and formatting citations. Full capability list (role capture, librarian recommendation, escalation, anonymized logging, staff dashboard) lives in the pre-narrowing project brief, `git show dd388a3:CLAUDE.md` — that document is still the source of truth for what Layer 2 must cover, this section only fixes *how* the orchestration graph is shaped.

Two considerations drove this shape: staying faithful to the spec's intent ("ranking, reasoning about which result best answers the user's actual question, and composing a natural-language answer all belong to Layer 2" — this implies a real reasoning step, not keyword dispatch), and not burning an LLM call plus a full OneSearch/RAG round trip on every trivial follow-up turn.

**Two deterministic fast paths (no LLM call, no OneSearch/RAG call):**
1. The user explicitly asks for a human ("talk to a real person", "human", etc.) → escalate immediately, passing conversation context to the live chat handoff.
2. The user asks to format a citation for a candidate already resolved earlier in the same conversation → format directly with the deterministic citation tool (APA/MLA/Chicago), skip OneSearch and RAG entirely.

**Default path (everything else):**

0. **Role** is captured once at the start of the conversation and kept in graph state. It is never sent to OneSearch; it steers planning and ranking below.
1. **Query planning (one small LLM call, structured output).** Input: the user's message, the role, and the last few turns. Output, validated against a schema:
   - `keywords`: 1-6 content words, no filler ("find me", "peer reviewed", "journals" are stripped out, they are not keywords),
   - `resource_type`: one of Layer 1's `RESOURCE_TYPES` or `null`,
   - `peer_reviewed`: bool,
   - `needs_clarification`: optional short question when the request is genuinely ambiguous (e.g. "peer reviewed journals" = journal titles vs. articles from peer-reviewed journals).
   Role informs defaults, e.g. an undergraduate asking about a broad topic leans to books / overview guides, a graduate student or faculty member leans to peer-reviewed articles. Explicit user wording always overrides the role default. Planning failures (bad JSON, unknown `resource_type`) fall back to `keywords = raw message`, no filters. This call can share a prompt with the fast-path classification so the turn costs one LLM round trip, not two (~1-2s of the 3-6s budget).
   Example: "find me peer reviewed journals on mental health" → `keywords="mental health"`, `resource_type="journals"`, `peer_reviewed=true`.
2. **Fan out in parallel** with the planned request: Layer 1 `search_with_fallback(keywords, resource_type=..., peer_reviewed=...)` and the LibGuides/librarian-directory RAG similarity search (which also uses the planned keywords, not the raw sentence). If OneSearch returns 0 results, refine once (two searches total): same keywords, all filters dropped, and tell the user which filters were relaxed.
3. **Rank + Compose (LLM)** sees both result sets and:
   - scores confidence and picks the top 1-3 candidates worth showing,
   - calls Layer 1's bounded `fetch_detail()` only for those chosen candidates (never looped over the full result set, this is the same rule Layer 1 enforces at the function-signature level, Layer 2 must not defeat it by calling it in a loop),
   - folds citation formatting into the composed answer when relevant,
   - escalates to a human librarian when confidence is low, instead of forcing an answer.

Every turn, regardless of which path it took, writes one anonymized log entry (role, course if given, topic category — never raw query text or an identifier).

## Reference scripts (already validated, use as the starting point)

Two scripts already prove the approach works end to end against real UB
data:
- `onesearch_probe.py` — plain `requests` version, demonstrates the comma
  encoding fix; will hit the bot-protection 400 on its own, kept as
  reference for the request-shape fix, not for direct reuse.
- `onesearch_probe_playwright.py` — working Playwright version. Confirms
  real data retrieval, the `skipDelivery` route-interception fix, and
  fetches the full-detail page for the top result. This is the closer
  starting point; the production adapter should factor its logic into a
  proper module (session minting separated from per-query search,
  cookie reuse instead of a fresh browser launch per call, as described
  above) rather than relaunching a browser every run.

## Suggested module structure for this phase

```
onesearch/
  __init__.py
  session.py       # Playwright session mint + cookie extraction, 15-min reuse
  client.py         # requests.Session-based search() and fetch_detail()
  parser.py         # field extraction per the tables above, name/contributor cleanup
  cache.py          # short-TTL query cache
  models.py         # dataclasses: SearchCandidate, ResourceDetail
tests/
  test_parser.py    # unit tests against saved sample JSON, not live calls
  test_client.py    # mocked, don't hit the live endpoint in CI
```

Save at least one real captured response (brief list and full detail) as a
fixture file once confirmed, so the parser has something to test against
without depending on live network access or UB's endpoint staying up
during CI runs.

## Acceptance criteria for this phase

- [ ] Session minted once via Playwright, cookies reused for ~15 min, not
      re-minted per search.
- [ ] Search returns recommendation fields correctly for a real query.
- [x] Detail fetch confirmed working live (2026-09-24) for both an online
      resource (link) and a physical item (call number + location).
- [ ] Detail fetch only ever runs for a bounded, small number of results,
      never looped over the full result set.
- [ ] Hard timeout + graceful fallback implemented.
- [ ] Basic response caching in place.
- [ ] Real, identifying User-Agent on every request.

## Development rules

These apply to all future code-building work in this repo, both layers.

- Build in phases, not as one large dump. A phase is one cohesive, testable unit — one module and its tests, or a tightly coupled pair (e.g. a scraper and the parser it feeds). Do not start the next phase until the current one is committed.
- After finishing each phase, explain in plain language what the code does before moving on. The person running this needs to actually understand what each piece does, not just receive a diff to trust blindly.
- Once a phase is reviewed and accepted, commit it, then `git push` to `origin`. Never batch multiple phases into a single commit or a single push.
