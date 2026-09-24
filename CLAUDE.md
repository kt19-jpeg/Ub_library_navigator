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
2. **Fan out in parallel** with the planned request: Layer 1 `search_with_fallback(keywords, resource_type=..., peer_reviewed=...)` and the LibGuides/librarian-directory RAG similarity search (which also uses the planned keywords, not the raw sentence). If OneSearch returns 0 results, refine once (drop `resource_type` first, then `peer_reviewed`) and tell the user which filter was relaxed.
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
