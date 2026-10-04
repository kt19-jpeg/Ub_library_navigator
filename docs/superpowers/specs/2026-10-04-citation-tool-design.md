# Citation Tool (Layer 2, Slice 5)

Date: 2026-10-04. Status: approved in conversation, pending written-spec review.

## Goal

Deterministic APA/MLA/Chicago citation formatting for every result the agent shows, rendered
inline (collapsed by default) with each result — not a conversational "cite that" fast path.

## Why not the conversational fast path (CLAUDE.md's original design)

CLAUDE.md's Layer 2 design names a fast path: "the user asks to format a citation for a candidate
already resolved earlier in the same conversation." Building that for real would require (a)
persisting structured candidate data across turns — today's `history` is plain
`{"user": str, "assistant": str}` text pairs, nothing structured — and (b) resolving which exact
prior result a follow-up message refers to, both genuinely ambiguous and out of proportion to
what this slice needs. Rendering a citation for every result the moment it's shown sidesteps both
problems entirely: there is never a "which one did you mean," because the citation sits right
next to the result it belongs to. The conversational fast path is deferred, not abandoned; noted
as a known follow-up, same pattern as the RAG pipeline's and evaluation suite's documented
limitations.

## Finding: the data doesn't exist yet, but is already free

Confirmed live 2026-10-04 against real UB OneSearch data: a real peer-reviewed article's `pnx.addata`
has everything a complete citation needs —

```
atitle, jtitle, volume, issue, spage, epage, pages, doi, issn, pub, addau
```

— and a real book's `addata` has:

```
btitle, pub, cop, isbn, aulast, aufirst, auinit, addau
```

**None of this currently flows into `SearchCandidate`** (`onesearch/models.py`): the parser only
keeps a single flattened `author` string and no publisher/journal/volume/pages/DOI/ISBN at all.
But `addata` is already part of the brief search response Layer 1 already fetches — this costs no
extra network call, only a parser change.

## Layer 1 change: new `SearchCandidate` fields

Added, not replacing the existing `author` field (nothing that reads it today breaks):

| Field | Type | Source |
|---|---|---|
| `authors` | `list[str] \| None` | `addata.addau` (already "Last, First" strings; present on both books and articles) |
| `journal_title` | `str \| None` | `addata.jtitle` (articles only; its absence is how the formatter tells a journal article from a book) |
| `volume`, `issue` | `str \| None` | `addata.volume`, `addata.issue` |
| `pages` | `str \| None` | `addata.pages[0]` if present, else built from `spage`/`epage` |
| `doi` | `str \| None` | `addata.doi` |
| `isbn` | `str \| None` | `addata.isbn[0]` |
| `publisher` | `str \| None` | `addata.pub` |
| `place_of_publication` | `str \| None` | `addata.cop` |

## The formatter: `agent/citation_tool.py`

`format_citation(candidate: SearchCandidate, style: Literal["apa", "mla", "chicago"]) -> str`.
Pure, deterministic string formatting, no LLM call — matches the original brief's "a deterministic
tool." Branches on `candidate.journal_title`: present → journal-article form; absent → book-like
form (using `publisher`/`place_of_publication`/`isbn`). Journal and book titles are rendered as
`*italic*` Markdown, which both the CLI's plain-text output and Streamlit's `st.write` already
render correctly — no new rendering mechanism needed.

**Documented simplification, not full style-guide fidelity:** real style guides have many
author-count edge cases (APA lists up to 20 authors before "et al.", MLA/Chicago cut off
differently). This implements the common case — 1-2 authors in full, 3+ as "first author et
al." — and states that limitation rather than claim complete bibliographic-standard compliance.
Missing fields (a book has no DOI; an older record may lack ISBN) are omitted cleanly, never
rendered as a literal "None" or empty parenthetical.

## Presentation

- **Streamlit** (`streamlit_app/Home.py`): one `st.expander("Cite this")` per shown result,
  closed by default, containing all 3 styles.
- **CLI** (`agent/cli.py`): a toggle alongside the existing `--debug` flag/approach (exact
  mechanism — flag vs. `/citations` command — decided in the plan) that shows the same block for
  every result when on, since the CLI has no native expand/collapse.
- **`compose.py`** stays UI-agnostic: it attaches the formatted citations (all 3 styles, per
  shown result) to its output data; each caller (CLI, Streamlit) decides how to display them,
  same separation of concerns `render_answer`/`render_rag_sections` already follow.

## Testing

All offline, using the real `addata` shapes captured live 2026-10-04 as fixtures (a peer-reviewed
journal article and a book), not synthetic data:

- `onesearch/parser.py`: the new fields parse correctly from real `addata`, and are `None`
  (not a crash) when absent.
- `agent/citation_tool.py`: each style's output against the real article and real book fixtures;
  missing-field graceful degradation; the 3+ authors "et al." case.
- `compose.py`: citations are attached to shown results, one per style, using the real formatter
  (not reimplemented inline).
- CLI / Streamlit rendering: covered by their existing test patterns (`FakeGraph` for the CLI;
  manual live check for Streamlit, same as every prior slice — no `AppTest` suite exists for
  `Home.py` yet, that's part of the still-separate, not-yet-planned "finish the Streamlit UI
  properly" work).

## Files

```
onesearch/models.py (modify)   new SearchCandidate fields
onesearch/parser.py (modify)   parse the new fields from addata
agent/citation_tool.py          format_citation() and the per-style formatters
agent/compose.py (modify)       attach citations to each shown result
agent/cli.py (modify)           citation toggle + rendering
streamlit_app/Home.py (modify)  st.expander per result
tests/test_parser.py (modify)   new field parsing, real fixtures
tests/test_agent_citation_tool.py        formatter tests
tests/test_agent_compose.py (modify)     citations attached to output
tests/fixtures/onesearch_article_addata.json  real captured article addata
tests/fixtures/onesearch_book_addata.json     real captured book addata
```

Dependencies: none added.
