# UB Library Navigator

This file is the project brief for Claude Code. Read this before starting any work in this repo. It captures the finalized scope, architecture, and conventions agreed on for this capstone project. Treat it as the source of truth alongside `docs/proposal.docx`.

## Project context

MS Data Science capstone project at the University at Buffalo, supervised by Dr. Justin Del Vecchio, built in coordination with the UB Libraries team.

The project originally started as a broad multi-tool agentic RAG system, but UB Libraries raised two concerns: the underlying content is too diverse for one system to confidently answer everything, and the needs of undergraduates, graduate students, and faculty differ enough that a single generic answer does not serve all of them well. The project was reshaped around those two concerns into a focused navigation and triage tool rather than a comprehensive answer engine.

## What this system is

The UB Library Navigator is a role-aware, first-tier triage agent. It does not try to answer every research question directly. It routes each request to the right resource, the right person, or the right tool, and escalates to a human librarian through the existing live chat system when it cannot confidently help. It sits in front of the existing live chat as an additional layer, not a replacement for it.

## Core capabilities

1. Role capture at the start of the conversation. The user selects undergraduate, graduate, or faculty. This is explicit, not inferred, to keep the system transparent and predictable.
2. Resource, database, and item discovery via OneSearch. UB Libraries' own OneSearch (their Primo-based discovery layer) is used as a first-pass narrowing search across databases, articles, and physical items, including book availability and shelf location. If the first pass looks weak, the query is refined once and searched again, two searches total, then the agent moves on regardless. This replaces what used to be two separate capabilities (resource recommendation and catalog lookup), merged because they hit the same underlying UB system.
3. RAG enrichment on the narrowed results. Once OneSearch has narrowed the field, a smaller RAG pass over the curated LibGuides and librarian-directory vector store adds the institutional context OneSearch itself doesn't carry: how to actually use a given resource, and which subject librarian owns that area.
4. Librarian recommendation. Surfaces the name and contact details of the subject librarian responsible for the relevant area, generated as part of the RAG enrichment step above.
5. Citation formatting tool. A deterministic tool that formats a source in APA, MLA, or Chicago style.
6. Escalation and handoff. When a question is ambiguous, out of scope, or confidence is low, hands off to a human librarian through the existing live chat system, passing along conversation context.
7. Conversation logging. Every session is logged in fully anonymous form, capturing role, course if provided, and a topic category rather than raw question text. No name or identifier is ever attached to a log entry. This exists purely so the library can see usage patterns over time.
8. Staff usage dashboard. Library-staff-only view summarizing query volume by role, topic category breakdown, and escalation rate.

## Architecture and tech stack

The stack is organized in layers. Each layer has one job.

**Presentation: Streamlit**
Renders the role selector and chat interface for end users, and a separate staff-only dashboard page. Deployed on Streamlit Community Cloud as a live public app.

**Agent orchestration: LangChain + LangGraph**
LangChain provides prompt templates, tool wrappers, and output parsers. LangGraph structures the agent as a graph of nodes: a router node classifies role and query type, then hands off to the matching node, the OneSearch tool (search, refine once, RAG-enrich), the citation tool, or escalation. LangGraph tracks conversation state across nodes.

**Two-stage retrieval: OneSearch narrows, then RAG enriches**
This is the key architectural pattern for this project and worth naming explicitly in the capstone report. UB's own database directory alone runs into the hundreds of entries, too much volume to embed and search well from scratch. Rather than building a local vector index over that entire catalog, the agent queries OneSearch live first, UB's own maintained discovery layer, to narrow the field down to a handful of relevant candidates (databases, articles, or physical items). Only then does a much smaller, local RAG pass run, over the curated LibGuides and librarian-directory content, to add the institutional context (how to use it, who owns it) that a generic discovery result never includes. Broad live search narrows; small local RAG enriches.

**Storage: Chroma + SQLite or Postgres**
Chroma holds embedded chunks of LibGuides, FAQs, and the librarian directory, the curated content used for RAG enrichment. It does not hold the database directory, since that discovery is now handled live via OneSearch instead of scraped and embedded. SQLite or Postgres is a separate, simpler database holding the anonymized session logs that power the dashboard. These two stores serve unrelated purposes and should not be conflated.

**Data pipeline: Playwright + BeautifulSoup, then OpenAI embeddings**
Playwright drives a real browser to load LibGuides pages that may render dynamically. BeautifulSoup parses the resulting HTML into clean text. LangChain text splitters chunk that text, and OpenAI's text-embedding-3-small model embeds each chunk into Chroma. This pipeline runs once initially and then again on a weekly schedule, and now covers a smaller scope than originally planned since the database directory is no longer scraped.

**External systems (pending confirmation): OneSearch/Primo discovery API, live chat platform**
OneSearch is UB Libraries' actual discovery layer (confirmed, not assumed), very likely built on the SUNY-wide Ex Libris Primo/Alma platform, and it covers books, articles, databases, and journals across UB, SUNY, and CUNY. What needs confirmation is whether UB Libraries can grant API access to it, covering both broad discovery search and item-level availability lookups, not just one or the other. The live chat platform currently in use also needs confirmation so the escalation node can integrate with it correctly. Do not hardcode assumptions about either system until confirmed. Build the OneSearch and escalation nodes with a clear interface boundary so the actual integration can be swapped in once details are confirmed.

**Automation: GitHub Actions**
Runs the scraping and embedding pipeline on a weekly schedule so the vector store does not go stale, and can run the evaluation suite on the same schedule.

**Evaluation: RAGAS + LangSmith**
RAGAS scores retrieval precision, recall, and answer faithfulness against a golden test set of 30 to 50 questions, tagged by role. LangSmith traces the internal steps LangGraph takes for a given request, which is what makes it possible to debug a bad answer rather than just observe that it happened.

## Data sources

- UB Libraries subject research guides (LibGuides)
- UB Libraries FAQ and citation-style help pages
- Subject librarian directory (maps subject areas to librarian names and contact details)
- OneSearch / Primo discovery API, queried live, not scraped or embedded, covers database, article, and item discovery
- Anonymized conversation logs, generated by the system itself, used only to power the internal dashboard

The UB Libraries database directory is no longer scraped separately. OneSearch already covers that discovery need live and better than a static scrape could, which also removes one full scraper and one source of staleness from the maintenance burden.

Before any large-scale scraping or API integration, the plan is to contact UB Libraries' web or digital services team for an official export or explicit permission, and to scope data collection to publicly available content and explicitly granted access only.

## Repo structure

```
ub-library-navigator/
├── .claude/settings.json          # attribution off (see Commit conventions below)
├── .github/workflows/weekly-refresh.yml
├── streamlit_app/
│   ├── Home.py                    # role selector + chat interface
│   └── pages/1_Staff_Dashboard.py # staff-only, needs its own access gate
├── agent/
│   ├── graph.py                   # LangGraph definition
│   ├── router.py
│   ├── onesearch_tool.py          # merged: OneSearch search+refine, then RAG enrichment
│   ├── citation_tool.py
│   └── escalation.py
├── scraper/
│   ├── scrape_libguides.py
│   ├── scrape_faqs.py
│   └── scrape_librarian_directory.py
├── pipeline/
│   ├── chunk.py
│   ├── embed.py
│   └── refresh.py
├── analytics/
│   ├── db.py
│   └── log_writer.py
├── evaluation/
│   ├── golden_test_set.json
│   ├── run_ragas.py
│   └── langsmith_config.py
├── data/raw/          # gitignored
├── data/processed/
├── vectorstore/       # gitignored, Chroma's persisted files
├── tests/
├── docs/proposal.docx
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Note: Streamlit's multipage convention means anything in `streamlit_app/pages/` becomes a nav item automatically, but Streamlit does not gate access on its own. The staff dashboard page needs its own access check built in. The method (UB SSO versus a shared staff login) is still pending confirmation from the library, so build this behind a simple, swappable interface for now rather than hardcoding a specific auth method.

## Commit conventions

Never add a Co-Authored-By line or a "Generated with Claude Code" line to any commit message or pull request description. This repo's `.claude/settings.json` sets attribution to empty strings, but treat this as a second, explicit layer of enforcement:

```json
{
  "attribution": {
    "commit": "",
    "pr": ""
  }
}
```

Commit message format: `type: short description`, for example `feat: add citation formatting node` or `fix: handle empty catalog response`. All commits should be authored under the repo owner's own git identity, not a generic or default one.

## Timeline (14 weeks)

| Weeks | Milestone |
|---|---|
| 1-2 | Outreach to UB Libraries for OneSearch/Primo discovery API access, live chat integration details, and the subject librarian directory; data acquisition begins |
| 3-4 | Build the embedding pipeline and vector store for LibGuides, FAQs, and the librarian directory |
| 5-6 | Build the LangGraph router, role handling, and the basic chat interface |
| 7-8 | Integrate the citation formatting tool, the merged OneSearch tool (search, refine, RAG enrichment), and the anonymized conversation logging pipeline |
| 9-10 | Build the escalation and live chat handoff logic; complete public deployment |
| 11-12 | Run RAGAS evaluation, including escalation accuracy, refine routing and prompts, and build the staff usage dashboard |
| 13 | Build the automated weekly refresh pipeline and feedback collection |
| 14 | Final polish, written report, and demo preparation |

## Known risks

- Role misclassification: a user may pick the wrong role, or a faculty member may want undergraduate-level guidance. Allow the role to be changed mid conversation.
- Escalation quality: too many handoffs undermines the point of the agent, too few undermines trust. Track escalation accuracy as an explicit evaluation metric.
- OneSearch API access and reliability: build a fallback that links to the manual OneSearch page if a live call fails.
- Two-stage retrieval latency: a live OneSearch search plus one refinement plus a RAG enrichment pass is slower than a single static lookup. Cap refinement at one retry (as decided), and show a loading state so this reads as thorough rather than slow.
- OneSearch coverage gaps: OneSearch narrows the field but doesn't carry institutional context like how to use a resource or who owns it. The RAG enrichment step over LibGuides and the librarian directory stays necessary, not optional, even though OneSearch now handles the broad discovery search.
- Librarian directory staleness: fold this directory into the same weekly refresh pipeline as the rest of the content.
- Re-identification in small groups: even with topic-only tagging and no name attached, a rare combination of course, role, and topic could be identifiable. Aggregate dashboard views and suppress any breakdown below a small count threshold.
- Dashboard access control: gate the dashboard behind proper staff authentication, never a shared or public link.
- Scraping permissions and content staleness: contact UB Libraries early, build the weekly refresh pipeline early rather than as a late addition.

## Open items still pending confirmation from UB Libraries management

Do not assume answers to these. Build with a clear interface boundary so these can be plugged in once confirmed.

- Whether UB Libraries can grant API access to OneSearch/Primo discovery search, covering both broad search (databases, articles, items) and item-level availability lookups, one confirmation now covers what used to be two separate asks
- Which live chat platform is currently in use, and how a handoff from the agent into that system should work
- Who maintains the subject librarian directory, and whether it can be shared or accessed programmatically
- Whether the library team wants input on how role-based recommendations are tuned
- How the staff dashboard should be access-restricted (UB SSO/Shibboleth versus a shared staff login)
- What topic and category taxonomy the library wants used for tagging conversations
- Whether this data collection needs any privacy, IRB, or institutional review sign-off, even though it is aggregate and anonymized by design

## Expected deliverables

- A live, publicly deployed navigation and recommendation agent, acting as a first-tier layer in front of the existing live chat
- A documented source code repository covering the data pipeline, agent architecture, OneSearch integration, logging, and deployment configuration
- A staff-only usage dashboard summarizing anonymized query patterns by role, topic category, and escalation rate
- An evaluation report presenting RAGAS metrics and escalation accuracy against the golden test set
- A final capstone report and presentation
