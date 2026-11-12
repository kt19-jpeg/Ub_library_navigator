import json
import logging
import re

from agent.citation_tool import format_citation
from agent.escalation import build_escalation_block
from agent.models import AgentState, FollowupAnswer, RagContext, Selection, SelectionItem
from onesearch.client import OneSearchClient
from onesearch.models import ResourceDetail, SearchCandidate

logger = logging.getLogger(__name__)

MAX_SHOWN = 3
DESCRIPTION_CHARS = 300

# A deliberately narrow, deterministic check (no LLM call) for the user directly
# asking about library services — contacting a librarian, citing, printing,
# browsing databases — rather than searching for resources on a topic. Mirrors
# the brief's "library_services" meta-question category. False negatives just
# fall through to the gentler one-line offer below; false positives just show
# the full footer a turn earlier than strictly needed. Either is benign.
LIBRARY_HELP_RE = re.compile(
    r"subject librarian|"
    r"(contact|talk to|speak (with|to)) (a |my )?librarian|"
    r"who (is|can i contact).*(librarian|contact)|"
    r"how (do|to) (i )?cite|citation (help|guide)|citing sources|"
    r"how (do|to) (i )?print|printing (help|guide)|"
    r"browse (the )?databases|list of databases",
    re.IGNORECASE,
)

LIBRARY_HELP_OFFER = (
    "Want a subject librarian, related help guides, or database suggestions for this topic too? "
    "Just ask."
)


def is_library_help_request(message: str) -> bool:
    return bool(LIBRARY_HELP_RE.search(message))


# Same deterministic, no-LLM-call pattern as LIBRARY_HELP_RE above. The
# Streamlit UI computes citations for every result regardless (cheap, local,
# no extra API call) but only renders them when the user's own message asked
# for them - citations cluttering every answer by default was a real
# usability complaint, the same shape as the librarian/help/database footer
# one above it.
CITATION_REQUEST_RE = re.compile(
    r"\bcit(e|ing|ation|ations)\b|\b(apa|mla)\b|\bchicago\b|"
    r"\bbibliography\b|\bworks cited\b|\breference list\b",
    re.IGNORECASE,
)


def wants_citations(message: str) -> bool:
    return bool(CITATION_REQUEST_RE.search(message))


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
    message: str,
    library_service: bool = False,
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
    # Only dump the full librarian/help/database footer when the user actually
    # asked about library services directly. Otherwise, having just handed them
    # resources, offer it instead of appending it unasked (confirmed as a real
    # usability complaint: patrons asking for resources don't want every answer
    # padded with a librarian contact and database links they didn't ask for).
    if library_service or is_library_help_request(message):
        rag_lines = render_rag_sections(rag_context)
        if rag_lines:
            out.append("")
            out.extend(rag_lines)
    elif not rag_context.is_empty():
        out.append("")
        out.append(LIBRARY_HELP_OFFER)
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


def render_human_help_answer(
    role: str, message: str, rag_context: RagContext, history: list[dict] | None = None
) -> str:
    """For a turn where the planner decided the user wants a person, not another
    automated search (QueryPlan.wants_human_help): whatever subject-librarian/
    help-guide/database info RAG found for the topic, plus the full live-chat
    escalation block. No OneSearch results are shown here — the user already
    said the catalog results weren't what they needed."""
    out = ["Let's get you some direct help with this."]
    rag_lines = render_rag_sections(rag_context)
    if rag_lines:
        out.append("")
        out.extend(rag_lines)
    out.append("")
    # include_current=False: `message` here is the dissatisfaction/human-help
    # trigger itself ("can you bring in human in loop i want more info"), not
    # a real topic - quoting it tells a librarian nothing; only prior turns
    # (the actual question the resources were for) are worth showing.
    out.append(build_escalation_block(role, message, [], history, include_current=False))
    return "\n".join(out)


def _render_no_answer(
    headline: str,
    notes: list[str],
    more_link: str,
    rag_context: RagContext,
    role: str,
    message: str,
    history: list[dict] | None = None,
) -> str:
    out = [headline]
    out.extend(f"Note: {n}" for n in notes)
    out.append(f"You can browse the full OneSearch results here: {more_link}")
    rag_lines = render_rag_sections(rag_context)
    if rag_lines:
        out.append("")
        out.extend(rag_lines)
    out.append("")
    out.append(build_escalation_block(role, message, notes, history))
    return "\n".join(out)


def render_library_service_answer(rag_context: RagContext, more_link: str) -> str:
    """A library-services follow-up ("who is the subject librarian for X", "how do I
    print") whose OneSearch lookup found nothing useful. The RAG info IS the answer,
    so it is presented as one - not framed as a failed search or pushed to escalation."""
    out = ["Here is who and what can help with that:", ""]
    out.extend(render_rag_sections(rag_context))
    out.append("")
    out.append(f"You can also browse OneSearch directly: {more_link}")
    return "\n".join(out)


def _is_library_service(plan, message: str) -> bool:
    return plan.topic_category == "library_services" or is_library_help_request(message)


FOLLOWUP_SYSTEM = """You answer a University at Buffalo library patron's follow-up question about results that were ALREADY shown to them.

You are given the patron's role, their question, the details of what was shown (JSON: OneSearch results of kind "onesearch_result", and library-guide suggestions of kind "subject_librarian", "database_subject" or "help_page"), and the previous answer text.
A patron may ask how the OneSearch results differ from the database or librarian suggestions: OneSearch results are individual items (books, articles) to read or open now; a database suggestion is a whole subject collection of databases to search yourself, and a subject librarian is a person who can help with research in that area. Explain such differences from the details given.
Answer the question directly and precisely, in plain text: 2-5 short sentences, then if comparing, one short line per result starting with "- " that names the result by title and states what is distinctive about it (type, author, year, subject, focus, access).
Rules:
- Use ONLY the provided details. Never invent content, authors' arguments, page counts, or availability.
- If the details do not contain enough to answer part of the question (descriptions are often missing), say plainly what you can and cannot tell from the catalog record, and suggest opening the record link for more.
- For judgment questions ("which is better for a beginner"), give a recommendation only if the details support it, and say it is based on the catalog description, year and type, not on reading the book.
- No numbered lists, no links, no headings."""

CAPABILITIES = (
    "I can find books, articles, journals, dissertations and videos in OneSearch, name your "
    "subject librarian and help guides, suggest databases, format APA/MLA/Chicago citations when "
    "you ask, and connect you to live chat with a librarian."
)
# Fixed replies, written in code: the planner only classifies a message as chit-chat, it
# never writes the reply, so it cannot answer a real question (or invent facts about shown
# results) under the chit-chat label.
CHITCHAT_REPLIES = {
    "greeting": f"Hi! {CAPABILITIES} What would you like to look for?",
    "thanks": "You're welcome! Ask me anything else about UB Libraries.",
    "about_agent": f"I'm the UB Library Navigator, an assistant for University at Buffalo Libraries. {CAPABILITIES}",
    "off_topic": "I help with UB Libraries resources, so I can't answer that. I'm happy to help you find books, articles or databases on a topic.",
}

FOLLOWUP_OFFER = "Want me to search for more on this topic, or look at a different angle?"
FOLLOWUP_FAILED = (
    "I couldn't compare those just now. You can open each record from the results above, "
    "or ask me to search again."
)


GUIDE_MARKERS = {
    "subject_librarian": "Subject librarian:",
    "database_subject": "Browse databases in",
    "help_pages": "Helpful pages:",
}
GUIDE_KEYWORDS_SHOWN = 10
LAST_ANSWER_CHARS = 3000


def _guide_entries(state: dict) -> list[dict]:
    """The librarian/database/help-page suggestions the last answer actually displayed
    (the footer only appears for some turns, so check the answer text). Database and
    librarian subjects carry a few of their keywords so a follow-up can explain what
    that subject area covers, not just that it exists."""
    rag = state.get("rag_context")
    answer = state.get("answer") or ""
    if rag is None or rag.is_empty():
        return []
    from pipeline.subject_keywords import load_subject_keywords

    keywords = load_subject_keywords()
    entries = []
    if rag.librarians and GUIDE_MARKERS["subject_librarian"] in answer:
        top = rag.librarians[0]
        subject = top.metadata.get("subject")
        entries.append({
            "kind": "subject_librarian", "subject": subject,
            "librarians": [l["name"] for l in top.metadata.get("librarians", [])],
            "covers": keywords.get(subject, [])[:GUIDE_KEYWORDS_SHOWN],
        })
    if rag.databases and GUIDE_MARKERS["database_subject"] in answer:
        top = rag.databases[0]
        subject = top.metadata.get("subject")
        entries.append({
            "kind": "database_subject", "subject": subject,
            "database_count": top.metadata.get("count"),
            "covers": keywords.get(subject, [])[:GUIDE_KEYWORDS_SHOWN],
        })
    if rag.help_topics and GUIDE_MARKERS["help_pages"] in answer:
        entries.extend(
            {"kind": "help_page", "title": t.metadata.get("title")} for t in rag.help_topics[:2]
        )
    return entries


def extract_shown_results(state: dict) -> list[dict]:
    """Structured details of what this turn actually showed - OneSearch results AND any
    librarian/database/help-page suggestions - kept by the app for the NEXT turn so a
    follow-up ("how do these differ?", "OneSearch results vs the database
    recommendation") can be answered from them instead of triggering a fresh search.
    Empty when the turn showed nothing."""
    shown = []
    result = state.get("search_result")
    items = state.get("selection") or []
    if result and items:
        by_id = {c.record_id: c for c in result.candidates}
        details = state.get("details") or {}
        for item in items:
            candidate = by_id.get(item.record_id)
            if candidate is None:
                continue
            view = _candidate_view(candidate)
            view.pop("record_id", None)
            detail = details.get(item.record_id)
            source = detail or candidate
            view["kind"] = "onesearch_result"
            view["location"] = " / ".join(p for p in (source.library, source.sub_location) if p) or None
            view["availability"] = source.availability_status
            view["why_shown"] = item.reason or None
            shown.append(view)
    shown.extend(_guide_entries(state))
    return shown


def answer_followup(
    model, role: str, message: str, shown_results: list[dict], last_answer: str = ""
) -> str:
    """One LLM call, no search: answer a question about what was already shown. Gets the
    stored structured details plus the previous answer's own text, so it still has
    something to work from when nothing structured was stored. Never raises; a model
    failure gives a plain fallback."""
    lines = [f"role: {role}", f"question: {message}", "results shown:"]
    lines.extend(f"{n}. {json.dumps(r)}" for n, r in enumerate(shown_results, start=1))
    if last_answer:
        lines.append("the previous answer shown to the patron, verbatim:")
        lines.append(last_answer[:LAST_ANSWER_CHARS])
    try:
        structured = model.with_structured_output(FollowupAnswer)
        response = structured.invoke([("system", FOLLOWUP_SYSTEM), ("human", "\n".join(lines))])
        if isinstance(response, dict):
            response = FollowupAnswer.model_validate(response)
        text = response.answer.strip()
        if not text:
            return FOLLOWUP_FAILED
    except Exception as exc:
        logger.warning("follow-up model call failed: %r", exc)
        return FOLLOWUP_FAILED
    return f"{text}\n\n{FOLLOWUP_OFFER}"


def build_citations(
    items: list[SelectionItem], candidates: list[SearchCandidate]
) -> dict[str, dict[str, str]]:
    """record_id -> {"apa": ..., "mla": ..., "chicago": ...}, one entry per shown
    item (never looped over the full candidate list — same bounded-by-selection
    rule fetch_detail() follows)."""
    by_id = {c.record_id: c for c in candidates}
    citations = {}
    for item in items:
        candidate = by_id.get(item.record_id)
        if candidate is None:
            continue
        citations[item.record_id] = {
            style: format_citation(candidate, style) for style in ("apa", "mla", "chicago")
        }
    return citations


def _with_out_of_scope_note(answer: str, plan) -> str:
    """Prepends the planner's up-front disclaimers ahead of whatever answer was
    composed: an out-of-scope one ("I can't write your notes for you...") and an
    unsupported-refinement one ("I can't filter by availability..."), so the user
    is never silently given results that ignore part of what they asked."""
    notes = [n for n in (plan.out_of_scope_note, getattr(plan, "unsupported_request_note", None)) if n]
    if notes:
        return "\n\n".join(notes) + f"\n\n{answer}"
    return answer


def compose(state: AgentState, model, client: OneSearchClient) -> dict:
    """Pick 1-3 results (LLM), fetch their access details (bounded, batched),
    and render the answer in code so links and locations are never LLM output."""
    plan = state["plan"]
    message = state["message"]
    history = state.get("history", [])
    result = state["search_result"]
    candidates = result.candidates
    notes = list(state.get("notes", []))
    rag_context = state.get("rag_context") or RagContext()
    more_link = result.fallback_link or client.fallback_link(plan.keywords)

    library_service = _is_library_service(plan, message)

    if not candidates and library_service and not rag_context.is_empty():
        return {
            "selection": [],
            "details": {},
            "answer": _with_out_of_scope_note(render_library_service_answer(rag_context, more_link), plan),
            "citations": {},
        }

    if not candidates:
        answer = _render_no_answer(
            f'I could not find results for "{plan.keywords}".',
            notes, more_link, rag_context, state["role"], message, history,
        )
        return {
            "selection": [],
            "details": {},
            "answer": _with_out_of_scope_note(answer, plan),
            "citations": {},
        }

    try:
        selection = select_results(model, state["role"], message, candidates, rag_context)
        items = validate_selection(selection.items, candidates)
        confidence = selection.confidence
    except Exception as exc:
        logger.warning("compose model call failed, showing top OneSearch matches: %r", exc)
        items = [SelectionItem(record_id=c.record_id, reason="") for c in candidates[:MAX_SHOWN]]
        confidence = "medium"
        notes.append("I could not rank these results, so these are the top matches from OneSearch.")

    if (confidence == "low" or not items) and library_service and not rag_context.is_empty():
        return {
            "selection": [],
            "details": {},
            "answer": _with_out_of_scope_note(render_library_service_answer(rag_context, more_link), plan),
            "citations": {},
        }

    if confidence == "low" or not items:
        answer = _render_no_answer(
            "I am not confident I found a good match for that.",
            notes, more_link, rag_context, state["role"], message, history,
        )
        return {
            "selection": [],
            "details": {},
            "answer": _with_out_of_scope_note(answer, plan),
            "citations": {},
        }

    try:
        details = client.fetch_detail([item.record_id for item in items])
    except Exception:
        details = {}
        notes.append("I could not load the full access details for these.")

    answer = render_answer(items, candidates, details, plan.keywords, notes, client, rag_context, message, library_service)
    citations = build_citations(items, candidates)
    return {
        "selection": items,
        "details": details,
        "answer": _with_out_of_scope_note(answer, plan),
        "citations": citations,
    }
