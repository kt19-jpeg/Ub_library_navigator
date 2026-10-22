import json

from agent.escalation import build_escalation_block
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
    out.append(build_escalation_block(role, message, [], history))
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


def compose(state: AgentState, model, client: OneSearchClient) -> dict:
    """Pick 1-3 results (LLM), fetch their access details (bounded, batched),
    and render the answer in code so links and locations are never LLM output."""
    plan = state["plan"]
    history = state.get("history", [])
    result = state["search_result"]
    candidates = result.candidates
    notes = list(state.get("notes", []))
    rag_context = state.get("rag_context") or RagContext()
    more_link = result.fallback_link or client.fallback_link(plan.keywords)

    if not candidates:
        answer = _render_no_answer(
            f'I could not find results for "{plan.keywords}".',
            notes, more_link, rag_context, state["role"], state["message"], history,
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
            "I am not confident I found a good match for that.",
            notes, more_link, rag_context, state["role"], state["message"], history,
        )
        return {"selection": [], "details": {}, "answer": answer}

    try:
        details = client.fetch_detail([item.record_id for item in items])
    except Exception:
        details = {}
        notes.append("I could not load the full access details for these.")

    answer = render_answer(items, candidates, details, plan.keywords, notes, client, rag_context)
    return {"selection": items, "details": details, "answer": answer}
