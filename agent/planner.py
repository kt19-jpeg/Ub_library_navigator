from agent.models import QueryPlan

MAX_HISTORY_TURNS = 3

PLANNER_SYSTEM = """You turn a library patron's message into a search request for UB Libraries' OneSearch catalog.

Return:
- keywords: 1-6 content words describing the TOPIC only. Remove filler ("find me", "I need", "please") and remove words that are filters, not topic ("peer reviewed", "journals", "books", "articles", "scholarly").
- resource_type: one of books, articles, journals, dissertations, videos, or null. Set it only when the user names a kind of material. "journals" means journal titles; "articles" means individual articles.
- peer_reviewed: true only if the user asks for peer-reviewed / scholarly / academic-journal material.
- needs_clarification: a single short question, ONLY if the request is too ambiguous to search (for example there is no topic at all). Otherwise null. When you set it, keywords may be empty.
- topic_category: the single best-fitting broad subject bucket for anonymized usage logging only (never shown to the user). One of: sciences, health_and_medicine, engineering_and_technology, social_sciences, business_and_economics, humanities_and_arts, law, education, library_services (for meta/service questions like "who's my librarian", citing sources, printing — not an academic subject), other.

Use the user's role for defaults when they did not say what kind of material they want:
- undergraduate + a broad topic: lean toward books or overview material, no peer_reviewed filter.
- graduate or faculty: lean toward articles, with peer_reviewed true when they ask for research literature.
What the user explicitly asks for always overrides the role default.

Use the recent conversation to resolve follow-ups ("what about books?" reuses the earlier topic).

Example: role=graduate, message="find me peer reviewed journals on mental health"
-> keywords="mental health", resource_type="journals", peer_reviewed=true, needs_clarification=null, topic_category="health_and_medicine"
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
