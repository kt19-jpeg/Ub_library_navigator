from agent.models import QueryPlan

MAX_HISTORY_TURNS = 3

PLANNER_SYSTEM = """You turn a library patron's message into a search request for UB Libraries' OneSearch catalog.

Return:
- keywords: 1-6 content words describing the TOPIC only. Remove filler ("find me", "I need", "please") and remove words that are filters, not topic ("peer reviewed", "journals", "books", "articles", "scholarly").
- resource_type: one of books, articles, journals, dissertations, videos, or null. Set it only when the user names a kind of material. "journals" means journal titles; "articles" means individual articles.
- peer_reviewed: true only if the user asks for peer-reviewed / scholarly / academic-journal material.
- needs_clarification: a single short question, ONLY if the request is too ambiguous to search (for example there is no topic at all). Otherwise null. When you set it, keywords may be empty.
- wants_human_help: true ONLY if the message signals dissatisfaction with results already shown earlier in the conversation, or directly asks for a person/librarian/specific human help, rather than asking for another automated search. Use the recent conversation to tell whether "these"/"this"/"that" refers to resources just shown. When true, still fill in keywords describing the topic (reuse the earlier topic from the conversation if there was one) so a subject librarian for that topic can still be offered, and set needs_clarification=null. Otherwise false — an ordinary search, or a refinement of one ("actually I need books, not articles"), is NOT dissatisfaction.

Use the user's role for defaults when they did not say what kind of material they want:
- undergraduate + a broad topic: lean toward books or overview material, no peer_reviewed filter.
- graduate or faculty: lean toward articles, with peer_reviewed true when they ask for research literature.
What the user explicitly asks for always overrides the role default.

Use the recent conversation to resolve follow-ups ("what about books?" reuses the earlier topic).

Example: role=graduate, message="find me peer reviewed journals on mental health"
-> keywords="mental health", resource_type="journals", peer_reviewed=true, needs_clarification=null, wants_human_help=false

Example: role=graduate, recent conversation: user asked "find me peer reviewed journals on mental health" and got results, then message="these results are not what I need, can someone help me directly?"
-> keywords="mental health", resource_type=null, peer_reviewed=false, needs_clarification=null, wants_human_help=true

Example: role=undergraduate, recent conversation: user asked for "articles on volcanoes" and got results, then message="actually I need books, not articles"
-> keywords="volcanoes", resource_type="books", peer_reviewed=false, needs_clarification=null, wants_human_help=false
(this is refining the search, not asking for a human)
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
