from dataclasses import dataclass, field

from agent.models import QueryPlan
from onesearch.client import OneSearchClient
from onesearch.models import SearchResult


@dataclass(frozen=True)
class SearchOutcome:
    result: SearchResult
    notes: list[str] = field(default_factory=list)


def _describe_filters(plan: QueryPlan) -> str:
    parts = []
    if plan.resource_type:
        parts.append(f"type: {plan.resource_type}")
    if plan.peer_reviewed:
        parts.append("peer-reviewed")
    return ", ".join(parts)


def run_onesearch(plan: QueryPlan, client: OneSearchClient) -> SearchOutcome:
    """Search OneSearch for the plan. If a filtered search finds nothing, refine
    exactly once (two searches total): same keywords, all filters dropped, and
    a note saying what was relaxed. This is the only place Layer 2 talks to Layer 1."""
    filters = _describe_filters(plan)
    try:
        result = client.search_with_fallback(
            plan.keywords,
            resource_type=plan.resource_type,
            peer_reviewed=plan.peer_reviewed,
        )
        notes: list[str] = []
        if not result.candidates and not result.timed_out and filters:
            result = client.search_with_fallback(plan.keywords)
            notes.append(f"Nothing matched with filters ({filters}), so I searched without them.")
        if result.timed_out:
            notes.append("OneSearch was slow to respond.")
        return SearchOutcome(result=result, notes=notes)
    except Exception:
        # HTTP errors and anything unexpected from Layer 1: degrade to the direct link.
        return SearchOutcome(
            result=SearchResult(
                candidates=[], timed_out=True, fallback_link=client.fallback_link(plan.keywords)
            ),
            notes=["OneSearch could not be reached."],
        )
