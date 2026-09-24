from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from onesearch.models import ResourceDetail, SearchResult

ROLES = ("undergraduate", "graduate", "faculty")

# Mirrors onesearch.client.RESOURCE_TYPES (a test enforces they stay equal).
ResourceType = Literal["books", "articles", "journals", "dissertations", "videos"]


class QueryPlan(BaseModel):
    """What the planner extracts from a natural-language message."""

    keywords: str = Field(
        description="1-6 content words to search for. Filler and filter words are removed."
    )
    resource_type: ResourceType | None = Field(
        default=None, description="Restrict to one resource type, or null for any."
    )
    peer_reviewed: bool = Field(
        default=False, description="True only if the user asked for peer-reviewed material."
    )
    needs_clarification: str | None = Field(
        default=None,
        description="A short question to ask the user if the request is genuinely ambiguous.",
    )


class SelectionItem(BaseModel):
    record_id: str = Field(description="record_id copied exactly from the candidate list.")
    reason: str = Field(description="One sentence on why this result fits the user.")


class Selection(BaseModel):
    """The compose LLM's choice of results to show."""

    items: list[SelectionItem] = Field(
        default_factory=list, description="The 1-3 best results, best first."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well these results answer the user's request."
    )


class AgentState(TypedDict, total=False):
    role: str
    message: str
    history: list[dict]  # [{"user": str, "assistant": str}], most recent last
    plan: QueryPlan
    search_result: SearchResult
    rag_context: list
    selection: list[SelectionItem]
    details: dict[str, ResourceDetail]
    answer: str
    notes: list[str]
    timings: dict[str, float]
