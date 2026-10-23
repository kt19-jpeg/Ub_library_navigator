import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from onesearch.models import ResourceDetail, SearchResult
from pipeline.records import Record

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


class RagContext(BaseModel):
    """Library-guide hits added alongside OneSearch results. Each list is already
    filtered by distance cutoff and capped, sorted best-first."""

    model_config = {"arbitrary_types_allowed": True}

    librarians: list[Record] = Field(default_factory=list)
    help_topics: list[Record] = Field(default_factory=list)
    databases: list[Record] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.librarians or self.help_topics or self.databases)


def merge_timings(current: dict, new: dict) -> dict:
    return {**current, **new}


class AgentState(TypedDict, total=False):
    role: str
    message: str
    history: list[dict]  # [{"user": str, "assistant": str}], most recent last
    plan: QueryPlan
    search_result: SearchResult
    rag_context: RagContext
    selection: list[SelectionItem]
    citations: dict[str, dict[str, str]]  # record_id -> {style: citation text}
    details: dict[str, ResourceDetail]
    answer: str
    notes: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], merge_timings]
