from dataclasses import dataclass


@dataclass(frozen=True)
class SearchCandidate:
    record_id: str
    title: str
    author: str | None
    resource_type: str | None
    format: str | None
    subject: str | None
    description: str | None
    date: str | None
    source: str | None
    delivery_category: str | None
    online_link: str | None
    call_number: str | None
    library: str | None
    sub_location: str | None
    availability_status: str | None
    copies_available: str | None


@dataclass(frozen=True)
class ResourceDetail:
    record_id: str
    online_link: str | None
    call_number: str | None
    library: str | None
    sub_location: str | None
    availability_status: str | None
    copies_available: str | None
    raw: dict


@dataclass(frozen=True)
class SearchResult:
    candidates: list[SearchCandidate]
    timed_out: bool
    fallback_link: str | None
