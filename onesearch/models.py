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
    # Citation data (agent/citation_tool.py), confirmed free in the brief response's
    # pnx.addata (no extra fetch): see onesearch/parser.py for real field provenance.
    authors: list[str] | None = None
    journal_title: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    isbn: str | None = None
    publisher: str | None = None
    place_of_publication: str | None = None


@dataclass(frozen=True)
class ResourceDetail:
    record_id: str
    online_link: str | None
    online_package: str | None
    call_number: str | None
    library: str | None
    sub_location: str | None
    availability_status: str | None


@dataclass(frozen=True)
class SearchResult:
    candidates: list[SearchCandidate]
    timed_out: bool
    fallback_link: str | None
