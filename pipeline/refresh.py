from dataclasses import dataclass, field

from pipeline.embed import EmbeddingUnavailableError, default_embedding_function, get_collection, sync_collection
from pipeline.records import Record
from pipeline.snapshot import DEFAULT_PATH, load_snapshot, save_snapshot
from pipeline.sources import SOURCES
from pipeline.subject_keywords import KEYWORDS_PATH, enrich_records, load_subject_keywords
from scraper.fetch import FetchError, fetch
from scraper.parse_databases import extract_database_subjects
from scraper.parse_help import extract_help_topics
from scraper.parse_librarians import extract_librarian_subjects

MIN_KEPT_FRACTION = 0.5

def _parser_for(record_type: str):
    # Looked up by name at call time (not captured into a dict at import time) so
    # that tests can monkeypatch e.g. pipeline.refresh.extract_help_topics.
    return {
        "librarian_subject": extract_librarian_subjects,
        "help_topic": extract_help_topics,
        "database_subject": extract_database_subjects,
    }[record_type]


RECORD_TYPES = ("librarian_subject", "help_topic", "database_subject")


@dataclass(frozen=True)
class RefreshResult:
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _update_one_source(source: dict, old_by_type: dict[str, list[Record]], fetch_fn, warnings: list[str]) -> list[Record]:
    """Returns the records this source should contribute to the new snapshot: the
    freshly parsed ones if the fetch+parse looks trustworthy, otherwise the old
    ones unchanged (a failed fetch or a suspicious drop in count never wipes data)."""
    record_type = source["type"]
    old_records = old_by_type.get(record_type, [])
    try:
        html = fetch_fn(source["url"])
    except FetchError as exc:
        warnings.append(f"{record_type}: fetch failed ({exc}), keeping previous data")
        return old_records

    new_records = _parser_for(record_type)(html)
    if old_records and len(new_records) < MIN_KEPT_FRACTION * len(old_records):
        warnings.append(
            f"{record_type}: parsed {len(new_records)} records, down from {len(old_records)} "
            "(possible layout change), keeping previous data"
        )
        return old_records
    return new_records


def refresh_all(
    sources: list[dict] = SOURCES,
    fetch_fn=fetch,
    snapshot_path=DEFAULT_PATH,
    persist_dir: str = "vectorstore",
    embedding_function=None,
    keywords_path=KEYWORDS_PATH,
) -> RefreshResult:
    """Fetches every source independently, keeps old data for any source that
    fails or looks broken, writes the merged snapshot, and syncs it to Chroma.
    A missing OR broken embedding backend (no OPENAI_API_KEY, an expired key, a
    rate limit, a network blip) still lets the snapshot update; only the Chroma
    sync step is skipped, with a warning - confirmed live 2026-11-03: an
    expired OPENAI_API_KEY raised a real openai.AuthenticationError, which
    isn't an EmbeddingUnavailableError, so it wasn't caught and crashed the
    whole script with a traceback instead of degrading gracefully (the
    snapshot itself had already been written by that point, unaffected, but
    the crash still looked like a hard failure). This is the exact function
    the scheduled weekly-refresh.yml job runs - a transient embedding-provider
    failure shouldn't take the whole job down."""
    old_records = load_snapshot(snapshot_path)
    old_by_type: dict[str, list[Record]] = {}
    for record in old_records:
        old_by_type.setdefault(record.type, []).append(record)

    warnings: list[str] = []
    new_records: list[Record] = []
    for source in sources:
        new_records.extend(_update_one_source(source, old_by_type, fetch_fn, warnings))

    save_snapshot(new_records, snapshot_path)

    try:
        collection = get_collection(
            persist_dir=persist_dir, embedding_function=embedding_function or default_embedding_function()
        )
        # Enrichment is applied here, at embedding time only - never to the saved
        # snapshot - so it can't be applied twice (see pipeline/subject_keywords.py).
        sync_collection(enrich_records(new_records, load_subject_keywords(keywords_path)), collection)
    except EmbeddingUnavailableError as exc:
        warnings.append(f"embedding sync skipped: {exc}")
    except Exception as exc:
        # Anything from the embedding provider itself (expired key, rate
        # limit, network error) - never crash the whole refresh over it.
        warnings.append(f"embedding sync failed: {exc}")

    counts = {record_type: 0 for record_type in RECORD_TYPES}
    for record in new_records:
        counts[record.type] = counts.get(record.type, 0) + 1
    return RefreshResult(counts=counts, warnings=warnings)


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    result = refresh_all()
    print(f"counts: {result.counts}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
