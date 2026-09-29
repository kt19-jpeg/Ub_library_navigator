import json

from agent.models import RagContext
from pipeline.embed import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, EmbeddingUnavailableError
from pipeline.records import Record

RAG_TOP_K = {
    LIBRARIAN_SUBJECT: 2,
    HELP_TOPIC: 3,
    DATABASE_SUBJECT: 2,
}
# Cosine distance from text-embedding-3-small; a starting point, tuned against real
# queries with agent/rag_live_check.py before this slice is considered done.
RAG_DISTANCE_CUTOFF = {
    LIBRARIAN_SUBJECT: 0.5,
    HELP_TOPIC: 0.5,
    DATABASE_SUBJECT: 0.5,
}


def _query_type(collection, keywords: str, record_type: str) -> list[Record]:
    result = collection.query(
        query_texts=[keywords], n_results=RAG_TOP_K[record_type], where={"type": record_type}
    )
    cutoff = RAG_DISTANCE_CUTOFF[record_type]
    hits = []
    if result["ids"]:
        for id_, metadata, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0]):
            if distance >= cutoff:
                continue
            record_metadata = {}
            for key, value in metadata.items():
                if key in ("type", "content_hash"):
                    continue
                if isinstance(value, str) and value[:1] in "[{":
                    try:
                        value = json.loads(value)
                    except ValueError:
                        pass
                record_metadata[key] = value
            hits.append(Record(id=id_, type=record_type, text="", metadata=record_metadata))
    return hits


def query_rag(keywords: str, collection) -> tuple[RagContext, list[str]]:
    """Looks up the librarian/help/database library guide for `keywords`. Never raises:
    if the collection can't be queried (e.g. no OPENAI_API_KEY for the embedding
    function), returns an empty RagContext and a note instead."""
    try:
        return (
            RagContext(
                librarians=_query_type(collection, keywords, LIBRARIAN_SUBJECT),
                help_topics=_query_type(collection, keywords, HELP_TOPIC),
                databases=_query_type(collection, keywords, DATABASE_SUBJECT),
            ),
            [],
        )
    except EmbeddingUnavailableError:
        return RagContext(), ["Library guide lookup was unavailable."]
    except Exception:
        return RagContext(), ["Library guide lookup failed."]
