import json
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record

COLLECTION_NAME = "records"
MANAGED_TYPES = (LIBRARIAN_SUBJECT, HELP_TOPIC, DATABASE_SUBJECT)


class EmbeddingUnavailableError(Exception):
    """No usable embedding backend (e.g. OPENAI_API_KEY is not set)."""


@dataclass(frozen=True)
class SyncResult:
    added: int
    updated: int
    deleted: int
    unchanged: int


def default_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingUnavailableError("OPENAI_API_KEY is not set")
    return OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")


def get_collection(persist_dir: str, embedding_function):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_function)


def open_rag_collection(persist_dir: str = "vectorstore"):
    """For callers (the CLI, the live check) that want RAG if it's available and
    silent degradation if not: returns the collection, or None if there's no
    OPENAI_API_KEY or the collection has never been populated (run refresh_all()
    first)."""
    try:
        collection = get_collection(persist_dir, default_embedding_function())
    except EmbeddingUnavailableError:
        return None
    if collection.count() == 0:
        return None
    return collection


def _to_chroma_metadata(record: Record) -> dict:
    """Chroma metadata values must be scalars; a Record's metadata can hold nested
    lists/dicts (e.g. librarian_subject's "librarians"), so those are JSON-encoded."""
    encoded = {}
    for key, value in record.metadata.items():
        encoded[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    encoded["type"] = record.type
    encoded["content_hash"] = record.content_hash
    return encoded


def sync_collection(records: list[Record], collection) -> SyncResult:
    """Adds new/changed records, deletes managed-type records no longer present,
    and skips records whose content_hash hasn't changed (no re-embedding cost)."""
    new_by_id = {r.id: r for r in records}

    existing = collection.get(where={"type": {"$in": list(MANAGED_TYPES)}})
    existing_hash = {
        id_: meta.get("content_hash")
        for id_, meta in zip(existing["ids"], existing["metadatas"])
    }

    to_delete = [id_ for id_ in existing_hash if id_ not in new_by_id]
    to_write = [r for r in records if existing_hash.get(r.id) != r.content_hash]
    unchanged = len(records) - len(to_write)
    added = sum(1 for r in to_write if r.id not in existing_hash)
    updated = len(to_write) - added

    if to_delete:
        collection.delete(ids=to_delete)
    if to_write:
        collection.upsert(
            ids=[r.id for r in to_write],
            documents=[r.text for r in to_write],
            metadatas=[_to_chroma_metadata(r) for r in to_write],
        )
    return SyncResult(added=added, updated=updated, deleted=len(to_delete), unchanged=unchanged)
