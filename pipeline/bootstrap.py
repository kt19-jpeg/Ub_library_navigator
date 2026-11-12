"""Builds/refreshes the Chroma store from the COMMITTED snapshot, with no scraping.

Why: `vectorstore/` is gitignored and not in the Docker image, and nothing built it on
Render, so the deployed app always had empty library-guide data (no subject librarian,
database or help-page suggestions) regardless of the API key. The snapshot
(data/processed/records.json) and keywords (data/subject_keywords.json) ARE committed,
so the store can be rebuilt from them at startup. sync_collection only embeds records
whose content hash changed, so a warm local store costs nothing and a fresh disk costs
one embedding pass (a few hundred short texts)."""

import logging
from pathlib import Path

from pipeline.embed import default_embedding_function, EmbeddingUnavailableError, get_collection, sync_collection
from pipeline.snapshot import DEFAULT_PATH, load_snapshot
from pipeline.subject_keywords import KEYWORDS_PATH, enrich_records, load_subject_keywords

logger = logging.getLogger(__name__)


def ensure_rag_collection(
    persist_dir: str = "vectorstore",
    snapshot_path: Path = DEFAULT_PATH,
    keywords_path: Path = KEYWORDS_PATH,
    embedding_function=None,
):
    """Returns a populated collection, or None (never raises) if RAG can't be used:
    no/invalid embedding key, no snapshot, or any failure while syncing."""
    try:
        embedding_function = embedding_function or default_embedding_function()
    except EmbeddingUnavailableError as exc:
        logger.warning("library-guide data unavailable: %s", exc)
        return None
    records = load_snapshot(snapshot_path)
    if not records:
        logger.warning("library-guide data unavailable: no snapshot at %s", snapshot_path)
        return None
    try:
        collection = get_collection(persist_dir, embedding_function)
        result = sync_collection(enrich_records(records, load_subject_keywords(keywords_path)), collection)
        logger.info("library-guide store ready: %s", result)
        return collection if collection.count() else None
    except Exception as exc:
        logger.warning("library-guide data unavailable (store sync failed): %r", exc)
        return None
