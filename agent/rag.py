import json

from agent.models import RagContext
from pipeline.embed import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, EmbeddingUnavailableError
from pipeline.records import Record

RAG_TOP_K = {
    LIBRARIAN_SUBJECT: 2,
    HELP_TOPIC: 3,
    DATABASE_SUBJECT: 2,
}
# Cosine distance from text-embedding-3-small. Tuned against real embeddings of the
# live 2026-09-28 data (372 records) with 11 test queries via agent/rag_live_check.py
# and a throwaway distance dump. text-embedding-3-small puts unrelated short phrases
# much closer together than word2vec-era intuition suggests, so these thresholds are
# per-type, not one global value:
#   - help_topic: real titles are natural-language and varied, so genuine matches
#     cluster low (0.59-0.79) with clear separation from irrelevant hits (1.26+).
#   - librarian_subject / database_subject: entries are short, formulaic phrases
#     ("<Subject> subject librarian: <Name>", "<Subject> databases (<N> available)"),
#     so even unrelated pairs sit close together; genuine matches ranged 1.08-1.55,
#     nonsense queries ("xylophone repair") started at 1.6. Cutoffs sit at that gap.
RAG_DISTANCE_CUTOFF = {
    LIBRARIAN_SUBJECT: 1.6,
    HELP_TOPIC: 1.1,
    DATABASE_SUBJECT: 1.6,
}
# A hit must also be within this margin of the BEST hit of its type. The absolute cutoff
# alone lets weak runners-up through (audit 2026-11-10: "cancer immunotherapy" returned
# Medicine plus Complementary Medicine). Margins are from the measured gaps: genuine
# second subjects sit within ~0.25 of the best (Pharmacy/Pharmaceutical Sciences +0.23),
# irrelevant ones are usually further. Help pages stay tight; their useful matches
# cluster low.
RAG_RELATIVE_MARGIN = {
    LIBRARIAN_SUBJECT: 0.25,
    HELP_TOPIC: 0.15,
    DATABASE_SUBJECT: 0.25,
}


# Embeddings alone rank a bare one-word query ("medicine") by how often subjects repeat
# the word in their keyword lists, so "Complementary Medicine" beat "Medicine"
# (confirmed live 2026-11-10). When the query IS a subject's name, or that name starts
# with it, nudge that subject ahead. Small on purpose: it breaks near-ties, it can't
# rescue a genuinely distant match.
NAME_MATCH_BONUS = {"exact": 0.1, "prefix": 0.05}
NAME_MATCH_POOL = 3  # extra candidates fetched so the named subject is in the pool


def _name_bonus(query: str, subject: str) -> float:
    q, name = query.strip().lower(), subject.strip().lower()
    if not q or not name:
        return 0.0
    if name == q:
        return NAME_MATCH_BONUS["exact"]
    if name.startswith(q + " "):
        return NAME_MATCH_BONUS["prefix"]
    return 0.0


def _query_type(collection, keywords: str, record_type: str) -> list[Record]:
    by_subject = record_type in (LIBRARIAN_SUBJECT, DATABASE_SUBJECT)
    pool = RAG_TOP_K[record_type] + (NAME_MATCH_POOL if by_subject else 0)
    result = collection.query(query_texts=[keywords], n_results=pool, where={"type": record_type})
    cutoff = RAG_DISTANCE_CUTOFF[record_type]
    hits = []
    if result["ids"]:
        rows = list(zip(result["ids"][0], result["metadatas"][0], result["distances"][0]))
        if by_subject:
            rows = [
                (i, m, d - _name_bonus(keywords, m.get("subject", ""))) for i, m, d in rows
            ]
        rows = sorted(rows, key=lambda row: row[2])[: RAG_TOP_K[record_type]]
        best = min((d for _, _, d in rows), default=None)
        margin = RAG_RELATIVE_MARGIN[record_type]
        for id_, metadata, distance in rows:
            if distance >= cutoff or distance > best + margin:
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
