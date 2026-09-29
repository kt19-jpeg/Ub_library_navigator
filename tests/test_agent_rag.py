import chromadb

from agent.models import RagContext
from agent.rag import query_rag
from pipeline.embed import COLLECTION_NAME, sync_collection
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record
from tests.fake_embeddings import FakeEmbeddingFunction


def _collection_with(tmp_path, records):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=FakeEmbeddingFunction())
    sync_collection(records, collection)
    return collection


def _lenient_cutoffs(monkeypatch):
    # The fake embedding function isn't semantic, so real cosine distances between
    # its vectors don't reflect real similarity. These tests check plumbing (does a
    # hit round-trip correctly, across all three types), not the cutoff's numeric
    # value, so the cutoff is raised high enough that nothing gets dropped.
    import agent.rag as rag_module
    for record_type in rag_module.RAG_DISTANCE_CUTOFF:
        monkeypatch.setitem(rag_module.RAG_DISTANCE_CUTOFF, record_type, 999.0)


def test_query_returns_hits_for_all_three_types(tmp_path, monkeypatch):
    _lenient_cutoffs(monkeypatch)
    records = [
        Record(id="librarian:anatomy", type=LIBRARIAN_SUBJECT, text="anatomy",
               metadata={"subject": "Anatomy", "librarians": [{"name": "Aronoff, Nell", "profile_url": "https://x"}]}),
        Record(id="help:citing-sources", type=HELP_TOPIC, text="Citing Sources",
               metadata={"title": "Citing Sources", "url": "https://y"}),
        Record(id="db:chemistry", type=DATABASE_SUBJECT, text="Chemistry databases (32 available)",
               metadata={"subject": "Chemistry", "count": 32, "browse_url": "https://z"}),
    ]
    collection = _collection_with(tmp_path, records)

    context, notes = query_rag("anatomy", collection)

    assert notes == []
    assert isinstance(context, RagContext)
    assert [r.id for r in context.librarians] == ["librarian:anatomy"]
    assert [r.id for r in context.help_topics] == ["help:citing-sources"]
    assert [r.id for r in context.databases] == ["db:chemistry"]


def test_hits_carry_back_their_original_metadata_including_nested_lists(tmp_path, monkeypatch):
    _lenient_cutoffs(monkeypatch)
    records = [
        Record(id="help:a", type=HELP_TOPIC, text="A", metadata={"title": "A", "url": "https://a"}),
        Record(id="librarian:anatomy", type=LIBRARIAN_SUBJECT, text="anatomy",
               metadata={"subject": "Anatomy", "librarians": [{"name": "Aronoff, Nell", "profile_url": "https://x"}]}),
    ]
    collection = _collection_with(tmp_path, records)

    context, _ = query_rag("a", collection)

    assert context.help_topics[0].metadata == {"title": "A", "url": "https://a"}
    assert context.librarians[0].metadata == {
        "subject": "Anatomy", "librarians": [{"name": "Aronoff, Nell", "profile_url": "https://x"}]
    }


def test_distant_hits_are_dropped_by_the_cutoff(tmp_path, monkeypatch):
    import agent.rag as rag_module
    monkeypatch.setitem(rag_module.RAG_DISTANCE_CUTOFF, HELP_TOPIC, -1.0)  # nothing can pass
    records = [Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})]
    collection = _collection_with(tmp_path, records)

    context, _ = query_rag("a", collection)

    assert context.help_topics == []


def test_empty_collection_gives_an_empty_context(tmp_path):
    collection = _collection_with(tmp_path, [])

    context, notes = query_rag("anything", collection)

    assert context.is_empty()
    assert notes == []


def test_query_failure_degrades_to_empty_context_with_a_note():
    class BrokenCollection:
        def query(self, **kwargs):
            raise RuntimeError("no embedding function configured")

    context, notes = query_rag("anything", BrokenCollection())

    assert context.is_empty()
    assert notes == ["Library guide lookup failed."]


def test_embedding_unavailable_gives_its_own_note():
    from pipeline.embed import EmbeddingUnavailableError

    class NoKeyCollection:
        def query(self, **kwargs):
            raise EmbeddingUnavailableError("OPENAI_API_KEY is not set")

    context, notes = query_rag("anything", NoKeyCollection())

    assert context.is_empty()
    assert notes == ["Library guide lookup was unavailable."]
