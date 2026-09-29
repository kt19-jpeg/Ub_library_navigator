from pipeline.embed import COLLECTION_NAME, get_collection, open_rag_collection, sync_collection
from pipeline.records import HELP_TOPIC, Record
from tests.fake_embeddings import FakeEmbeddingFunction


def test_returns_none_when_no_openai_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert open_rag_collection(str(tmp_path)) is None


def test_returns_none_when_collection_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("pipeline.embed.default_embedding_function", lambda: FakeEmbeddingFunction())
    assert open_rag_collection(str(tmp_path)) is None


def test_returns_the_collection_once_it_has_data(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("pipeline.embed.default_embedding_function", lambda: FakeEmbeddingFunction())
    collection = get_collection(str(tmp_path), FakeEmbeddingFunction())
    sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})], collection)

    opened = open_rag_collection(str(tmp_path))

    assert opened is not None
    assert opened.name == COLLECTION_NAME
    assert opened.count() == 1
