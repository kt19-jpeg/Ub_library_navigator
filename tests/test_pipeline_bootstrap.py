import logging

from pipeline.bootstrap import ensure_rag_collection
from pipeline.embed import EmbeddingUnavailableError
from pipeline.records import DATABASE_SUBJECT, Record
from pipeline.snapshot import save_snapshot
from tests.fake_embeddings import FakeEmbeddingFunction


def _snapshot(tmp_path):
    path = tmp_path / "records.json"
    save_snapshot([Record(id="db:law", type=DATABASE_SUBJECT, text="Law databases (3 available)",
                          metadata={"subject": "Law", "count": 3})], path)
    return path


def test_builds_a_populated_store_from_the_snapshot_with_keywords_applied(tmp_path):
    kw = tmp_path / "k.json"
    kw.write_text('{"Law": ["contracts", "torts"]}')

    collection = ensure_rag_collection(
        persist_dir=str(tmp_path / "vs"), snapshot_path=_snapshot(tmp_path),
        keywords_path=kw, embedding_function=FakeEmbeddingFunction(),
    )

    assert collection.count() == 1
    assert collection.get(ids=["db:law"])["documents"][0] == "Law: contracts, torts"


def test_a_second_call_is_a_no_op_not_a_re_embed(tmp_path, monkeypatch):
    kwargs = dict(persist_dir=str(tmp_path / "vs"), snapshot_path=_snapshot(tmp_path),
                  keywords_path=tmp_path / "none.json", embedding_function=FakeEmbeddingFunction())
    ensure_rag_collection(**kwargs)

    def explode(self, input):
        raise AssertionError("should not re-embed unchanged records")

    monkeypatch.setattr(FakeEmbeddingFunction, "__call__", explode)
    assert ensure_rag_collection(**kwargs).count() == 1


def test_returns_none_without_a_key(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        "pipeline.bootstrap.default_embedding_function",
        lambda: (_ for _ in ()).throw(EmbeddingUnavailableError("OPENAI_API_KEY is not set")),
    )
    with caplog.at_level(logging.WARNING):
        assert ensure_rag_collection(persist_dir=str(tmp_path / "vs"), snapshot_path=_snapshot(tmp_path)) is None
    assert "OPENAI_API_KEY" in caplog.text


def test_returns_none_without_a_snapshot(tmp_path):
    assert ensure_rag_collection(
        persist_dir=str(tmp_path / "vs"), snapshot_path=tmp_path / "missing.json",
        embedding_function=FakeEmbeddingFunction(),
    ) is None


def test_a_broken_key_degrades_to_none_instead_of_raising(tmp_path, caplog):
    class Broken:
        def __call__(self, input):
            raise RuntimeError("401 expired")

    with caplog.at_level(logging.WARNING):
        result = ensure_rag_collection(
            persist_dir=str(tmp_path / "vs"), snapshot_path=_snapshot(tmp_path),
            keywords_path=tmp_path / "none.json", embedding_function=Broken(),
        )
    assert result is None and "store sync failed" in caplog.text
