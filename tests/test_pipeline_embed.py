import chromadb

from pipeline.embed import COLLECTION_NAME, sync_collection
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, Record
from tests.fake_embeddings import FakeEmbeddingFunction


def _collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=FakeEmbeddingFunction())


def test_first_sync_adds_every_record(tmp_path):
    collection = _collection(tmp_path)
    records = [Record(id="help:a", type=HELP_TOPIC, text="A", metadata={"url": "u"})]

    result = sync_collection(records, collection)

    assert (result.added, result.updated, result.deleted, result.unchanged) == (1, 0, 0, 0)
    assert collection.count() == 1


def test_second_sync_with_no_changes_touches_nothing(tmp_path):
    collection = _collection(tmp_path)
    records = [Record(id="help:a", type=HELP_TOPIC, text="A", metadata={"url": "u"})]
    sync_collection(records, collection)

    result = sync_collection(records, collection)

    assert (result.added, result.updated, result.deleted, result.unchanged) == (0, 0, 0, 1)


def test_changed_text_is_updated_not_readded(tmp_path):
    collection = _collection(tmp_path)
    sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})], collection)

    result = sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A changed", metadata={})], collection)

    assert (result.added, result.updated) == (0, 1)
    assert collection.get(ids=["help:a"])["documents"] == ["A changed"]


def test_record_missing_from_new_list_is_deleted(tmp_path):
    collection = _collection(tmp_path)
    sync_collection([Record(id="help:a", type=HELP_TOPIC, text="A", metadata={})], collection)

    result = sync_collection([], collection)

    assert result.deleted == 1
    assert collection.count() == 0


def test_only_managed_types_are_touched(tmp_path):
    collection = _collection(tmp_path)
    collection.upsert(
        ids=["other:1"], documents=["unrelated"], metadatas=[{"type": "some_future_type", "content_hash": "x"}]
    )

    result = sync_collection([], collection)

    assert result.deleted == 0
    assert collection.count() == 1  # the unrelated record survives


def test_syncs_two_managed_types_independently(tmp_path):
    collection = _collection(tmp_path)
    records = [
        Record(id="help:a", type=HELP_TOPIC, text="A", metadata={}),
        Record(id="db:b", type=DATABASE_SUBJECT, text="B", metadata={}),
    ]

    result = sync_collection(records, collection)

    assert result.added == 2
    assert collection.count() == 2
