from pipeline.records import HELP_TOPIC, LIBRARIAN_SUBJECT, Record
from tests.fake_embeddings import FakeEmbeddingFunction
from pipeline.refresh import refresh_all
from scraper.fetch import FetchError


def _sources():
    return [
        {"type": LIBRARIAN_SUBJECT, "url": "https://a/librarians"},
        {"type": HELP_TOPIC, "url": "https://a/help"},
    ]


def _snapshot_with(path, records):
    from pipeline.snapshot import save_snapshot
    save_snapshot(records, path)


def test_first_run_with_no_prior_snapshot_writes_parsed_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.refresh.extract_librarian_subjects",
        lambda html: [Record(id="librarian:a", type=LIBRARIAN_SUBJECT, text="a", metadata={})],
    )
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:a", type=HELP_TOPIC, text="a", metadata={})],
    )
    result = refresh_all(
        sources=_sources(), fetch_fn=lambda url: "<html></html>",
        snapshot_path=tmp_path / "records.json", persist_dir=str(tmp_path / "vectorstore"),
        embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts == {LIBRARIAN_SUBJECT: 1, HELP_TOPIC: 1, "database_subject": 0}
    assert result.warnings == []


def test_fetch_failure_keeps_that_types_old_records_and_warns(tmp_path, monkeypatch):
    path = tmp_path / "records.json"
    old = [Record(id="librarian:a", type=LIBRARIAN_SUBJECT, text="a", metadata={})]
    _snapshot_with(path, old)
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:new", type=HELP_TOPIC, text="new", metadata={})],
    )

    def flaky_fetch(url):
        if "librarians" in url:
            raise FetchError("HTTP 500")
        return "<html></html>"

    result = refresh_all(
        sources=_sources(), fetch_fn=flaky_fetch, snapshot_path=path,
        persist_dir=str(tmp_path / "vectorstore"), embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts[LIBRARIAN_SUBJECT] == 1  # kept, not lost
    assert result.counts[HELP_TOPIC] == 1  # the working source still updated
    assert any("fetch failed" in w for w in result.warnings)

    from pipeline.snapshot import load_snapshot
    expected = old + [Record(id="help:new", type=HELP_TOPIC, text="new", metadata={})]
    assert sorted(load_snapshot(path), key=lambda r: r.id) == sorted(expected, key=lambda r: r.id)


def test_suspicious_drop_in_count_keeps_old_records_and_warns(tmp_path, monkeypatch):
    path = tmp_path / "records.json"
    old = [Record(id=f"help:{i}", type=HELP_TOPIC, text=str(i), metadata={}) for i in range(10)]
    _snapshot_with(path, old)
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:0", type=HELP_TOPIC, text="0", metadata={})],  # 1 of 10: looks broken
    )
    monkeypatch.setattr("pipeline.refresh.extract_librarian_subjects", lambda html: [])

    result = refresh_all(
        sources=[{"type": HELP_TOPIC, "url": "https://a/help"}],
        fetch_fn=lambda url: "<html></html>", snapshot_path=path,
        persist_dir=str(tmp_path / "vectorstore"), embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts[HELP_TOPIC] == 10  # old data kept
    assert any("possible layout change" in w for w in result.warnings)


def test_a_real_update_fully_replaces_removed_entries(tmp_path, monkeypatch):
    path = tmp_path / "records.json"
    old = [Record(id=f"help:{i}", type=HELP_TOPIC, text=str(i), metadata={}) for i in range(4)]
    _snapshot_with(path, old)
    monkeypatch.setattr(
        "pipeline.refresh.extract_help_topics",
        lambda html: [Record(id="help:0", type=HELP_TOPIC, text="0", metadata={}),
                       Record(id="help:1", type=HELP_TOPIC, text="1", metadata={})],  # 2 of 4: not suspicious (>=50%)
    )
    monkeypatch.setattr("pipeline.refresh.extract_librarian_subjects", lambda html: [])

    result = refresh_all(
        sources=[{"type": HELP_TOPIC, "url": "https://a/help"}],
        fetch_fn=lambda url: "<html></html>", snapshot_path=path,
        persist_dir=str(tmp_path / "vectorstore"), embedding_function=FakeEmbeddingFunction(),
    )

    assert result.counts[HELP_TOPIC] == 2  # help:2 and help:3 really removed
    assert result.warnings == []


def test_missing_embedding_key_still_writes_the_snapshot(tmp_path, monkeypatch):
    from pipeline.embed import EmbeddingUnavailableError

    monkeypatch.setattr("pipeline.refresh.extract_librarian_subjects", lambda html: [])
    monkeypatch.setattr("pipeline.refresh.extract_help_topics", lambda html: [])
    monkeypatch.setattr(
        "pipeline.refresh.default_embedding_function",
        lambda: (_ for _ in ()).throw(EmbeddingUnavailableError("OPENAI_API_KEY is not set")),
    )

    result = refresh_all(
        sources=_sources(), fetch_fn=lambda url: "<html></html>",
        snapshot_path=tmp_path / "records.json", persist_dir=str(tmp_path / "vectorstore"),
        embedding_function=None,
    )

    assert any("embedding sync skipped" in w for w in result.warnings)
    from pipeline.snapshot import load_snapshot
    assert load_snapshot(tmp_path / "records.json") == []
