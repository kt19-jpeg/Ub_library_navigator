from pipeline.records import Record, make_unique_id, slugify


def test_slugify_lowercases_and_collapses_punctuation():
    assert slugify("African & African-American Studies") == "african-african-american-studies"


def test_slugify_handles_empty_input():
    assert slugify("!!!") == "x"


def test_make_unique_id_adds_suffix_on_collision():
    seen: set[str] = set()
    first = make_unique_id("help", "Cafe & Hours", seen)
    second = make_unique_id("help", "Cafe: Hours", seen)  # slugifies the same
    assert first == "help:cafe-hours"
    assert second == "help:cafe-hours-2"


def test_content_hash_changes_when_text_or_metadata_changes():
    a = Record(id="x", type="help_topic", text="A", metadata={"url": "u"})
    b = Record(id="x", type="help_topic", text="B", metadata={"url": "u"})
    c = Record(id="x", type="help_topic", text="A", metadata={"url": "v"})
    assert a.content_hash != b.content_hash
    assert a.content_hash != c.content_hash


def test_content_hash_is_stable_regardless_of_metadata_key_order():
    a = Record(id="x", type="t", text="A", metadata={"a": 1, "b": 2})
    b = Record(id="x", type="t", text="A", metadata={"b": 2, "a": 1})
    assert a.content_hash == b.content_hash


def test_round_trips_through_dict():
    r = Record(id="db:x", type="database_subject", text="X (3)", metadata={"count": 3})
    assert Record.from_dict(r.to_dict()) == r
