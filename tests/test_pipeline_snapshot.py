import json

from pipeline.records import Record
from pipeline.snapshot import load_snapshot, save_snapshot


def test_load_missing_file_returns_empty_list(tmp_path):
    assert load_snapshot(tmp_path / "missing.json") == []


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "sub" / "records.json"
    records = [
        Record(id="b:2", type="t", text="B", metadata={}),
        Record(id="a:1", type="t", text="A", metadata={"x": 1}),
    ]

    save_snapshot(records, path)

    assert load_snapshot(path) == sorted(records, key=lambda r: r.id)


def test_save_writes_sorted_by_id_for_stable_diffs(tmp_path):
    path = tmp_path / "records.json"
    save_snapshot(
        [Record(id="z", type="t", text="Z"), Record(id="a", type="t", text="A")], path
    )

    data = json.loads(path.read_text())
    assert [item["id"] for item in data] == ["a", "z"]
