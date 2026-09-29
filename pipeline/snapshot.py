import json
from pathlib import Path

from pipeline.records import Record

DEFAULT_PATH = Path("data/processed/records.json")


def load_snapshot(path: Path = DEFAULT_PATH) -> list[Record]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [Record.from_dict(item) for item in data]


def save_snapshot(records: list[Record], path: Path = DEFAULT_PATH) -> None:
    """Writes sorted by id so a refresh's git diff only shows what actually changed."""
    ordered = sorted(records, key=lambda r: r.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in ordered], indent=2) + "\n")
