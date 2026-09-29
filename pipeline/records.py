import hashlib
import json
import re
from dataclasses import dataclass, field

LIBRARIAN_SUBJECT = "librarian_subject"
HELP_TOPIC = "help_topic"
DATABASE_SUBJECT = "database_subject"


@dataclass(frozen=True)
class Record:
    id: str
    type: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = json.dumps({"text": self.text, "metadata": self.metadata}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }

    @staticmethod
    def from_dict(data: dict) -> "Record":
        return Record(id=data["id"], type=data["type"], text=data["text"], metadata=data["metadata"])


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "x"


def make_unique_id(prefix: str, text: str, seen: set[str]) -> str:
    """`prefix:slug`, with a numeric suffix if that id was already used this run
    (e.g. two subjects/titles that slugify to the same string)."""
    base = f"{prefix}:{slugify(text)}"
    candidate = base
    n = 2
    while candidate in seen:
        candidate = f"{base}-{n}"
        n += 1
    seen.add(candidate)
    return candidate
