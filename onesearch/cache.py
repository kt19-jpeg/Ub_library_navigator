import re
import time

from onesearch.models import SearchCandidate


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class QueryCache:
    def __init__(self, ttl_seconds: float):
        self._ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, list[SearchCandidate]]] = {}

    def get(self, query: str) -> list[SearchCandidate] | None:
        key = _normalize(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, results = entry
        if time.monotonic() - stored_at > self._ttl_seconds:
            del self._store[key]
            return None
        return results

    def set(self, query: str, results: list[SearchCandidate]) -> None:
        self._store[_normalize(query)] = (time.monotonic(), results)
