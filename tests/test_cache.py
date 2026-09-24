import time

from onesearch.cache import QueryCache
from onesearch.models import SearchCandidate


def _candidate(record_id="1"):
    return SearchCandidate(
        record_id=record_id, title="t", author=None, resource_type=None,
        format=None, subject=None, description=None, date=None, source=None,
        delivery_category=None, online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
        copies_available=None,
    )


def test_cache_hit_within_ttl():
    cache = QueryCache(ttl_seconds=60)
    cache.set("machine learning", [_candidate()])
    assert cache.get("machine learning") == [_candidate()]


def test_cache_normalizes_query_text():
    cache = QueryCache(ttl_seconds=60)
    cache.set("  Machine   Learning  ", [_candidate()])
    assert cache.get("machine learning") == [_candidate()]


def test_cache_miss_after_ttl_expires():
    cache = QueryCache(ttl_seconds=0.05)
    cache.set("machine learning", [_candidate()])
    time.sleep(0.1)
    assert cache.get("machine learning") is None
