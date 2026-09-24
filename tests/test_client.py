import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from onesearch.cache import QueryCache
from onesearch.client import OneSearchClient
from onesearch.models import SearchResult
from onesearch.session import SessionCookies, SessionManager

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_session_manager():
    cookies = SessionCookies(jsessionid="J", urm_st="1", urm_se="900001", institute="UB")
    return SessionManager(mint_fn=lambda: cookies)


def test_search_returns_parsed_candidates():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_brief_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    candidates = client.search("machine learning")

    assert len(candidates) == 2
    assert candidates[0].title == "Machine Learning for Coastal Systems"


def test_search_builds_query_string_with_literal_commas():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")

    called_url = http_get.call_args.args[0]
    assert "any,contains," in called_url
    assert "%2C" not in called_url


def test_search_forces_skip_delivery_and_pc_availability():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")

    called_url = http_get.call_args.args[0]
    assert "skipDelivery=N" in called_url
    assert "pcAvailability=true" in called_url


def test_search_uses_identifying_user_agent():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")

    headers = http_get.call_args.kwargs["headers"]
    assert "kavyansht21@gmail.com" in headers["User-Agent"]


def test_search_hits_cache_on_second_call():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_brief_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("machine learning")
    client.search("machine learning")

    assert http_get.call_count == 1


def test_fetch_detail_only_calls_for_given_record_ids():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_detail_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    details = client.fetch_detail(["alma991234567890001"])

    assert http_get.call_count == 1
    assert set(details.keys()) == {"alma991234567890001"}
    assert details["alma991234567890001"].call_number == "QA76.87 .W66 2021"


def test_fetch_detail_bounded_to_multiple_explicit_ids_only():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_detail_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.fetch_detail(["a", "b", "c"])

    assert http_get.call_count == 3


def test_search_with_fallback_returns_candidates_when_fast():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_brief_response.json").read_text()
    )
    http_get = MagicMock(return_value=response)

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    result = client.search_with_fallback("machine learning", timeout_seconds=5)

    assert isinstance(result, SearchResult)
    assert result.timed_out is False
    assert len(result.candidates) == 2
    assert result.fallback_link is None


def test_search_with_fallback_times_out_gracefully():
    def _slow_get(*args, **kwargs):
        time.sleep(0.2)
        raise AssertionError("should not be reached before timeout fires")

    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=_slow_get,
    )
    result = client.search_with_fallback("machine learning", timeout_seconds=0.05)

    assert result.timed_out is True
    assert result.candidates == []
    assert "search.lib.buffalo.edu" in result.fallback_link
