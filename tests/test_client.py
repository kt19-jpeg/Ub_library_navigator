import json
from pathlib import Path
from unittest.mock import MagicMock

from onesearch.cache import QueryCache
from onesearch.client import OneSearchClient
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
