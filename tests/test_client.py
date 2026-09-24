import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


def _search_url(**filters):
    response = MagicMock()
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("mental health", **filters)
    return http_get.call_args.args[0]


def test_search_without_filters_sends_no_facets():
    assert "multiFacets" not in _search_url()


def test_search_sends_resource_type_and_peer_reviewed_facets():
    url = _search_url(resource_type="journals", peer_reviewed=True)
    assert (
        "multiFacets=facet_rtype,include,journals|,|facet_tlevel,include,peer_reviewed"
        in url.replace("%7C", "|")
    )
    assert "%2C" not in url


def test_search_peer_reviewed_alone():
    url = _search_url(peer_reviewed=True)
    assert "multiFacets=facet_tlevel,include,peer_reviewed" in url
    assert "facet_rtype" not in url


def test_search_rejects_unknown_resource_type():
    with pytest.raises(ValueError, match="resource_type"):
        _search_url(resource_type="bogus")


def test_cache_key_distinguishes_filters():
    response = MagicMock()
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search("mental health")
    client.search("mental health", resource_type="journals")
    client.search("mental health", resource_type="journals")

    assert http_get.call_count == 2


def test_search_with_fallback_passes_filters_through():
    response = MagicMock()
    response.json.return_value = {"docs": []}
    http_get = MagicMock(return_value=response)
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=http_get,
    )
    client.search_with_fallback("mental health", resource_type="books", peer_reviewed=True)
    assert "facet_rtype,include,books" in http_get.call_args.args[0]


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


def _edelivery_post():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = json.loads(
        (FIXTURES / "onesearch_edelivery_response.json").read_text()
    )
    return MagicMock(return_value=response)


def test_fetch_detail_makes_one_batched_call_for_all_ids():
    http_post = _edelivery_post()
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_post=http_post,
    )
    details = client.fetch_detail(["alma990019605390204803", "alma9939979830304803", "other"])

    assert http_post.call_count == 1
    assert http_post.call_args.args[0].endswith("/edelivery?vid=01SUNY_BUF:everything")
    assert http_post.call_args.kwargs["json"] == [
        {"recId": "alma990019605390204803", "sharedDigitalCandidates": None},
        {"recId": "alma9939979830304803", "sharedDigitalCandidates": None},
        {"recId": "other", "sharedDigitalCandidates": None},
    ]
    assert set(details) == {"alma990019605390204803", "alma9939979830304803", "other"}


def test_fetch_detail_resolves_absolute_online_link():
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_post=_edelivery_post(),
    )
    detail = client.fetch_detail(["alma990019605390204803"])["alma990019605390204803"]

    assert detail.online_link.startswith(
        "https://search.lib.buffalo.edu/view/action/uresolver.do?"
    )
    assert detail.online_package == "view online"


def test_fetch_detail_merges_physical_holdings_from_prior_search():
    brief = MagicMock()
    brief.json.return_value = json.loads(
        (FIXTURES / "onesearch_brief_response.json").read_text()
    )
    edelivery = MagicMock()
    edelivery.json.return_value = {"beaconO22": "1"}
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_get=MagicMock(return_value=brief),
        http_post=MagicMock(return_value=edelivery),
    )
    client.search("machine learning")
    detail = client.fetch_detail(["alma991234567890001"])["alma991234567890001"]

    assert detail.call_number == "QA76.87 .W66 2021"
    assert detail.library == "Silverman Library"
    assert detail.online_link is None


def test_fetch_detail_with_no_ids_makes_no_call():
    http_post = _edelivery_post()
    client = OneSearchClient(
        session_manager=_fake_session_manager(),
        cache=QueryCache(ttl_seconds=60),
        http_post=http_post,
    )
    assert client.fetch_detail([]) == {}
    assert http_post.call_count == 0


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
