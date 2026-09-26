from unittest.mock import MagicMock

from agent.models import QueryPlan
from agent.onesearch_tool import run_onesearch
from onesearch.models import SearchCandidate, SearchResult


def _candidate(record_id="1"):
    return SearchCandidate(
        record_id=record_id, title="t", author=None, resource_type=None, format=None,
        subject=None, description=None, date=None, source=None, delivery_category=None,
        online_link=None, call_number=None, library=None, sub_location=None,
        availability_status=None,
    )


def _result(candidates=(), timed_out=False, link=None):
    return SearchResult(candidates=list(candidates), timed_out=timed_out, fallback_link=link)


def test_passes_keywords_and_filters_to_layer1():
    client = MagicMock()
    client.search_with_fallback.return_value = _result([_candidate()])
    plan = QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True)

    outcome = run_onesearch(plan, client)

    client.search_with_fallback.assert_called_once_with(
        "mental health", resource_type="journals", peer_reviewed=True
    )
    assert len(outcome.result.candidates) == 1
    assert outcome.notes == []


def test_refines_once_without_filters_when_filtered_search_is_empty():
    client = MagicMock()
    client.search_with_fallback.side_effect = [_result(), _result([_candidate()])]
    plan = QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True)

    outcome = run_onesearch(plan, client)

    assert client.search_with_fallback.call_count == 2
    assert client.search_with_fallback.call_args_list[1].args == ("mental health",)
    assert client.search_with_fallback.call_args_list[1].kwargs == {}
    assert len(outcome.result.candidates) == 1
    assert "type: journals, peer-reviewed" in outcome.notes[0]


def test_stops_after_one_refinement_even_if_still_empty():
    client = MagicMock()
    client.search_with_fallback.return_value = _result()

    outcome = run_onesearch(QueryPlan(keywords="zzz", peer_reviewed=True), client)

    assert client.search_with_fallback.call_count == 2
    assert outcome.result.candidates == []


def test_no_refinement_when_no_filters_were_set():
    client = MagicMock()
    client.search_with_fallback.return_value = _result()

    run_onesearch(QueryPlan(keywords="zzz"), client)

    assert client.search_with_fallback.call_count == 1


def test_timeout_is_not_refined_and_keeps_fallback_link():
    client = MagicMock()
    client.search_with_fallback.return_value = _result(timed_out=True, link="https://x/y")

    outcome = run_onesearch(QueryPlan(keywords="a", peer_reviewed=True), client)

    assert client.search_with_fallback.call_count == 1
    assert outcome.result.timed_out is True
    assert outcome.result.fallback_link == "https://x/y"
    assert any("slow" in n for n in outcome.notes)


def test_layer1_exception_degrades_to_direct_link():
    client = MagicMock()
    client.search_with_fallback.side_effect = RuntimeError("HTTP 500")
    client.fallback_link.return_value = "https://search.lib.buffalo.edu/discovery/search?query=any,contains,a"

    outcome = run_onesearch(QueryPlan(keywords="a"), client)

    assert outcome.result.candidates == []
    assert outcome.result.timed_out is True
    assert outcome.result.fallback_link.startswith("https://search.lib.buffalo.edu/")
    assert any("could not be reached" in n for n in outcome.notes)
