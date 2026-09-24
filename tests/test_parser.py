import json
from pathlib import Path

from onesearch.parser import parse_brief_response, parse_detail_response

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def test_parse_brief_response_extracts_recommendation_fields():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert len(candidates) == 2
    first = candidates[0]
    assert first.record_id == "alma991234567890001"
    assert first.title == "Machine Learning for Coastal Systems"
    assert first.call_number == "QA76.87 .W66 2021"
    assert first.library == "Silverman Library"


def test_parse_brief_response_splits_creator_on_internal_separator():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[0].author == "Wong, Kau-Fui Vincent, author."


def test_parse_brief_response_falls_back_to_contributor():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[1].author == "UB Data Science Working Group"


def test_parse_brief_response_falls_back_creationdate_to_addata_date():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[1].date == "2019"


def test_parse_brief_response_online_link_present():
    candidates = parse_brief_response(_load("onesearch_brief_response.json"))
    assert candidates[1].online_link == (
        "https://search.lib.buffalo.edu/discovery/openurl?id=alma991234567890002"
    )


def test_parse_detail_response():
    detail = parse_detail_response(
        "alma991234567890001", _load("onesearch_detail_response.json")
    )
    assert detail.call_number == "QA76.87 .W66 2021"
    assert detail.library == "Silverman Library"
    assert detail.record_id == "alma991234567890001"
