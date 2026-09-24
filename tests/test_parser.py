import json
from pathlib import Path

from onesearch.parser import parse_brief_doc, parse_brief_response, parse_edelivery_response

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
    # Absolutised from GetIt1; the thumbnail in delivery.link must be ignored.
    assert candidates[1].online_link == (
        "https://search.lib.buffalo.edu/view/action/uresolver.do?operation="
        "resolveService&package_service_id=1&institutionId=4803&customerId=4800&VE=true"
    )


def test_parse_brief_response_physical_record_has_no_online_link():
    assert parse_brief_response(_load("onesearch_brief_response.json"))[0].online_link is None


def test_parse_brief_doc_handles_explicit_null_bestlocation():
    # Real UB responses can carry "bestlocation": null (not just an absent
    # key) for online-only resources — confirmed against live data 2026-09-24.
    doc = {
        "pnx": {
            "display": {"title": ["Some Online Resource"]},
            "control": {"recordid": ["alma1"]},
        },
        "delivery": {"deliveryCategory": ["Alma-E"], "bestlocation": None},
    }
    candidate = parse_brief_doc(doc)
    assert candidate.call_number is None
    assert candidate.library is None


def test_parse_edelivery_response_maps_record_ids_to_absolute_urls():
    access = parse_edelivery_response(_load("onesearch_edelivery_response.json"))
    url, package = access["alma990019605390204803"]
    assert url.startswith("https://search.lib.buffalo.edu/view/action/uresolver.do?")
    assert package == "view online"


def test_parse_edelivery_response_ignores_envelope_and_serviceless_records():
    access = parse_edelivery_response(
        {"beaconO22": "1", "alma1": {"electronicServices": []}, "alma2": None}
    )
    assert access == {}
