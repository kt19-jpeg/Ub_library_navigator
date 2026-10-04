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


def test_parse_brief_doc_extracts_citation_fields_from_a_real_article():
    doc = _load("onesearch_article_doc.json")
    candidate = parse_brief_doc(doc)

    assert candidate.authors[0] == "Smith, Daniel G. A."
    assert len(candidate.authors) == 48
    assert candidate.journal_title == "The Journal of chemical physics"
    assert candidate.volume == "155"
    assert candidate.issue == "20"
    assert candidate.pages == "204801"  # single-page: trailing "-" stripped
    assert candidate.doi == "10.1063/5.0059356"
    assert candidate.publisher == "American Institute of Physics"
    assert candidate.isbn is None


def test_parse_brief_doc_extracts_citation_fields_from_a_real_book():
    doc = _load("onesearch_book_doc.json")
    candidate = parse_brief_doc(doc)

    assert candidate.authors == ["Brooker, Charlie", "Repper, Julie."]
    assert candidate.isbn == "9780443103834"
    assert candidate.publisher == "Churchill Livingstone Elsevier"
    assert candidate.place_of_publication == "Edinburgh"
    assert candidate.journal_title is None  # the book/article distinguisher
    assert candidate.doi is None


def test_parse_brief_doc_handles_page_range():
    doc = _load("onesearch_article_doc.json")
    doc["pnx"]["addata"]["spage"] = ["3276"]
    doc["pnx"]["addata"]["epage"] = ["3289"]
    candidate = parse_brief_doc(doc)
    assert candidate.pages == "3276-3289"


def test_parse_brief_doc_citation_fields_are_none_when_addata_absent():
    doc = {"pnx": {"display": {"title": ["X"]}, "control": {"recordid": ["r1"]}}}
    candidate = parse_brief_doc(doc)
    assert candidate.authors is None
    assert candidate.journal_title is None
    assert candidate.pages is None
