import json
from pathlib import Path

from agent.citation_tool import format_citation
from onesearch.models import SearchCandidate
from onesearch.parser import parse_brief_doc

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _candidate(**overrides):
    base = dict(
        record_id="1", title="A Title", author=None, resource_type=None, format=None,
        subject=None, description=None, date=None, source=None, delivery_category=None,
        online_link=None, call_number=None, library=None, sub_location=None,
        availability_status=None,
    )
    base.update(overrides)
    return SearchCandidate(**base)


ARTICLE = parse_brief_doc(_load("onesearch_article_doc.json"))
BOOK = parse_brief_doc(_load("onesearch_book_doc.json"))


def test_apa_article_has_authors_year_journal_volume_issue_pages_doi():
    citation = format_citation(ARTICLE, "apa")
    assert citation.startswith("Smith, D. G. A., et al. (2021).")
    assert "*The Journal of chemical physics*, *155*(20), 204801." in citation
    assert citation.endswith("https://doi.org/10.1063/5.0059356")


def test_mla_article_uses_quoted_title_and_vol_no_pp():
    citation = format_citation(ARTICLE, "mla")
    assert '"Quantum Chemistry Common Driver' in citation
    assert "vol. 155, no. 20, 2021, pp. 204801." in citation


def test_chicago_article_uses_volume_no_issue_parens_year_colon_pages():
    citation = format_citation(ARTICLE, "chicago")
    assert "*The Journal of chemical physics* 155, no. 20 (2021): 204801." in citation


def test_apa_book_has_two_authors_in_full_with_ampersand():
    citation = format_citation(BOOK, "apa")
    assert citation == "Brooker, C., & Repper, J. (2009). *Mental health : from policy to practice*. Churchill Livingstone Elsevier."


def test_mla_book_second_author_in_natural_order_with_marc_period_stripped():
    citation = format_citation(BOOK, "mla")
    assert "Brooker, Charlie, and Julie Repper." in citation
    assert "Julie. Repper" not in citation  # the MARC trailing-period artifact


def test_chicago_book_includes_place_of_publication():
    citation = format_citation(BOOK, "chicago")
    assert citation.startswith("Brooker, Charlie, and Julie Repper.")
    assert "Edinburgh: Churchill Livingstone Elsevier, 2009." in citation


def test_three_or_more_authors_collapse_to_et_al():
    for style in ("apa", "mla", "chicago"):
        citation = format_citation(ARTICLE, style)
        assert "et al." in citation
        assert "Lolinco" not in citation  # the 2nd of 48 real authors is not listed


def test_single_author_no_et_al():
    candidate = _candidate(title="Solo Work", authors=["Jones, Mary"], date="2020")
    for style in ("apa", "mla", "chicago"):
        assert "et al." not in format_citation(candidate, style)


def test_missing_authors_degrades_to_unknown_author_not_a_crash():
    candidate = _candidate(title="No Author Work", date="2020")
    for style in ("apa", "mla", "chicago"):
        citation = format_citation(candidate, style)
        assert "Unknown author" in citation
        assert "None" not in citation


def test_missing_date_gives_nd_with_no_double_period():
    candidate = _candidate(title="Undated Work", authors=["Jones, Mary"])
    for style in ("mla", "chicago"):
        citation = format_citation(candidate, style)
        assert citation.endswith("n.d.")
        assert "n.d.." not in citation


def test_missing_doi_omits_the_doi_url_entirely():
    article_no_doi = _candidate(
        title="An Article", authors=["Jones, Mary"], journal_title="Some Journal",
        volume="1", issue="2", pages="3-4", date="2020",
    )
    citation = format_citation(article_no_doi, "apa")
    assert "doi.org" not in citation
    assert citation.endswith("3-4.")


def test_book_fields_never_appear_in_an_article_citation_and_vice_versa():
    # ARTICLE has no isbn/publisher-as-book-field confusion: journal_title alone
    # must be what routes to the article branch, not the presence of a publisher
    # (both books and articles carry a "publisher"/addata.pub).
    assert "isbn" not in format_citation(ARTICLE, "chicago").lower()


def test_title_whitespace_from_source_data_is_stripped():
    assert "practice *" not in format_citation(BOOK, "apa")
    assert "practice*" in format_citation(BOOK, "apa")


def test_article_with_no_volume_issue_pages_never_leaks_a_literal_none():
    # Regression: confirmed live 2026-10-04 against a real journal-title record
    # (OneSearch can return a journal's own entry, not a specific article in it,
    # with journal_title set but no volume/issue/pages) — the Chicago formatter
    # used to interpolate a missing candidate.volume straight into an f-string,
    # rendering the literal word "None".
    candidate = _candidate(
        title="The journal of mental health policy and economics",
        journal_title="The journal of mental health policy and economics",
        authors=["Some Publisher, issuing body"],
    )
    for style in ("apa", "mla", "chicago"):
        citation = format_citation(candidate, style)
        assert "None" not in citation


def test_mla_article_with_no_pages_ends_cleanly_with_a_period_not_a_comma():
    # Regression: confirmed live 2026-10-04 — the year was always followed by a
    # comma assuming page numbers would come next; when pages are missing, the
    # citation dangled on a trailing comma with no closing period.
    candidate = _candidate(
        title="A Journal Entry", journal_title="Some Journal",
        authors=["Jones, Mary"], date="2020",
    )
    citation = format_citation(candidate, "mla")
    assert citation.endswith("2020.")
    assert not citation.endswith(",")
