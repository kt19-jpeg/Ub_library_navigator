from pathlib import Path

from pipeline.records import DATABASE_SUBJECT
from scraper.parse_databases import BROWSE_URL, extract_database_subjects

FIXTURE = (Path(__file__).parent / "fixtures/pages/databases.html").read_text(errors="ignore")


def test_extracts_many_real_subjects():
    records = extract_database_subjects(FIXTURE)
    assert len(records) > 50  # confirmed live 2026-09-25: 70+ subject categories
    assert all(r.type == DATABASE_SUBJECT for r in records)


def test_a_known_subject_has_the_right_count_and_text():
    by_subject = {r.metadata["subject"]: r for r in extract_database_subjects(FIXTURE)}
    chemistry = by_subject["Chemistry"]
    assert chemistry.metadata["count"] == 32
    assert chemistry.text == "Chemistry databases (32 available)"
    assert chemistry.id == "db:chemistry"


def test_browse_url_is_subject_specific_not_the_blanket_link():
    # Regression: every subject used to get the exact same BROWSE_URL with no
    # query string - confirmed live 2026-11-03 that az.php?s=<id> genuinely
    # pre-filters server-side, so each subject should link one level deeper
    # than that, not all collapse to the identical generic page.
    by_subject = {r.metadata["subject"]: r for r in extract_database_subjects(FIXTURE)}
    chemistry_url = by_subject["Chemistry"].metadata["browse_url"]
    biology_url = by_subject["Biological Sciences"].metadata["browse_url"]
    assert chemistry_url == f"{BROWSE_URL}?s=94171"
    assert biology_url == f"{BROWSE_URL}?s=93976"
    assert chemistry_url != biology_url
    assert chemistry_url != BROWSE_URL


def test_falls_back_to_the_blanket_url_when_an_option_has_no_value():
    html = """
    <select id="s-lg-sel-subjects">
      <option>Chemistry (32)</option>
    </select>
    """
    records = extract_database_subjects(html)
    assert records[0].metadata["browse_url"] == BROWSE_URL


def test_all_subjects_placeholder_is_skipped():
    subjects = {r.metadata["subject"] for r in extract_database_subjects(FIXTURE)}
    assert "All Subjects" not in subjects


def test_ids_are_unique():
    records = extract_database_subjects(FIXTURE)
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))


def test_missing_select_gives_no_records():
    assert extract_database_subjects("<html></html>") == []
