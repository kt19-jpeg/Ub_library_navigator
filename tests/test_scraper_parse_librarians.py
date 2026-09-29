from pathlib import Path

from pipeline.records import LIBRARIAN_SUBJECT
from scraper.parse_librarians import extract_librarian_subjects

FIXTURE = (Path(__file__).parent / "fixtures/pages/librarians.html").read_text(errors="ignore")


def test_extracts_many_real_subjects():
    records = extract_librarian_subjects(FIXTURE)
    assert len(records) > 100  # confirmed live 2026-09-25: 154 <tr> rows across 21 letter tables
    assert all(r.type == LIBRARIAN_SUBJECT for r in records)


def test_single_librarian_subject():
    records = extract_librarian_subjects(FIXTURE)
    by_subject = {r.metadata["subject"]: r for r in records}
    art = by_subject["Art"]
    assert art.metadata["librarians"] == [
        {"name": "Kicey, Michael", "profile_url": "https://library.buffalo.edu/staff/mkicey"}
    ]
    assert art.text == "Art subject librarian: Kicey, Michael"


def test_subject_with_multiple_librarians():
    records = extract_librarian_subjects(FIXTURE)
    by_subject = {r.metadata["subject"]: r for r in records}
    anatomy = by_subject["Anatomy"]
    assert [l["name"] for l in anatomy.metadata["librarians"]] == ["Aronoff, Nell", "Mueller, Laurel"]
    assert anatomy.text == "Anatomy subject librarian: Aronoff, Nell, Mueller, Laurel"


def test_ids_are_stable_and_unique():
    records = extract_librarian_subjects(FIXTURE)
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))
    assert "librarian:anatomy" in ids


def test_ignores_rows_with_no_link_or_empty_subject():
    html = "<table><tr><th>Subject Area</th><th>Librarian Name</th></tr><tr><td>Empty</td><td></td></tr></table>"
    assert extract_librarian_subjects(html) == []


def test_empty_html_gives_no_records():
    assert extract_librarian_subjects("<html></html>") == []
