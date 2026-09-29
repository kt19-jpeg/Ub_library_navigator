from pathlib import Path

from pipeline.records import HELP_TOPIC
from scraper.parse_help import extract_help_topics

FIXTURE = (Path(__file__).parent / "fixtures/pages/help.html").read_text(errors="ignore")


def test_extracts_many_real_topics():
    records = extract_help_topics(FIXTURE)
    assert len(records) > 80  # confirmed live 2026-09-25: 100+ topics on the page
    assert all(r.type == HELP_TOPIC for r in records)


def test_a_known_topic_has_the_right_title_and_url():
    by_title = {r.metadata["title"]: r for r in extract_help_topics(FIXTURE)}
    citing = by_title["Citing Sources"]
    assert citing.metadata["url"] == "http://research.lib.buffalo.edu/citingsources"
    assert citing.text == "Citing Sources"
    assert citing.id == "help:citing-sources"


def test_ids_are_unique():
    records = extract_help_topics(FIXTURE)
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))


def test_missing_container_gives_no_records():
    assert extract_help_topics("<html><body>nothing here</body></html>") == []


def test_link_with_no_text_is_skipped():
    html = '<div class="help-browse"><ul><li><a href="https://x"></a></li></ul></div>'
    assert extract_help_topics(html) == []
