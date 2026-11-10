import json

from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record
from pipeline.subject_keywords import (
    KeywordBatch,
    SubjectKeywords,
    enrich_records,
    generate_keywords,
    load_subject_keywords,
    subjects_in,
)
from tests.fakes import FakeChatModel


def _db(subject, text=None):
    return Record(
        id=f"db:{subject.lower()}", type=DATABASE_SUBJECT,
        text=text or f"{subject} databases (5 available)", metadata={"subject": subject},
    )


def test_enrich_rewords_the_embedded_text_of_database_and_librarian_records():
    librarian = Record(
        id="librarian:medicine", type=LIBRARIAN_SUBJECT,
        text="Medicine subject librarian: Doe, Jane", metadata={"subject": "Medicine"},
    )
    out = enrich_records([_db("Medicine"), librarian], {"Medicine": ["oncology", "immunology"]})

    # boilerplate (count, librarian name) is dropped from the embedded text; it stays in metadata
    assert out[0].text == "Medicine: oncology, immunology"
    assert out[1].text == "Medicine: oncology, immunology"


def test_enrich_leaves_records_without_keywords_and_other_types_alone():
    help_page = Record(id="help:a", type=HELP_TOPIC, text="Citing", metadata={"subject": "Medicine"})
    records = [_db("Law"), help_page]

    out = enrich_records(records, {"Medicine": ["oncology"]})

    assert out == records  # Law has no keywords; a help page is never enriched


def test_enrich_never_mutates_the_originals_so_it_cannot_double_apply():
    original = _db("Medicine")
    enrich_records([original], {"Medicine": ["oncology"]})
    assert original.text == "Medicine databases (5 available)"


def test_changed_text_changes_the_content_hash_so_it_gets_re_embedded():
    original = _db("Medicine")
    enriched = enrich_records([original], {"Medicine": ["oncology"]})[0]
    assert enriched.content_hash != original.content_hash


def test_load_returns_empty_for_a_missing_or_corrupt_file(tmp_path):
    assert load_subject_keywords(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_subject_keywords(bad) == {}


def test_load_reads_a_real_file(tmp_path):
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"Law": ["contracts"]}))
    assert load_subject_keywords(path) == {"Law": ["contracts"]}


def test_subjects_in_is_the_sorted_unique_union_of_both_subject_types():
    librarian = Record(id="l:1", type=LIBRARIAN_SUBJECT, text="x", metadata={"subject": "Art"})
    assert subjects_in([_db("Law"), _db("Art"), librarian]) == ["Art", "Law"]


def test_generate_keeps_only_requested_subjects_and_cleans_keywords():
    model = FakeChatModel(KeywordBatch(items=[
        SubjectKeywords(subject="Medicine", keywords=["Oncology ", "oncology", "", " cancer,", "x" * 3]),
        SubjectKeywords(subject="Invented Subject", keywords=["nope"]),
    ]))

    result = generate_keywords(["Medicine", "Law"], model)

    assert result == {"Medicine": ["Oncology", "cancer", "xxx"]}  # deduped, trimmed; invented dropped; Law missing


def test_generate_caps_the_keyword_count():
    model = FakeChatModel(KeywordBatch(items=[
        SubjectKeywords(subject="Medicine", keywords=[f"term{i}" for i in range(40)]),
    ]))
    assert len(generate_keywords(["Medicine"], model)["Medicine"]) == 25


def test_generate_skips_a_failed_batch_instead_of_aborting_the_run():
    model = FakeChatModel(RuntimeError("provider down"), KeywordBatch(items=[
        SubjectKeywords(subject="Law", keywords=["contracts"]),
    ]))

    result = generate_keywords(["Medicine", "Law"], model, batch_size=1)

    assert result == {"Law": ["contracts"]}
