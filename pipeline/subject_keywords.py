"""Related-topic keywords for subject-style records (database + librarian subjects).

Why this exists: those records are embedded as only "<Subject> databases (N available)"
or "<Subject> subject librarian: <Name>". That is almost no meaning for a topical query,
so "cancer immunotherapy" landed closer to "Complementary Medicine" (it shares the word
"therapy") than to "Medicine" - confirmed in a 13-query audit on 2026-11-10. Appending a
short list of real topics/sub-fields per subject gives the embedding something to match.

The keywords are generated once by an LLM and committed (data/subject_keywords.json) so
they are reviewable in a diff and cost nothing at refresh time or in CI. They are applied
only at embedding time (enrich_records), never written into the scraped snapshot, so the
snapshot stays exactly what the sites said and enrichment can never be applied twice.

Regenerate/extend: `python -m pipeline.subject_keywords` (only subjects missing from the
file are generated; `--force` redoes all)."""

import json
import sys
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.records import DATABASE_SUBJECT, LIBRARIAN_SUBJECT, Record

KEYWORDS_PATH = Path("data/subject_keywords.json")
ENRICHED_TYPES = (LIBRARIAN_SUBJECT, DATABASE_SUBJECT)
MAX_KEYWORDS = 25
BATCH_SIZE = 20

GENERATION_SYSTEM = """You help a university library's search tool understand what its subject areas cover.

You are given the names of subject areas used by the University at Buffalo Libraries (for subject-librarian assignments and database lists). For EACH one, return 15-25 short keywords: specific sub-fields, named topics, methods, conditions/technologies/periods/schools of thought as relevant, and the common search terms or synonyms a student or researcher would use for that subject.

Prefer SPECIFIC over generic. For "Medicine", good keywords are: oncology, cancer, immunology, cardiology, neurology, infectious disease, clinical trials, surgery, pharmacology, epidemiology, radiology. Bad keywords are generic words that fit anything: research, diagnosis, treatment, healthcare, practice.

Rules:
- For a broad subject (Medicine, Biology, Engineering, Law, Business...), cover its major branches and specialties and the main topic areas inside it, not just a few generic ones.
- Base keywords on what the subject area realistically covers in a library. Some names are places, collections, or formats (e.g. "Buffalo, NY", "Maps", "Streaming Video", "Love Canal"): for those, give the topics that material actually concerns or is about.
- If you are not sure what a name covers, return only the keywords you are confident about rather than guessing.
- Return one item per subject, with the subject name copied exactly as given."""


class SubjectKeywords(BaseModel):
    subject: str = Field(description="The subject name, copied exactly as given.")
    keywords: list[str] = Field(description="15-25 short, specific topical keywords for this subject.")


class KeywordBatch(BaseModel):
    items: list[SubjectKeywords] = Field(default_factory=list)


def load_subject_keywords(path: Path = KEYWORDS_PATH) -> dict[str, list[str]]:
    """Missing or unreadable file just means no enrichment - never an error."""
    try:
        data = json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def enrich_records(records: list[Record], keywords: dict[str, list[str]]) -> list[Record]:
    """Re-words the embedded text of database/librarian subject records that have keywords
    as "<Subject>: <keyword>, <keyword>, ...". The scraped boilerplate it replaces
    ("databases (32 available)", a librarian's name) carries nothing about topic and only
    dilutes the embedding; everything useful (counts, names, URLs) stays in metadata.
    Returns new Records; the originals are untouched."""
    enriched = []
    for record in records:
        if record.type in ENRICHED_TYPES:
            subject = record.metadata.get("subject", "")
            related = keywords.get(subject)
            if related:
                record = replace(record, text=f"{subject}: {', '.join(related)}")
        enriched.append(record)
    return enriched


def subjects_in(records: list[Record]) -> list[str]:
    names = {
        r.metadata.get("subject")
        for r in records
        if r.type in ENRICHED_TYPES and r.metadata.get("subject")
    }
    return sorted(names)


def _clean(raw_keywords: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for keyword in raw_keywords:
        keyword = " ".join(str(keyword).split()).strip(" ,.;")
        if keyword and keyword.lower() not in seen:
            seen.add(keyword.lower())
            cleaned.append(keyword)
    return cleaned[:MAX_KEYWORDS]


def generate_keywords(subjects: list[str], model, batch_size: int = BATCH_SIZE) -> dict[str, list[str]]:
    """One structured LLM call per batch. Only subjects that were asked for are accepted
    (an invented or renamed subject is dropped), and a failed batch is skipped rather
    than aborting the run - the caller reports what is still missing."""
    result: dict[str, list[str]] = {}
    structured = model.with_structured_output(KeywordBatch)
    for start in range(0, len(subjects), batch_size):
        batch = subjects[start : start + batch_size]
        try:
            response = structured.invoke([("system", GENERATION_SYSTEM), ("human", "\n".join(batch))])
            if isinstance(response, dict):
                response = KeywordBatch.model_validate(response)
        except Exception as exc:
            print(f"warning: batch starting at {batch[0]!r} failed: {exc}", file=sys.stderr)
            continue
        wanted = set(batch)
        for item in response.items:
            if item.subject in wanted:
                cleaned = _clean(item.keywords)
                if cleaned:
                    result[item.subject] = cleaned
    return result


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from agent.llm import LLMConfigError, get_chat_model
    from pipeline.snapshot import load_snapshot

    load_dotenv()
    force = "--force" in (argv if argv is not None else sys.argv[1:])
    snapshot = load_snapshot()
    if not snapshot:
        print("No snapshot found: run `python -m pipeline.refresh` first.", file=sys.stderr)
        return 2
    existing = {} if force else load_subject_keywords()
    missing = [s for s in subjects_in(snapshot) if s not in existing]
    if not missing:
        print(f"All {len(existing)} subjects already have keywords.")
        return 0
    try:
        model = get_chat_model()
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    generated = generate_keywords(missing, model)
    merged = {**existing, **generated}
    KEYWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEYWORDS_PATH.write_text(json.dumps(dict(sorted(merged.items())), indent=2, ensure_ascii=False) + "\n")
    still_missing = [s for s in missing if s not in generated]
    print(f"Generated keywords for {len(generated)}/{len(missing)} subjects; {len(merged)} total saved.")
    if still_missing:
        print(f"Still missing ({len(still_missing)}): {still_missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
