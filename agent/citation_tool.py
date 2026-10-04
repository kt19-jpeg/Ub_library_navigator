from typing import Literal

from onesearch.models import SearchCandidate

CitationStyle = Literal["apa", "mla", "chicago"]

MAX_AUTHORS_IN_FULL = 2
"""Simplification, not full style-guide fidelity: real APA lists up to 20 authors
before "et al.", MLA/Chicago cut off differently. This project lists 1-2 authors
in full and uses "et al." from the 3rd author on, for every style."""


def _strip_marc_period(name: str) -> str:
    """Real source data can carry a trailing "." on a full first name as a MARC
    formatting artifact, not a true initial (confirmed live 2026-10-04: "Repper,
    Julie."). A real initial is one letter before the dot ("Julie K."); anything
    longer is the artifact, stripped so it doesn't look like a truncated name."""
    if name.endswith("."):
        word_before_dot = name[:-1].rsplit(" ", 1)[-1]
        if len(word_before_dot) > 1:
            return name[:-1]
    return name


def _split_name(full: str) -> tuple[str, str]:
    """"Last, First Middle" -> ("Last", "First Middle"). No comma -> (full, "")."""
    if "," in full:
        last, _, rest = full.partition(",")
        return last.strip(), _strip_marc_period(rest.strip())
    return full.strip(), ""


def _initials(first_middle: str) -> str:
    parts = first_middle.replace(".", " ").split()
    return " ".join(f"{p[0]}." for p in parts if p)


def _apa_name(full: str) -> str:
    last, first_middle = _split_name(full)
    initials = _initials(first_middle)
    return f"{last}, {initials}" if initials else last


def _natural_name(full: str) -> str:
    """"Last, First" -> "First Last", for a non-first MLA/Chicago author."""
    last, first_middle = _split_name(full)
    return f"{first_middle} {last}".strip() if first_middle else last


def _last_first_name(full: str) -> str:
    """"Last, First" kept as-is (trailing period artifacts from source data aside),
    for the first MLA/Chicago author."""
    last, first_middle = _split_name(full)
    return f"{last}, {first_middle}" if first_middle else last


def _authors_apa(authors: list[str]) -> str:
    if len(authors) == 1:
        return _apa_name(authors[0])
    if len(authors) == 2:
        return f"{_apa_name(authors[0])}, & {_apa_name(authors[1])}"
    return f"{_apa_name(authors[0])}, et al."


def _authors_mla_chicago(authors: list[str]) -> str:
    if len(authors) == 1:
        return _last_first_name(authors[0]) + "."
    if len(authors) == 2:
        return f"{_last_first_name(authors[0])}, and {_natural_name(authors[1])}."
    return f"{_last_first_name(authors[0])}, et al."


def _year(date: str | None) -> str:
    if date and len(date) >= 4 and date[:4].isdigit():
        return date[:4]
    return "n.d."


def _year_sentence(year: str) -> str:
    """`year` + a closing period, without doubling it when year is already
    "n.d." (which carries its own period)."""
    return year if year.endswith(".") else f"{year}."


def _join(*parts: str | None, sep: str = " ") -> str:
    """Joins only the pieces that are actually present, so a missing field never
    shows up as a literal "None" or a dangling separator."""
    return sep.join(p for p in parts if p)


def _is_article(candidate: SearchCandidate) -> bool:
    return bool(candidate.journal_title)


def _format_apa(candidate: SearchCandidate) -> str:
    authors = _authors_apa(candidate.authors) if candidate.authors else "Unknown author"
    year = _year(candidate.date)
    if _is_article(candidate):
        issue = f"({candidate.issue})" if candidate.issue else ""
        volume_issue = _join(f"*{candidate.volume}*" if candidate.volume else None, issue, sep="")
        locator = _join(volume_issue, candidate.pages, sep=", ")
        doi = f"https://doi.org/{candidate.doi}" if candidate.doi else None
        return _join(
            f"{authors} ({year}).",
            f"{candidate.title.strip()}.",
            _join(f"*{candidate.journal_title}*", locator, sep=", ") + ".",
            doi,
        )
    return _join(f"{authors} ({year}).", f"*{candidate.title.strip()}*.", f"{candidate.publisher}." if candidate.publisher else None)


def _format_mla(candidate: SearchCandidate) -> str:
    authors = _authors_mla_chicago(candidate.authors) if candidate.authors else "Unknown author."
    year = _year(candidate.date)
    if _is_article(candidate):
        vol = f"vol. {candidate.volume}," if candidate.volume else None
        no = f"no. {candidate.issue}," if candidate.issue else None
        pp = f"pp. {candidate.pages}." if candidate.pages else None
        year_piece = f"{year}," if pp else _year_sentence(year)
        return _join(authors, f'"{candidate.title.strip()}."', f"*{candidate.journal_title}*,", vol, no, year_piece, pp)
    return _join(authors, f"*{candidate.title.strip()}*.", f"{candidate.publisher}," if candidate.publisher else None, _year_sentence(year))


def _format_chicago(candidate: SearchCandidate) -> str:
    authors = _authors_mla_chicago(candidate.authors) if candidate.authors else "Unknown author."
    year = _year(candidate.date)
    if _is_article(candidate):
        no = f", no. {candidate.issue}" if candidate.issue else ""
        pp = f": {candidate.pages}" if candidate.pages else ""
        volume = f" {candidate.volume}" if candidate.volume else ""
        volume_part = f"*{candidate.journal_title}*{volume}{no} ({year}){pp}."
        return _join(authors, f'"{candidate.title.strip()}."', volume_part)
    publisher_part = _join(
        f"{candidate.place_of_publication}:" if candidate.place_of_publication else None,
        f"{candidate.publisher}," if candidate.publisher else None,
        _year_sentence(year),
    )
    return _join(authors, f"*{candidate.title.strip()}*.", publisher_part)


_FORMATTERS = {"apa": _format_apa, "mla": _format_mla, "chicago": _format_chicago}


def format_citation(candidate: SearchCandidate, style: CitationStyle) -> str:
    """A deterministic citation string in the given style. No LLM call. Missing
    fields are omitted cleanly rather than rendered as "None" or a stray comma."""
    return _FORMATTERS[style](candidate)
