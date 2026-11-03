import re

from bs4 import BeautifulSoup

from pipeline.records import DATABASE_SUBJECT, Record, make_unique_id

BROWSE_URL = "https://research.lib.buffalo.edu/az.php"
_COUNT_RE = re.compile(r"^(?P<subject>.+?)\s*\((?P<count>\d+)\)$")


def extract_database_subjects(html: str) -> list[Record]:
    """Reads the "Database Subject Filter" <select> on the A-Z Databases page:
    each <option value="<subject_id>"> is "<subject> (<count>)". The individual
    database list itself is loaded by JavaScript from an endpoint robots.txt
    disallows ("Disallow: /az.php?"), so we never fetch it ourselves - but that
    same disallow rule is proof the query-string form is a real, working URL on
    their site, just one we don't crawl. Confirmed live 2026-11-03: GETting
    az.php?s=<subject_id> with plain curl (no JS) comes back with that option
    already marked selected="selected" server-side, so linking a human straight
    to a subject-filtered view - never fetching it with our own scraper, robots.txt
    only restricts that - is "one level deeper" than the one blanket link every
    subject used to get, at zero scraping-politeness cost."""
    soup = BeautifulSoup(html, "html.parser")
    select = soup.find("select", id="s-lg-sel-subjects")
    if select is None:
        return []
    records: list[Record] = []
    seen_ids: set[str] = set()
    for option in select.find_all("option"):
        label = option.get_text(strip=True)
        match = _COUNT_RE.match(label)
        if not match:
            continue  # e.g. "All Subjects", which has no count
        subject = match.group("subject")
        count = int(match.group("count"))
        subject_id = (option.get("value") or "").strip()
        browse_url = f"{BROWSE_URL}?s={subject_id}" if subject_id else BROWSE_URL
        records.append(
            Record(
                id=make_unique_id("db", subject, seen_ids),
                type=DATABASE_SUBJECT,
                text=f"{subject} databases ({count} available)",
                metadata={"subject": subject, "count": count, "browse_url": browse_url},
            )
        )
    return records
