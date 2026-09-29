import re

from bs4 import BeautifulSoup

from pipeline.records import DATABASE_SUBJECT, Record, make_unique_id

BROWSE_URL = "https://research.lib.buffalo.edu/az.php"
_COUNT_RE = re.compile(r"^(?P<subject>.+?)\s*\((?P<count>\d+)\)$")


def extract_database_subjects(html: str) -> list[Record]:
    """Reads the "Database Subject Filter" <select> on the A-Z Databases page:
    each <option> is "<subject> (<count>)". The individual database list itself
    is loaded by JavaScript from an endpoint robots.txt disallows, so only the
    subject categories and counts are indexed; BROWSE_URL is the one link given
    to the user to look at the actual databases."""
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
        records.append(
            Record(
                id=make_unique_id("db", subject, seen_ids),
                type=DATABASE_SUBJECT,
                text=f"{subject} databases ({count} available)",
                metadata={"subject": subject, "count": count, "browse_url": BROWSE_URL},
            )
        )
    return records
