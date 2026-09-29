from bs4 import BeautifulSoup

from pipeline.records import LIBRARIAN_SUBJECT, Record, make_unique_id


def extract_librarian_subjects(html: str) -> list[Record]:
    """Reads the subject -> librarian(s) table(s) on the "By Subject" ask-a-librarian
    page. A subject with more than one librarian (e.g. "Anatomy") becomes one record
    with multiple entries in metadata["librarians"]."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[Record] = []
    seen_ids: set[str] = set()
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 2:
            continue  # header rows ("Subject Area" / "Librarian Name") have <th>, not <td>
        subject = cells[0].get_text(strip=True)
        links = cells[1].find_all("a")
        if not subject or not links:
            continue
        librarians = [
            {"name": a.get_text(strip=True), "profile_url": a["href"]}
            for a in links
            if a.get_text(strip=True) and a.get("href")
        ]
        if not librarians:
            continue
        names = ", ".join(l["name"] for l in librarians)
        records.append(
            Record(
                id=make_unique_id("librarian", subject, seen_ids),
                type=LIBRARIAN_SUBJECT,
                text=f"{subject} subject librarian: {names}",
                metadata={"subject": subject, "librarians": librarians},
            )
        )
    return records
