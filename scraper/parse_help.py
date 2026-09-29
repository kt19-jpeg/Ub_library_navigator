from bs4 import BeautifulSoup

from pipeline.records import HELP_TOPIC, Record, make_unique_id


def extract_help_topics(html: str) -> list[Record]:
    """Reads the Help A-Z topic index (div.help-browse > ul > li > a). Each entry is
    a title linking somewhere else on (or off) the library site; this only indexes
    the title and URL, it does not follow the link."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="help-browse")
    if container is None:
        return []
    records: list[Record] = []
    seen_ids: set[str] = set()
    for link in container.find_all("a"):
        title = link.get_text(strip=True)
        url = link.get("href")
        if not title or not url:
            continue
        records.append(
            Record(
                id=make_unique_id("help", title, seen_ids),
                type=HELP_TOPIC,
                text=title,
                metadata={"title": title, "url": url},
            )
        )
    return records
