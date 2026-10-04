from onesearch.models import SearchCandidate

SITE_ROOT = "https://search.lib.buffalo.edu"


def _first(values: list | None) -> str | None:
    if not values:
        return None
    return values[0]


def _clean_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    return raw.split("$$Q")[0]


def _absolute_url(url: str) -> str:
    # Resolver links come back host-relative ("/view/action/uresolver.do?...").
    return f"{SITE_ROOT}{url}" if url.startswith("/") else url


def _online_link_from_getit(delivery: dict) -> str | None:
    # `delivery.link[]` is NOT the access link: real records carry thumbnail
    # and "About requesting" entries there. The access link is in GetIt1
    # (confirmed against live data 2026-09-24).
    for group in delivery.get("GetIt1") or []:
        for link in group.get("links") or []:
            if link.get("isLinktoOnline") and link.get("link"):
                return _absolute_url(link["link"])
    return None


def _citation_authors(addata: dict) -> list[str] | None:
    # Confirmed live 2026-10-04: which key holds the author list varies by record,
    # not cleanly by type — a real journal article had "au" with "addau" absent;
    # a real book had "addau" with "au" absent. Both are already "Last, First"
    # strings (no $$Q markers), unlike display.creator.
    authors = addata.get("au") or addata.get("addau")
    return list(authors) if authors else None


def _citation_pages(addata: dict) -> str | None:
    spage = _first(addata.get("spage"))
    epage = _first(addata.get("epage"))
    if spage and epage:
        return f"{spage}-{epage}"
    if spage:
        return spage
    # Confirmed live 2026-10-04: a single-page article's "pages" value itself
    # carries a trailing "-" with no epage ("204801-"); strip it rather than
    # show a dangling dash in a citation.
    pages = _first(addata.get("pages"))
    return pages.rstrip("-") if pages else None


def parse_brief_doc(doc: dict) -> SearchCandidate:
    # Real responses can carry an explicit `null` for these keys (not just
    # omit them), and dict.get(key, default) only falls back to default
    # when the key is absent — confirmed against live data 2026-09-24.
    pnx = doc.get("pnx") or {}
    display = pnx.get("display") or {}
    addata = pnx.get("addata") or {}
    control = pnx.get("control") or {}
    delivery = doc.get("delivery") or {}
    bestlocation = delivery.get("bestlocation") or {}

    author = _clean_name(_first(display.get("creator")))
    if author is None:
        author = _clean_name(_first(display.get("contributor")))

    date = _first(display.get("creationdate")) or _first(addata.get("date"))

    return SearchCandidate(
        authors=_citation_authors(addata),
        journal_title=_first(addata.get("jtitle")),
        volume=_first(addata.get("volume")),
        issue=_first(addata.get("issue")),
        pages=_citation_pages(addata),
        doi=_first(addata.get("doi")),
        isbn=_first(addata.get("isbn")),
        publisher=_first(addata.get("pub")),
        place_of_publication=_first(addata.get("cop")),
        record_id=_first(control.get("recordid")),
        title=_first(display.get("title")),
        author=author,
        resource_type=_first(display.get("type")),
        format=_first(display.get("format")),
        subject=_first(display.get("subject")),
        description=_first(display.get("description")),
        date=date,
        source=_first(display.get("source")),
        delivery_category=_first(delivery.get("deliveryCategory")),
        online_link=_online_link_from_getit(delivery),
        call_number=bestlocation.get("callNumber"),
        library=bestlocation.get("mainLocation"),
        sub_location=bestlocation.get("subLocation"),
        availability_status=bestlocation.get("availabilityStatus"),
    )


def parse_brief_response(response_json: dict) -> list[SearchCandidate]:
    return [parse_brief_doc(doc) for doc in response_json.get("docs", [])]


def parse_edelivery_response(response_json: dict) -> dict[str, tuple[str, str | None]]:
    """Maps record_id -> (online_url, package_name) for records that have an
    electronic service. The envelope also carries non-record keys such as
    `beaconO22`; those and records with no electronic service are skipped."""
    access = {}
    for record_id, entry in response_json.items():
        if not isinstance(entry, dict):
            continue
        services = entry.get("electronicServices") or []
        url = _online_link_from_getit(entry) or next(
            (_absolute_url(s["serviceUrl"]) for s in services if s.get("serviceUrl")), None
        )
        if url is None:
            continue
        package = next((s.get("packageName") for s in services if s.get("packageName")), None)
        access[record_id] = (url, package)
    return access
