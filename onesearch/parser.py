from onesearch.models import ResourceDetail, SearchCandidate


def _first(values: list | None) -> str | None:
    if not values:
        return None
    return values[0]


def _clean_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    return raw.split("$$Q")[0]


def parse_brief_doc(doc: dict) -> SearchCandidate:
    pnx = doc.get("pnx", {})
    display = pnx.get("display", {})
    addata = pnx.get("addata", {})
    control = pnx.get("control", {})
    delivery = doc.get("delivery", {})
    bestlocation = delivery.get("bestlocation", {})

    author = _clean_name(_first(display.get("creator")))
    if author is None:
        author = _clean_name(_first(display.get("contributor")))

    date = _first(display.get("creationdate")) or _first(addata.get("date"))

    return SearchCandidate(
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
        online_link=_first(delivery.get("link")),
        call_number=bestlocation.get("callNumber"),
        library=bestlocation.get("mainLocation"),
        sub_location=bestlocation.get("subLocation"),
        availability_status=bestlocation.get("itemAvailabilityStatus"),
        copies_available=bestlocation.get("number"),
    )


def parse_brief_response(response_json: dict) -> list[SearchCandidate]:
    return [parse_brief_doc(doc) for doc in response_json.get("docs", [])]


def parse_detail_response(record_id: str, response_json: dict) -> ResourceDetail:
    delivery = response_json.get("delivery", {})
    bestlocation = delivery.get("bestlocation", {})
    return ResourceDetail(
        record_id=record_id,
        online_link=_first(delivery.get("link")),
        call_number=bestlocation.get("callNumber"),
        library=bestlocation.get("mainLocation"),
        sub_location=bestlocation.get("subLocation"),
        availability_status=bestlocation.get("itemAvailabilityStatus"),
        copies_available=bestlocation.get("number"),
        raw=response_json,
    )
