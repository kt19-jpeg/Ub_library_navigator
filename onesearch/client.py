import concurrent.futures
import urllib.parse

import requests

from onesearch.cache import QueryCache
from onesearch.models import ResourceDetail, SearchCandidate, SearchResult
from onesearch.parser import parse_brief_response, parse_edelivery_response
from onesearch.session import ONESEARCH_USER_AGENT, SessionManager

BASE_URL = "https://search.lib.buffalo.edu/primaws/rest/pub/pnxs"
EDELIVERY_URL = "https://search.lib.buffalo.edu/primaws/rest/pub/edelivery"


def _build_edelivery_url(vid: str) -> str:
    # Confirmed live 2026-09-24: /edelivery only needs `vid` (no replay of
    # the search-context params). /delivery is not used at all: it 500s
    # outside the browser, and skipDelivery=N on the search call already
    # returns the same physical location data.
    return f"{EDELIVERY_URL}?{urllib.parse.urlencode({'vid': vid}, safe=':')}"


# Values confirmed live 2026-09-24 for facet_rtype; an unknown value does not
# error, it silently returns 0 hits, so callers are validated up front.
RESOURCE_TYPES = frozenset({"books", "articles", "journals", "dissertations", "videos"})


def _build_facets(resource_type: str | None, peer_reviewed: bool) -> str | None:
    facets = []
    if resource_type is not None:
        if resource_type not in RESOURCE_TYPES:
            raise ValueError(
                f"unknown resource_type {resource_type!r}; expected one of {sorted(RESOURCE_TYPES)}"
            )
        facets.append(f"facet_rtype,include,{resource_type}")
    if peer_reviewed:
        facets.append("facet_tlevel,include,peer_reviewed")
    return "|,|".join(facets) or None


def _build_search_url(query: str, vid: str, facets: str | None = None) -> str:
    params = {
        "q": f"any,contains,{query}",
        "vid": vid,
        "skipDelivery": "N",
        "pcAvailability": "true",
        "scope": "MyInst_and_CI",
        "tab": "Everything",
        "offset": "0",
        "limit": "10",
    }
    if facets:
        params["multiFacets"] = facets
    # requests' params= dict percent-encodes commas (%2C), which the
    # backend rejects. Build the query string manually and keep commas
    # literal (CLAUDE.md, confirmed finding 1).
    query_string = urllib.parse.urlencode(
        params, safe=",", quote_via=urllib.parse.quote
    )
    return f"{BASE_URL}?{query_string}"


class OneSearchClient:
    def __init__(
        self,
        session_manager: SessionManager,
        cache: QueryCache,
        http_get=requests.get,
        http_post=requests.post,
        vid: str = "01SUNY_BUF:everything",
    ):
        self._session_manager = session_manager
        self._cache = cache
        self._http_get = http_get
        self._http_post = http_post
        # Physical holdings arrive with search(); remember them so
        # fetch_detail() only needs the one extra call for online access.
        self._seen: dict[str, SearchCandidate] = {}
        self._vid = vid

    def search(
        self,
        query: str,
        resource_type: str | None = None,
        peer_reviewed: bool = False,
    ) -> list[SearchCandidate]:
        """`query` is keywords only (Layer 2 turns natural language into
        keywords + filters). `resource_type` is one of RESOURCE_TYPES;
        `peer_reviewed` restricts to peer-reviewed material."""
        facets = _build_facets(resource_type, peer_reviewed)
        cache_key = f"{query}|{facets or ''}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._remember(cached)
            return cached

        cookies = self._session_manager.get_cookies()
        url = _build_search_url(query, self._vid, facets)
        response = self._http_get(
            url,
            headers={"User-Agent": ONESEARCH_USER_AGENT},
            cookies={
                "JSESSIONID": cookies.jsessionid,
                "urm_st": cookies.urm_st,
                "urm_se": cookies.urm_se,
                "institute": cookies.institute,
            },
            timeout=10,
        )
        response.raise_for_status()
        candidates = parse_brief_response(response.json())
        self._cache.set(cache_key, candidates)
        self._remember(candidates)
        return candidates

    def _remember(self, candidates: list[SearchCandidate]) -> None:
        for candidate in candidates:
            self._seen[candidate.record_id] = candidate

    def fetch_detail(self, record_ids: list[str]) -> dict[str, ResourceDetail]:
        # Bounded by definition: callers (Layer 2) pass only the 1-3 ids
        # they decided to show. Never call this with a full result set
        # (CLAUDE.md: "never loop fetch_detail() over every search result").
        # One batched POST covers every id, so there is no per-id fan-out.
        if not record_ids:
            return {}
        cookies = self._session_manager.get_cookies()
        response = self._http_post(
            _build_edelivery_url(self._vid),
            json=[{"recId": rid, "sharedDigitalCandidates": None} for rid in record_ids],
            headers={"User-Agent": ONESEARCH_USER_AGENT},
            cookies={
                "JSESSIONID": cookies.jsessionid,
                "urm_st": cookies.urm_st,
                "urm_se": cookies.urm_se,
                "institute": cookies.institute,
            },
            timeout=10,
        )
        response.raise_for_status()
        online = parse_edelivery_response(response.json())

        details = {}
        for rid in record_ids:
            known = self._seen.get(rid)
            url, package = online.get(rid, (None, None))
            details[rid] = ResourceDetail(
                record_id=rid,
                online_link=url or (known.online_link if known else None),
                online_package=package,
                call_number=known.call_number if known else None,
                library=known.library if known else None,
                sub_location=known.sub_location if known else None,
                availability_status=known.availability_status if known else None,
            )
        return details

    def fallback_link(self, query: str) -> str:
        """Direct OneSearch results page for `query` (keywords only, no filters)."""
        return (
            "https://search.lib.buffalo.edu/discovery/search?"
            f"query=any,contains,{urllib.parse.quote(query)}&vid={self._vid}"
        )

    def record_link(self, record_id: str) -> str:
        """OneSearch full-record page for one record id."""
        return (
            "https://search.lib.buffalo.edu/discovery/fulldisplay?"
            f"docid={record_id}&context=L&vid={self._vid}"
        )

    def search_with_fallback(
        self,
        query: str,
        timeout_seconds: float = 9.0,
        resource_type: str | None = None,
        peer_reviewed: bool = False,
    ) -> SearchResult:
        # Validate before spawning the worker so a bad filter raises here
        # instead of surfacing as a swallowed future exception.
        _build_facets(resource_type, peer_reviewed)
        # Mint (or reuse) the session before starting the clock: a cold mint
        # takes a few seconds by itself and is not what this timeout bounds.
        self._session_manager.get_cookies()
        # No `with` block: its exit would wait for a slow worker and defeat the timeout.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self.search, query, resource_type, peer_reviewed)
            try:
                candidates = future.result(timeout=timeout_seconds)
                return SearchResult(candidates=candidates, timed_out=False, fallback_link=None)
            except concurrent.futures.TimeoutError:
                return SearchResult(
                    candidates=[], timed_out=True, fallback_link=self.fallback_link(query)
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
