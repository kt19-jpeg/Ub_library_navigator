import time
from collections.abc import Callable
from dataclasses import dataclass

SESSION_WINDOW_SECONDS = 15 * 60  # matches confirmed urm_st/urm_se gap: 900,000 ms
ONESEARCH_USER_AGENT = (
    "UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)"
)


@dataclass(frozen=True)
class SessionCookies:
    jsessionid: str
    urm_st: str
    urm_se: str
    institute: str


def mint_session_via_playwright() -> SessionCookies:
    """Loads the real OneSearch search page in headless Chromium so its own
    JS fires the request, satisfying bot-mitigation checks that a bare
    `requests` call fails (see CLAUDE.md finding 2). Route-intercepts to
    force skipDelivery=N and pcAvailability=true (finding 4), then extracts
    the session cookies the response sets."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(user_agent=ONESEARCH_USER_AGENT)

        def _force_delivery_params(route):
            request = route.request
            url = request.url.replace("skipDelivery=Y", "skipDelivery=N")
            if "pcAvailability=" not in url:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}pcAvailability=true"
            route.continue_(url=url)

        context.route("**/primaws/rest/pub/pnxs**", _force_delivery_params)
        page = context.new_page()
        # vid=01SUNY_BUF:everything confirmed live 2026-09-24 — the site
        # redirects a bare vid=01SUNY_BUF:UB request to this value; using
        # :UB directly 400s on the very first bootstrap call the page makes.
        # Wait for the page's own search response rather than "networkidle":
        # the site's analytics keep the network busy, so networkidle timed out
        # at 30s (confirmed live 2026-09-24) while this returns in 2-4s.
        with page.expect_response(
            lambda r: "/primaws/rest/pub/pnxs" in r.url and r.status == 200, timeout=30000
        ):
            page.goto(
                "https://search.lib.buffalo.edu/discovery/search?query=any,contains,test&vid=01SUNY_BUF:everything",
                wait_until="domcontentloaded",
            )

        cookies = {c["name"]: c["value"] for c in context.cookies()}
        browser.close()

    return SessionCookies(
        jsessionid=cookies["JSESSIONID"],
        urm_st=cookies["urm_st"],
        urm_se=cookies["urm_se"],
        institute=cookies["institute"],
    )


class SessionManager:
    def __init__(
        self,
        mint_fn: Callable[[], SessionCookies] = mint_session_via_playwright,
        session_window_seconds: float = SESSION_WINDOW_SECONDS,
    ):
        self._mint_fn = mint_fn
        self._session_window_seconds = session_window_seconds
        self._cookies: SessionCookies | None = None
        self._minted_at: float | None = None

    def _is_expired(self) -> bool:
        if self._cookies is None or self._minted_at is None:
            return True
        return (time.monotonic() - self._minted_at) > self._session_window_seconds

    def get_cookies(self) -> SessionCookies:
        if self._is_expired():
            self._cookies = self._mint_fn()
            self._minted_at = time.monotonic()
        return self._cookies
