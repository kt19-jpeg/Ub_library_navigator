import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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


def _mint_timeout_ms() -> int:
    """How long to wait for the page's own search response. 180s default: a cold mint takes
    ~5s on a full CPU, ~22s at 0.25 CPU and ~100s at Render's free ~0.1 CPU (measured in Docker).
    Override with ONESEARCH_MINT_TIMEOUT_MS."""
    try:
        return int(os.getenv("ONESEARCH_MINT_TIMEOUT_MS", "180000"))
    except ValueError:
        return 180000


def mint_session_via_playwright() -> SessionCookies:
    """Loads the real OneSearch search page in headless Chromium so its own
    JS fires the request, satisfying bot-mitigation checks that a bare
    `requests` call fails (see CLAUDE.md finding 2). Route-intercepts to
    force skipDelivery=N and pcAvailability=true (finding 4), then extracts
    the session cookies the response sets."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        # --disable-dev-shm-usage: containers get a tiny /dev/shm, which crashes Chromium.
        browser = playwright.chromium.launch(args=["--disable-dev-shm-usage", "--disable-gpu"])
        context = browser.new_context(user_agent=ONESEARCH_USER_AGENT)
        # Only the page's own search request matters here, not how it looks. Skipping
        # images/fonts/media cuts the CPU work, which is what decides success on a
        # small host: Render's free plan (~0.1 CPU) timed out at 30s, 2026-11-12.
        context.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "font", "media")
            else route.fallback(),
        )

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
            lambda r: "/primaws/rest/pub/pnxs" in r.url and r.status == 200, timeout=_mint_timeout_ms()
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
        self._last_used: float | None = None
        # Single-flight: two visitors arriving together must not launch two Chromiums
        # (on a 512MB / ~0.1 CPU host that alone can kill the process).
        self._lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None

    def _is_expired(self) -> bool:
        if self._cookies is None or self._minted_at is None:
            return True
        return (time.monotonic() - self._minted_at) > self._session_window_seconds

    def _age(self) -> float:
        return float("inf") if self._minted_at is None else time.monotonic() - self._minted_at

    def _mint_locked(self) -> None:
        self._cookies = self._mint_fn()
        self._minted_at = time.monotonic()

    def get_cookies(self) -> SessionCookies:
        self._last_used = time.monotonic()
        if self._is_expired():
            with self._lock:
                if self._is_expired():  # another caller may have minted while we waited
                    self._mint_locked()
        return self._cookies

    def warm_up(
        self,
        refresh_margin_seconds: float = 240,
        idle_limit_seconds: float = 1800,
        poll_seconds: float = 30,
        retry_seconds: float = 120,
    ) -> None:
        """Mint now in a background thread, then keep the session fresh while the app is
        being used, so a search rarely waits for a mint. A cold mint takes ~100s on a
        ~0.1 CPU host (Render free plan), so the refresh starts `refresh_margin_seconds`
        before expiry - the old session stays valid and in use until the new one lands.
        Stops refreshing after `idle_limit_seconds` without a search (no pointless load on
        OneSearch). Failures are logged and retried; never raised. Safe to call twice."""
        if self._warm_thread is not None:
            return

        def loop():
            while True:
                try:
                    needs = self._cookies is None or (
                        self._age() > self._session_window_seconds - refresh_margin_seconds
                        and self._last_used is not None
                        and time.monotonic() - self._last_used < idle_limit_seconds
                    )
                    if needs and self._lock.acquire(blocking=False):
                        try:
                            self._mint_locked()
                            logger.info("OneSearch session minted (background)")
                        finally:
                            self._lock.release()
                    time.sleep(poll_seconds)
                except Exception as exc:
                    logger.warning("OneSearch background session mint failed: %r", exc)
                    time.sleep(retry_seconds)

        self._warm_thread = threading.Thread(target=loop, name="onesearch-session-warmup", daemon=True)
        self._warm_thread.start()
