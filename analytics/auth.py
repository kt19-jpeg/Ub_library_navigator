import os


class StaffAuthError(Exception):
    """Raised when the configured auth method can't be used at all (unset or
    unrecognized). Distinct from a wrong password, which just fails the check."""


def _check_password(secret: str) -> bool:
    expected = os.getenv("STAFF_DASHBOARD_PASSWORD")
    if not expected:
        raise StaffAuthError(
            "STAFF_DASHBOARD_AUTH_METHOD=password but STAFF_DASHBOARD_PASSWORD is not set."
        )
    return secret == expected


# Registry of swappable auth methods (CLAUDE.md: "the method is still pending
# confirmation from the library... build this behind a simple, swappable
# interface"). Adding real SSO later means adding a key here, not rewriting
# the dashboard page.
_METHODS = {
    "password": _check_password,
}


def check_staff_access(secret: str) -> bool:
    """True if `secret` grants staff access under the configured method.
    Fails closed: an unset or unrecognized STAFF_DASHBOARD_AUTH_METHOD denies
    access rather than raising past the dashboard page, so a misconfiguration
    never accidentally opens the gate."""
    method = os.getenv("STAFF_DASHBOARD_AUTH_METHOD", "").strip()
    check = _METHODS.get(method)
    if check is None:
        return False
    try:
        return check(secret)
    except StaffAuthError:
        return False


def staff_auth_configured() -> bool:
    """True if a recognized auth method is fully configured (used by the
    dashboard page to show a clear setup message instead of a silent lockout)."""
    method = os.getenv("STAFF_DASHBOARD_AUTH_METHOD", "").strip()
    if method not in _METHODS:
        return False
    if method == "password":
        return bool(os.getenv("STAFF_DASHBOARD_PASSWORD"))
    return True
