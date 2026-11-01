import os


class AuthError(Exception):
    """Raised when the configured auth method can't be used at all (unset or
    unrecognized). Distinct from a wrong password, which just fails the check."""


class PasswordGate:
    """A swappable shared-password gate, keyed by two env vars: one naming the
    method ("password" is the only one implemented), one holding the secret.
    Fails closed: an unset or unrecognized method denies access rather than
    raising past the caller, so a misconfiguration never accidentally opens
    the gate. Adding real SSO later means adding a key to `_methods` on
    whichever gate needs it, not rewriting the page that uses it."""

    def __init__(self, method_env: str, password_env: str):
        self._method_env = method_env
        self._password_env = password_env
        self._methods = {"password": self._check_password}

    def _check_password(self, secret: str) -> bool:
        expected = os.getenv(self._password_env)
        if not expected:
            raise AuthError(f"{self._method_env}=password but {self._password_env} is not set.")
        return secret == expected

    def check_access(self, secret: str) -> bool:
        method = os.getenv(self._method_env, "").strip()
        check = self._methods.get(method)
        if check is None:
            return False
        try:
            return check(secret)
        except AuthError:
            return False

    def is_configured(self) -> bool:
        method = os.getenv(self._method_env, "").strip()
        if method not in self._methods:
            return False
        if method == "password":
            return bool(os.getenv(self._password_env))
        return True


# Two independent gates, each with its own env vars and secret - a cloud
# deploy's general access password is not the same credential as the staff
# dashboard's, and must stay swappable separately from it.
_staff_gate = PasswordGate("STAFF_DASHBOARD_AUTH_METHOD", "STAFF_DASHBOARD_PASSWORD")
_app_gate = PasswordGate("APP_ACCESS_METHOD", "APP_ACCESS_PASSWORD")


def check_staff_access(secret: str) -> bool:
    return _staff_gate.check_access(secret)


def staff_auth_configured() -> bool:
    return _staff_gate.is_configured()


def check_app_access(secret: str) -> bool:
    """Gates the whole chat app, not just the staff dashboard - needed once
    the app is reachable over a public URL rather than just localhost, since
    every chat turn spends a real OpenAI API key."""
    return _app_gate.check_access(secret)


def app_auth_configured() -> bool:
    return _app_gate.is_configured()
